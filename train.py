import math
import os
from datetime import datetime
import time

import wandb
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as T
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import gymnasium as gym
import gym_pusht  

ROBOT_DOF = 2
PREDICTION_HORIZON = 16
ACTION_CHUNK_SIZE = 8
N_DENOISING_STEPS = 10
DATASET_ID = "lerobot/pusht"
# DiT sized to ~47.5M (matching lerobot's DiT backbone): hidden 600, 6 layers, 8 heads
N_HEADS = 8
N_LAYERS = 6
HIDDEN_DIM = 600
TOTAL_STEPS = 200_000
LR = 1e-4
LOG_EVERY = 200
EVAL_EVERY = 25_000
EVAL_EPISODES = 20
SAVE_EVERY = 25_000
RUN_NAME = f"pusht-dit-{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}"
CHECKPOINT_DIR = f"checkpoints_{RUN_NAME}"
BATCH_SIZE = 64
WANDB_PROJECT = "pushT-slim"
WANDB_ENTITY = "robot_learning_collective"

# PushT positions and actions are pixel coordinates in [0, 512]
# We normalize everything the model sees to [-1, 1]
# so it lives at the same scale as the Gaussian noise used by flow matching,
# then unnormalize the model's output before sending it back to the env.
COORD_MIN, COORD_MAX = 0.0, 512.0
# PushT frames are 96x96. lerobot-style: crop at native resolution (no upsample)
# and let SpatialSoftmax pool the small feature map (see ImageEncoder).
NATIVE_IMG_SIZE = 96
CROP_SIZE = 84  # random crop (train) / center crop (eval), ~0.875 of native
NUM_KEYPOINTS = 32  # SpatialSoftmax keypoints
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

def normalize(x):
    return 2.0 * (x - COORD_MIN) / (COORD_MAX - COORD_MIN) - 1.0


def unnormalize(x):
    return (x + 1.0) / 2.0 * (COORD_MAX - COORD_MIN) + COORD_MIN


# =============================================================================
# 1. Environment
# =============================================================================
class PushTAdapter:
    """Translates between the PushT env and the policy.

    Keeps all PushT/coordinate specifics (obs layout, normalization, tensor
    plumbing) in one place so the eval loop can stay env-agnostic: it only
    sees model-ready tensors going in and env-ready actions coming out.
    """

    def __init__(self, device):
        self.device = device

    def observe(self, obs):
        # env obs dict -> batched, normalized (images, states) on device.
        # Feed the raw native-resolution frame; the encoder owns cropping (center
        # crop in eval, random crop in train) so train and eval share one path.
        image = torch.from_numpy(obs["pixels"]).permute(2, 0, 1).float() / 255.0
        state = normalize(torch.from_numpy(obs["agent_pos"]).float())
        return image.unsqueeze(0).to(self.device), state.unsqueeze(0).to(self.device)

    def act(self, actions):
        # policy output in [-1, 1] -> env actions in [0, 512]
        return unnormalize(actions).squeeze(0).cpu().numpy()


def evaluate(policy, device, n_episodes, render, n_steps):
    render_mode = "human" if render else "rgb_array"
    env = gym.make(
        "gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode=render_mode
    )
    adapter = PushTAdapter(device)
    policy.eval()

    successes = 0
    max_rewards = []
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        max_reward = -float("inf")

        info = {}
        while not done:
            images_in, states_in = adapter.observe(obs)

            # Predict a full action horizon, then map it back to env actions.
            actions = policy.inference(images_in, states_in, n_steps=n_steps)
            actions = adapter.act(actions)

            # Receding horizon: execute one chunk, then replan from the new state.
            for action in actions[:ACTION_CHUNK_SIZE]:
                obs, reward, terminated, truncated, info = env.step(action)
                max_reward = max(max_reward, float(reward))
                if render:
                    time.sleep(0.05)
                done = terminated or truncated
                if done:
                    break

        if info.get("is_success", False):
            successes += 1
        max_rewards.append(max_reward)

    env.close()
    policy.train()
    return {
        "success_rate": successes / n_episodes,
        "avg_max_reward": sum(max_rewards) / n_episodes,
    }


# =============================================================================
# 2. Dataset
# =============================================================================
class PushTDataset(Dataset):
    def __init__(self, dataset_id, prediction_horizon):
        self.dataset = LeRobotDataset(dataset_id)
        self.prediction_horizon = prediction_horizon

        # decode every frame once and keep it in RAM; afterwards each __getitem__
        print("Caching dataset in memory...")
        self.cache = {}
        for idx in range(len(self.dataset)):
            item = self.dataset[idx]
            self.cache[idx] = {
                "observation.image": item["observation.image"],
                "observation.state": item["observation.state"],
                "action": item["action"],
            }
        self.episode_indices = np.array(self.dataset.hf_dataset["episode_index"])
        print("Done caching.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        current_episode = self.episode_indices[idx]

        # Return the raw native-resolution frame ([0, 1], 3x96x96). Cropping and
        # normalization happen inside the encoder so train/eval stay in sync.
        img = self.cache[idx]["observation.image"]

        obs = normalize(self.cache[idx]["observation.state"])

        actions = torch.zeros(self.prediction_horizon, 2)
        mask = torch.zeros(self.prediction_horizon, 2)
        
        for k in range(self.prediction_horizon):
            i = idx + k
            if i < len(self.dataset) and self.episode_indices[i] == current_episode:
                actions[k] = normalize(self.cache[i]["action"])
                mask[k] = 1.0

        return img, obs, actions.flatten(), mask.flatten()

# =============================================================================
# 3. Model
# =============================================================================
class SpatialSoftmax(nn.Module):
    """Spatial soft-argmax pooling (Finn et al. 2015), ported from lerobot/robomimic.

    Turns a (B, C, H, W) feature map into (B, K, 2) keypoint coordinates: the
    softmax-weighted "center of mass" of each channel's activations. Unlike global
    average pooling, this preserves *where* features fire, so it stays informative
    even on the tiny feature maps a stock ResNet produces from small inputs (an
    84x84 crop -> 512x3x3). A 1x1 conv first remaps C -> num_kp keypoint channels.
    """

    def __init__(self, input_shape, num_kp=None):
        super().__init__()
        self._in_c, self._in_h, self._in_w = input_shape
        if num_kp is not None:
            self.nets = nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c

        pos_x, pos_y = np.meshgrid(
            np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h)
        )
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1)).float()
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forward(self, features):
        if self.nets is not None:
            features = self.nets(features)
        batch = features.shape[0]
        features = features.reshape(-1, self._in_h * self._in_w)
        attention = F.softmax(features, dim=-1)
        keypoints = attention @ self.pos_grid  # (B * out_c, 2)
        return keypoints.reshape(batch, self._out_c, 2)


class ImageEncoder(nn.Module):
    """lerobot/Diffusion-Policy-style vision encoder.

    Solves the "small image + ResNet built for 224" problem the way lerobot does:
    keep the stock ResNet stem, feed a native-resolution crop (no upsampling), and
    let SpatialSoftmax pool the resulting tiny feature map into keypoints — instead
    of preserving resolution via a modified stem or upsampling the input to 224.
    Cropping lives here (random in train, center in eval) so train and eval share
    a single code path.
    """

    def __init__(self, out_dim, crop_size=CROP_SIZE, num_kp=NUM_KEYPOINTS):
        super().__init__()
        weights = models.ResNet34_Weights.DEFAULT
        resnet = models.resnet34(weights=weights)

        # Reuse the exact normalization the pretrained weights expect, sourced
        # from the weights metadata so it can't drift out of sync.
        preprocess = weights.transforms()
        self.register_buffer(
            "img_mean", torch.tensor(preprocess.mean).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "img_std", torch.tensor(preprocess.std).view(1, 3, 1, 1)
        )

        # Same crop for the whole batch (torchvision RandomCrop), matching lerobot.
        self.random_crop = T.RandomCrop(crop_size)
        self.center_crop = T.CenterCrop(crop_size)

        # Drop avgpool + fc; keep the stock stem and conv stages -> feature map.
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # Dry run to get the feature-map shape that SpatialSoftmax must index.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, crop_size, crop_size)
            feat_shape = self.backbone(dummy).shape[1:]  # (C, H, W)
        self.pool = SpatialSoftmax(feat_shape, num_kp=num_kp)
        self.fc = nn.Linear(num_kp * 2, out_dim)

    def forward(self, images):
        # images: (batch, 3, 96, 96) in [0, 1]
        images = self.random_crop(images) if self.training else self.center_crop(images)
        images = (images - self.img_mean) / self.img_std

        x = self.backbone(images)
        x = self.pool(x).flatten(1)  # (batch, num_kp * 2)
        return self.fc(x)


class StateEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(ROBOT_DOF, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim)
        )

    def forward(self, state):
        # state: (batch, ROBOT_DOF)
        return self.mlp(state)


# turns the scalar flow-time t into a vector "fingerprint" the network can use,
# rather than a single float which is hard for an MLP to interact with
class TimestepEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.mlp = nn.Sequential(
            nn.Linear(out_dim, out_dim * 4), nn.GELU(), nn.Linear(out_dim * 4, out_dim)
        )

    def forward(self, t):
        # t: (batch,)
        half = self.out_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half) / half).to(t.device)
        x = t[:, None] * freqs[None, :]  # (batch, half)
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)  # (batch, out_dim)
        return self.mlp(x)


class AdaLNBlock(nn.Module):
    def __init__(self, hidden_dim, n_heads):
        super().__init__()
        # turns the conditioning vector into scale1, shift1, alpha1, scale2, shift2, alpha2
        self.adaLN_modulation = nn.Linear(hidden_dim, hidden_dim * 6)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)

    def forward(self, x, cond):
        # x: (batch, PREDICTION_HORIZON, hidden_dim), cond: (batch, hidden_dim)
        scale1, shift1, alpha1, scale2, shift2, alpha2 = self.adaLN_modulation(
            cond
        ).chunk(6, dim=-1)

        # attention sub-block with adaptive LayerNorm
        x1 = self.norm1(x)
        x1 = scale1.unsqueeze(1) * x1 + shift1.unsqueeze(1)
        x1, _ = self.attention(x1, x1, x1)
        x = x + alpha1.unsqueeze(1) * x1

        # MLP sub-block with adaptive LayerNorm
        x2 = self.norm2(x)
        x2 = scale2.unsqueeze(1) * x2 + shift2.unsqueeze(1)
        x = x + alpha2.unsqueeze(1) * self.mlp(x2)
        return x


class DiTPolicy(nn.Module):
    def __init__(
        self,
        hidden_dim=HIDDEN_DIM,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        prediction_horizon=PREDICTION_HORIZON,
    ):
        super().__init__()
        # conditioning = image + state + timestep, all hidden_dim wide
        self.image_encoder = ImageEncoder(out_dim=hidden_dim)
        self.timestep_encoder = TimestepEncoder(out_dim=hidden_dim)
        self.state_encoder = StateEncoder(out_dim=hidden_dim)

        self.action_proj = nn.Linear(ROBOT_DOF, hidden_dim)
        self.action_out = nn.Linear(hidden_dim, ROBOT_DOF)

        self.action_pos_emb = nn.Parameter(
            torch.zeros(1, prediction_horizon, hidden_dim)
        )

        self.cond_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        self.blocks = nn.ModuleList(
            [
                AdaLNBlock(hidden_dim=hidden_dim, n_heads=n_heads)
                for _ in range(n_layers)
            ]
        )

    def forward(self, x_t, t, images, obs):
        # pure network: noisy actions + time + context -> predicted velocity
        # x_t: (batch, PREDICTION_HORIZON, ROBOT_DOF), t: (batch,)
        # images: (batch, 3, 96, 96), obs: (batch, ROBOT_DOF)
        images_cond = self.image_encoder(images)
        obs_cond = self.state_encoder(obs)
        timestep_cond = self.timestep_encoder(t)
        cond = torch.cat([images_cond, obs_cond, timestep_cond], dim=-1)
        cond = self.cond_proj(cond)

        x = self.action_proj(x_t)
        x = x + self.action_pos_emb
        
        for block in self.blocks:
            x = block(x, cond)
        return self.action_out(x)

    def training_step(self, images, obs, actions, masks):
        B = actions.shape[0]
        actions = actions.reshape(B, PREDICTION_HORIZON, ROBOT_DOF)
        masks = masks.reshape(B, PREDICTION_HORIZON, ROBOT_DOF)

        t = torch.rand(B).to(actions.device)
        noise = torch.randn_like(actions)
        x_t = (1 - t[:, None, None]) * noise + t[:, None, None] * actions

        v_pred = self.forward(x_t, t, images, obs)
        loss = torch.nn.functional.mse_loss(v_pred, actions - noise, reduction="none")
        # Masked mean over valid (in-episode) timesteps. Averaging — rather than
        # summing over the horizon — keeps the loss on the same scale as an
        # unmasked mean, so it's comparable across runs and doesn't inflate the
        # effective learning rate by ~PREDICTION_HORIZON.
        return (loss * masks).sum() / masks.sum().clamp(min=1)

    @torch.no_grad()
    def inference(self, images, obs, n_steps=N_DENOISING_STEPS):
        batch_size = images.shape[0]
        x = torch.randn(batch_size, PREDICTION_HORIZON, ROBOT_DOF).to(images.device)

        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((batch_size,), i / n_steps).to(images.device)
            v = self.forward(x, t, images, obs)
            x = x + v * dt
        return x.clamp(-1.0, 1.0) 


# =============================================================================
# 4. Training
# =============================================================================
def save_checkpoint(policy, optimizer, global_step, loss, checkpoint_dir):
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{global_step}.pt")

    # torch.compile wraps the model and stores the real model in _orig_mod.
    # Save the real model so checkpoint keys match a normal, uncompiled model.
    model = policy._orig_mod if hasattr(policy, "_orig_mod") else policy
    torch.save(
        {
            "step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to {checkpoint_path}")


def train(
    policy,
    dataloader,
    total_steps=TOTAL_STEPS,
    lr=LR,
    log_every=LOG_EVERY,
    eval_every=EVAL_EVERY,
    save_every=SAVE_EVERY,
    checkpoint_dir=CHECKPOINT_DIR,
):
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Training on {device}")
    policy = policy.to(device)
    policy = torch.compile(policy)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    os.makedirs(checkpoint_dir, exist_ok=True)

    last_loss = 0.0
    global_step = 0
    dataloader_iter = iter(dataloader)

    for global_step in range(1, total_steps + 1):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)

        images, obs, actions, mask = [t.to(device) for t in batch]

        loss = policy.training_step(images, obs, actions, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        last_loss = loss.item()

        # collect everything due on this step into one log call so wandb sees
        # a single record per step (log and eval cadences can coincide)
        log_dict = {}
        if global_step % log_every == 0:
            log_dict["loss"] = last_loss
            print(f"step {global_step}: loss={last_loss:.4f}")

        if global_step % eval_every == 0:
            log_dict.update(evaluate(
                policy, device, n_episodes=EVAL_EPISODES, render=False,
                n_steps=N_DENOISING_STEPS,
            ))

        if log_dict:
            wandb.log(log_dict, step=global_step)

        if global_step % save_every == 0:
            save_checkpoint(
                policy, optimizer, global_step, last_loss, checkpoint_dir
            )

    # final eval + checkpoint, unless the last step already triggered them
    final_dict = {}
    if total_steps % eval_every != 0:
        final_dict.update(evaluate(
            policy, device, n_episodes=EVAL_EPISODES, render=False,
            n_steps=N_DENOISING_STEPS,
        ))
    if final_dict:
        wandb.log(final_dict, step=global_step)
    if total_steps % save_every != 0:
        save_checkpoint(policy, optimizer, global_step, last_loss, checkpoint_dir)

def main():

    policy = DiTPolicy()
    dataset = PushTDataset(
        dataset_id=DATASET_ID,
        prediction_horizon=PREDICTION_HORIZON,
    )
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )

    steps_per_epoch = len(dataloader)
    print(
        f"steps/epoch={steps_per_epoch}, training for {TOTAL_STEPS} total steps "
        f"(~{math.ceil(TOTAL_STEPS / steps_per_epoch)} passes over the data)"
    )

    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=RUN_NAME,
        config={
            "total_steps": TOTAL_STEPS,
            "steps_per_epoch": steps_per_epoch,
            "log_every": LOG_EVERY,
            "eval_every": EVAL_EVERY,
            "eval_episodes": EVAL_EPISODES,
            "save_every": SAVE_EVERY,
            "lr": LR,
            "n_layers": N_LAYERS,
            "n_heads": N_HEADS,
            "hidden_dim": HIDDEN_DIM,
            "backbone": "resnet34",
            "pooling": "spatial_softmax",
            "crop_size": CROP_SIZE,
            "num_keypoints": NUM_KEYPOINTS,
            "robot_dof": ROBOT_DOF,
            "prediction_horizon": PREDICTION_HORIZON,
            "action_chunk_size": ACTION_CHUNK_SIZE,
            "n_denoising_steps": N_DENOISING_STEPS,
            "batch_size": BATCH_SIZE,
        },
    )

    train(policy=policy, dataloader=dataloader)

    wandb.finish()


if __name__ == "__main__":
    main()