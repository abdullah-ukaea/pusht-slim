import math
import os
from datetime import datetime
import time

import wandb
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
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
CHECKPOINT_DIR = "checkpoints"
BATCH_SIZE = 64
WANDB_PROJECT = "pushT-slim"
WANDB_ENTITY = "robot_learning_collective"

# PushT positions and actions are pixel coordinates in [0, 512]
# We normalize everything the model sees to [-1, 1]
# so it lives at the same scale as the Gaussian noise used by flow matching,
# then unnormalize the model's output before sending it back to the env.
COORD_MIN, COORD_MAX = 0.0, 512.0

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
        # env obs dict -> batched, normalized (images, states) on device
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
        self.valid_indices = self._determine_valid_indices()

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
        print("Done caching.")

    def _determine_valid_indices(self):
        valid_indices = []
        episode_indices = np.array(self.dataset.hf_dataset["episode_index"])

        for idx in range(len(self.dataset)):
            episode = episode_indices[idx]

            # Need enough future actions for one predicted horizon.
            future_idx = idx + self.prediction_horizon - 1
            if future_idx >= len(self.dataset):
                continue

            # and those future frames must be in the same episode
            future_episode = episode_indices[future_idx].item()
            if future_episode == episode:
                valid_indices.append(idx)

        return valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]

        obs = self.cache[start_idx]["observation.state"]
        actions = torch.stack(
            [
                self.cache[i]["action"]
                for i in range(start_idx, start_idx + self.prediction_horizon)
            ]
        )
        images = self.cache[start_idx]["observation.image"]

        # normalize positions and actions to [-1, 1]; images are already in [0, 1]
        obs = normalize(obs)
        actions = normalize(actions)

        return {"images": images, "obs": obs, "actions": actions}


# =============================================================================
# 3. Model
# =============================================================================
class ImageEncoder(nn.Module):
    def __init__(self, out_dim):
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

        resnet.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )

        resnet.maxpool = nn.Identity()

        feat_dim = resnet.fc.in_features  # 512 for resnet34
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        self.fc = nn.Linear(feat_dim, out_dim)

    def forward(self, images):
        # images: (batch, 3, 96, 96)
        images = (images - self.img_mean) / self.img_std
        
        x = self.backbone(images)
        x = x.flatten(1) 
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

    def training_step(self, images, obs, actions):
        t = torch.rand(actions.shape[0]).to(actions.device)
        noise = torch.randn_like(actions)
        x_t = (1 - t[:, None, None]) * noise + t[:, None, None] * actions

        v_pred = self.forward(x_t, t, images, obs)
        target = actions - noise  
        return torch.nn.functional.mse_loss(v_pred, target)

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

        images = batch["images"].to(device)
        obs = batch["obs"].to(device)
        actions = batch["actions"].to(device)

        loss = policy.training_step(images, obs, actions)
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
    run_name = f"pusht-dit-{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}"

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
        name=run_name,
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