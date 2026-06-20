import math
import os
from datetime import datetime
import time

import wandb
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import gymnasium as gym
import gym_pusht  

N_OBS = 1
N_ACTIONS = 16
N_ACTION_STEPS = 8
DATASET_ID = "lerobot/pusht"
# DiT sized to ~47.5M (matching lerobot's DiT backbone): hidden 600, 6 layers, 8 heads
N_HEADS = 8
N_LAYERS = 6
HIDDEN_DIM = 600
COND_DIM = HIDDEN_DIM * (2 * N_OBS + 1)
BACKBONE = "resnet34"
TOTAL_STEPS = 200_000
LR = 1e-4
LOG_EVERY = 200
EVAL_EVERY = 10_000
SAVE_EVERY = 10_000
CHECKPOINT_DIR = "checkpoints"
BATCH_SIZE = 64

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
def evaluate(policy, device, n_episodes=10, render=False, n_steps=10):
    render_mode = "human" if render else "rgb_array"
    env = gym.make(
        "gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode=render_mode
    )
    policy.eval()

    successes = 0
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False

        # build initial observation buffers (state is normalized to match training)
        image = torch.from_numpy(obs["pixels"]).permute(2, 0, 1).float() / 255.0
        state = normalize(torch.from_numpy(obs["agent_pos"]).float())

        # stack n_obs frames (repeat first frame to fill the history buffer)
        images_buf = image.unsqueeze(0).repeat(N_OBS, 1, 1, 1)  # (n_obs, 3, 96, 96)
        states_buf = state.unsqueeze(0).repeat(N_OBS, 1)         # (n_obs, 2)

        info = {}
        while not done:
            images_in = images_buf.unsqueeze(0).to(device)  # (1, n_obs, 3, 96, 96)
            states_in = states_buf.unsqueeze(0).to(device)  # (1, n_obs, 2)

            # generate a chunk of actions (in normalized [-1, 1] space)
            actions = policy.inference(images_in, states_in, n_steps=n_steps)  # (1, n_actions, 2)
            # back to the env's [0, 512] coordinate space
            actions = unnormalize(actions).squeeze(0).cpu().numpy()  # (n_actions, 2)

            # receding horizon: execute only the first N_ACTION_STEPS, then replan
            for action in actions[:N_ACTION_STEPS]:
                obs, reward, terminated, truncated, info = env.step(action)
                if render:
                    time.sleep(0.05)
                done = terminated or truncated

                # update the observation history after every executed step
                image = torch.from_numpy(obs["pixels"]).permute(2, 0, 1).float() / 255.0
                state = normalize(torch.from_numpy(obs["agent_pos"]).float())
                images_buf = torch.roll(images_buf, -1, dims=0)
                images_buf[-1] = image
                states_buf = torch.roll(states_buf, -1, dims=0)
                states_buf[-1] = state

                if done:
                    break

        if info.get("is_success", False):
            successes += 1

    env.close()
    policy.train()
    return successes / n_episodes


# =============================================================================
# 2. Dataset
# =============================================================================
class PushTDataset(Dataset):
    def __init__(self, dataset_id=DATASET_ID, n_obs=N_OBS, n_actions=N_ACTIONS):
        self.dataset = LeRobotDataset(dataset_id)
        self.n_obs = n_obs
        self.n_actions = n_actions
        self.valid_indices = self._determine_valid_indices()

        # decode every frame once and keep it in RAM; afterwards each __getitem__
        print("Caching dataset in memory...")
        self.cache = {}
        for idx in tqdm(range(len(self.dataset))):
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
        frame_indices = np.array(self.dataset.hf_dataset["frame_index"])

        for idx in range(len(self.dataset)):
            episode = episode_indices[idx]
            frame = frame_indices[idx]

            # need n_obs frames behind this one
            if frame < self.n_obs - 1:
                continue

            # need n_actions frames ahead of this one
            future_idx = idx + self.n_actions - 1
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

        obs = torch.stack(
            [
                self.cache[i]["observation.state"]
                for i in range(start_idx - self.n_obs + 1, start_idx + 1)
            ]
        )
        actions = torch.stack(
            [
                self.cache[i]["action"]
                for i in range(start_idx, start_idx + self.n_actions)
            ]
        )
        images = torch.stack(
            [
                self.cache[i]["observation.image"]
                for i in range(start_idx - self.n_obs + 1, start_idx + 1)
            ]
        )

        # normalize positions and actions to [-1, 1]; images are already in [0, 1]
        obs = normalize(obs)
        actions = normalize(actions)

        return {"images": images, "obs": obs, "actions": actions}


# =============================================================================
# 3. Model
# =============================================================================
_RESNETS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
    "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT),
}


class ImageEncoder(nn.Module):
    def __init__(self, out_dim, backbone="resnet18"):
        super().__init__()
        ctor, weights = _RESNETS[backbone]
        resnet = ctor(weights=weights)

        resnet.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )

        resnet.maxpool = nn.Identity()

        feat_dim = resnet.fc.in_features  # 512 for resnet18/34
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        self.fc = nn.Linear(feat_dim, out_dim)

    def forward(self, images):
        # images: (batch, n_obs, 3, 96, 96)
        b, n, c, h, w = images.shape
        images = images.view(b * n, c, h, w)  

        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        images = (images - mean) / std
        
        x = self.backbone(images)
        x = x.flatten(1) 
        x = self.fc(x)
        x = x.view(b, n, -1)  # (batch, n_obs, out_dim)
        return x.reshape(b, n * x.shape[-1]) 


class StateEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim)
        )

    def forward(self, state):
        # state: (batch, n_obs, 2)
        b, n, s = state.shape
        states = state.view(b * n, s)
        x = self.mlp(states)
        x = x.view(b, n, -1)
        return x.reshape(b, n * x.shape[-1]) 


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
        # x: (batch, n_actions, hidden_dim), cond: (batch, hidden_dim)
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
    def __init__(self, hidden_dim=HIDDEN_DIM, cond_dim=None, n_heads=N_HEADS, n_layers=N_LAYERS, n_actions=N_ACTIONS, backbone=BACKBONE):
        super().__init__()
        # conditioning = image + state (each n_obs frames) + timestep, all hidden_dim wide
        if cond_dim is None:
            cond_dim = hidden_dim * (2 * N_OBS + 1)
        self.image_encoder = ImageEncoder(out_dim=hidden_dim, backbone=backbone)
        self.timestep_encoder = TimestepEncoder(out_dim=hidden_dim)
        self.state_encoder = StateEncoder(out_dim=hidden_dim)

        self.action_proj = nn.Linear(2, hidden_dim)
        self.action_out = nn.Linear(hidden_dim, 2)

        self.action_pos_emb = nn.Parameter(torch.zeros(1, n_actions, hidden_dim))

        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim * 4),
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
        # x_t: (batch, n_actions, 2), t: (batch,)
        # images: (batch, n_obs, 3, 96, 96), obs: (batch, n_obs, 2)
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
    def inference(self, images, obs, n_steps=10):
        batch_size = images.shape[0]
        x = torch.randn(batch_size, N_ACTIONS, 2).to(images.device)

        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((batch_size,), i / n_steps).to(images.device)
            v = self.forward(x, t, images, obs)
            x = x + v * dt
        return x.clamp(-1.0, 1.0) 


# =============================================================================
# 4. Training
# =============================================================================
def run_eval(policy, device, global_step):
    eval_dict = {}
    for n_steps in [1, 2, 5, 10, 20, 50]:
        success_rate = evaluate(policy, device, n_episodes=20, n_steps=n_steps)
        eval_dict[f"success_rate/n_steps_{n_steps}"] = success_rate
        print(f"step {global_step}: n_steps={n_steps}, success_rate={success_rate:.2f}")
    return eval_dict


def save_checkpoint(policy, optimizer, global_step, loss, checkpoint_dir):
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{global_step}.pt")
    torch.save(
        {
            "step": global_step,
            "model_state_dict": policy.state_dict(),
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
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    os.makedirs(checkpoint_dir, exist_ok=True)

    global_step = 0
    running_loss = 0.0
    running_count = 0
    last_loss = 0.0
    pbar = tqdm(total=total_steps, desc="train")
    done = False
    while not done:
        for batch in dataloader:
            images = batch["images"].to(device)
            obs = batch["obs"].to(device)
            actions = batch["actions"].to(device)

            loss = policy.training_step(images, obs, actions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            last_loss = loss.item()
            running_loss += last_loss
            running_count += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{last_loss:.4f}")

            # collect everything due on this step into one log call so wandb sees
            # a single record per step (log and eval cadences can coincide)
            log_dict = {}
            if global_step % log_every == 0:
                avg_loss = running_loss / running_count
                log_dict["avg_loss"] = avg_loss
                print(f"step {global_step}: avg_loss={avg_loss:.4f}")
                running_loss = 0.0
                running_count = 0

            if global_step % eval_every == 0:
                log_dict.update(run_eval(policy, device, global_step))

            if log_dict:
                wandb.log(log_dict, step=global_step)

            if global_step % save_every == 0:
                save_checkpoint(
                    policy, optimizer, global_step, last_loss, checkpoint_dir
                )

            if global_step >= total_steps:
                done = True
                break

    # final eval + checkpoint, unless the last step already triggered them
    final_dict = {}
    if total_steps % eval_every != 0:
        final_dict.update(run_eval(policy, device, global_step))
    if running_count > 0:
        final_dict["avg_loss"] = running_loss / running_count
    if final_dict:
        wandb.log(final_dict, step=global_step)
    if total_steps % save_every != 0:
        save_checkpoint(policy, optimizer, global_step, last_loss, checkpoint_dir)
    pbar.close()


# =============================================================================
# 5. Main
# =============================================================================
WANDB_PROJECT = "pushT-slim"
WANDB_ENTITY = "robot_learning_collective"


def main():
    run_name = f"pusht-dit-{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}"

    policy = DiTPolicy()
    dataset = PushTDataset()
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
            "save_every": SAVE_EVERY,
            "lr": LR,
            "n_layers": N_LAYERS,
            "n_heads": N_HEADS,
            "hidden_dim": HIDDEN_DIM,
            "backbone": BACKBONE,
            "n_obs": N_OBS,
            "n_actions": N_ACTIONS,
            "n_action_steps": N_ACTION_STEPS,
            "batch_size": BATCH_SIZE,
        },
    )

    train(policy=policy, dataloader=dataloader)

    wandb.finish()


if __name__ == "__main__":
    main()