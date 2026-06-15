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
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import gymnasium as gym
import gym_pusht  

N_OBS = 2
N_ACTIONS = 16
N_ACTION_STEPS = 8  
DATASET_ID = "lerobot/pusht"
N_HEADS = 4
N_LAYERS = 2
HIDDEN_DIM = 128
N_EPOCHS = 150
LR = 1e-4
SAVE_EVERY = 50
EVAL_EVERY = 1
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
def evaluate(policy, device, n_episodes=10, render=False):
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
            actions = policy.inference(images_in, states_in)  # (1, n_actions, 2)
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
class ImageEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc = nn.Linear(15488, out_dim)

    def forward(self, images):
        # images: (batch, n_obs, 3, 96, 96)
        b, n, c, h, w = images.shape
        images = images.view(b * n, c, h, w)  
        x = self.cnn(images)
        x = self.fc(x)
        x = x.view(b, n, -1)  # (batch, n_obs, out_dim)
        return x.mean(dim=1)  # average over the observation frames


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
        return x.mean(dim=1)


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
        self.adaLN_modulation = nn.Linear(hidden_dim * 3, hidden_dim * 6)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        # zero-init the modulation so each block starts as an identity function
        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)

    def forward(self, x, cond):
        # x: (batch, n_actions, hidden_dim), cond: (batch, hidden_dim * 3)
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
    def __init__(self, hidden_dim=HIDDEN_DIM, n_heads=N_HEADS, n_layers=N_LAYERS):
        super().__init__()
        self.image_encoder = ImageEncoder(out_dim=hidden_dim)
        self.timestep_encoder = TimestepEncoder(out_dim=hidden_dim)
        self.state_encoder = StateEncoder(out_dim=hidden_dim)

        self.action_proj = nn.Linear(2, hidden_dim)
        self.action_out = nn.Linear(hidden_dim, 2)

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

        x = self.action_proj(x_t)
        for block in self.blocks:
            x = block(x, cond)
        return self.action_out(x)

    def training_step(self, images, obs, actions):
        # corrupt the clean actions to a random point on the noise->action line,
        # then learn to predict that line's (constant) velocity
        t = torch.rand(actions.shape[0]).to(actions.device)
        noise = torch.randn_like(actions)
        x_t = (1 - t[:, None, None]) * noise + t[:, None, None] * actions

        v_pred = self.forward(x_t, t, images, obs)
        target = actions - noise  # true velocity of the straight path
        return torch.nn.functional.mse_loss(v_pred, target)

    @torch.no_grad()
    def inference(self, images, obs, n_steps=10):
        # start from pure noise and follow the predicted velocity field to t=1
        batch_size = images.shape[0]
        x = torch.randn(batch_size, N_ACTIONS, 2).to(images.device)

        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((batch_size,), i / n_steps).to(images.device)
            v = self.forward(x, t, images, obs)
            x = x + v * dt
        return x  # normalized actions in [-1, 1]


# =============================================================================
# 4. Training
# =============================================================================
def train(
    policy,
    dataloader,
    n_epochs=N_EPOCHS,
    lr=LR,
    save_every=SAVE_EVERY,
    eval_every=EVAL_EVERY,
    checkpoint_dir=CHECKPOINT_DIR,
):
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Training on {device}")
    policy = policy.to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(n_epochs):
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"epoch {epoch}")
        for batch in pbar:
            images = batch["images"].to(device)
            obs = batch["obs"].to(device)
            actions = batch["actions"].to(device)

            loss = policy.training_step(images, obs, actions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        log_dict = {"epoch": epoch, "avg_loss": avg_loss}
        print(f"epoch {epoch}: avg_loss={avg_loss:.4f}")

        if epoch % eval_every == 0 or epoch == n_epochs - 1:
            success_rate = evaluate(policy, device)
            log_dict["success_rate"] = success_rate
            print(f"epoch {epoch}: success_rate={success_rate:.2f}")

        wandb.log(log_dict)

        if epoch % save_every == 0 or epoch == n_epochs - 1:
            checkpoint_path = os.path.join(
                checkpoint_dir, f"checkpoint_epoch_{epoch}.pt"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": policy.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                },
                checkpoint_path,
            )
            print(f"Saved checkpoint to {checkpoint_path}")


# =============================================================================
# 5. Main
# =============================================================================
def main():
    run_name = f"pusht-dit-{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}"
    wandb.init(
        project="pusht-slim",
        name=run_name,
        config={
            "n_epochs": N_EPOCHS,
            "lr": LR,
            "n_layers": N_LAYERS,
            "n_heads": N_HEADS,
            "hidden_dim": HIDDEN_DIM,
            "n_obs": N_OBS,
            "n_actions": N_ACTIONS,
            "n_action_steps": N_ACTION_STEPS,
            "batch_size": BATCH_SIZE,
        },
    )

    policy = DiTPolicy()
    dataset = PushTDataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    train(policy=policy, dataloader=dataloader)

    wandb.finish()


if __name__ == "__main__":
    main()