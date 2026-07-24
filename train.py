import math
import os
from datetime import datetime
import time

import wandb
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.io import read_video
from huggingface_hub import hf_hub_download
import gymnasium as gym
import gym_pusht

ROBOT_DOF = 2
PREDICTION_HORIZON = 16
ACTION_CHUNK_SIZE = 8
N_DENOISING_STEPS = 10
DATASET_ID = "lerobot/pusht"
# DiT sized to ~200M: hidden 896, 12 layers, 8 heads. Model scale was the
# dominant success-rate lever in our experiments: ~47M plateaued at SR ~0.5,
# 200M reaches ~0.6-0.7.
N_HEADS = 8
N_LAYERS = 12
HIDDEN_DIM = 896
TOTAL_STEPS = 100_000
LR = 1e-4
WARMUP_STEPS = 500   # linear LR warmup
LR_MIN = 1e-6        # cosine decay floor
GRAD_CLIP = 10.0  # max grad norm; the pre-clip norm is logged to watch for instability
LOG_EVERY = 200
EVAL_EVERY = 10_000
EVAL_EPISODES = 50  # SR noise at 20 episodes was +-0.11; 50 brings it to ~+-0.07
EVAL_VIDEOS = 3     # rollout videos logged to wandb per eval
SAVE_EVERY = 20_000
RUN_NAME = f"exp-dinov2-vits14-200m-100k-{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}"
CHECKPOINT_DIR = f"checkpoints_{RUN_NAME}"
BATCH_SIZE = 512
NUM_WORKERS = 8  # dataloader workers
WANDB_PROJECT = "pushT-slim"
WANDB_ENTITY = "robot_learning_collective"

# PushT frames are uint8; positions and actions are pixel coordinates in
# [0, 512]. The policy consumes this raw data and emits pixel-space actions —
# all normalization lives inside the model: images are scaled to ImageNet
# stats (see ImageEncoder), coordinates are mapped to [-1, 1] so they live at
# the same scale as the Gaussian noise used by flow matching.
COORD_MIN, COORD_MAX = 0.0, 512.0
# PushT frames are 96x96; crop at native resolution (no upsample) and let
# SpatialSoftmax pool the small feature map (see ImageEncoder).
BACKBONE = "dinov2_vits14"  # DINOv2 ViT-S/14, self-supervised pretrained
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
def evaluate(policy, device, n_episodes=EVAL_EPISODES, n_videos=EVAL_VIDEOS):
    env = gym.make(
        "gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array"
    )
    policy.eval()

    successes = 0
    max_rewards = []
    videos = []  # (T, H, W, C) uint8 render frames of the first n_videos episodes
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        max_reward = -float("inf")
        frames = [env.render()]  # 680x680 rgb_array

        info = {}
        while not done:
            # env obs dict -> batched tensors on device; the policy takes raw
            # uint8 frames and pixel-space state, and returns pixel-space actions
            image = torch.from_numpy(obs["pixels"]).permute(2, 0, 1)  # HWC -> CHW
            state = torch.from_numpy(obs["agent_pos"]).float()
            actions = policy.inference(
                image[None].to(device), state[None].to(device)
            )
            actions = actions.squeeze(0).cpu().numpy()

            # Receding horizon: execute one chunk, then replan from the new state.
            for action in actions[:ACTION_CHUNK_SIZE]:
                obs, reward, terminated, truncated, info = env.step(action)
                max_reward = max(max_reward, float(reward))
                frames.append(env.render())
                done = terminated or truncated
                if done:
                    break

        if info.get("is_success", False):
            successes += 1
        max_rewards.append(max_reward)
        if episode < n_videos:
            videos.append(np.stack(frames))

    fps = env.metadata["render_fps"]
    env.close()
    policy.train()
    metrics = {
        "success_rate": successes / n_episodes,
        "avg_max_reward": sum(max_rewards) / n_episodes,
    }
    if videos:
        # All videos under one key -> one wandb panel (a key per video makes
        # one panel per video). wandb.Video wants (T, C, H, W).
        metrics["eval_videos"] = [
            wandb.Video(v.transpose(0, 3, 1, 2), fps=fps, format="mp4")
            for v in videos
        ]
    print(
        f"[eval] success_rate={metrics['success_rate']:.3f} "
        f"avg_max_reward={metrics['avg_max_reward']:.3f} (n_episodes={n_episodes})"
    )
    return metrics


# =============================================================================
# 2. Dataset
# =============================================================================
class PushTDataset(Dataset):
    """PushT, loaded straight from the HF hub without lerobot.

    The dataset (v3 format) is just two files: one MP4 with every frame of
    every episode concatenated in order, and one parquet with the matching
    per-frame state/action/episode metadata (row i <-> video frame i). We
    decode the whole video once up front (~7s) and keep everything in RAM
    (~0.7 GB as uint8), so __getitem__ is pure tensor slicing. Data is served
    raw; the model owns all normalization.
    """

    def __init__(self, dataset_id, prediction_horizon):
        video_path = hf_hub_download(
            dataset_id, "videos/observation.image/chunk-000/file-000.mp4",
            repo_type="dataset",
        )
        meta_path = hf_hub_download(
            dataset_id, "data/chunk-000/file-000.parquet", repo_type="dataset"
        )

        meta = pd.read_parquet(meta_path)
        states = torch.from_numpy(np.stack(meta["observation.state"])).float()
        actions = torch.from_numpy(np.stack(meta["action"])).float()
        episode_ids = torch.tensor(meta["episode_index"].to_numpy())

        print("Decoding video...")
        frames, _, _ = read_video(video_path, pts_unit="sec", output_format="TCHW")
        assert len(frames) == len(meta), (
            f"video has {len(frames)} frames but metadata has {len(meta)} rows"
        )
        print(f"Done: {len(frames)} frames.")

        self.prediction_horizon = prediction_horizon
        self.images = frames  # uint8, native resolution
        self.states = states
        self.actions = actions
        self.episode_ids = episode_ids

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Future actions over the horizon, zero-padded and masked out past the
        # end of the episode. Episodes are contiguous, so the frames still in
        # idx's episode are exactly the prefix of the window.
        end = min(idx + self.prediction_horizon, len(self))
        n = int((self.episode_ids[idx:end] == self.episode_ids[idx]).sum())

        actions = torch.zeros(self.prediction_horizon, ROBOT_DOF)
        mask = torch.zeros(self.prediction_horizon, ROBOT_DOF)
        actions[:n] = self.actions[idx : idx + n]
        mask[:n] = 1.0

        return self.images[idx], self.states[idx], actions, mask

# =============================================================================
# 3. Model
# =============================================================================
class ImageEncoder(nn.Module):
    """Vision encoder: DINOv2 ViT-S/14 (self-supervised pretrained), fine-tuned
    end-to-end.

    The 84x84 crop is exactly 6x14 pixels, so the ViT sees a 6x6 patch grid with
    no resizing. Its 36 patch tokens are reshaped back into a (embed_dim, 6, 6)
    feature map and pooled by SpatialSoftmax -> keypoints -> linear. Cropping
    lives here (random in train, center in eval) so train and eval share a
    single code path.
    """

    def __init__(self, out_dim, crop_size=CROP_SIZE, num_kp=NUM_KEYPOINTS):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", BACKBONE)
        self.patch_size = self.backbone.patch_size  # 14
        assert crop_size % self.patch_size == 0, "crop must be a multiple of the ViT patch size"
        self._grid = crop_size // self.patch_size  # 6

        # ImageNet stats, matching DINOv2's pretraining normalization.
        self.register_buffer("img_mean", torch.tensor(IMG_MEAN).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor(IMG_STD).view(1, 3, 1, 1))

        # Same crop window for the whole batch (matching lerobot's RandomCrop).
        # We index with tensors instead of torchvision's RandomCrop because the
        # latter calls torch.randint(...).item(), and that .item() forces a
        # CUDA sync + a torch.compile graph break at the top of the model.
        self.crop_size = crop_size
        self.center_crop = T.CenterCrop(crop_size)
        
        self.fc = nn.Linear(self.backbone.embed_dim, out_dim)

    def random_crop(self, images):
        # One random crop window for the whole batch, expressed entirely in
        # tensor ops (no .item()) so it stays inside the compiled graph.
        _, _, h, w = images.shape
        size = self.crop_size
        top = torch.randint(0, h - size + 1, (1,), device=images.device)
        left = torch.randint(0, w - size + 1, (1,), device=images.device)
        rows = top + torch.arange(size, device=images.device)
        cols = left + torch.arange(size, device=images.device)
        return images[:, :, rows[:, None], cols[None, :]]

    def forward(self, images):
        # images: (batch, 3, 96, 96) uint8
        images = images.float() / 255.0
        images = self.random_crop(images) if self.training else self.center_crop(images)
        images = (images - self.img_mean) / self.img_std

        # (B, 36, embed_dim) patch tokens -> (B, embed_dim, 6, 6) feature map
        tokens = self.backbone.forward_features(images)["x_norm_patchtokens"]
        x = tokens.mean(dim=1)
        return self.fc(x)


class StateEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(ROBOT_DOF, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim)
        )

    def forward(self, state):
        # state: (batch, ROBOT_DOF) in pixel coordinates
        return self.mlp(normalize(state))


# turns the scalar flow-time t into a vector "fingerprint" the network can use,
# rather than a single float which is hard for an MLP to interact with
class TimestepEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(out_dim, out_dim * 4), nn.GELU(), nn.Linear(out_dim * 4, out_dim)
        )
        half = out_dim // 2
        self.register_buffer(
            "freqs", torch.exp(-math.log(10000) * torch.arange(half) / half)
        )

    def forward(self, t):
        # t: (batch,)
        x = t[:, None] * self.freqs[None, :]  # (batch, half)
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
        x1 = (1 + scale1.unsqueeze(1)) * x1 + shift1.unsqueeze(1)
        x1, _ = self.attention(x1, x1, x1)
        x = x + alpha1.unsqueeze(1) * x1

        # MLP sub-block with adaptive LayerNorm
        x2 = self.norm2(x)
        x2 = (1 + scale2.unsqueeze(1)) * x2 + shift2.unsqueeze(1)
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

    def encode_observation(self, images, obs):
        images_cond = self.image_encoder(images)
        obs_cond = self.state_encoder(obs)
        return images_cond, obs_cond

    def vector_field(self, x_t, t, images_cond, obs_cond):
        timestep_cond = self.timestep_encoder(t)
        cond = torch.cat([images_cond, obs_cond, timestep_cond], dim=-1)
        cond = self.cond_proj(cond)

        x = self.action_proj(x_t)
        x = x + self.action_pos_emb

        for block in self.blocks:
            x = block(x, cond)
        return self.action_out(x)

    def forward(self, x_t, t, images, obs):
        # pure network: noisy actions + time + context -> predicted velocity
        # x_t: (batch, PREDICTION_HORIZON, ROBOT_DOF) in [-1, 1], t: (batch,)
        # images: (batch, 3, 96, 96) uint8, obs: (batch, ROBOT_DOF) in pixels
        images_cond, obs_cond = self.encode_observation(images, obs)
        return self.vector_field(x_t, t, images_cond, obs_cond)

    @torch.no_grad()
    def inference(self, images, obs, n_steps=N_DENOISING_STEPS):
        batch_size, device = images.shape[0], images.device
        x = torch.randn(batch_size, PREDICTION_HORIZON, ROBOT_DOF, device=device)
        images_cond, obs_cond = self.encode_observation(images, obs)

        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((batch_size,), i / n_steps, device=device)
            v = self.vector_field(x, t, images_cond, obs_cond)
            x = x + v * dt
        return unnormalize(x.clamp(-1.0, 1.0))  # -> pixel-space actions


# =============================================================================
# 4. Training
# =============================================================================
def save_checkpoint(policy, optimizer, scheduler, global_step, loss):
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_step_{global_step}.pt")

    # torch.compile wraps the model and stores the real model in _orig_mod.
    # Save the real model so checkpoint keys match a normal, uncompiled model.
    model = policy._orig_mod if hasattr(policy, "_orig_mod") else policy
    torch.save(
        {
            "step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": loss,
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to {checkpoint_path}")


# Kept as a free function rather than a DiTPolicy method on purpose: we compile
# the module (torch.compile(policy)) and call policy(...) directly, so the loss
# must live outside forward. A method would route through the uncompiled
# self.forward, making the compile a no-op.
def flow_matching_loss(policy, images, obs, actions, masks):
    # actions, masks: (batch, PREDICTION_HORIZON, ROBOT_DOF); actions in pixels
    actions = normalize(actions)  # the flow runs at noise scale, in [-1, 1]
    t = torch.rand(len(actions), device=actions.device)
    noise = torch.randn_like(actions)
    x_t = (1 - t[:, None, None]) * noise + t[:, None, None] * actions
    v_pred = policy(x_t, t, images, obs)
    loss = F.mse_loss(v_pred, actions - noise, reduction="none")
    # Masked mean over valid (in-episode) timesteps. Averaging — rather than
    # summing over the horizon — keeps the loss on the same scale as an unmasked
    # mean, so it's comparable across runs and doesn't inflate the effective
    # learning rate by ~PREDICTION_HORIZON.
    return (loss * masks).sum() / masks.sum().clamp(min=1)


def train(policy, dataloader):
    device = torch.device("cuda")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    policy = policy.to(device)
    policy = torch.compile(policy)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=LR, weight_decay=1e-4)

    # Linear warmup then cosine decay to LR_MIN (lerobot-style schedule).
    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return (step + 1) / WARMUP_STEPS
        progress = (step - WARMUP_STEPS) / max(1, TOTAL_STEPS - WARMUP_STEPS)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return LR_MIN / LR + (1.0 - LR_MIN / LR) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    last_loss = 0.0
    dataloader_iter = iter(dataloader)

    # Rolling per-step timers (reset each log window) so we can see whether we're
    # data-loading bound or compute bound. data_time covers fetching/collating
    # the batch and the host->device copy; compute_time covers the forward,
    # backward and optimizer step (loss.item() below forces a CUDA sync, so the
    # measured compute time includes the GPU work rather than just the launch).
    data_time = 0.0
    compute_time = 0.0

    for global_step in range(1, TOTAL_STEPS + 1):
        t_data_start = time.perf_counter()
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)

        images, obs, actions, mask = [
            t.to(device, non_blocking=True) for t in batch
        ]
        t_compute_start = time.perf_counter()

        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = flow_matching_loss(policy, images, obs, actions, mask)
        optimizer.zero_grad()
        loss.backward()
        # clip_grad_norm_ returns the total grad norm pre-clip
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        last_loss = loss.item()
        t_step_end = time.perf_counter()

        data_time += t_compute_start - t_data_start
        compute_time += t_step_end - t_compute_start

        # collect everything due on this step into one log call so wandb sees
        # a single record per step (log and eval cadences can coincide)
        log_dict = {}
        if global_step % LOG_EVERY == 0:
            data_ms = data_time / LOG_EVERY * 1e3
            compute_ms = compute_time / LOG_EVERY * 1e3
            step_ms = data_ms + compute_ms
            it_per_s = 1e3 / step_ms if step_ms > 0 else 0.0
            grad_norm_val = grad_norm.item()
            log_dict["loss"] = last_loss
            log_dict["lr"] = scheduler.get_last_lr()[0]
            log_dict["grad_norm"] = grad_norm_val
            log_dict["time/data_ms"] = data_ms
            log_dict["time/compute_ms"] = compute_ms
            log_dict["time/step_ms"] = step_ms
            log_dict["time/it_per_s"] = it_per_s
            print(
                f"step {global_step}: loss={last_loss:.4f} grad_norm={grad_norm_val:.3f} | "
                f"data={data_ms:.1f}ms compute={compute_ms:.1f}ms "
                f"step={step_ms:.1f}ms ({it_per_s:.1f} it/s)"
            )
            data_time = 0.0
            compute_time = 0.0

        if global_step % EVAL_EVERY == 0:
            log_dict.update(evaluate(policy, device))

        if log_dict:
            wandb.log(log_dict, step=global_step)

        if global_step % SAVE_EVERY == 0:
            save_checkpoint(policy, optimizer, scheduler, global_step, last_loss)

    # final eval + checkpoint, unless the last step already triggered them
    if TOTAL_STEPS % EVAL_EVERY != 0:
        wandb.log(evaluate(policy, device), step=global_step)
    if TOTAL_STEPS % SAVE_EVERY != 0:
        save_checkpoint(policy, optimizer, scheduler, global_step, last_loss)

def main():
    policy = DiTPolicy()
    dataset = PushTDataset(
        dataset_id=DATASET_ID,
        prediction_horizon=PREDICTION_HORIZON,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
        prefetch_factor=4 if NUM_WORKERS > 0 else None,
    )

    steps_per_epoch = len(dataloader)
    print(
        f"steps/epoch={steps_per_epoch}, training for {TOTAL_STEPS} total steps "
        f"(~{math.ceil(TOTAL_STEPS / steps_per_epoch)} passes over the data)"
    )

    # the run config is every UPPERCASE scalar/string constant above (nanoGPT-style)
    config = {
        k: v for k, v in globals().items()
        if k.isupper() and isinstance(v, (int, float, str))
    }
    config["steps_per_epoch"] = steps_per_epoch
    wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name=RUN_NAME, config=config)

    train(policy, dataloader)

    wandb.finish()


if __name__ == "__main__":
    main()
