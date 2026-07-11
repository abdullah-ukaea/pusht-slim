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
# dominant success-rate lever in the experiment rounds (EXPERIMENTS.md on the
# feat/zaringleb/claude-flow-experiments branch): ~47M plateaued at SR ~0.5,
# 200M reaches ~0.6-0.7.
N_HEADS = 8
N_LAYERS = 12
HIDDEN_DIM = 896
TOTAL_STEPS = 100_000
LR = 1e-4
WARMUP_STEPS = 500   # linear LR warmup
LR_MIN = 1e-6        # cosine decay floor
# Gradient clipping a la nanoGPT. torch.nn.utils.clip_grad_norm_ returns the
# total pre-clip grad norm, which we log to watch for instability. Set to a
# float (nanoGPT uses 1.0) to also clip; None measures the norm without clipping.
GRAD_CLIP = None
LOG_EVERY = 200
EVAL_EVERY = 10_000
EVAL_EPISODES = 50  # SR noise at 20 episodes was +-0.11; 50 brings it to ~+-0.07
SAVE_EVERY = 25_000
RUN_NAME = f"exp-dinov2-vits14-200m-100k-{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}"
CHECKPOINT_DIR = f"checkpoints_{RUN_NAME}"
BATCH_SIZE = 512
# Background dataloader workers. With num_workers=0 the loader runs in the train
# process and each step blocks on CPU collation, which (with the GPU step at
# ~27ms) makes training data-loading bound. A few workers prefetch batches so
# the GPU step becomes the bottleneck.
NUM_WORKERS = 8
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
    metrics = {
        "success_rate": successes / n_episodes,
        "avg_max_reward": sum(max_rewards) / n_episodes,
    }
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
    (~2.8 GB), so __getitem__ is pure tensor slicing.
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
        # Raw native-resolution frames in [0, 1]. Cropping and normalization
        # happen inside the encoder so train/eval stay in sync.
        self.images = frames.float().div_(255.0)
        self.states = normalize(states)
        self.actions = normalize(actions)
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

        return self.images[idx], self.states[idx], actions.flatten(), mask.flatten()

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
    """Vision encoder: DINOv2 ViT-S/14 (self-supervised pretrained), fine-tuned
    end-to-end, replacing the from-scratch ResNet18+GN of main.

    The 84x84 crop is exactly 6x14 pixels, so the ViT sees a 6x6 patch grid with
    no resizing. Its 36 patch tokens are reshaped back into a (embed_dim, 6, 6)
    feature map and pooled by the same SpatialSoftmax -> keypoints -> linear
    pipeline main uses, so only the backbone changes. Cropping lives here
    (random in train, center in eval) so train and eval share a single code path.
    """

    def __init__(self, out_dim, crop_size=CROP_SIZE, num_kp=NUM_KEYPOINTS):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
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

        feat_shape = (self.backbone.embed_dim, self._grid, self._grid)
        self.pool = SpatialSoftmax(feat_shape, num_kp=num_kp)
        self.fc = nn.Linear(num_kp * 2, out_dim)

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
        # images: (batch, 3, 96, 96) in [0, 1]
        images = self.random_crop(images) if self.training else self.center_crop(images)
        images = (images - self.img_mean) / self.img_std

        # (B, 36, embed_dim) patch tokens -> (B, embed_dim, 6, 6) feature map
        tokens = self.backbone.forward_features(images)["x_norm_patchtokens"]
        b, n, c = tokens.shape
        x = tokens.transpose(1, 2).reshape(b, c, self._grid, self._grid)
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
def save_checkpoint(policy, optimizer, scheduler, global_step, loss, checkpoint_dir):
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{global_step}.pt")

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
    B = actions.shape[0]
    actions = actions.reshape(B, PREDICTION_HORIZON, ROBOT_DOF)
    masks = masks.reshape(B, PREDICTION_HORIZON, ROBOT_DOF)

    t = torch.rand(B, device=actions.device)
    noise = torch.randn_like(actions)
    x_t = (1 - t[:, None, None]) * noise + t[:, None, None] * actions
    v_pred = policy(x_t, t, images, obs)
    loss = nn.functional.mse_loss(v_pred, actions - noise, reduction="none")
    # Masked mean over valid (in-episode) timesteps. Averaging — rather than
    # summing over the horizon — keeps the loss on the same scale as an unmasked
    # mean, so it's comparable across runs and doesn't inflate the effective
    # learning rate by ~PREDICTION_HORIZON.
    return (loss * masks).sum() / masks.sum().clamp(min=1)


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
    device = torch.device("cuda")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    policy = policy.to(device)
    policy = torch.compile(policy)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)

    # Linear warmup then cosine decay to LR_MIN (lerobot-style schedule).
    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return (step + 1) / WARMUP_STEPS
        progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return LR_MIN / lr + (1.0 - LR_MIN / lr) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    os.makedirs(checkpoint_dir, exist_ok=True)

    last_loss = 0.0
    global_step = 0
    dataloader_iter = iter(dataloader)

    # Rolling per-step timers (reset each log window) so we can see whether we're
    # data-loading bound or compute bound. data_time covers fetching/collating
    # the batch and the host->device copy; compute_time covers the forward,
    # backward and optimizer step (loss.item() below forces a CUDA sync, so the
    # measured compute time includes the GPU work rather than just the launch).
    data_time = 0.0
    compute_time = 0.0

    for global_step in range(1, total_steps + 1):
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
        # nanoGPT-style: clip_grad_norm_ returns the total grad norm pre-clip.
        # max_norm=inf measures without actually clipping (GRAD_CLIP=None).
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(),
            max_norm=GRAD_CLIP if GRAD_CLIP is not None else float("inf"),
        )
        optimizer.step()
        scheduler.step()

        last_loss = loss.item()
        t_step_end = time.perf_counter()

        data_time += t_compute_start - t_data_start
        compute_time += t_step_end - t_compute_start

        # collect everything due on this step into one log call so wandb sees
        # a single record per step (log and eval cadences can coincide)
        log_dict = {}
        if global_step % log_every == 0:
            data_ms = data_time / log_every * 1e3
            compute_ms = compute_time / log_every * 1e3
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

        if global_step % eval_every == 0:
            log_dict.update(evaluate(
                policy, device, n_episodes=EVAL_EPISODES, render=False,
                n_steps=N_DENOISING_STEPS,
            ))

        if log_dict:
            wandb.log(log_dict, step=global_step)

        if global_step % save_every == 0:
            save_checkpoint(
                policy, optimizer, scheduler, global_step, last_loss, checkpoint_dir
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
        save_checkpoint(policy, optimizer, scheduler, global_step, last_loss, checkpoint_dir)

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
            "warmup_steps": WARMUP_STEPS,
            "lr_min": LR_MIN,
            "n_layers": N_LAYERS,
            "n_heads": N_HEADS,
            "hidden_dim": HIDDEN_DIM,
            "backbone": "dinov2-vits14-pretrained",
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