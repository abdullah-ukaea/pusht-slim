# Copyright (c) Sudeep Dasari, 2023

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import copy
import math
import os
from datetime import datetime

import numpy as np
import wandb
import gymnasium as gym
import gym_pusht
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import models

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ── Config ─────────────────────────────────────────────────────────────────────
N_ACTIONS = 16
N_ACTION_STEPS = 8
HIDDEN_DIM = 512 
BATCH_SIZE = 64
TOTAL_STEPS = 200_000
LR = 1e-4
LOG_EVERY = 200
EVAL_EVERY = 10_000
SAVE_EVERY = 10_000
CHECKPOINT_DIR = "checkpoints_dit"
DATASET_ID = "lerobot/pusht"
# ResNet.n_tokens is hardcoded to 49, assuming 224×224 → 7×7 spatial tokens
IMG_SIZE = 224
COORD_MIN, COORD_MAX = 0.0, 512.0
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]


def normalize(x):
    return 2.0 * (x - COORD_MIN) / (COORD_MAX - COORD_MIN) - 1.0


def unnormalize(x):
    return (x + 1.0) / 2.0 * (COORD_MAX - COORD_MIN) + COORD_MIN


# ── Dataset ────────────────────────────────────────────────────────────────────
class PushTDataset(Dataset):
    def __init__(self, dataset_id=DATASET_ID, n_actions=N_ACTIONS):
        self.dataset = LeRobotDataset(dataset_id)
        self.n_actions = n_actions

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
        self.episode_indices = np.array(self.dataset.hf_dataset["episode_index"])
        print("Done caching.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        current_episode = self.episode_indices[idx]

        img = TF.resize(
            self.cache[idx]["observation.image"],
            [IMG_SIZE, IMG_SIZE],
            antialias=True,
        )
        img = TF.normalize(img, IMG_MEAN, IMG_STD)

        obs = normalize(self.cache[idx]["observation.state"])

        actions = torch.zeros(self.n_actions, 2)
        mask = torch.zeros(self.n_actions, 2)
        
        for k in range(self.n_actions):
            i = idx + k
            if i < len(self.dataset) and self.episode_indices[i] == current_episode:
                actions[k] = normalize(self.cache[i]["action"])
                mask[k] = 1.0

        return img, obs, actions.flatten(), mask.flatten()


# ── Model ──────────────────────────────────────────────────────────────────────
def _construct_resnet(size, norm, weights=None):
    if size == 18:
        w = models.ResNet18_Weights
        m = models.resnet18(norm_layer=norm)
    elif size == 34:
        w = models.ResNet34_Weights
        m = models.resnet34(norm_layer=norm)
    else:
        raise NotImplementedError(f"Missing size: {size}")

    if weights is not None:
        w = w.verify(weights).get_state_dict(progress=True)
        m.load_state_dict(w)
    return m


class ResNet(nn.Module):
    def __init__(
        self,
        size,
        weights=None,
        avg_pool=True,
    ):
        super().__init__()
        norm_layer = nn.BatchNorm2d
        model = _construct_resnet(size, norm_layer, weights)
        model.fc = nn.Identity()
        if not avg_pool:
            model.avgpool = nn.Identity()
        self.model = model
        self._size, self._avg_pool = size, avg_pool

    def forward(self, x):
        if self._avg_pool:
            return self.model(x)[:, None]
        B = x.shape[0]
        x = self.model(x)
        x = x.reshape((B, self.embed_dim, -1))
        return x.transpose(1, 2)

    @property
    def embed_dim(self):
        return {18: 512, 34: 512, 50: 2048}[self._size]

    @property
    def n_tokens(self):
        if self._avg_pool:
            return 1
        return 49

class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)

        Returns:
            Tensor of shape (seq_len, batch_size, d_model) with positional encodings added
        """
        pe = self.pe[: x.shape[0]]
        pe = pe.repeat((1, x.shape[1], 1))
        return pe.detach().clone()


class _TimeNetwork(nn.Module):
    def __init__(self, time_dim, out_dim, learnable_w=False):
        assert time_dim % 2 == 0, "time_dim must be even!"
        half_dim = int(time_dim // 2)
        super().__init__()

        w = np.log(10000) / (half_dim - 1)
        w = torch.exp(torch.arange(half_dim) * -w).float()
        self.register_parameter("w", nn.Parameter(w, requires_grad=learnable_w))

        self.out_net = nn.Sequential(
            nn.Linear(time_dim, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        assert len(x.shape) == 1, "assumes 1d input timestep array"
        x = x[:, None] * self.w[None]
        x = torch.cat((torch.cos(x), torch.sin(x)), dim=1)
        return self.out_net(x)


class _SelfAttnEncoder(nn.Module):
    def __init__(
        self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.GELU(approximate="tanh")

    def forward(self, src, pos):
        q = k = src + pos
        src2, _ = self.self_attn(q, k, value=src, need_weights=False)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class _ShiftScaleMod(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.act = nn.SiLU()
        self.scale = nn.Linear(dim, dim)
        self.shift = nn.Linear(dim, dim)

    def forward(self, x, c):
        c = self.act(c)
        return x * self.scale(c)[None] + self.shift(c)[None]

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.scale.weight)
        nn.init.xavier_uniform_(self.shift.weight)
        nn.init.zeros_(self.scale.bias)
        nn.init.zeros_(self.shift.bias)


class _ZeroScaleMod(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.act = nn.SiLU()
        self.scale = nn.Linear(dim, dim)

    def forward(self, x, c):
        c = self.act(c)
        return x * self.scale(c)[None]

    def reset_parameters(self):
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.scale.bias)


class _DiTDecoder(nn.Module):
    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.GELU(approximate="tanh")

        # create modulation layers
        self.attn_mod1 = _ShiftScaleMod(d_model)
        self.attn_mod2 = _ZeroScaleMod(d_model)
        self.mlp_mod1 = _ShiftScaleMod(d_model)
        self.mlp_mod2 = _ZeroScaleMod(d_model)

    def forward(self, x, t, cond):
        # process the conditioning vector first
        cond = torch.mean(cond, dim=0)
        cond = cond + t

        x2 = self.attn_mod1(self.norm1(x), cond)
        x2, _ = self.self_attn(x2, x2, x2, need_weights=False)
        x = self.attn_mod2(self.dropout1(x2), cond) + x

        x2 = self.mlp_mod1(self.norm2(x), cond)
        x2 = self.linear2(self.dropout2(self.activation(self.linear1(x2))))
        x2 = self.mlp_mod2(self.dropout3(x2), cond)
        return x + x2

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for s in (self.attn_mod1, self.attn_mod2, self.mlp_mod1, self.mlp_mod2):
            s.reset_parameters()


class _FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, out_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, t, cond):
        # process the conditioning vector first
        cond = torch.mean(cond, dim=0)
        cond = cond + t

        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=1)
        x = x * scale[None] + shift[None]
        x = self.linear(x)
        return x.transpose(0, 1)

    def reset_parameters(self):
        for p in self.parameters():
            nn.init.zeros_(p)


class _TransformerEncoder(nn.Module):
    def __init__(self, base_module, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(base_module) for _ in range(num_layers)]
        )

        for l in self.layers:
            l.reset_parameters()

    def forward(self, src, pos):
        x, outputs = src, []
        for layer in self.layers:
            x = layer(x, pos)
            outputs.append(x)
        return outputs

class _TransformerDecoder(_TransformerEncoder):
    def forward(self, src, t, all_conds):
        x = src
        for layer, cond in zip(self.layers, all_conds):
            x = layer(x, t, cond)
        return x

class _DiTNoiseNet(nn.Module):
    def __init__(
        self,
        ac_dim,
        ac_chunk,
        time_dim=256,
        hidden_dim=512,
        num_blocks=6,
        dropout=0.1,
        dim_feedforward=2048,
        nhead=8,
        activation="gelu",
    ):
        super().__init__()

        # positional encoding blocks
        self.enc_pos = _PositionalEncoding(hidden_dim)
        self.register_parameter(
            "dec_pos",
            nn.Parameter(torch.empty(ac_chunk, 1, hidden_dim), requires_grad=True),
        )
        nn.init.xavier_uniform_(self.dec_pos.data)

        # input encoder mlps
        self.time_net = _TimeNetwork(time_dim, hidden_dim)
        self.ac_proj = nn.Sequential(
            nn.Linear(ac_dim, ac_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ac_dim, hidden_dim),
        )

        # encoder blocks
        encoder_module = _SelfAttnEncoder(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.encoder = _TransformerEncoder(encoder_module, num_blocks)

        # decoder blocks
        decoder_module = _DiTDecoder(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.decoder = _TransformerDecoder(decoder_module, num_blocks)

        # turns predicted tokens into epsilons
        self.eps_out = _FinalLayer(hidden_dim, ac_dim)


    def forward(self, noise_actions, time, obs_enc, enc_cache=None):
        if enc_cache is None:
            enc_cache = self.forward_enc(obs_enc)
        return enc_cache, self.forward_dec(noise_actions, time, enc_cache)
    
    def forward_enc(self, obs_enc):
        obs_enc = obs_enc.transpose(0, 1)
        pos = self.enc_pos(obs_enc)
        enc_cache = self.encoder(obs_enc, pos)
        return enc_cache

    def forward_dec(self, noise_actions, time, enc_cache):
        time_enc = self.time_net(time)
        
        ac_tokens = self.ac_proj(noise_actions)
        ac_tokens = ac_tokens.transpose(0, 1)
        ## ?? 
        dec_in = ac_tokens + self.dec_pos

        # apply decoder
        dec_out = self.decoder(dec_in, time_enc, enc_cache)

        # apply final epsilon prediction layer
        return self.eps_out(dec_out, time_enc, enc_cache[-1])

class _BatchNorm1DHelper(nn.BatchNorm1d):
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.transpose(1, 2)
            x = super().forward(x)
            return x.transpose(1, 2)
        return super().forward(x)

    
class FlowMatchingTransformerAgent(nn.Module):
    def __init__(
        self,
        image_encoder,
        odim,
        ac_dim,
        ac_chunk,
        dropout=0,
        noise_net_kwargs=dict(),
    ):

        # initialize obs and img tokenizers
        super().__init__()

        self.image_encoder = image_encoder

        self._token_dim = image_encoder.embed_dim 
        self._n_tokens = image_encoder.n_tokens + 1 # added 1 for observation token

        # handle obs tokenization strategies
        self._obs_proc = nn.Sequential(
            nn.Dropout(p=0.2), nn.Linear(odim, self._token_dim)
        )

        norm = _BatchNorm1DHelper(self._token_dim)

        self.post_proc = nn.Sequential(norm, nn.Dropout(dropout))

        self.noise_net = _DiTNoiseNet(
            ac_dim=ac_dim,
            ac_chunk=ac_chunk,
            **noise_net_kwargs,
        )
        self.ac_dim, self.ac_chunk = ac_dim, ac_chunk


    def tokenize_obs(self, imgs, obs):
        image_tokens = self.image_encoder(imgs)
        obs_token = self._obs_proc(obs)[:, None]
        tokens = torch.cat((image_tokens, obs_token), 1)
        return self.post_proc(tokens)

    def forward(self, imgs, obs, ac_flat, mask_flat):
        # get observation encoding and sample noise/timesteps
        B, device = obs.shape[0], obs.device
        s_t = self.tokenize_obs(imgs, obs)

        actions = ac_flat.reshape(B, self.ac_chunk, self.ac_dim)
        mask = mask_flat.reshape((B, self.ac_chunk, self.ac_dim))
        timestep = torch.rand(B, device=device)
        noise = torch.randn_like(actions)
        noise_acs = (1 - timestep[:, None, None]) * noise + timestep[:, None, None] * actions
        _, v_pred = self.noise_net(noise_acs, timestep, s_t)

        loss = nn.functional.mse_loss(v_pred, actions-noise)
        loss = (loss * mask).sum(1)  # mask the loss to only consider "real" acs
        return loss.mean()

    @torch.no_grad()
    def get_actions(self, imgs, obs, n_steps=10):
        B, device = obs.shape[0], obs.device
        s_t = self.tokenize_obs(imgs, obs)
        enc_cache = self.noise_net.forward_enc(s_t)
        x = torch.randn(B, self.ac_chunk, self.ac_dim, device=device)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((B,), i / n_steps, device=device)
            _, v = self.noise_net(x, t, s_t, enc_cache)
            x = x + v * dt
        return x.clamp(-1.0, 1.0)


# ── Evaluation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(agent, device, n_episodes=10, n_steps=10):
    env = gym.make(
        "gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array"
    )
    agent.eval()
    successes = 0

    for _ in range(n_episodes):
        obs_env, _ = env.reset()
        done = False
        img_t = torch.from_numpy(obs_env["pixels"]).permute(2, 0, 1).float() / 255.0
        state_t = normalize(torch.from_numpy(obs_env["agent_pos"]).float())

        while not done:
            img_in = TF.normalize(
                TF.resize(img_t, [IMG_SIZE, IMG_SIZE]), IMG_MEAN, IMG_STD
            )
            imgs = img_in.unsqueeze(0).to(device)
            obs = state_t.unsqueeze(0).to(device)

            acs = agent.get_actions(imgs, obs, n_steps=n_steps)  # (1, 16, 2) normalized
            acs = unnormalize(acs.squeeze(0).cpu().numpy())  # (16, 2) in [0, 512]

            for ac in acs[:N_ACTION_STEPS]:
                obs_env, _, terminated, truncated, info = env.step(ac)
                done = terminated or truncated
                img_t = (
                    torch.from_numpy(obs_env["pixels"]).permute(2, 0, 1).float() / 255.0
                )
                state_t = normalize(torch.from_numpy(obs_env["agent_pos"]).float())
                if done:
                    break

        if info.get("is_success", False):
            successes += 1

    env.close()
    agent.train()
    return successes / n_episodes


# ── Training loop ─────────────────────────────────────────────────────────────
def train(agent, loader):
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Training on {device}")
    agent = agent.to(device)
    optimizer = torch.optim.AdamW(agent.parameters(), lr=LR, weight_decay=1e-4)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    step = 0
    running_loss, running_n = 0.0, 0
    pbar = tqdm(total=TOTAL_STEPS, desc="train")

    while step < TOTAL_STEPS:
        for imgs, obs, ac_flat in loader:
            imgs = imgs.to(device)
            obs = obs.to(device)
            ac_flat = ac_flat.to(device)
            mask_flat = mask_flat.to(device)

            loss = agent(imgs, obs, ac_flat, mask_flat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            running_loss += loss.item()
            running_n += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            log = {}
            if step % LOG_EVERY == 0:
                log["train/loss"] = running_loss / running_n
                running_loss, running_n = 0.0, 0

            if step % EVAL_EVERY == 0:
                for n in [1, 5, 10]:
                    sr = evaluate(agent, device, n_episodes=20, n_steps=n)
                    log[f"eval/success_n{n}"] = sr
                    print(f"step {step}: n_steps={n}  success={sr:.2f}")

            if log:
                wandb.log(log, step=step)

            if step % SAVE_EVERY == 0:
                ckpt = os.path.join(CHECKPOINT_DIR, f"step_{step}.pt")
                torch.save(
                    {
                        "step": step,
                        "model": agent.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    ckpt,
                )
                print(f"Saved {ckpt}")

            if step >= TOTAL_STEPS:
                break

    pbar.close()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    wandb.init(
        project="pushT-slim",
        entity="robot_learning_collective",
        name=f"dit-pusht-slim-{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}",
        config=dict(
            total_steps=TOTAL_STEPS,
            lr=LR,
            hidden_dim=HIDDEN_DIM,
            n_actions=N_ACTIONS,
            batch_size=BATCH_SIZE,
        ),
    )
    resnet = ResNet(
        size=34,
        weights="IMAGENET1K_V1",
        avg_pool=False,  # keep 7×7 = 49 spatial tokens per image
    )

    agent = FlowMatchingTransformerAgent(
        image_encoder=resnet,
        odim=2,  
        ac_dim=2,
        ac_chunk=N_ACTIONS,
        dropout=0.1,
        noise_net_kwargs=dict(
            time_dim=256,
            hidden_dim=HIDDEN_DIM,
            num_blocks=6,
            dim_feedforward=HIDDEN_DIM * 4,
            nhead=8,
            dropout=0.1,
            activation="gelu",
        ),
    )
    dataset = PushTDataset()
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    print(
        f"Steps/epoch: {len(loader)},  ~{math.ceil(TOTAL_STEPS / len(loader))} epochs"
    )
    train(agent, loader)
    wandb.finish()

if __name__ == "__main__":
    main()
