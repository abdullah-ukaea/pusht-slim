# RunPod Setup

Steps to run `train.py` on a fresh RunPod box (RTX 4090).

## 1. System deps
```bash
apt-get update && apt-get install -y tmux ffmpeg
```
- `ffmpeg` provides `libavutil.so.56`, required by `torchcodec` (lerobot's video decoder). Without it, training crashes on dataset load.
- These install to the system root, which is **ephemeral** — reinstall after every VM stop.

## 2. Python env (uv) — installed on the persistent `/workspace` volume
**Important:** `/root` (incl. `~/.local`, the default uv home) is wiped on VM stop, while `/workspace` persists. Install uv *and* its managed Pythons under `/workspace` so the env survives restarts.

First-time setup:
```bash
export UV_INSTALL_DIR=/workspace/uv/bin
export UV_PYTHON_INSTALL_DIR=/workspace/uv/python
export UV_CACHE_DIR=/workspace/uv/cache
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/workspace/uv/bin:$PATH"
cd /workspace/pusht-slim
uv venv --python 3.12          # interpreter lands in /workspace/uv/python
uv pip install -r requirements.txt
```

After a VM restart, uv + Python + the project `.venv` are all still on `/workspace`, so just re-export the paths (no re-download, no reinstall):
```bash
source /workspace/uv/env.sh    # sets PATH + UV_PYTHON_INSTALL_DIR + UV_CACHE_DIR
```
The project `.venv/bin/python` symlinks into `/workspace/uv/python/...`, so `source .venv/bin/activate` works directly after a restart — only the apt deps in step 1 need reinstalling.

## 3. W&B auth
```bash
export WANDB_API_KEY=$(tr -d '[:space:]' < ~/.wandb_personal_key)
```

## 4. Train (in tmux)
```bash
tmux new-session -d -s train
tmux send-keys -t train 'source .venv/bin/activate && \
  export WANDB_API_KEY=$(tr -d "[:space:]" < ~/.wandb_personal_key) && \
  export SDL_VIDEODRIVER=dummy && \
  python -u train.py --batch-size 64 --steps 200000 2>&1 | tee train.log' Enter
```
- `SDL_VIDEODRIVER=dummy`: headless pygame for eval rollouts.
- `python -u`: unbuffered output so `train.log` is live.

## Notes
- `train.py` is **step-based**: `--steps` (total optimizer steps), `--log-every` (default 200), `--eval-every` (default 10000), `--save-every` (default 10000), plus `--batch-size`, `--lr`, `--wandb-project`, `--wandb-entity`.
- Checkpoints are written as `checkpoints/checkpoint_step_<N>.pt`.
- `N_OBS = 1` (single-frame observation history).
- Defaults target W&B project `pushT-slim` under entity `robot_learning_collective`.
- Watch: `tmux attach -t train` (detach `Ctrl-b d`) or `tail -f train.log`.
