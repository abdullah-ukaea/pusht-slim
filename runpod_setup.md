# RunPod Setup

Steps to run `train.py` on a fresh RunPod box (RTX 4090).

No system deps needed: the dataset is two files (MP4 + parquet) pulled straight
from the HF hub, and video decoding goes through torchvision's bundled PyAV.

## 1. Python env (uv) — installed on the persistent `/workspace` volume
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
The project `.venv/bin/python` symlinks into `/workspace/uv/python/...`, so `source .venv/bin/activate` works directly after a restart.

## 2. W&B auth
```bash
export WANDB_API_KEY=$(tr -d '[:space:]' < ~/.wandb_personal_key)
```

## 3. Train (detached with setsid, survives the shell)
```bash
cd /workspace/pusht-slim
setsid bash -c 'source .venv/bin/activate && \
  export WANDB_API_KEY=$(tr -d "[:space:]" < ~/.wandb_personal_key) && \
  export SDL_VIDEODRIVER=dummy && \
  exec python -u train.py' > train.log 2>&1 < /dev/null & disown
tail -f train.log
```
- `setsid ... & disown`: the run gets its own session, detached from the shell.
- `SDL_VIDEODRIVER=dummy`: headless pygame for eval rollouts.
- `python -u`: unbuffered output so `train.log` is live.
