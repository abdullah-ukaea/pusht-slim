# RunPod setup

Running `train.py` on a fresh RunPod box (tested on RTX 4090). No system deps:
the dataset comes straight from the HF hub and video decoding uses torchvision's
bundled PyAV.

## Python env

RunPod wipes `/root` on VM stop but keeps `/workspace`, so install uv, its
managed Python, and the venv all under `/workspace`:

```bash
export UV_INSTALL_DIR=/workspace/uv/bin
export UV_PYTHON_INSTALL_DIR=/workspace/uv/python
export UV_CACHE_DIR=/workspace/uv/cache
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/workspace/uv/bin:$PATH"

cd /workspace/pusht-slim
uv venv --python 3.12
uv pip install -r requirements.txt
```

After a VM restart everything is still there — just run
`source /workspace/uv/env.sh` (sets PATH + uv env vars), then
`source .venv/bin/activate`.

## Train

Launch detached so the run survives the SSH session:

```bash
cd /workspace/pusht-slim
setsid bash -c 'source .venv/bin/activate && \
  export WANDB_API_KEY=$(tr -d "[:space:]" < ~/.wandb_personal_key) && \
  export SDL_VIDEODRIVER=dummy && \
  exec python -u train.py' > train.log 2>&1 < /dev/null & disown
tail -f train.log
```

`SDL_VIDEODRIVER=dummy` makes pygame render headless for eval rollouts;
`python -u` keeps `train.log` unbuffered.
