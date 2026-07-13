# pusht-slim

A minimal, single-file imitation-learning pipeline for the [PushT](https://github.com/huggingface/gym-pusht) task, in the spirit of [nanoGPT](https://github.com/karpathy/nanoGPT): dataset, model, training loop, and eval all live in `train.py` (~600 lines), configured by plain constants at the top of the file.

The policy is a ~200M-parameter [DiT](https://arxiv.org/abs/2212.09748) trained with [flow matching](https://arxiv.org/abs/2210.02747) on the [`lerobot/pusht`](https://huggingface.co/datasets/lerobot/pusht) demonstrations. It reaches a success rate of ~0.6–0.7 after 100k steps (~3.5 h on an RTX 4090).

## How it works

- **Data** — the hub dataset is just an MP4 (every frame of every episode, concatenated) plus a parquet with per-frame state/action/episode metadata. `PushTDataset` decodes the video once (~7 s) and keeps everything in RAM (~2.8 GB), so `__getitem__` is pure tensor slicing. No lerobot dependency.
- **Observation encoding** — frames are cropped to 84×84 (random in train, center in eval) and passed through a pretrained DINOv2 ViT-S/14, fine-tuned end-to-end. Its 6×6 patch grid is pooled by SpatialSoftmax into 32 keypoints. The 2-DoF robot state goes through a small MLP.
- **Action generation** — a DiT with adaLN conditioning (image + state + flow time) predicts the velocity field of a flow from Gaussian noise to a 16-step action chunk. Inference is 10 Euler steps.
- **Control** — receding horizon: predict 16 actions, execute 8, replan.
- **Eval** — every 10k steps the policy is rolled out in `gym-pusht` for 50 episodes; success rate, average max reward, and 3 rollout videos are logged to wandb.

## Run

```bash
pip install -r requirements.txt
export WANDB_API_KEY=...   # metrics and eval videos go to wandb
python train.py
```

Set `WANDB_ENTITY` / `WANDB_PROJECT` at the top of `train.py` to your own wandb account. All hyperparameters are constants in the same block. Checkpoints land in `checkpoints_<run-name>/`.

For running on a cloud GPU box, see [runpod_setup.md](runpod_setup.md).
