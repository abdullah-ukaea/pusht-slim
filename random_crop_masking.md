# Diff vs `feat/anjana/random-crop_masks`

One change on top of Anjana's branch: **restore the stock ResNet stem in `ImageEncoder`**.

Anjana's branch kept `main`'s small-image stem (`conv1` 3x3 **stride-1** +
`maxpool = Identity`), which was fine for 96x96 frames but **OOMs** on the new 224x224
crops.

Why it OOMs — the stem normally downsamples 4x (conv1 stride-2 = 2x, maxpool stride-2 =
2x) *before* layer1. Removing both keeps every feature map at 4x the height and width,
i.e. **16x more spatial elements**, through all stages:

| stage  | stock stem (224 in) | stride-1 / no-maxpool stem |
|--------|---------------------|----------------------------|
| conv1  | 64 x 112 x 112      | 64 x 224 x 224             |
| layer1 | 64 x 56 x 56        | 64 x 224 x 224             |
| layer2 | 128 x 28 x 28       | 128 x 112 x 112            |
| layer3 | 256 x 14 x 14       | 256 x 56 x 56              |
| layer4 | 512 x 7 x 7         | 512 x 28 x 28              |

Training has to keep every block's activations around for the backward pass, and that
memory scales as `batch x channels x H x W`. The 16x blow-up hits hardest in the early
high-resolution stages (layer1 holds 64 x 224 x 224 ~= 3.2M floats *per sample* vs 0.2M;
at batch 64 / fp32 that's ~0.8 GB for a single activation vs ~0.05 GB), and summed over
all blocks it exhausts GPU memory. Restoring the stock stride-2 conv1 + maxpool brings the
4x stem downsample back and training fits.

## Status

wandb run: https://wandb.ai/robot_learning_collective/pushT-slim/runs/e6191j7v

**SR=0 was a train/eval preprocessing mismatch — confirmed and fixed.** Training feeds the
encoder `resize(256) -> random-crop(224)` images, but eval (`PushTAdapter.observe`) was
passing native-resolution frames straight through. The model only ever saw 224x224 crops,
so at eval it got an out-of-distribution scale/resolution and never succeeded.

Verified with an A/B eval on `checkpoints/checkpoint_step_75000.pt` (a *mid*-training
checkpoint), comparing the old eval preprocessing against the training-matched one:

| eval preprocessing                         | success_rate | avg_max_reward |
|--------------------------------------------|--------------|----------------|
| `raw` (native frame, old `observe`)        | 0.000        | 0.113          |
| `resizecrop` (`resize(256)->center-crop`)  | 0.250        | 0.836          |

**Fix:** `PushTAdapter.observe` now applies `resize(256) -> center-crop(224)` (center crop
for eval, vs the training random crop). See `eval_check.py` for the A/B harness.

## "Loss much higher" was a reduction-scale artifact — also fixed

The loss looked alarming (~4.0 at start, ~0.5–1.0 at step 75k) but the run was actually
converging fine. The branch had changed the loss reduction from `main`'s full mean
(`mse_loss(...)`) to a **sum over the 16-step prediction horizon**:

```python
(loss * masks).sum(1).mean()   # sums over PREDICTION_HORIZON=16
```

Masks are full for almost every sample (only the last 15 frames of an episode are partially
masked), so this just multiplies the reported loss by ~16. Dividing by 16, the run tracks
the baselines almost exactly:

| | start | converged |
|--------------------------|-------|-----------------------|
| baselines (full-mean)    | ~0.37 | ~0.02–0.04            |
| this run (sum-over-16)   | ~4.0  | ~0.5–1.0 (step 75k)   |
| this run ÷ 16            | ~0.25 | ~0.03–0.06            |

Summing also inflated the gradient magnitude ~16x (effective LR ~16x higher), which —
together with the new random-crop augmentation — explains the bouncy loss curve.

**Fix:** use a proper masked mean so the loss is on the baseline scale and the gradient
magnitude is comparable:

```python
(loss * masks).sum() / masks.sum().clamp(min=1)
```
