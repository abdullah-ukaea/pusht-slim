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

**Caveat — "loss not converging" is a separate issue.** Training loss is computed entirely
on the (already-correct) training pipeline, so the eval mismatch does not explain it. With
the eval fixed, a mid-training checkpoint already reaches 25% SR, so the run was healthier
than the SR=0 metric implied. The loss behavior (bouncing ~0.3–1.4 around step 80k) still
warrants a separate look.
