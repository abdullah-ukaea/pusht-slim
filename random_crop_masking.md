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

Loss not converging, SR=0. Most likely a train/eval mismatch: training uses 224x224 crops
but eval (`PushTAdapter.observe`) still feeds native-resolution frames — the eval path
needs the same `resize(256) -> center-crop(224)` preprocessing.
