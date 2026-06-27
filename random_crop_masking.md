# Diff vs `feat/anjana/random-crop_masks`

One change on top of Anjana's branch: **restore the stock ResNet stem in `ImageEncoder`**.

- Anjana's branch kept `main`'s small-image stem (`conv1` stride-1 + `maxpool = Identity`),
  which was tuned for 96x96 frames.
- With the branch's new 224x224 random crops that stem keeps activations at full
  resolution, which **OOMs** on the GPU. Restoring the stock stride-2 `conv1` + maxpool
  downsamples 4x in the stem, shrinking the early feature maps so training fits in memory.

## Status

wandb run: https://wandb.ai/robot_learning_collective/pushT-slim/runs/e6191j7v

Loss not converging, SR=0. Most likely a train/eval mismatch: training uses 224x224 crops
but eval (`PushTAdapter.observe`) still feeds native-resolution frames — the eval path
needs the same `resize(256) -> center-crop(224)` preprocessing.
