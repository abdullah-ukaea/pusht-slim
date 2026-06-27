# Random crop augmentation + action loss masking

This branch builds on the refactored `main` (`DiTPolicy`, step-based `train.py`) and
adds image augmentation and a masked flow-matching loss. It is the rebased version of
`feat/anjana/random-crop_masks`.

## What changed (`train.py`)

1. **Random-crop image augmentation.**
   - New constants `IMG_SIZE = 224`, `PRE_CROP_SIZE = 256`.
   - In `PushTDataset.__getitem__`, each cached frame is resized to `256x256` and then
     `T.RandomCrop(224)` is applied, giving the model jittered 224x224 views instead of
     always seeing the exact same framing.

2. **Loss masking instead of `_determine_valid_indices`.**
   - Removed `_determine_valid_indices`. Previously we *dropped* every frame that did not
     have a full `prediction_horizon` of future actions inside the same episode, so the
     last `horizon-1` frames of every episode were never used as training samples.
   - Now **every** frame is a valid sample (`__len__ == len(dataset)`). For frames near the
     end of an episode we still build a full-length action tensor, but pad the missing
     future steps with zeros and set a `mask` to `0` there (and `1` for real steps).
   - `DiTPolicy.training_step` now takes `masks`, computes the per-element MSE with
     `reduction="none"`, multiplies by the mask, and reduces. This way the padded /
     out-of-episode action slots contribute nothing to the gradient.

3. **ImageEncoder stem restored for 224x224 inputs.**
   - `main` had replaced the ResNet stem (`conv1` stride-2 + maxpool) with a stride-1 conv
     and `Identity` maxpool, which made sense for the small 96x96 frames.
   - With 224x224 crops that stem keeps activations at full resolution and blows up memory
     while diverging from how the pretrained weights expect to be used, so the stock
     stride-2 conv + maxpool stem is restored (4x early downsample).

4. **Removed the stray `facebook/dinov3-vitb16` block** that the original branch added at
   import time (an unused experiment that also forced a heavy model download).

## Why

- **Augmentation:** random cropping is cheap regularization to reduce overfitting on the
  fixed PushT camera framing.
- **Masking:** keeps the full dataset (no discarded tail-of-episode frames) and lets the
  model learn from short horizons near episode boundaries without being penalized for the
  nonexistent future actions.

## How

- Rebased the single `wip` commit from `feat/anjana/random-crop_masks` onto the refactored
  `main`, re-expressing the masking/crop logic against the new `DiTPolicy` structure
  (constants/obs were renamed during the refactor) and dropping the dinov3 block and the
  unrelated `dit_stripped.py` edits.

## Status / open issues

Training run: https://wandb.ai/robot_learning_collective/pushT-slim/runs/e6191j7v

Observed in this run: **loss does not converge and success rate stays at 0**. Two things to
investigate before trusting the results:

- **Loss scale:** the masked loss sums the per-element MSE over the prediction-horizon
  dimension (`(loss * masks).sum(1).mean()`) rather than averaging, so its absolute value
  is much larger than `main`'s mean MSE and is not directly comparable. This alone does not
  explain SR=0 but makes the curve look "huge".
- **Train/eval image-size mismatch (most likely cause of SR=0):** training now feeds
  224x224 crops, but `PushTAdapter.observe` in eval still passes the env's native-resolution
  frames straight through with no resize/crop. The policy is evaluated on a resolution it
  never trained on. The eval path needs the same `resize(256) -> center-crop(224)`
  preprocessing as training.
