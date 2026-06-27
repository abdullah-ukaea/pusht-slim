"""Quick A/B eval to test the train/eval preprocessing-mismatch hypothesis.

Loads a `train.py` DiTPolicy checkpoint and runs the env eval under two image
preprocessings:

  - "raw":        what `PushTAdapter.observe` does today (native-res frame, no
                  resize/crop) -- the suspected-buggy path.
  - "resizecrop": training-matched `resize(256) -> center-crop(224)`.

If the hypothesis is right, "raw" reproduces SR~0 while "resizecrop" recovers
meaningful success.
"""

import argparse
from pathlib import Path

import torch
import torchvision.transforms.functional as TF

import train as T


def observe_factory(mode, device):
    def observe(self, obs):
        image = torch.from_numpy(obs["pixels"]).permute(2, 0, 1).float() / 255.0
        if mode == "resizecrop":
            image = TF.resize(image, [T.PRE_CROP_SIZE, T.PRE_CROP_SIZE], antialias=True)
            image = TF.center_crop(image, [T.IMG_SIZE, T.IMG_SIZE])
        state = T.normalize(torch.from_numpy(obs["agent_pos"]).float())
        return image.unsqueeze(0).to(device), state.unsqueeze(0).to(device)

    return observe


def load_policy(checkpoint_path, device):
    policy = T.DiTPolicy().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    policy.load_state_dict(state_dict)
    return policy, ckpt.get("step")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/checkpoint_step_75000.pt"),
    )
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--n-steps", type=int, default=T.N_DENOISING_STEPS)
    p.add_argument(
        "--modes",
        nargs="+",
        default=["raw", "resizecrop"],
        choices=["raw", "resizecrop"],
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy, step = load_policy(args.checkpoint, device)
    print(f"Loaded {args.checkpoint} (step={step}) on {device}")

    results = {}
    for mode in args.modes:
        T.PushTAdapter.observe = observe_factory(mode, device)
        metrics = T.evaluate(
            policy,
            device,
            n_episodes=args.episodes,
            render=False,
            n_steps=args.n_steps,
        )
        results[mode] = metrics
        print(
            f"[{mode}] success_rate={metrics['success_rate']:.3f} "
            f"avg_max_reward={metrics['avg_max_reward']:.3f}"
        )

    print("\n=== summary ===")
    for mode, m in results.items():
        print(
            f"{mode:>11}: SR={m['success_rate']:.3f}  "
            f"avg_max_reward={m['avg_max_reward']:.3f}"
        )


if __name__ == "__main__":
    main()
