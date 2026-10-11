"""Sample deterministic RGB LR patches for CondiQuant calibration."""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch


def sample_patches(root, samples=100, patch_size=64, seed=3407):
    paths = sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
    if not paths:
        raise FileNotFoundError(f"No calibration images in {root}")
    rng = np.random.default_rng(seed)
    patches = []
    for index in rng.permutation(len(paths)):
        image = cv2.imread(str(paths[index]), cv2.IMREAD_COLOR)
        if image is None or min(image.shape[:2]) < patch_size:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        top = rng.integers(image.shape[0] - patch_size + 1)
        left = rng.integers(image.shape[1] - patch_size + 1)
        crop = image[top:top + patch_size, left:left + patch_size].copy()
        patches.append(torch.from_numpy(crop).permute(2, 0, 1).float() / 255.)
        if len(patches) == samples:
            return torch.stack(patches)
    raise ValueError(f"Need {samples} images at least {patch_size}px; found {len(patches)} usable images")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lr-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    if args.samples < 1 or args.patch_size < 1:
        parser.error("samples and patch-size must be positive")
    patches = sample_patches(args.lr_dir, args.samples, args.patch_size, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"lq": patches}, output)
    print(f"Saved {tuple(patches.shape)} RGB float32 patches to {output}")


if __name__ == "__main__":
    main()
