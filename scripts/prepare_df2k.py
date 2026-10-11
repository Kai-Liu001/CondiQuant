"""Prepare a deterministic DF2K subset and MATLAB-style bicubic LR images."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from basicsr.utils.matlab_functions import imresize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hr-dir", required=True)
    parser.add_argument("--output-dir", default="datasets/DF2K")
    parser.add_argument("--images", type=int, default=100)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--scales", type=int, nargs="+", choices=[2, 3, 4], default=[2, 3, 4])
    args = parser.parse_args()
    torch.set_num_threads(1)
    sources = sorted(p for p in Path(args.hr_dir).rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if len(sources) < args.images:
        raise ValueError(f"Need {args.images} HR images; found {len(sources)}")
    order = np.random.default_rng(args.seed).permutation(len(sources))[:args.images]
    root = Path(args.output_dir)
    (root / "HR").mkdir(parents=True, exist_ok=True)
    for scale in args.scales:
        (root / "LR_bicubic" / f"X{scale}").mkdir(parents=True, exist_ok=True)
    selected = []
    for count, index in enumerate(order, 1):
        source = sources[index]
        name = f"{index:05d}_{source.stem}"
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot read {source}")
        # A common multiple keeps every scale's pairs aligned.
        height, width = image.shape[:2]
        image = image[:height // 12 * 12, :width // 12 * 12]
        if min(image.shape[:2]) < 192:
            raise ValueError(f"Image too small for 64px LR patches: {source}")
        hr = root / "HR" / f"{name}.png"
        if not hr.exists():
            if not cv2.imwrite(str(hr), image):
                raise OSError(f"Could not write {hr}")
        for scale in args.scales:
            output = root / "LR_bicubic" / f"X{scale}" / f"{name}x{scale}.png"
            if not output.exists():
                resized = imresize(image.astype(np.float32) / 255., 1. / scale)
                result = np.clip(np.round(resized * 255.), 0, 255).astype(np.uint8)
                if not cv2.imwrite(str(output), result):
                    raise OSError(f"Could not write {output}")
        selected.append(str(source.relative_to(Path(args.hr_dir))))
        print(f"[{count}/{args.images}] {source.name}", flush=True)
    (root / "selection.txt").write_text("\n".join(selected) + "\n")


if __name__ == "__main__":
    main()
