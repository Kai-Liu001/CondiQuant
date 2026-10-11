"""Download the official SwinIR-light full-precision checkpoints."""
import argparse
from pathlib import Path
from urllib.request import urlopen
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scales", type=int, nargs="+", choices=[2, 3, 4], default=[2, 3, 4])
    parser.add_argument("--output-dir", default="experiments/pretrained_models/SwinIR")
    args = parser.parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for scale in args.scales:
        name = f"002_lightweightSR_DIV2K_s64w8_SwinIR-S_x{scale}.pth"
        dest = root / name
        if dest.exists():
            print(f"Already exists: {dest}")
            continue
        url = f"https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{name}"
        temporary = dest.with_suffix(".download")
        try:
            with urlopen(url, timeout=120) as response, temporary.open("wb") as stream:
                shutil.copyfileobj(response, stream)
            temporary.replace(dest)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"Downloaded {dest}")


if __name__ == "__main__":
    main()
