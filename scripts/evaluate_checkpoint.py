"""Evaluate a checkpoint on all five benchmarks and compare with paper Table 3."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml
from basicsr.data import build_dataset
from basicsr.archs import build_network
from basicsr.metrics import calculate_metric
from basicsr.models import build_model
from basicsr.utils import tensor2img

COUNTS = {"Set5": 5, "Set14": 14, "B100": 100, "Urban100": 100, "Manga109": 109}


def compare_results(measured, scale, bits, psnr_tolerance=0.05, ssim_tolerance=0.0005):
    with (ROOT / "docs/paper_results.csv").open() as stream:
        references = {r["dataset"]: r for r in csv.DictReader(stream)
                      if int(r["scale"]) == scale and int(r["bits"]) == bits}
    if set(references) != set(COUNTS) or set(measured) != set(COUNTS):
        raise ValueError("Verification requires all five benchmark references and measurements")
    rows = []
    for dataset, ref in references.items():
        result = measured[dataset]
        pd = result["psnr"] - float(ref["psnr"])
        sd = result["ssim"] - float(ref["ssim"])
        rows.append({"dataset": dataset, **result,
                     "paper_psnr": float(ref["psnr"]), "paper_ssim": float(ref["ssim"]),
                     "psnr_delta": pd, "ssim_delta": sd,
                     "meets_paper": pd >= -psnr_tolerance and sd >= -ssim_tolerance})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument("--scale", required=True, type=int, choices=[2, 3, 4])
    parser.add_argument("--bits", required=True, type=int, choices=[2, 3, 4])
    parser.add_argument("--calibration")
    parser.add_argument("--fp-checkpoint", required=True)
    parser.add_argument("--dataset-root", default="datasets/benchmark")
    parser.add_argument("--report", required=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--fp-only", action="store_true")
    parser.add_argument("--psnr-tolerance", type=float, default=0.05)
    parser.add_argument("--ssim-tolerance", type=float, default=0.0005)
    args = parser.parse_args()
    if not args.fp_only and (not args.checkpoint or not args.calibration):
        parser.error("quantized evaluation requires --checkpoint and --calibration")
    if args.psnr_tolerance < 0 or args.ssim_tolerance < 0:
        parser.error("verification tolerances must be nonnegative")
    opt = yaml.safe_load((ROOT / f"options/test/test_condiquant_x{args.scale}.yml").read_text())
    opt.update(bit=args.bits, num_gpu=0 if args.cpu else 1, is_train=False, dist=False, rank=0)
    opt["path"]["pretrain_network_Q"] = args.checkpoint
    opt["pathFP"]["pretrain_network_FP"] = args.fp_checkpoint
    opt["cali_data"] = args.calibration
    opt["quant"]["fp_test"] = args.fp_only
    torch.manual_seed(opt["manual_seed"])
    if args.fp_only:
        network = build_network(opt["network_Q"])
        weights = torch.load(args.fp_checkpoint, map_location="cpu", weights_only=True)
        network.load_state_dict(weights.get("params", weights.get("params_ema", weights)), strict=True)
        network = network.to("cpu" if args.cpu else "cuda").eval()
    else:
        model = build_model(opt)
    measured = {}
    for dataopt in opt["datasets"].values():
        name = dataopt["name"]
        dataopt.update(phase="test", scale=args.scale,
                       dataroot_gt=str(Path(args.dataset_root) / name / "HR"),
                       dataroot_lq=str(Path(args.dataset_root) / name / "LR_bicubic" / f"X{args.scale}"))
        dataset = build_dataset(dataopt)
        if len(dataset) != COUNTS[name]:
            raise ValueError(f"{name}: expected {COUNTS[name]} images, found {len(dataset)}")
        totals = {"psnr": 0., "ssim": 0.}
        for index in range(len(dataset)):
            item = dataset[index]
            if args.fp_only:
                lq = item["lq"].unsqueeze(0).to(next(network.parameters()).device)
                height, width = lq.shape[-2:]
                window = opt["network_Q"]["window_size"]
                ph, pw = (-height) % window, (-width) % window
                padded = torch.cat([lq, lq.flip(2)], 2)[:, :, :height + ph, :]
                padded = torch.cat([padded, padded.flip(3)], 3)[:, :, :, :width + pw]
                with torch.no_grad():
                    prediction = network(padded)[:, :, :height * args.scale, :width * args.scale]
            else:
                model.feed_data({"lq": item["lq"].unsqueeze(0)})
                model.test()
                prediction = model.output
            image = tensor2img(prediction)
            target = tensor2img(item["gt"])
            for metric, settings in opt["val"]["metrics"].items():
                totals[metric] += float(calculate_metric({"img": image, "img2": target}, settings))
        measured[name] = {k: v / len(dataset) for k, v in totals.items()}
        measured[name]["images"] = len(dataset)
        print(f"{name}: PSNR {measured[name]['psnr']:.4f}, SSIM {measured[name]['ssim']:.6f}", flush=True)
    ref_bits = 32 if args.fp_only else args.bits
    rows = compare_results(measured, args.scale, ref_bits, args.psnr_tolerance, args.ssim_tolerance)
    checkpoint = args.fp_checkpoint if args.fp_only else args.checkpoint
    checksum = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    report = {"checkpoint": Path(checkpoint).name, "sha256": checksum,
              "scale": args.scale, "bits": ref_bits, "fp_only": args.fp_only,
              "psnr_tolerance": args.psnr_tolerance, "ssim_tolerance": args.ssim_tolerance,
              "meets_paper": all(r["meets_paper"] for r in rows), "results": rows}
    report["torch_version"] = torch.__version__
    report["evaluated_at"] = datetime.now(timezone.utc).isoformat()
    report["protocol"] = {"test_y_channel": True, "crop_border": args.scale,
                          "self_ensemble": False, "full_image": True}
    if args.cpu:
        report["device"] = {"name": "CPU"}
    else:
        device = torch.cuda.get_device_properties(torch.cuda.current_device())
        report["device"] = {"name": device.name, "memory_bytes": device.total_memory}
    report["fp_checkpoint_sha256"] = hashlib.sha256(Path(args.fp_checkpoint).read_bytes()).hexdigest()
    if not args.fp_only:
        report["calibration"] = Path(args.calibration).name
        report["calibration_sha256"] = hashlib.sha256(Path(args.calibration).read_bytes()).hexdigest()
    dest = Path(args.report)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Report: {dest}; meets paper: {report['meets_paper']}")
    return 0 if report["meets_paper"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
