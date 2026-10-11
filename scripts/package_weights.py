"""Package verified CondiQuant checkpoints, calibration files, and reports."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", default="experiments")
    parser.add_argument("--reports", default="results/verification")
    parser.add_argument("--calibration", default="keydata")
    parser.add_argument("--output", default="release_weights/CondiQuant_verified")
    args = parser.parse_args()
    entries = []
    # Validate everything before creating a release directory.
    for scale in (2, 3, 4):
        for bits in (2, 3, 4):
            report_path = Path(args.reports) / f"x{scale}_w{bits}a{bits}.json"
            report = json.loads(report_path.read_text())
            if not report.get("meets_paper") or report.get("fp_only"):
                raise ValueError(f"Unverified checkpoint: {report_path}")
            if report.get("scale") != scale or report.get("bits") != bits or len(report.get("results", [])) != 5:
                raise ValueError(f"Incorrect verification metadata: {report_path}")
            source = Path(args.experiments) / f"train_CondiQuant_x{scale}_bit{bits}" / "models" / report["checkpoint"]
            config = source.parents[1] / f"train_condiquant_x{scale}.yml"
            settings = yaml.safe_load(config.read_text())
            if settings.get("bit") != bits or settings.get("scale") != scale:
                raise ValueError(f"Training configuration does not match checkpoint: {config}")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            if digest != report["sha256"]:
                raise ValueError(f"Checkpoint changed since evaluation: {source}")
            calibration = Path(args.calibration) / f"cali_data_x{scale}.pth"
            if not calibration.is_file():
                raise FileNotFoundError(calibration)
            if hashlib.sha256(calibration.read_bytes()).hexdigest() != report.get("calibration_sha256"):
                raise ValueError(f"Calibration changed since evaluation: {calibration}")
            entries.append((scale, bits, source, calibration, config, report_path, report))
    output = Path(args.output)
    archive = output.with_suffix(".zip")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if archive.exists():
        raise FileExistsError(archive)
    output.mkdir(parents=True)
    manifest = []
    for scale, bits, source, calibration, config, report_path, report in entries:
        name = f"CondiQuant_x{scale}_w{bits}a{bits}.pth"
        shutil.copy2(source, output / name)
        shutil.copy2(calibration, output / calibration.name)
        packaged_report = {**report, "checkpoint": name,
                           "source_checkpoint": report["checkpoint"],
                           "calibration": calibration.name}
        (output / report_path.name).write_text(json.dumps(packaged_report, indent=2) + "\n")
        shutil.copy2(config, output / f"train_x{scale}_w{bits}a{bits}.yml")
        manifest.append({**packaged_report,
                         "calibration_sha256": hashlib.sha256(calibration.read_bytes()).hexdigest()})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "README.txt").write_text(
        "CondiQuant SwinIR-light checkpoints: scales 2/3/4, W2A2/W3A3/W4A4.\n"
        "These are floating-point fake-quantization checkpoints.\n\n"
        "Keep each checkpoint together with its calibration file, training YAML,\n"
        "and evaluation report. SHA-256 checksums are recorded in manifest.json.\n"
        "Every model was evaluated on the five complete benchmarks. The allowed\n"
        "deficits are recorded in its report; inspect the actual deltas there.\n\n"
        "From a CondiQuant checkout, install dependencies and run:\n"
        "  python scripts/download_pretrained.py\n"
        "  cp /path/to/this/bundle/cali_data_x*.pth keydata/\n"
        "  bash scripts/test.sh 2 4 /path/to/this/bundle/CondiQuant_x2_w4a4.pth\n\n"
        "See the repository README for dataset layout and full verification commands.\n"
    )
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as stream:
        for path in sorted(output.iterdir()):
            stream.write(path, arcname=path.name)
    print(f"Created {archive}; nine verified checkpoints")


if __name__ == "__main__":
    main()
