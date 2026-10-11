"""Exercise upload-bundle integrity independently of GPU training."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

import yaml

ROOT = Path(__file__).resolve().parents[1]


class WeightBundleTests(unittest.TestCase):
    def test_package_preserves_verified_files_and_rejects_changed_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reports, calibration = root / "reports", root / "calibration"
            reports.mkdir()
            calibration.mkdir()
            originals = {}
            for scale in (2, 3, 4):
                cal = calibration / f"cali_data_x{scale}.pth"
                cal.write_bytes(f"calibration fixture {scale}".encode())
                for bits in (2, 3, 4):
                    experiment = root / "experiments" / f"train_CondiQuant_x{scale}_bit{bits}"
                    (experiment / "models").mkdir(parents=True)
                    weight = experiment / "models/net_Q_latest.pth"
                    weight.write_bytes(f"checkpoint fixture {scale} {bits}".encode())
                    (experiment / f"train_condiquant_x{scale}.yml").write_text(
                        yaml.safe_dump(dict(scale=scale, bit=bits)))
                    digest = hashlib.sha256(weight.read_bytes()).hexdigest()
                    originals[f"CondiQuant_x{scale}_w{bits}a{bits}.pth"] = digest
                    report = dict(scale=scale, bits=bits, meets_paper=True, fp_only=False,
                                  checkpoint=weight.name, sha256=digest,
                                  calibration_sha256=hashlib.sha256(cal.read_bytes()).hexdigest(),
                                  results=[{} for _ in range(5)])
                    (reports / f"x{scale}_w{bits}a{bits}.json").write_text(json.dumps(report))
            command = [sys.executable, str(ROOT / "scripts/package_weights.py"),
                       "--experiments", str(root / "experiments"), "--reports", str(reports),
                       "--calibration", str(calibration)]
            output = root / "bundle"
            result = subprocess.run(command + ["--output", str(output)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            with zipfile.ZipFile(output.with_suffix(".zip")) as archive:
                for name, digest in originals.items():
                    self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), digest)
                self.assertEqual(len(json.loads(archive.read("manifest.json"))), 9)
                self.assertIn("train_x4_w2a2.yml", archive.namelist())
                self.assertEqual(json.loads(archive.read("x4_w2a2.json"))["checkpoint"], "CondiQuant_x4_w2a2.pth")
            changed = root / "experiments/train_CondiQuant_x2_bit2/models/net_Q_latest.pth"
            changed.write_bytes(b"changed after evaluation")
            rejected = root / "changed_bundle"
            result = subprocess.run(command + ["--output", str(rejected)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("changed since evaluation", result.stderr)
            self.assertFalse(rejected.exists())


if __name__ == "__main__":
    unittest.main()
