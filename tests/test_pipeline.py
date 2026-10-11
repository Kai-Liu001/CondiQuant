import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import cv2
import numpy as np

import torch
import yaml
from basicsr.archs import build_network
from basicsr.archs.quant_arch import FakeQuantizerAct, FakeQuantizerWeight, change_singular
from basicsr.models import build_model
from basicsr.data.lr_calibration_dataset import LRCalibrationDataset
from basicsr.train import create_train_val_dataloader, train_pipeline
from basicsr.utils.options import copy_opt_file
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_low_bit_quantizers_and_gradients(self):
        for bit in (2, 3, 4):
            for shape in ((4, 6), (2, 8, 6), (2, 3, 8, 6)):
                x = torch.randn(shape, requires_grad=True)
                q = FakeQuantizerAct(bit)
                q(x)
                y = q(x)
                self.assertTrue(torch.isfinite(y).all())
                self.assertEqual(y.shape, x.shape)
                y.square().mean().backward()
                self.assertTrue(torch.isfinite(q.lower_bound.grad).all())
            q = FakeQuantizerWeight(bit)
            q(torch.zeros(4, 6))
            self.assertTrue(torch.isfinite(q(torch.zeros(4, 6))).all())

    def test_svd_preserves_input_and_improves_condition(self):
        weight = torch.diag(torch.tensor([10., 1., .01]))
        original = weight.clone()
        output = change_singular(weight, torch.eye(3).unsqueeze(0), lambda_reg=.05,
                                 learning_rate=.001, num_iterations=10)
        self.assertTrue(torch.equal(weight, original))
        self.assertLess(torch.linalg.cond(output), torch.linalg.cond(original))
        self.assertEqual(output.device, weight.device)

    def test_train_save_reload_and_reject_wrong_bit(self):
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as temp:
            d = Path(temp)
            netopt = dict(type="SwinIR", upscale=2, img_size=8, window_size=4,
                          img_range=1., depths=[1], embed_dim=12, num_heads=[3],
                          mlp_ratio=2, upsampler="pixelshuffledirect")
            fp = build_network(netopt)
            torch.save({"params": fp.state_dict()}, d / "fp.pth")
            torch.save({"lq": torch.rand(2, 3, 8, 8)}, d / "cali.pth")
            opt = yaml.safe_load((ROOT / "options/train/train_condiquant_x2.yml").read_text())
            opt.update(network_Q=netopt, num_gpu=0, is_train=True, dist=False, rank=0,
                       bit=4, cali_data=str(d / "cali.pth"))
            opt["pathFP"]["pretrain_network_FP"] = str(d / "fp.pth")
            opt["path"]["models"] = str(d)
            opt["path"]["training_states"] = str(d)
            opt["condiquant"]["num_iterations"] = 1
            model = build_model(copy.deepcopy(opt))
            sample = {"lq": torch.rand(2, 3, 8, 8)}
            initial = {n: p.detach().clone() for n, p in model.net_Q.named_parameters() if "bound" in n}
            model.feed_data(sample)
            model.optimize_parameters(1)
            self.assertTrue(any(not torch.equal(initial[n], p) for n, p in model.net_Q.named_parameters() if n in initial))
            self.assertTrue(all(not p.requires_grad for p in model.net_F.parameters()))
            model.test()
            expected = model.output.clone()
            self.assertEqual(tuple(expected.shape), (2, 3, 16, 16))
            self.assertTrue(torch.isfinite(expected).all())
            model.save(0, 1)
            state = torch.load(d / "1.state", weights_only=True)
            self.assertEqual(state["iter"], 1)
            self.assertEqual(len(state["optimizers"]), 1)
            opt.update(is_train=False)
            opt["path"]["pretrain_network_Q"] = str(d / "net_Q_1.pth")
            loaded = build_model(copy.deepcopy(opt))
            loaded.feed_data(sample)
            loaded.test()
            torch.testing.assert_close(expected, loaded.output, atol=0, rtol=0)
            opt["bit"] = 2
            with self.assertRaisesRegex(ValueError, "requested 2-bit"):
                build_model(copy.deepcopy(opt))

    def test_all_paper_configurations_consistent(self):
        for scale in (2, 3, 4):
            for kind in ("train", "test"):
                opt = yaml.safe_load((ROOT / f"options/{kind}/{kind}_condiquant_x{scale}.yml").read_text())
                self.assertEqual(opt["scale"], scale)
                self.assertEqual(opt["network_Q"]["upscale"], scale)
                for metric in opt["val"]["metrics"].values():
                    self.assertEqual(metric["crop_border"], scale)
                if kind == "test":
                    self.assertIn(f"_bit{opt['bit']}/", opt["path"]["pretrain_network_Q"])

    def test_lr_calibration_color_range_and_crop(self):
        with tempfile.TemporaryDirectory() as temp:
            image = np.zeros((16, 16, 3), dtype=np.uint8)
            image[..., 2] = 255  # BGR red
            cv2.imwrite(str(Path(temp) / "red.png"), image)
            dataset = LRCalibrationDataset(dict(dataroot_lq=temp, scale=2, gt_size=16))
            item = dataset[0]["lq"]
            self.assertEqual(tuple(item.shape), (3, 8, 8))
            self.assertTrue(torch.equal(item[0], torch.ones(8, 8)))
            self.assertTrue(torch.equal(item[1:], torch.zeros(2, 8, 8)))

    def test_training_budget_with_incomplete_batch(self):
        with tempfile.TemporaryDirectory() as temp:
            for index in range(5):
                cv2.imwrite(str(Path(temp) / f"{index}.png"), np.zeros((8, 8, 3), np.uint8))
            opt = dict(num_gpu=0, dist=False, world_size=1, rank=0, manual_seed=0,
                       train=dict(total_iter=7), datasets=dict(train=dict(
                           name="tiny", type="LRCalibrationDataset", phase="train",
                           dataroot_lq=temp, scale=2, gt_size=16, batch_size_per_gpu=2,
                           num_worker_per_gpu=0)))
            loader, _, _, epochs, iterations = create_train_val_dataloader(opt, Mock())
            self.assertGreaterEqual(len(loader) * epochs, iterations)

    def test_config_snapshot_preserves_overrides(self):
        with tempfile.TemporaryDirectory() as temp:
            opt = yaml.safe_load((ROOT / "options/train/train_condiquant_x2.yml").read_text())
            opt["bit"] = 2
            opt["condiquant"]["learning_rate"] = .001
            opt["path"]["models"] = "/runtime/path"
            copy_opt_file("train_condiquant_x2.yml", temp, opt)
            saved = yaml.safe_load((Path(temp) / "train_condiquant_x2.yml").read_text())
            self.assertEqual(saved["bit"], 2)
            self.assertEqual(saved["condiquant"]["learning_rate"], .001)
            self.assertNotIn("models", saved["path"])

    def test_final_validation_reports_actual_iteration(self):
        with tempfile.TemporaryDirectory() as temp:
            opt = yaml.safe_load((ROOT / "options/train/train_condiquant_x2.yml").read_text())
            opt.update(num_gpu=0, is_train=True, auto_resume=False, dist=False, rank=0, world_size=1)
            opt["train"]["total_iter"] = 7
            opt["logger"]["print_freq"] = 100
            opt["val"]["val_freq"] = 7
            for name in ("experiments_root", "models", "training_states", "log", "visualization"):
                opt["path"][name] = str(Path(temp) / name)
            model = Mock()
            loaders = ([{}, {}], Mock(), [Mock()], 4, 7)
            args = SimpleNamespace(opt=str(ROOT / "options/train/train_condiquant_x2.yml"))
            with patch("basicsr.train.parse_options", return_value=(opt, args)), \
                    patch("basicsr.train.create_train_val_dataloader", return_value=loaders), \
                    patch("basicsr.train.build_model", return_value=model), \
                    patch("basicsr.train.get_root_logger", return_value=Mock()), \
                    patch("basicsr.train.get_env_info", return_value="test"):
                train_pipeline(temp)
            self.assertEqual([call.args[0] for call in model.optimize_parameters.call_args_list], list(range(1, 8)))
            self.assertEqual([call.args[1] for call in model.validation.call_args_list], [7, 7])

    def test_metric_comparison_requires_all_benchmarks(self):
        spec = importlib.util.spec_from_file_location("evaluation", ROOT / "scripts/evaluate_checkpoint.py")
        evaluation = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evaluation)
        with self.assertRaisesRegex(ValueError, "all five"):
            evaluation.compare_results({"Set5": {"psnr": 40., "ssim": 1.}}, 2, 4)
        measured = {name: {"psnr": 40., "ssim": 1.} for name in evaluation.COUNTS}
        measured["Urban100"]["psnr"] = 31.
        rows = evaluation.compare_results(measured, 2, 4)
        self.assertFalse(all(row["meets_paper"] for row in rows))


if __name__ == "__main__":
    unittest.main()
