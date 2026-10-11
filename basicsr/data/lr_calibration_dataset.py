"""LR-only calibration images; distillation targets come from the FP teacher."""
from pathlib import Path
import random

import cv2
from torch.utils.data import Dataset

from basicsr.data.transforms import augment
from basicsr.utils import img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class LRCalibrationDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        root = Path(opt["dataroot_lq"])
        self.paths = sorted(p for p in root.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
        if not self.paths:
            raise FileNotFoundError(f"No calibration images in {root}")
        self.patch_size = opt.get("gt_size", 64 * opt["scale"]) // opt["scale"]
        self.images = []
        for path in self.paths:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None or min(image.shape[:2]) < self.patch_size:
                raise ValueError(f"Invalid or too-small calibration image: {path}")
            self.images.append(image)  # Keep uint8 images cached; normalize only cropped patches.

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        height, width = image.shape[:2]
        size = self.patch_size
        top, left = random.randint(0, height - size), random.randint(0, width - size)
        patch = image[top:top + size, left:left + size].copy()
        patch = augment([patch], self.opt.get("use_hflip", True), self.opt.get("use_rot", True))
        lq = img2tensor(patch, bgr2rgb=True, float32=True) / 255.
        return {"lq": lq, "lq_path": str(self.paths[index])}
