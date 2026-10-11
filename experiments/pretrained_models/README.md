# Models and verification status

Download the official SwinIR-light FP models with `python scripts/download_pretrained.py`. The source is the [official SwinIR release](https://github.com/JingyunLiang/SwinIR/releases/tag/v0.0).

All nine combinations of ×2/×3/×4 and W2A2/W3A3/W4A4 have been prepared and evaluated on the five complete benchmarks against the camera-ready paper's Table 3. The [measured results and reports](../../docs/verification/README.md) include every metric, paper delta, image count, and checkpoint checksum. All 45 comparisons pass the stated 0.05 dB / 0.0005 SSIM deficit criterion.

The upload archive is named `CondiQuant_verified.zip`. It contains nine checkpoints, three calibration files, nine effective training YAML files, nine reports, and a checksum manifest. Download links will be added once the authors upload this bundle.

The original FP checkpoints are starting points, not CondiQuant quantized models. A `.pth` file produced by this repository is a fake-quantization checkpoint containing FP tensors, calibrated bounds, fixed bit-width values, and learnable activation biases.
