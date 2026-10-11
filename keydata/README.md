# Calibration data

Generate one calibration file for each scale using `scripts/prepare_calibration.py`, or run `bash scripts/prepare_calibration.sh` to prepare all three.

Files are named `cali_data_x2.pth`, `cali_data_x3.pth`, and `cali_data_x4.pth`. Each contains `{'lq': tensor}` with shape `[100, 3, 64, 64]`, RGB channel order, float32 dtype, and values in `[0, 1]`. Default sampling seed is 3407. Preserve the exact file alongside its trained checkpoint for evaluation.
