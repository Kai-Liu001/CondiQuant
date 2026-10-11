#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for SCALE in 2 3 4; do
  python scripts/prepare_calibration.py \
    --lr-dir "datasets/DF2K/LR_bicubic/X${SCALE}" \
    --output "keydata/cali_data_x${SCALE}.pth"
done
