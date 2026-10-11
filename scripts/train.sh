#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
SCALE=${1:-2}
BITS=${2:-4}
python -m basicsr.train -opt "options/train/train_condiquant_x${SCALE}.yml" \
  --force_yml "bit=${BITS}" "name=train_CondiQuant_x${SCALE}_bit${BITS}" "${@:3}"
