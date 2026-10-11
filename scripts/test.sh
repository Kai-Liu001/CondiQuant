#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
SCALE=${1:-2}
BITS=${2:-4}
CHECKPOINT=${3:-experiments/train_CondiQuant_x${SCALE}_bit${BITS}/models/net_Q_latest.pth}
python -m basicsr.test -opt "options/test/test_condiquant_x${SCALE}.yml" \
  --force_yml "bit=${BITS}" "name=test_CondiQuant_x${SCALE}_bit${BITS}" \
  "path:pretrain_network_Q=${CHECKPOINT}" "${@:4}"
