#!/usr/bin/env bash
set -euo pipefail

narh-yolo-align build \
  --ref best.pt \
  --target best_int8.onnx \
  --source raw_field_frames/ \
  --out consistency_workspace/ \
  --max-images 5000 \
  --batch-size 8
