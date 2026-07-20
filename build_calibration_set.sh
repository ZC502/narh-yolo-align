#!/usr/bin/env bash
set -euo pipefail

# Minimal NARH-YOLO Align v0.1 test workflow.
#
# Usage:
#   bash examples/build_calibration_set.sh best.pt best_int8.onnx raw_field_frames consistency_workspace
#
# Or with environment overrides:
#   MAX_IMAGES=500 BATCH_SIZE=4 DEVICE=cpu bash examples/build_calibration_set.sh best.pt best_int8.onnx raw_field_frames out
#
# If you only have a .pt model, first export an ONNX target, for example:
#   yolo export model=best.pt format=onnx imgsz=640 opset=12 simplify=True
#
# For v0.1, start with ONNX FP32 / ONNX INT8 before trying TensorRT/RKNN/Hailo.

REF="${1:-best.pt}"
TARGET="${2:-best_int8.onnx}"
SOURCE="${3:-raw_field_frames}"
OUT="${4:-consistency_workspace}"

IMGSZ="${IMGSZ:-640}"
DEVICE="${DEVICE:-cpu}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_IMAGES="${MAX_IMAGES:-5000}"
MAX_CAL_IMAGES="${MAX_CAL_IMAGES:-500}"
NMS_CONF="${NMS_CONF:-0.01}"
DEPLOY_CONF="${DEPLOY_CONF:-0.25}"
IOU="${IOU:-0.70}"
MATCH_IOU="${MATCH_IOU:-0.50}"
REP_RATIO="${REP_RATIO:-0.70}"

if [[ ! -f "$REF" ]]; then
  echo "[ERROR] Reference .pt model not found: $REF" >&2
  exit 1
fi

if [[ ! -f "$TARGET" ]]; then
  echo "[ERROR] Target ONNX model not found: $TARGET" >&2
  echo "Export one first, for example:" >&2
  echo "  yolo export model=$REF format=onnx imgsz=$IMGSZ opset=12 simplify=True" >&2
  exit 1
fi

if [[ ! -d "$SOURCE" ]]; then
  echo "[ERROR] Source image folder not found: $SOURCE" >&2
  exit 1
fi

echo "[NARH-YOLO Align] Running label-free calibration dataset build"
echo "  ref:       $REF"
echo "  target:    $TARGET"
echo "  source:    $SOURCE"
echo "  out:       $OUT"
echo "  imgsz:     $IMGSZ"
echo "  device:    $DEVICE"
echo "  max imgs:  $MAX_IMAGES"
echo ""

narh-yolo-align build \
  --ref "$REF" \
  --target "$TARGET" \
  --source "$SOURCE" \
  --out "$OUT" \
  --imgsz "$IMGSZ" \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" \
  --max-images "$MAX_IMAGES" \
  --max-calibration-images "$MAX_CAL_IMAGES" \
  --nms-conf "$NMS_CONF" \
  --deploy-conf "$DEPLOY_CONF" \
  --iou "$IOU" \
  --match-iou "$MATCH_IOU" \
  --representative-ratio "$REP_RATIO"

echo ""
echo "[NARH-YOLO Align] Done"
echo "Open report:"
echo "  $OUT/report.md"
echo ""
echo "Calibration set:"
echo "  $OUT/calibration_set/"
