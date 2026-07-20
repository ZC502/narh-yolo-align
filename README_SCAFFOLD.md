# NARH-YOLO Align v0.1 Scaffold

Minimal skeleton for:

```text
PyTorch FP32 reference
vs.
ONNX FP32 / ONNX INT8 target
        ↓
label-free residual classification
        ↓
residual-guided calibration_set
```

v0.1 design constraints:

- Preprocessing parity: both backends receive the same tensor.
- NMS alignment: both backends are evaluated through one shared postprocess path.
- OOM protection: streaming image loader, `--max-images`, bounded batch size.
