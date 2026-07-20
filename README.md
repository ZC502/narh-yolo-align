# NARH-YOLO Align 🎯

**Label-Free Calibration Dataset Builder for YOLO Edge Export**

Stop randomly picking images for INT8 calibration.

NARH-YOLO Align takes an unlabeled folder of production-like images, compares a PyTorch FP32 YOLO model against an exported ONNX model, and automatically builds a residual-guided calibration dataset.

>**The goal is to significantly reduce cross-backend accuracy discrepancies, eliminating the need to compare images one by one to identify quantization accuracy drops and avoid random image sampling by trial and error. On-site unlabeled images can be imported to automatically locate and export drift types, generate residual-guided calibration datasets, and cut cross-backend accuracy discrepancies by over 40% in a single run (the average reduction range of cross-backend recall discrepancies stands at 40% to 60%).**

It helps answer one practical deployment question:

"Which field images make my exported edge model behave differently from the original PyTorch model?"

The output is simple:
```
raw_field_frames/ (Sleeping data)
       ↓
NARH-YOLO Align
       ↓
diagnostic_buckets/ (Classified by drift type)
calibration_set/    (Ready for PTQ/Export)
report.md           (Parity summary)
```
No labels are required.

### 🚀 Quick Start (v0.1)

NARH-YOLO Align currently supports `PyTorch FP32` vs `ONNX (FP32/INT8)` backend parity checks.

(TensorRT, RKNN, and Hailo backends are planned after the ONNX path is stable).
```
narh-yolo-align build \
 --ref best.pt \
 --target best_int8.onnx \
 --source raw_field_frames/ \
 --out consistency_workspace/
```
### 📂 What You Get (The Output)

NARH-YOLO Align instantly acts as a cross-backend residual classifier. It categorizes your unlabeled images based on how severely the ONNX export degraded the predictions.

**1. Diagnostic Buckets**

Your chaotic folder of field images is sorted into interpretable engineering buckets:
```
consistency_workspace/diagnostic_buckets/
├── critical_disagreement/     # Target backend misses reference detections or class flips
├── threshold_crossing/        # Score drops below deploy threshold (e.g., drops below 0.25)
├── score_drift_heavy/         # Matched detections but confidence drift is large
├── box_iou_drift/             # Matched detections but box geometry shifts significantly
└── representative_consistent/ # Stable, highly consistent production-domain frames
```

**2. The Calibration Set**

NARH-YOLO Align automatically generates a perfectly mixed dataset designed specifically to guide your next INT8 quantization export.
```
consistency_workspace/calibration_set/
├── images/ 
└── data.yaml 
```

⚠️ **Important Note:** `data.yaml` is generated exclusively for INT8 calibration/export workflows. NARH-YOLO Align does not create YOLO ground-truth labels. It builds a label-free calibration image set.

**3. The Parity Report (`report.md`)**

A highly readable summary of your export degradation, perfect for engineering audits:

| Metric                     | Value          |
| -------------------------- | -------------- |
| **Reference Backend**          | PyTorch FP32   |
| **Target Backend**              | ONNX INT8      |
| **Critical Disagreement Frames**  | 37           |
| **Threshold Crossing Frames**   | 128            |
| **Calibration Set Size**        | 500 images     |

*(A detailed breakdown of top residual frames is included in the full report)*.

### ⚖️ Default Calibration Mix Strategy

By default, NARH-YOLO Align builds the calibration set using the following ratio to ensure the quantizer sees both standard production variance and edge-case deployment failures:

| Category                          | Default ratio | Purpose                                                         |
| --------------------------------- | ------------- | --------------------------------------------------------------- |
| Representative consistent frames | 70%           | Preserve the normal production-domain distribution.             |
| High-residual frames              | 30%           | Expose the quantizer to deployment-sensitive drift cases.       |

*(Note: Rare/small/low-confidence sampling and customizable ratios will be added in later versions)*.

### 🧠 Why "NARH"?

NARH stands for **Non-Associative Residual Hypothesis**.

In this project, the idea is simple: Different deployment paths are not assumed to be equivalent.

Instead of trusting that a PyTorch FP32 model and an INT8 ONNX model will behave exactly the same in the physical world, NARH-YOLO Align measures cross-backend output residuals between these execution paths. It then uses the highest-residual frames to build a better calibration dataset, effectively reducing deployment drift.

### 🔗 Ecosystem: YERP Factory

NARH-YOLO Align can use any image folder.

>**It can seamlessly interface with the Ultralytics YOLO export quantization workflow, serving as a scenario-specific supplement to official general export tools and helping enterprise users improve the consistency of edge deployment.**

In the future, field hard-cases captured automatically at the edge by **YERP Factory** can be used as direct production-like input frames for NARH-YOLO Align, creating a seamless, label-free MLOps loop from edge detection to local calibration.
