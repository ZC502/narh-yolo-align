from __future__ import annotations

from typing import Any, List, Sequence, Tuple

import numpy as np

from narh_yolo_align.matching import Detection
from narh_yolo_align.backends.pytorch_yolo import PreprocessBatch


EPS = 1e-9


def shared_postprocess(
    raw_preds: Any,
    batch: PreprocessBatch,
    *,
    nms_conf: float = 0.01,
    iou: float = 0.70,
    max_det: int = 300,
    class_agnostic: bool = False,
    has_objectness: bool = False,
) -> List[List[Detection]]:
    """Shared decode + NMS for both PyTorch and ONNX outputs.

    This function is the core v0.1 parity gate:
    - Both reference (.pt) and target (.onnx) raw outputs must flow through
      this identical decode and NMS implementation.
    - Residuals are only meaningful if postprocess is unified.

    Expected input shape after normalization:
        [B, N, C]
    where:
        B = batch size
        N = number of anchors / predictions
        C = 4 + num_classes           (YOLOv8/11/26 detect common case)
          or 5 + num_classes          (if objectness exists, older style)

    Args:
        raw_preds:
            Raw backend output. May be:
            - torch.Tensor
            - tuple(torch.Tensor, ...)
            - list[np.ndarray]
            - np.ndarray
        batch:
            Shared preprocessing batch with ratio_pad metadata.
        nms_conf:
            Internal low confidence threshold for retaining boxes before NMS.
            This should typically be lower than the actual deployment threshold.
        iou:
            NMS IoU threshold.
        max_det:
            Maximum number of detections kept per image.
        class_agnostic:
            If True, perform class-agnostic NMS.
        has_objectness:
            If True, interpret predictions as [x, y, w, h, obj, cls...].
            For YOLOv8/11/26 detect exports, this is usually False.

    Returns:
        List[List[Detection]]:
            Outer list over images in batch, inner list is sorted detections.
    """
    preds = _unwrap_and_normalize_predictions(raw_preds)

    batch_size = preds.shape[0]
    if batch_size != len(batch.meta):
        raise ValueError(
            f"Postprocess batch size mismatch: preds batch={batch_size}, "
            f"meta batch={len(batch.meta)}"
        )

    outputs: List[List[Detection]] = []
    for i in range(batch_size):
        per_image = preds[i]  # [N, C]
        meta = batch.meta[i]
        dets = _decode_one_image(
            per_image,
            meta=meta,
            nms_conf=nms_conf,
            iou=iou,
            max_det=max_det,
            class_agnostic=class_agnostic,
            has_objectness=has_objectness,
        )
        outputs.append(dets)
    return outputs


def _unwrap_and_normalize_predictions(raw_preds: Any) -> np.ndarray:
    """Unwrap backend outputs and normalize shape to [B, N, C].

    Handles:
    - PyTorch raw outputs as Tensor or tuple(Tensor, ...)
    - ONNX Runtime outputs as list[np.ndarray]
    - Already-normalized np.ndarray

    Shape heuristic:
    Many YOLO exports come as [B, C, N] e.g. [1, 84, 8400].
    We standardize this to [B, N, C].
    """
    # 1) unwrap common outer containers
    if isinstance(raw_preds, (tuple, list)):
        if len(raw_preds) == 0:
            raise ValueError("raw_preds is an empty tuple/list")
        raw_preds = raw_preds[0]

    # 2) torch.Tensor -> numpy
    if hasattr(raw_preds, "detach"):
        raw_preds = raw_preds.detach().cpu().numpy()

    if not isinstance(raw_preds, np.ndarray):
        raw_preds = np.asarray(raw_preds)

    # 3) normalize dimensionality
    if raw_preds.ndim == 2:
        # [N, C] -> [1, N, C]
        raw_preds = raw_preds[None, ...]
    elif raw_preds.ndim != 3:
        raise ValueError(
            f"Unsupported raw prediction ndim={raw_preds.ndim}, "
            f"expected 2D or 3D array after unwrap"
        )

    # 4) normalize to [B, N, C]
    # Common case from YOLO export: [B, 84, 8400] -> transpose
    # Heuristic: middle dim is channels if it is "small", last dim is anchors if large.
    b, d1, d2 = raw_preds.shape
    if d1 <= 256 and d2 > d1:
        raw_preds = np.transpose(raw_preds, (0, 2, 1))  # [B, C, N] -> [B, N, C]

    # Final sanity
    if raw_preds.ndim != 3:
        raise ValueError("Prediction normalization failed to produce a 3D tensor")

    if raw_preds.shape[-1] < 6:
        # Need at least x,y,w,h + 2 class logits/scores or obj+cls
        raise ValueError(
            f"Prediction channel dimension too small after normalization: shape={raw_preds.shape}"
        )

    return np.ascontiguousarray(raw_preds)


def _decode_one_image(
    pred: np.ndarray,
    *,
    meta: dict,
    nms_conf: float,
    iou: float,
    max_det: int,
    class_agnostic: bool,
    has_objectness: bool,
) -> List[Detection]:
    """Decode one image worth of predictions into canonical Detection objects."""
    if pred.ndim != 2:
        raise ValueError(f"Expected per-image prediction shape [N, C], got {pred.shape}")

    if pred.shape[0] == 0:
        return []

    # Parse columns
    xywh = pred[:, :4]

    if has_objectness:
        if pred.shape[1] < 6:
            return []
        obj = pred[:, 4]
        cls_scores = pred[:, 5:]
        if cls_scores.shape[1] == 0:
            return []
        cls_ids = np.argmax(cls_scores, axis=1)
        cls_conf = cls_scores[np.arange(cls_scores.shape[0]), cls_ids]
        scores = obj * cls_conf
    else:
        cls_scores = pred[:, 4:]
        if cls_scores.shape[1] == 0:
            return []
        cls_ids = np.argmax(cls_scores, axis=1)
        scores = cls_scores[np.arange(cls_scores.shape[0]), cls_ids]

    # Filter by low internal threshold
    keep = scores >= float(nms_conf)
    if not np.any(keep):
        return []

    xywh = xywh[keep]
    scores = scores[keep]
    cls_ids = cls_ids[keep]

    # Convert model-space xywh -> xyxy
    boxes = _xywh_to_xyxy(xywh)

    # Map back to original image coordinates
    boxes = _scale_boxes_to_original(boxes, meta)

    # Clip to image bounds
    h0, w0 = meta["original_shape"]
    boxes = _clip_boxes(boxes, w=w0, h=h0)

    # Remove degenerate boxes
    wh = boxes[:, 2:4] - boxes[:, 0:2]
    valid = (wh[:, 0] > 1.0) & (wh[:, 1] > 1.0)
    if not np.any(valid):
        return []

    boxes = boxes[valid]
    scores = scores[valid]
    cls_ids = cls_ids[valid]

    # NMS
    keep_idx = _nms(
        boxes,
        scores,
        cls_ids,
        iou_threshold=float(iou),
        max_det=max_det,
        class_agnostic=class_agnostic,
    )
    if len(keep_idx) == 0:
        return []

    boxes = boxes[keep_idx]
    scores = scores[keep_idx]
    cls_ids = cls_ids[keep_idx]

    # Sort by confidence descending for deterministic output
    order = np.argsort(-scores)
    detections: List[Detection] = []
    for j in order:
        detections.append(
            Detection(
                xyxy=boxes[j].astype(np.float32),
                score=float(scores[j]),
                cls=int(cls_ids[j]),
            )
        )
    return detections


def _xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    """Convert [x_c, y_c, w, h] -> [x1, y1, x2, y2]."""
    out = np.empty_like(xywh, dtype=np.float32)
    out[:, 0] = xywh[:, 0] - xywh[:, 2] / 2.0
    out[:, 1] = xywh[:, 1] - xywh[:, 3] / 2.0
    out[:, 2] = xywh[:, 0] + xywh[:, 2] / 2.0
    out[:, 3] = xywh[:, 1] + xywh[:, 3] / 2.0
    return out


def _scale_boxes_to_original(boxes: np.ndarray, meta: dict) -> np.ndarray:
    """Inverse letterbox mapping from model space to original image space."""
    ratio_pad = meta.get("ratio_pad", None)
    if ratio_pad is None:
        raise ValueError("Missing ratio_pad in preprocess metadata")

    (gain_w, gain_h), (pad_w, pad_h) = ratio_pad
    gain_w = float(gain_w)
    gain_h = float(gain_h)
    pad_w = float(pad_w)
    pad_h = float(pad_h)

    out = boxes.astype(np.float32).copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - pad_w) / max(gain_w, EPS)
    out[:, [1, 3]] = (out[:, [1, 3]] - pad_h) / max(gain_h, EPS)
    return out


def _clip_boxes(boxes: np.ndarray, *, w: int, h: int) -> np.ndarray:
    """Clip xyxy boxes to image bounds."""
    out = boxes.copy()
    out[:, 0] = np.clip(out[:, 0], 0, max(w - 1, 0))
    out[:, 1] = np.clip(out[:, 1], 0, max(h - 1, 0))
    out[:, 2] = np.clip(out[:, 2], 0, max(w - 1, 0))
    out[:, 3] = np.clip(out[:, 3], 0, max(h - 1, 0))
    return out


def _nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    cls_ids: np.ndarray,
    *,
    iou_threshold: float,
    max_det: int,
    class_agnostic: bool,
) -> List[int]:
    """Pure NumPy NMS.

    Returns indices into the input arrays.
    """
    if boxes.shape[0] == 0:
        return []

    order = np.argsort(-scores)
    keep: List[int] = []

    if class_agnostic:
        return _nms_single_class(
            boxes,
            scores,
            order,
            iou_threshold=iou_threshold,
            max_det=max_det,
        )

    # class-aware NMS
    for cls in np.unique(cls_ids[order]):
        cls_mask = cls_ids[order] == cls
        cls_order = order[cls_mask]
        cls_keep = _nms_single_class(
            boxes,
            scores,
            cls_order,
            iou_threshold=iou_threshold,
            max_det=max_det,
        )
        keep.extend(cls_keep)

    # Global score ordering after per-class keep
    keep = sorted(keep, key=lambda idx: float(scores[idx]), reverse=True)
    if max_det > 0:
        keep = keep[:max_det]
    return keep


def _nms_single_class(
    boxes: np.ndarray,
    scores: np.ndarray,
    order: np.ndarray,
    *,
    iou_threshold: float,
    max_det: int,
) -> List[int]:
    """Standard greedy NMS on a subset of indices."""
    keep: List[int] = []
    idxs = order.copy()

    while idxs.size > 0:
        i = int(idxs[0])
        keep.append(i)

        if max_det > 0 and len(keep) >= max_det:
            break

        if idxs.size == 1:
            break

        rest = idxs[1:]
        ious = _iou_one_to_many(boxes[i], boxes[rest])
        idxs = rest[ious < iou_threshold]

    return keep


def _iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU between one xyxy box and many xyxy boxes."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_a = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    area_b = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])

    union = area_a + area_b - inter
    return inter / np.maximum(union, EPS)
