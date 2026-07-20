from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from narh_yolo_align.matching import MatchResult


@dataclass
class ImageResidual:
    image_path: str
    bucket: str
    residual_score: float
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


def classify_residual(
    image_path: str,
    match: MatchResult,
    *,
    deploy_conf: float = 0.25,
    heavy_score_drift_threshold: float = 0.10,
    iou_drift_threshold: float = 0.75,
) -> ImageResidual:
    """Classify one image into the strongest residual bucket."""
    reasons: List[str] = []
    metrics: Dict[str, Any] = {}

    missing_count = len(match.missing_ref)
    extra_count = len(match.extra_target)
    class_flips = 0
    threshold_crossings = 0
    heavy_score_drifts = 0
    box_iou_drifts = 0
    max_score_drift = 0.0
    min_iou = 1.0 if match.pairs else 0.0

    for pair in match.pairs:
        ref, target = pair.ref, pair.target
        score_drift = abs(ref.score - target.score)
        max_score_drift = max(max_score_drift, score_drift)
        min_iou = min(min_iou, pair.iou)
        if ref.cls != target.cls:
            class_flips += 1
        if ref.score >= deploy_conf and target.score < deploy_conf:
            threshold_crossings += 1
        if score_drift >= heavy_score_drift_threshold:
            heavy_score_drifts += 1
        if pair.iou < iou_drift_threshold:
            box_iou_drifts += 1

    metrics.update({
        "matched_pairs": len(match.pairs),
        "missing_ref": missing_count,
        "extra_target": extra_count,
        "class_flips": class_flips,
        "threshold_crossings": threshold_crossings,
        "heavy_score_drifts": heavy_score_drifts,
        "box_iou_drifts": box_iou_drifts,
        "max_score_drift": max_score_drift,
        "min_iou": min_iou,
    })

    if missing_count > 0 or class_flips > 0:
        bucket = "critical_disagreement"
        reasons.append("target backend missed reference detections or class changed")
        residual_score = 100.0 + missing_count * 10.0 + class_flips * 10.0 + max_score_drift
    elif threshold_crossings > 0:
        bucket = "threshold_crossing"
        reasons.append("score dropped below deployment confidence threshold")
        residual_score = 80.0 + threshold_crossings * 10.0 + max_score_drift
    elif heavy_score_drifts > 0:
        bucket = "score_drift_heavy"
        reasons.append("matched detections show large confidence drift")
        residual_score = 60.0 + heavy_score_drifts * 10.0 + max_score_drift
    elif box_iou_drifts > 0:
        bucket = "box_iou_drift"
        reasons.append("matched boxes shifted after export")
        residual_score = 40.0 + box_iou_drifts * 10.0 + (1.0 - min_iou)
    else:
        bucket = "representative_consistent"
        reasons.append("reference and target outputs are consistent under current thresholds")
        residual_score = 0.0

    return ImageResidual(image_path=image_path, bucket=bucket, residual_score=float(residual_score), reasons=reasons, metrics=metrics)
