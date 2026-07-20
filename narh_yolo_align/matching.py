from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class Detection:
    """Canonical detection after shared postprocess."""

    xyxy: np.ndarray
    score: float
    cls: int


@dataclass
class MatchedPair:
    ref: Detection
    target: Detection
    iou: float


@dataclass
class MatchResult:
    pairs: List[MatchedPair]
    missing_ref: List[Detection]
    extra_target: List[Detection]


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Compute IoU between two xyxy boxes."""
    ax1, ay1, ax2, ay2 = a.astype(float)
    bx1, by1, bx2, by2 = b.astype(float)

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0 else float(inter / denom)


def greedy_match(
    ref: List[Detection],
    target: List[Detection],
    *,
    iou_match_threshold: float = 0.50,
    class_agnostic: bool = True,
) -> MatchResult:
    """Greedy IoU matching for v0.1.

    class_agnostic=True lets the tool detect class flips after spatial matching.
    """
    candidates = []
    for i, r in enumerate(ref):
        for j, t in enumerate(target):
            if not class_agnostic and r.cls != t.cls:
                continue
            iou = box_iou(r.xyxy, t.xyxy)
            if iou >= iou_match_threshold:
                candidates.append((iou, i, j))
    candidates.sort(reverse=True, key=lambda x: x[0])

    used_ref = set()
    used_target = set()
    pairs: List[MatchedPair] = []
    for iou, i, j in candidates:
        if i in used_ref or j in used_target:
            continue
        used_ref.add(i)
        used_target.add(j)
        pairs.append(MatchedPair(ref=ref[i], target=target[j], iou=float(iou)))

    missing = [d for i, d in enumerate(ref) if i not in used_ref]
    extra = [d for j, d in enumerate(target) if j not in used_target]
    return MatchResult(pairs=pairs, missing_ref=missing, extra_target=extra)
