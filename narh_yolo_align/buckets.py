from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List
import shutil

from narh_yolo_align.residuals import ImageResidual


BUCKETS = [
    "critical_disagreement",
    "threshold_crossing",
    "score_drift_heavy",
    "box_iou_drift",
    "representative_consistent",
]


def group_by_bucket(residuals: Iterable[ImageResidual]) -> Dict[str, List[ImageResidual]]:
    grouped: Dict[str, List[ImageResidual]] = defaultdict(list)
    for r in residuals:
        grouped[r.bucket].append(r)
    for items in grouped.values():
        items.sort(key=lambda x: x.residual_score, reverse=True)
    return dict(grouped)


def materialize_buckets(grouped: Dict[str, List[ImageResidual]], *, out_dir: str | Path, copy_images: bool = True) -> None:
    """Create diagnostic bucket directories."""
    root = Path(out_dir) / "diagnostic_buckets"
    root.mkdir(parents=True, exist_ok=True)
    for bucket in BUCKETS:
        bdir = root / bucket
        bdir.mkdir(parents=True, exist_ok=True)
        for r in grouped.get(bucket, []):
            src = Path(r.image_path)
            dst = bdir / src.name
            if copy_images and src.exists():
                shutil.copy2(src, dst)
