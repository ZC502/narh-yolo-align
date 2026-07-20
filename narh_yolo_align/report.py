from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import json

from narh_yolo_align.residuals import ImageResidual


def write_report(residuals: List[ImageResidual], selected_calibration: List[ImageResidual], *, out_dir: str | Path, reference_backend: str, target_backend: str) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts: Dict[str, int] = {}
    for r in residuals:
        counts[r.bucket] = counts.get(r.bucket, 0) + 1
    top = sorted(residuals, key=lambda r: r.residual_score, reverse=True)[:20]

    report_json = {
        "schema": "narh-yolo-align.report.v0.1",
        "reference_backend": reference_backend,
        "target_backend": target_backend,
        "image_count": len(residuals),
        "bucket_counts": counts,
        "calibration_set_size": len(selected_calibration),
        "top_residual_frames": [
            {"image_path": r.image_path, "bucket": r.bucket, "residual_score": r.residual_score, "reasons": r.reasons, "metrics": r.metrics}
            for r in top
        ],
    }
    (out_dir / "report.json").write_text(json.dumps(report_json, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["# NARH-YOLO Align Report\n", "## Summary\n", "| Item | Value |", "|---|---:|"]
    lines.append(f"| Images scanned | {len(residuals)} |")
    lines.append(f"| Reference backend | {reference_backend} |")
    lines.append(f"| Target backend | {target_backend} |")
    for bucket, count in sorted(counts.items()):
        lines.append(f"| {bucket} | {count} |")
    lines.append(f"| Calibration set size | {len(selected_calibration)} |\n")

    lines += ["## Calibration Mix\n", "| Category | Count |", "|---|---:|"]
    cal_counts: Dict[str, int] = {}
    for r in selected_calibration:
        cal_counts[r.bucket] = cal_counts.get(r.bucket, 0) + 1
    for bucket, count in sorted(cal_counts.items()):
        lines.append(f"| {bucket} | {count} |")

    lines += ["\n## Top Residual Frames\n", "| Rank | Image | Bucket | Residual Score | Reason |", "|---:|---|---|---:|---|"]
    for i, r in enumerate(top, start=1):
        reason = "; ".join(r.reasons)
        lines.append(f"| {i} | `{Path(r.image_path).name}` | `{r.bucket}` | {r.residual_score:.3f} | {reason} |")

    lines.append("\n> Note: label-free backend parity uses the PyTorch FP32 output as a reference path, not as ground-truth labels.")
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
