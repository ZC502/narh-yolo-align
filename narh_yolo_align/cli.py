from __future__ import annotations

import argparse
from pathlib import Path
from typing import List
import hashlib
import json

from narh_yolo_align.image_loader import list_images, iter_image_batches
from narh_yolo_align.backends.pytorch_yolo import PyTorchYOLOBackend
from narh_yolo_align.backends.onnx_yolo import ONNXYOLOBackend
from narh_yolo_align.postprocess import shared_postprocess
from narh_yolo_align.matching import greedy_match
from narh_yolo_align.residuals import ImageResidual, classify_residual
from narh_yolo_align.buckets import group_by_bucket, materialize_buckets
from narh_yolo_align.calibration_builder import build_calibration_set
from narh_yolo_align.report import write_report


def _tensor_hash(batch) -> str:
    """Hash shared preprocessing tensor for parity debugging.

    Works for both torch.Tensor and numpy.ndarray.
    """
    arr = batch.tensor
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()

    # Defensive: make hashing stable even if the array is non-contiguous.
    arr = arr.copy(order="C") if hasattr(arr, "flags") and not arr.flags["C_CONTIGUOUS"] else arr
    return hashlib.sha256(arr.tobytes()).hexdigest()


def build(args: argparse.Namespace) -> int:
    """Run one-shot label-free backend residual diagnosis and calibration-set build."""
    if args.nms_conf >= args.deploy_conf:
        raise SystemExit(
            f"--nms-conf ({args.nms_conf}) must be lower than --deploy-conf "
            f"({args.deploy_conf}) so threshold_crossing cases are observable."
        )

    if not (0.0 < args.representative_ratio < 1.0):
        raise SystemExit("--representative-ratio must be between 0 and 1.")

    paths = list_images(
        args.source,
        max_images=args.max_images,
        seed=args.seed,
        shuffle=not args.no_shuffle,
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not paths:
        raise SystemExit(f"No images found under {args.source}")

    print("[NARH-YOLO Align] Label-free calibration dataset builder")
    print(f"[NARH-YOLO Align] Reference backend: PyTorch FP32 ({args.ref})")
    print(f"[NARH-YOLO Align] Target backend: ONNX target ({args.target})")
    print(f"[NARH-YOLO Align] Images selected: {len(paths)}")
    print(f"[NARH-YOLO Align] Output workspace: {out_dir}")
    print("[NARH-YOLO Align] Loading backends...")

    ref = PyTorchYOLOBackend(args.ref, device=args.device, imgsz=args.imgsz)
    target = ONNXYOLOBackend(args.target)

    residuals: List[ImageResidual] = []

    debug_path = out_dir / "preprocess_hashes.jsonl"
    with debug_path.open("w", encoding="utf-8") as debug_f:
        for batch_idx, records in enumerate(
            iter_image_batches(paths, batch_size=args.batch_size),
            start=1,
        ):
            images = [r.image_bgr for r in records]
            pp = ref.preprocess(images)

            debug_f.write(
                json.dumps(
                    {
                        "batch": batch_idx,
                        "image_count": len(records),
                        "preprocess_tensor_sha256": _tensor_hash(pp),
                        "preprocessed_shape": pp.preprocessed_shape,
                        "meta": pp.meta,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            ref_raw = ref.raw_forward(pp)
            target_raw = target.raw_forward(pp)

            # Shared decode + NMS: this is the v0.1 parity gate.
            # Both raw outputs must pass through the same NumPy postprocess path.
            ref_dets = shared_postprocess(
                ref_raw,
                pp,
                nms_conf=args.nms_conf,
                iou=args.iou,
                max_det=args.max_det,
                class_agnostic=args.class_agnostic_nms,
                has_objectness=args.has_objectness,
            )
            target_dets = shared_postprocess(
                target_raw,
                pp,
                nms_conf=args.nms_conf,
                iou=args.iou,
                max_det=args.max_det,
                class_agnostic=args.class_agnostic_nms,
                has_objectness=args.has_objectness,
            )

            for rec, rd, td in zip(records, ref_dets, target_dets):
                match = greedy_match(
                    rd,
                    td,
                    iou_match_threshold=args.match_iou,
                    class_agnostic=True,  # required to detect class flips
                )
                residual = classify_residual(
                    str(rec.path),
                    match,
                    deploy_conf=args.deploy_conf,
                    score_drift_threshold=args.score_drift_threshold,
                    heavy_score_drift_threshold=args.heavy_score_drift_threshold,
                    iou_drift_threshold=args.iou_drift_threshold,
                )
                residuals.append(residual)

            print(
                f"[NARH-YOLO Align] Processed batch {batch_idx} "
                f"({len(residuals)}/{len(paths)} images)"
            )

    if not residuals:
        raise SystemExit(
            "No valid images were processed. Check image paths and OpenCV decoding support."
        )

    grouped = group_by_bucket(residuals)
    materialize_buckets(grouped, out_dir=out_dir, copy_images=not args.manifest_only)

    selected = build_calibration_set(
        grouped,
        out_dir=out_dir,
        max_images=args.max_calibration_images,
        representative_ratio=args.representative_ratio,
    )

    write_report(
        residuals,
        selected,
        out_dir=out_dir,
        reference_backend="PyTorch FP32",
        target_backend="ONNX target",
    )

    print("")
    print("[NARH-YOLO Align] Done")
    print(f"[NARH-YOLO Align] Report: {out_dir / 'report.md'}")
    print(f"[NARH-YOLO Align] JSON: {out_dir / 'report.json'}")
    print(f"[NARH-YOLO Align] Buckets: {out_dir / 'diagnostic_buckets'}")
    print(f"[NARH-YOLO Align] Calibration set: {out_dir / 'calibration_set'}")
    print(
        "[NARH-YOLO Align] Note: generated data.yaml is for INT8 calibration/export, "
        "not supervised YOLO training labels."
    )
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="narh-yolo-align",
        description=(
            "Build a label-free residual-guided calibration dataset for YOLO edge export."
        ),
    )
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("build", help="Build diagnostic buckets and calibration set.")
    p.add_argument("--ref", required=True, help="Reference PyTorch YOLO .pt model.")
    p.add_argument("--target", required=True, help="Target ONNX model.")
    p.add_argument("--source", required=True, help="Folder of unlabeled production-like images.")
    p.add_argument("--out", default="consistency_workspace", help="Output workspace.")

    p.add_argument("--imgsz", type=int, default=640, help="Image size used by Ultralytics preprocessing.")
    p.add_argument("--device", default=None, help="PyTorch device, e.g. cpu, cuda:0. Default uses Ultralytics default.")
    p.add_argument("--batch-size", type=int, default=8, help="Batch size for sequential inference.")
    p.add_argument("--max-images", type=int, default=5000, help="OOM protection: max images sampled from source.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for source image sampling.")
    p.add_argument("--no-shuffle", action="store_true", help="Disable source image shuffling before max-images cap.")
    p.add_argument("--manifest-only", action="store_true", help="Do not copy bucket images; still writes reports and calibration set.")

    # Important: internal NMS filtering must be lower than deployment threshold,
    # otherwise threshold_crossing cannot be detected.
    p.add_argument(
        "--nms-conf",
        type=float,
        default=0.01,
        help="Low internal threshold used by shared postprocess to preserve drifted boxes.",
    )
    p.add_argument("--deploy-conf", type=float, default=0.25, help="Actual deployment confidence threshold.")
    p.add_argument("--iou", type=float, default=0.70, help="Common NMS IoU threshold.")
    p.add_argument("--max-det", type=int, default=300, help="Maximum detections kept per image after shared NMS.")
    p.add_argument(
        "--class-agnostic-nms",
        action="store_true",
        help="Use class-agnostic shared NMS. Default is class-aware NMS.",
    )
    p.add_argument(
        "--has-objectness",
        action="store_true",
        help="Interpret predictions as [x,y,w,h,obj,cls...]. Default assumes YOLOv8/11 style [x,y,w,h,cls...].",
    )

    p.add_argument("--match-iou", type=float, default=0.50, help="IoU threshold for backend matching.")
    p.add_argument("--score-drift-threshold", type=float, default=0.05)
    p.add_argument("--heavy-score-drift-threshold", type=float, default=0.10)
    p.add_argument("--iou-drift-threshold", type=float, default=0.75)

    p.add_argument("--max-calibration-images", type=int, default=500)
    p.add_argument(
        "--representative-ratio",
        type=float,
        default=0.70,
        help="Default calibration mix ratio for representative_consistent frames.",
    )

    p.set_defaults(func=build)
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        raise SystemExit(2)
    raise SystemExit(args.func(args))
