from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple

import numpy as np
import torch
from ultralytics import YOLO


@dataclass
class PreprocessBatch:
    """Shared preprocessed tensor and transform metadata.

    v0.1 parity rule:
    The ONNX backend should receive this exact tensor after conversion to numpy.
    Do not reimplement preprocessing separately in the ONNX adapter.

    Coordinate rule:
    `meta[i]["ratio_pad"]` is required by postprocess.py to map model-space
    boxes back to the original image coordinate system.
    """

    tensor: torch.Tensor  # BCHW float tensor, normalized, letterboxed
    original_shapes: List[Tuple[int, int]]  # [(h, w), ...]
    preprocessed_shape: Tuple[int, int]  # (h, w)
    meta: List[dict]


def _compute_letterbox_ratio_pad(
    original_shape: Tuple[int, int],
    preprocessed_shape: Tuple[int, int],
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Compute ratio/pad metadata for inverse box scaling.

    This mirrors standard centered letterbox geometry:
    - preserve aspect ratio
    - scale by min(new_h / old_h, new_w / old_w)
    - pad equally on both sides

    Note:
    This assumes the Ultralytics preprocessing path used ordinary centered
    letterbox behavior. If you later enable unusual options such as scale_fill,
    non-centered padding, or backend-specific pre-transform behavior, add a
    parity test and adapt this function accordingly.
    """
    old_h, old_w = original_shape
    new_h, new_w = preprocessed_shape

    if old_h <= 0 or old_w <= 0 or new_h <= 0 or new_w <= 0:
        raise ValueError(
            f"Invalid shapes for ratio_pad: original={original_shape}, "
            f"preprocessed={preprocessed_shape}"
        )

    gain = min(new_h / old_h, new_w / old_w)
    unpad_w = round(old_w * gain)
    unpad_h = round(old_h * gain)

    pad_w = (new_w - unpad_w) / 2.0
    pad_h = (new_h - unpad_h) / 2.0

    # Ultralytics scale_boxes commonly represents ratio_pad as:
    # ((gain_w, gain_h), (pad_w, pad_h)).
    return (float(gain), float(gain)), (float(pad_w), float(pad_h))


class PyTorchYOLOBackend:
    """Reference backend using Ultralytics YOLO .pt model.

    Important:
    This backend owns preprocessing in v0.1. The ONNX target backend should
    reuse the exact output tensor from `preprocess`.
    """

    def __init__(self, weights: str, *, device: str | None = None, imgsz: int = 640):
        self.model = YOLO(weights)
        self.device = device
        self.imgsz = imgsz

    def preprocess(self, images_bgr: Sequence[np.ndarray]) -> PreprocessBatch:
        """Run Ultralytics-native preprocessing and expose the exact tensor.

        Preprocessing parity is the first invariant of this project:
        `.pt` and `.onnx` must receive the same preprocessed tensor, otherwise
        residuals measure preprocessing bugs instead of export/quantization drift.
        """
        if not images_bgr:
            raise ValueError("preprocess() received an empty image batch")

        predictor = self.model.predictor
        if predictor is None:
            # Lazy setup through Ultralytics. Keep tensor hashes in cli.py as
            # a regression check because predictor internals can change.
            self.model.predict(
                images_bgr[0],
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )
            predictor = self.model.predictor

        # predictor.preprocess expects a list of BGR numpy images.
        tensor = predictor.preprocess(list(images_bgr))
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Ultralytics predictor.preprocess did not return a torch.Tensor")

        original_shapes = [(int(im.shape[0]), int(im.shape[1])) for im in images_bgr]
        preprocessed_shape = (int(tensor.shape[-2]), int(tensor.shape[-1]))

        meta: List[dict] = []
        for original_shape in original_shapes:
            ratio_pad = _compute_letterbox_ratio_pad(
                original_shape=original_shape,
                preprocessed_shape=preprocessed_shape,
            )
            meta.append(
                {
                    "original_shape": original_shape,
                    "preprocessed_shape": preprocessed_shape,
                    "ratio_pad": ratio_pad,
                    "preprocess_owner": "ultralytics_pytorch",
                    "preprocess_parity_note": (
                        "ONNX backend must consume this exact tensor. "
                        "Postprocess must use ratio_pad to scale boxes back."
                    ),
                }
            )

        return PreprocessBatch(
            tensor=tensor,
            original_shapes=original_shapes,
            preprocessed_shape=preprocessed_shape,
            meta=meta,
        )

    @torch.no_grad()
    def raw_forward(self, batch: PreprocessBatch) -> Any:
        """Run raw model forward on the shared tensor.

        Shared-NMS rule:
        This returns raw predictions before project-level postprocess. Both
        PyTorch and ONNX outputs must be decoded and filtered by one shared
        postprocess implementation.
        """
        model = self.model.model
        model.eval()

        x = batch.tensor
        if self.device:
            x = x.to(self.device)
            model = model.to(self.device)

        preds = model(x)
        return preds
