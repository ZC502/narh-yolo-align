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
    """

    tensor: torch.Tensor
    original_shapes: List[Tuple[int, int]]
    preprocessed_shape: Tuple[int, int]
    meta: List[dict]


class PyTorchYOLOBackend:
    """Reference backend using Ultralytics YOLO .pt model.

    This backend owns preprocessing in v0.1. The ONNX target backend should
    reuse the output tensor from `preprocess`.
    """

    def __init__(self, weights: str, *, device: str | None = None, imgsz: int = 640):
        self.model = YOLO(weights)
        self.device = device
        self.imgsz = imgsz

    def preprocess(self, images_bgr: Sequence[np.ndarray]) -> PreprocessBatch:
        """Run Ultralytics-native preprocessing and expose the exact tensor.

        Implementation note:
        Ultralytics internals can change between versions. Before trusting
        residual metrics, verify tensor parity with the generated SHA256 hash.
        """
        predictor = self.model.predictor
        if predictor is None:
            self.model.predict(images_bgr[0], imgsz=self.imgsz, device=self.device, verbose=False)
            predictor = self.model.predictor

        tensor = predictor.preprocess(list(images_bgr))
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Ultralytics predictor.preprocess did not return a torch.Tensor")

        original_shapes = [(int(im.shape[0]), int(im.shape[1])) for im in images_bgr]
        preprocessed_shape = (int(tensor.shape[-2]), int(tensor.shape[-1]))
        meta = [
            {
                "original_shape": original_shapes[i],
                "preprocessed_shape": preprocessed_shape,
                "preprocess_owner": "ultralytics_pytorch",
            }
            for i in range(len(images_bgr))
        ]

        return PreprocessBatch(tensor=tensor, original_shapes=original_shapes, preprocessed_shape=preprocessed_shape, meta=meta)

    @torch.no_grad()
    def raw_forward(self, batch: PreprocessBatch) -> Any:
        """Run raw model forward on the shared tensor.

        Shared-NMS rule:
        This returns raw predictions before project-level postprocess.
        Compare outputs only after one common postprocess function is applied.
        """
        model = self.model.model
        model.eval()
        x = batch.tensor
        if self.device:
            x = x.to(self.device)
            model = model.to(self.device)
        return model(x)
