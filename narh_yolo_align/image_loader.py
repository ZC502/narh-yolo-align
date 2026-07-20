from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence
import random

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class ImageRecord:
    """One image path and loaded pixels."""

    path: Path
    image_bgr: np.ndarray


def list_images(
    source: str | Path,
    *,
    max_images: int = 5000,
    seed: int = 42,
    shuffle: bool = True,
) -> List[Path]:
    """List image files without loading them into memory.

    OOM protection:
    - Only file paths are collected.
    - `max_images` caps work by default.
    - Images are loaded lazily by `iter_image_batches`.
    """
    source = Path(source)
    if source.is_file() and source.suffix.lower() in IMAGE_EXTS:
        paths = [source]
    else:
        paths = [p for p in source.rglob("*") if p.suffix.lower() in IMAGE_EXTS]

    paths = sorted(paths)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(paths)

    if max_images and max_images > 0:
        paths = paths[:max_images]

    return paths


def iter_image_batches(
    paths: Sequence[Path],
    *,
    batch_size: int = 8,
) -> Iterator[List[ImageRecord]]:
    """Yield batches of loaded BGR images. Invalid images are skipped."""
    batch: List[ImageRecord] = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        batch.append(ImageRecord(path=p, image_bgr=img))
        if len(batch) >= batch_size:
            yield batch
            batch = []

    if batch:
        yield batch
