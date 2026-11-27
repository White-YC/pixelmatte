"""Trimap helpers. A trimap marks every pixel as background (0), unknown (128) or foreground (255)."""
import numpy as np
from PIL import Image

BACKGROUND, UNKNOWN, FOREGROUND = 0, 128, 255


def normalize_trimap(trimap) -> np.ndarray:
    """Any single-channel trimap (PIL image or array) -> uint8 array with the three values 0 / 128 / 255.
    Values up to 10 count as background and values from 245 as foreground; everything in between is unknown."""
    t = np.asarray(trimap.convert("L") if isinstance(trimap, Image.Image) else trimap)
    if t.ndim == 3:
        t = t[..., 0]
    if t.dtype != np.uint8:
        t = np.clip(np.round(t.astype(np.float32) * (255.0 if t.max() <= 1.0 else 1.0)), 0, 255).astype(np.uint8)
    out = np.full(t.shape, UNKNOWN, np.uint8)
    out[t <= 10] = BACKGROUND
    out[t >= 245] = FOREGROUND
    return out
