"""OpenCV panels: depth colormap + compositing for teleop debug."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2
import numpy as np


def colorize_depth_metres(depth_hw: np.ndarray) -> np.ndarray:
    """
    Single-channel depth values → coloured BGR for debugging.

    OpenCV **TURBO** emphasises teal/green in the mid-range; modest depth ranges
    then look uniformly “solid green”. We map with **MAGMA** (purple → pink → yellow)
    and stretch with 1–99.5% percentiles against valid pixels only.
    """
    d = np.asarray(depth_hw, dtype=np.float32)
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    valid = np.isfinite(d) & (np.abs(d) > 1e-10)

    cmap = getattr(cv2, "COLORMAP_MAGMA", cv2.COLORMAP_INFERNO)

    h, w = d.shape[:2]

    if not np.any(valid):
        mono = np.zeros((h, w), dtype=np.uint8)
        return cv2.applyColorMap(mono, cmap)

    lo = float(np.percentile(d[valid], 1.0))
    hi = float(np.percentile(d[valid], 99.5))

    if hi <= lo + 1e-12:
        # Flat scalar field — single hue through magma mid / upper range
        mono = np.full((h, w), 220, dtype=np.uint8)
        return cv2.applyColorMap(mono, cmap)

    scaled = np.zeros((h, w), dtype=np.float32)
    scaled[valid] = np.clip((d[valid] - lo) / (hi - lo + 1e-18), 0.0, 1.0)
    u8 = (scaled * 255.0).astype(np.uint8)
    u8[~valid] = 0
    return cv2.applyColorMap(u8, cmap)


def compose_h_resize(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    target_h: int = 480,
) -> np.ndarray:
    """Place two panels side-by-side, scale to equal height."""
    def _scale(im: np.ndarray) -> np.ndarray:
        hh, ww = im.shape[:2]
        scale = float(target_h) / float(max(hh, 1))
        nw = max(1, int(round(ww * scale)))
        return cv2.resize(im, (nw, target_h), interpolation=cv2.INTER_AREA)

    return np.hstack([_scale(left_bgr), _scale(right_bgr)])


def annotate_lines(
    img_bgr: np.ndarray,
    lines: Sequence[str],
    origin: Tuple[int, int] = (12, 28),
) -> np.ndarray:
    out = img_bgr.copy()
    y = origin[1]
    for ln in lines:
        cv2.putText(
            out,
            ln[:120],
            (origin[0], y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (30, 255, 30),
            2,
            cv2.LINE_AA,
        )
        y += 22
    return out


__all__ = ["colorize_depth_metres", "compose_h_resize", "annotate_lines"]
