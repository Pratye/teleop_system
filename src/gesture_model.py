from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np


class TFLiteGestureClassifier:
    """
    Kinivi-style keypoint classifier wrapper.

    Expects 21 hand landmarks in pixel coordinates and applies the common
    preprocessing (relative coords from wrist + max-abs normalization).
    """

    def __init__(self, model_path: Path, labels_path: Optional[Path] = None) -> None:
        try:
            import tensorflow as tf  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("tensorflow is required for gesture-control mode") from exc

        self._interp = tf.lite.Interpreter(model_path=str(model_path))
        self._interp.allocate_tensors()
        self._inp = self._interp.get_input_details()[0]
        self._out = self._interp.get_output_details()[0]
        self.labels: List[str] = []
        if labels_path is not None and labels_path.is_file():
            self.labels = [ln.strip() for ln in labels_path.read_text(encoding="utf-8").splitlines() if ln.strip()]

    @staticmethod
    def _preprocess(hand_xy: Sequence[Tuple[float, float]]) -> np.ndarray:
        pts = np.asarray(hand_xy, dtype=np.float32)
        if pts.shape[0] < 21:
            raise ValueError("need 21 hand landmarks")
        base = pts[0].copy()
        rel = pts - base
        vec = rel.reshape(-1)
        maxv = float(np.max(np.abs(vec))) if vec.size else 1.0
        if maxv < 1e-6:
            maxv = 1.0
        vec = vec / maxv
        return vec.astype(np.float32)

    def predict(self, hand_xy: Sequence[Tuple[float, float]]) -> Tuple[int, float, str]:
        x = self._preprocess(hand_xy)
        # Typical shape is [1, 42]
        x = np.expand_dims(x, axis=0).astype(np.float32)
        self._interp.set_tensor(self._inp["index"], x)
        self._interp.invoke()
        y = self._interp.get_tensor(self._out["index"]).reshape(-1).astype(np.float32)
        idx = int(np.argmax(y))
        conf = float(y[idx]) if y.size else 0.0
        label = self.labels[idx] if 0 <= idx < len(self.labels) else f"class_{idx}"
        return idx, conf, label

