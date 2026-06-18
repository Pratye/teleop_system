"""
iPhone 16 Pro LiDAR depth source via the Record3D USB streaming protocol.

Prerequisites
-------------
1.  Install the **Record3D** app on your iPhone:
      https://apps.apple.com/app/record3d-lidar-3d-scanner/id1Record3D
2.  Install the Python client:
      pip install record3d
3.  Connect your iPhone via USB-C and trust the computer.
4.  In the Record3D app → top-right menu → enable "USB Streaming".

Usage in main.py
----------------
    mjpython -m src.main --mode sim \\
        --iphone-depth        # enable this source (depth from LiDAR)
        --show-cv

The source replaces the Depth Anything / Depth Pro depth backend.
Pose estimation (MediaPipe) still runs on the same RGB frames.

How it works
------------
Record3D streams pairs of (RGB frame, LiDAR depth map) at up to 30 fps
over USB.  The depth map is metric (metres), so we don't need any
monocular estimation — we get ground-truth depth directly from the
iPhone's LiDAR sensor.

Coordinate note
---------------
The depth map is aligned to the main (wide) camera.  Use the
Record3D intrinsics (provided per frame) to back-project any pixel to
a 3D point.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Shared frame dataclass
# ---------------------------------------------------------------------------

@dataclass
class IPhoneFrame:
    rgb_bgr: np.ndarray          # H×W×3 BGR
    depth_m: np.ndarray          # H×W float32, metric metres
    focal_px: float              # fx ≈ fy in pixels (from Record3D intrinsics)
    t_mono: float = field(default_factory=time.monotonic)


# ---------------------------------------------------------------------------
# Record3D streaming source
# ---------------------------------------------------------------------------

class Record3DSource:
    """
    Streams RGB + LiDAR depth from an iPhone 16 Pro over USB.

    Parameters
    ----------
    device_idx : int
        Record3D device index (0 = first connected iPhone, default).
    rgb_wh : tuple
        Target (width, height) for the RGB output delivered to the pipeline.
    depth_wh : tuple
        Target (width, height) for the depth output (usually the same).
    """

    def __init__(
        self,
        device_idx: int = 0,
        rgb_wh: Tuple[int, int] = (640, 480),
        depth_wh: Tuple[int, int] = (256, 192),
    ) -> None:
        try:
            from record3d import Record3DSession  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "record3d is not installed.  Run:  pip install record3d\n"
                "Then connect your iPhone 16 Pro via USB and enable "
                "'USB Streaming' in the Record3D app."
            ) from exc

        self._rgb_wh = rgb_wh
        self._depth_wh = depth_wh
        self._latest: Optional[IPhoneFrame] = None
        self._lock = threading.Lock()
        self._ready = threading.Event()

        self._session = Record3DSession()
        self._session.on_new_frame = self._on_frame
        self._session.on_stream_stopped = self._on_stopped
        self._session.connect(session_id=self._session.get_devices()[device_idx])

    # ------------------------------------------------------------------

    def _on_frame(self) -> None:
        s = self._session
        try:
            # RGB comes as RGBA from Record3D; depth in metres
            rgba = s.get_rgb_frame()       # H×W×4 uint8
            depth = s.get_depth_frame()    # H×W float32 (metres)
            intr = s.camera_matrix         # 3×3

            bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)

            # Resize to requested resolution
            bgr_out = cv2.resize(bgr, self._rgb_wh, interpolation=cv2.INTER_LINEAR)
            depth_out = cv2.resize(depth, self._depth_wh, interpolation=cv2.INTER_NEAREST)

            # Focal length: average of fx and fy scaled to output size
            fx = float(intr[0, 0]) * self._rgb_wh[0] / bgr.shape[1]
            fy = float(intr[1, 1]) * self._rgb_wh[1] / bgr.shape[0]
            focal_px = (fx + fy) * 0.5

            frame = IPhoneFrame(
                rgb_bgr=bgr_out,
                depth_m=depth_out.astype(np.float32),
                focal_px=focal_px,
            )
            with self._lock:
                self._latest = frame
            self._ready.set()
        except Exception:  # noqa: BLE001
            pass

    def _on_stopped(self) -> None:
        print("[Record3D] stream stopped — check USB connection.", flush=True)

    # ------------------------------------------------------------------

    def get_latest(self, timeout: float = 0.5) -> Optional[IPhoneFrame]:
        """Block until a new frame arrives, or return None on timeout."""
        self._ready.wait(timeout=timeout)
        self._ready.clear()
        with self._lock:
            return self._latest

    def release(self) -> None:
        try:
            self._session.disconnect()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# Adapter for DepthEstimatePipeline-compatible snapshot() interface
# ---------------------------------------------------------------------------

class Record3DPipeline:
    """
    Drop-in replacement for ``DepthEstimatePipeline`` when using
    the iPhone 16 Pro LiDAR depth.

    ``snapshot()`` returns:
        (t, rgb_pose, rgb_depth, depth_fresh, depth_m)

    where ``rgb_pose`` and ``rgb_depth`` are the same BGR→RGB image
    (MediaPipe is run on the same frame), and ``depth_m`` is the raw
    LiDAR depth map — no monocular estimation needed.
    """

    def __init__(
        self,
        device_idx: int = 0,
        pose_wh: Tuple[int, int] = (640, 480),
        depth_wh: Tuple[int, int] = (256, 192),
    ) -> None:
        self._source = Record3DSource(device_idx, pose_wh, depth_wh)
        self.last_focal_px: Optional[float] = None
        # Expose a minimal _camera shim so main.py at_end checks work
        self._camera = _LiveCamShim()
        print(
            "[Record3D] connected — streaming RGB + LiDAR depth from iPhone 16 Pro",
            flush=True,
        )

    def snapshot(
        self,
    ) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray], bool, Optional[np.ndarray]]:
        """
        Returns (t, rgb_pose_hwc, rgb_depth_hwc, depth_fresh, depth_metres_hw).
        """
        frame = self._source.get_latest(timeout=0.25)
        if frame is None:
            return time.monotonic(), None, None, False, None

        self.last_focal_px = frame.focal_px

        # Both pose and depth branches use the same full-res RGB
        rgb = cv2.cvtColor(frame.rgb_bgr, cv2.COLOR_BGR2RGB)
        return frame.t_mono, rgb, rgb, True, frame.depth_m

    def release(self) -> None:
        self._source.release()


class _LiveCamShim:
    """Minimal shim so main.py's at_end / _cap checks don't crash."""
    @property
    def at_end(self) -> bool:
        return False

    @property
    def _cap(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Convenience: list connected Record3D devices
# ---------------------------------------------------------------------------

def list_record3d_devices() -> None:
    try:
        from record3d import Record3DSession  # type: ignore[import]
        s = Record3DSession()
        devices = s.get_devices()
        if not devices:
            print("No Record3D devices found.  Connect iPhone via USB and enable USB Streaming.")
        else:
            for i, d in enumerate(devices):
                print(f"  [{i}] {d}")
    except ImportError:
        print("record3d not installed.  pip install record3d")


__all__ = [
    "IPhoneFrame",
    "Record3DPipeline",
    "Record3DSource",
    "list_record3d_devices",
]
