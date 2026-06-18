"""Monocular RGB capture + optional depth (Depth Pro or Depth Anything V2 on Hugging Face)."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, List, Optional, Tuple

import cv2
import numpy as np

try:
    from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT as _DP_CONF
except Exception:  # noqa: BLE001
    _DP_CONF = None


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class DepthFrameRecord:
    t_mono: float
    depth_metres: np.ndarray
    focal_length_px_used: Optional[float]


class DepthCircularBuffer:
    def __init__(self, maxlen: int = 4) -> None:
        self._buf: Deque[DepthFrameRecord] = deque(maxlen=maxlen)

    def push(self, rec: DepthFrameRecord) -> None:
        self._buf.append(rec)

    def latest(self) -> Optional[DepthFrameRecord]:
        return self._buf[-1] if self._buf else None

    def history_ns(self, n_last: int) -> List[DepthFrameRecord]:
        return list(self._buf)[-n_last:]


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

def rgb_resized_pair_from_bgr(
    bgr: np.ndarray,
    pose_wh: Tuple[int, int],
    depth_wh: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pose + depth-inference RGB tensors (matches ``DepthCamera.rgb_resized_pair``).


    Pure function—does not open ``cv2.VideoCapture`` (safe for offline tooling).

    """
    rgb_pose = cv2.cvtColor(cv2.resize(bgr, pose_wh, interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
    rgb_depth = cv2.cvtColor(cv2.resize(bgr, depth_wh, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    return rgb_pose, rgb_depth


class DepthCamera:
    """
    Wraps ``cv2.VideoCapture`` for either a live camera or a video file.

    Parameters
    ----------
    device_index : int or str
        Integer → live camera index.
        String  → path to a video file (MOV / MP4 / AVI …).
    loop_video : bool
        When *True* and the source is a file, seeks back to frame 0 on EOF
        so the video plays indefinitely.  Ignored for live cameras.
    """

    def __init__(
        self,
        device_index: "int | str" = 0,
        capture_size: Tuple[int, int] = (320, 240),
        loop_video: bool = True,
        capture_fps: Optional[int] = None,
    ) -> None:
        # A string source is either a local video file or a live stream URL.
        # URLs start with a scheme (rtsp://, http://, https://, udp://, …).
        _is_url = isinstance(device_index, str) and "://" in str(device_index)
        self._is_file = isinstance(device_index, str) and not _is_url
        self._loop = loop_video and self._is_file

        # On macOS, AVFoundation is the fastest backend for live cameras.
        backend = cv2.CAP_AVFOUNDATION if isinstance(device_index, int) else cv2.CAP_ANY
        self._cap = cv2.VideoCapture(device_index, backend)

        if self._cap.isOpened() and not self._is_file:
            # Minimise internal frame queue so read_bgr() returns the *latest* frame,
            # not one that has been sitting in the buffer for 100-200 ms.
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # Request resolution and FPS *before* the first grab so the driver
            # negotiates the right mode from the start.
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, capture_size[0])
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_size[1])
            if capture_fps is not None:
                self._cap.set(cv2.CAP_PROP_FPS, float(capture_fps))

        # Video metadata (only meaningful when is_file)
        self.video_fps: float = float(self._cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.video_frame_count: int = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    @property
    def at_end(self) -> bool:
        """True when a non-looping video has been exhausted."""
        if not self._is_file or self._loop:
            return False
        pos = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
        return self.video_frame_count > 0 and pos >= self.video_frame_count

    def read_bgr(self) -> Tuple[float, Optional[np.ndarray]]:
        # For live cameras: drain any stale frames accumulated since the last
        # read so we always process the *most recent* frame.
        if not self._is_file:
            drained = 0
            while self._cap.grab() and drained < 4:
                drained += 1
        t = time.monotonic()
        ok, bgr = self._cap.read()
        if not ok:
            if self._loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, bgr = self._cap.read()
            if not ok:
                return t, None
        return t, bgr

    def release(self) -> None:
        self._cap.release()

    def rgb_resized_pair(
        self,
        bgr: Optional[np.ndarray],
        pose_wh: Tuple[int, int],
        depth_wh: Tuple[int, int],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if bgr is None:
            return None, None
        rp, rd = rgb_resized_pair_from_bgr(bgr, pose_wh, depth_wh)
        return rp, rd


# ---------------------------------------------------------------------------
# Depth Pro backend
# ---------------------------------------------------------------------------

class DepthProBackend:
    """Wraps Apple's Depth Pro (optional dependency)."""

    def __init__(self, checkpoint_path: Path, device_str: Optional[str]) -> None:
        from dataclasses import replace
        import torch
        from PIL import Image
        from depth_pro.depth_pro import create_model_and_transforms

        self._torch = torch
        self._Image = Image

        if device_str:
            self._device = torch.device(device_str)
        elif torch.backends.mps.is_available():
            self._device = torch.device("mps")
        elif torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")

        if _DP_CONF is None:
            raise ImportError("depth_pro not installed; clone apple/ml-depth-pro and pip install -e .")

        # float16 on accelerators halves resident memory; CPU always float32
        _precision = (
            torch.float16
            if self._device.type in ("mps", "cuda")
            else torch.float32
        )
        config = replace(_DP_CONF, checkpoint_uri=str(checkpoint_path))
        self._model, self._transform = create_model_and_transforms(
            config, device=self._device, precision=_precision
        )

    def infer_depth(self, rgb_hwc: np.ndarray, focal_px: float) -> Tuple[np.ndarray, Optional[float]]:
        torch = self._torch
        im = self._Image.fromarray(rgb_hwc.astype(np.uint8), mode="RGB")
        inp_b = self._transform(im).unsqueeze(0)
        fx_t = torch.tensor(focal_px, device=self._device, dtype=torch.float32) if focal_px > 1e-3 else None

        with torch.inference_mode():
            out = self._model.infer(inp_b, f_px=fx_t)
            depth_t = out["depth"]
            depth_map = (
                np.squeeze(depth_t.float().detach().cpu().numpy())
                if isinstance(depth_t, torch.Tensor)
                else np.asarray(depth_t).squeeze()
            )

        inferred_f = None if focal_px > 1e-3 else float(out["focallength_px"])

        # Release device tensors and flush allocator cache immediately
        del inp_b, fx_t, out, depth_t
        if self._device.type == "mps":
            try:
                torch.mps.empty_cache()
            except AttributeError:
                pass
        elif self._device.type == "cuda":
            torch.cuda.empty_cache()

        return depth_map.astype(np.float32), inferred_f


# ---------------------------------------------------------------------------
# Depth Anything V2 (Hugging Face transformers) — small, fast vs Depth Pro
# ---------------------------------------------------------------------------

_DEFAULT_DA_HUB = "depth-anything/Depth-Anything-V2-Small-hf"


def _depth_anything_relative_np_to_approx_metres(d_hwx: np.ndarray) -> np.ndarray:
    """Map relative / disparity-like map into a plausible metre range for viz + IK."""
    d = np.maximum(np.asarray(d_hwx, dtype=np.float32), 1e-6)
    flat = d.reshape(-1)
    pm = flat[np.isfinite(flat)]
    if pm.size < 64:
        return np.full_like(d, 0.75, dtype=np.float32)
    lo = float(np.percentile(pm, 5.0))
    hi = float(np.percentile(pm, 95.0))
    span = hi - lo
    if span < 1e-6:
        span = 1e-6
    norm = np.clip((d - lo) / span, 0.0, 1.0)
    return (0.35 + norm * (2.85 - 0.35)).astype(np.float32)


def _depth_anything_v2_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent / "third_party" / "Depth-Anything-V2"


def _resolve_hf_pipeline_device(device_str: Optional[str]) -> Optional["int | str"]:
    """Map teleop `--depth-device` to Hugging Face `pipeline(..., device=...)`."""
    try:
        import torch
    except ImportError:
        return None
    ds = None if device_str is None else device_str.strip().lower()
    if ds == "" or ds is None:
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return 0
        return None  # transformers default = CPU for latest API
    if ds == "cpu":
        return -1 if hasattr(torch, "cuda") else None  # pragma: no cover
    if ds == "cuda":
        return 0 if torch.cuda.is_available() else (-1 if hasattr(torch, "cuda") else None)
    if ds == "mps":
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return -1 if hasattr(torch, "cuda") else None
    return None


class DepthAnythingBackend:
    """
    Lightweight monocular depth via Hugging Face ``depth-anything``.

    Outputs **pseudo-metric** depth (metres) by percentile-stretching the model's
    relative map into ~0.35–2.8 m — good enough for VIS + depth-back-projection
    fusion; not replacement for calibrated metric sensors.
    """

    def __init__(self, hub_id: str, device_str: Optional[str]) -> None:
        try:
            from transformers import pipeline  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "Depth Anything backend needs `pip install transformers` "
                "(and usually `pillow`, `accelerate`)."
            ) from e

        self._hub_id = hub_id or _DEFAULT_DA_HUB
        dev = _resolve_hf_pipeline_device(device_str)
        # device=None lets transformers pick; maps to CPU reliably on unsupported HW
        try:
            if dev is None:
                self._pipe = pipeline(
                    task="depth-estimation",
                    model=self._hub_id,
                )
            else:
                self._pipe = pipeline(
                    task="depth-estimation",
                    model=self._hub_id,
                    device=dev,
                )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Failed loading depth model {self._hub_id!r}. "
                "Install transformers and try `--depth-device cpu` if GPU backends fail."
            ) from exc

    def infer_depth(self, rgb_hwc: np.ndarray, focal_px: float) -> Tuple[np.ndarray, Optional[float]]:
        del focal_px  # DA does not expose focal-length estimate
        from PIL import Image  # noqa: PLC0415

        h, w = rgb_hwc.shape[:2]
        pil = Image.fromarray(np.ascontiguousarray(rgb_hwc.astype(np.uint8)), mode="RGB")
        out = self._pipe(pil)

        # HF API: pipelines return {"depth": PIL.Image} (docs); some versions use "predicted_depth".
        def _to_numpy(obj: object) -> np.ndarray:
            if hasattr(obj, "detach"):
                return np.squeeze(np.asarray(obj.detach().cpu().numpy(), dtype=np.float32))
            if isinstance(obj, Image.Image):
                return np.asarray(obj.convert("F"), dtype=np.float32)
            return np.squeeze(np.asarray(obj, dtype=np.float32))

        pred: Optional[object]
        if isinstance(out, dict):
            pred = out.get("depth", out.get("predicted_depth"))
        else:
            pred = getattr(out, "depth", None) or getattr(out, "predicted_depth", None)
        if pred is None:
            raise KeyError("depth-estimation output has no 'depth' or 'predicted_depth'")
        d = _to_numpy(pred)

        if d.ndim != 2:
            raise ValueError(f"unexpected depth shape {d.shape}")

        dh, dw = d.shape[:2]
        if dh != h or dw != w:
            d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)

        return _depth_anything_relative_np_to_approx_metres(d), None


class DepthAnythingTorchBackend:
    """
    Depth Anything **V2** small/medium checkpoints as ``*.pth`` (official PyTorch releases).

    Expects upstream ``DepthAnything/Depth-Anything-V2`` at ``third_party/Depth-Anything-V2``.

    """

    MODEL_CFG = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }

    def __init__(self, ckpt_path: Path, device_str: Optional[str], encoder: str = "vits") -> None:
        import sys
        import torch
        import torch.nn.functional as F

        repo = _depth_anything_v2_repo_root()
        pkg = repo / "depth_anything_v2"
        if not pkg.is_dir():
            raise ImportError(
                "Depth Anything V2 sources not found. Clone into teleop_system/third_party:\n"
                "  git clone https://github.com/DepthAnything/Depth-Anything-V2.git "
                f"{repo}",
            )

        r = str(repo)
        if r not in sys.path:
            sys.path.insert(0, r)

        torch_mod = torch
        self._torch = torch_mod
        self._F = F

        ds = "" if device_str is None else device_str.strip().lower()
        if ds in ("", "auto"):
            if getattr(torch.backends, "mps", None) is not None and torch_mod.backends.mps.is_available():  # type: ignore[union-attr]
                self._device = torch_mod.device("mps")
            elif torch_mod.cuda.is_available():
                self._device = torch_mod.device("cuda")
            else:
                self._device = torch_mod.device("cpu")
        elif ds == "cpu":
            self._device = torch_mod.device("cpu")
        elif ds == "mps":
            self._device = torch_mod.device("mps") if torch_mod.backends.mps.is_available() else torch_mod.device("cpu")
        elif ds == "cuda":
            self._device = torch_mod.device("cuda") if torch_mod.cuda.is_available() else torch_mod.device("cpu")
        else:
            raise ValueError(f"unsupported device_str for Depth Anything Torch: {device_str!r}")

        enc = encoder.strip().lower()
        if enc not in self.MODEL_CFG:
            raise ValueError(f"encoder must be one of {list(self.MODEL_CFG)}; got {encoder!r}")

        from depth_anything_v2.dpt import DepthAnythingV2

        cfg = dict(self.MODEL_CFG[enc])
        model = DepthAnythingV2(**cfg)
        ck = Path(ckpt_path)
        sd = torch_mod.load(str(ck), map_location="cpu")
        model.load_state_dict(sd)
        self._model = model.to(self._device).eval()

        self._input_size = 518

    def infer_depth(self, rgb_hwc: np.ndarray, focal_px: float) -> Tuple[np.ndarray, Optional[float]]:
        del focal_px
        torch = self._torch
        F = self._F

        bgr = cv2.cvtColor(np.ascontiguousarray(rgb_hwc, dtype=np.uint8), cv2.COLOR_RGB2BGR)

        with torch.inference_mode():
            image_t, (h0, w0) = self._model.image2tensor(bgr, input_size=self._input_size)
            image_t = image_t.to(self._device)

            depth = self._model.forward(image_t)

            depth = F.interpolate(
                depth[:, None],
                (h0, w0),
                mode="bilinear",
                align_corners=True,
            )[0, 0].float()
            d_np = depth.cpu().numpy()

        return _depth_anything_relative_np_to_approx_metres(d_np), None


_da_backend_lock = threading.Lock()
_depth_backend_lock = threading.Lock()
_depth_backend_singleton: Optional[DepthProBackend] = None
_depth_anything_singleton: Optional[DepthAnythingBackend] = None
_depth_anything_singleton_key: Tuple[str, Optional[str]] = ("", None)
_depth_anything_torch_singleton: Optional[DepthAnythingTorchBackend] = None
_depth_anything_torch_key: Tuple[str, Optional[str], str] = ("", None, "")


def get_or_create_depth_backend(path: Path, device_str: Optional[str]) -> DepthProBackend:
    global _depth_backend_singleton  # noqa: PLW0603
    if _depth_backend_singleton is None:
        with _depth_backend_lock:
            if _depth_backend_singleton is None:
                _depth_backend_singleton = DepthProBackend(path, device_str)
    return _depth_backend_singleton


def get_or_create_depth_anything_backend(
    hub_id: str,
    device_str: Optional[str],
) -> DepthAnythingBackend:
    global _depth_anything_singleton  # noqa: PLW0603
    global _depth_anything_singleton_key  # noqa: PLW0603
    key = (hub_id, device_str)
    if _depth_anything_singleton is None or _depth_anything_singleton_key != key:
        with _da_backend_lock:
            if _depth_anything_singleton is None or _depth_anything_singleton_key != key:
                _depth_anything_singleton = DepthAnythingBackend(hub_id, device_str)
                _depth_anything_singleton_key = key
    return _depth_anything_singleton


def get_or_create_depth_anything_torch_backend(
    ckpt_path: Path,
    device_str: Optional[str],
    encoder: str,
) -> DepthAnythingTorchBackend:
    global _depth_anything_torch_singleton  # noqa: PLW0603
    global _depth_anything_torch_key  # noqa: PLW0603
    key = (str(Path(ckpt_path).resolve()), device_str, encoder.strip().lower())
    if _depth_anything_torch_singleton is None or _depth_anything_torch_key != key:
        with _da_backend_lock:
            if _depth_anything_torch_singleton is None or _depth_anything_torch_key != key:
                _depth_anything_torch_singleton = DepthAnythingTorchBackend(ckpt_path, device_str, encoder)
                _depth_anything_torch_key = key
    return _depth_anything_torch_singleton



# ---------------------------------------------------------------------------
# Async depth worker — runs depth inference in a background thread so the main
# loop (MediaPipe + IK + MuJoCo) is never blocked during inference.
# ---------------------------------------------------------------------------

class _AsyncDepthWorker:
    """
    Background thread that keeps one depth inference in flight.

    Backend can be DepthProBackend or DepthAnythingBackend (duck-typed infer_depth).
    """

    def __init__(self, backend: Any, focal_px: float,
                 min_interval_s: float = 2.0) -> None:
        self._backend = backend
        self._focal_px = float(focal_px)
        self._min_interval = float(min_interval_s)

        self._lock = threading.Lock()
        self._inbox: Optional[np.ndarray] = None
        self._latest: Optional[np.ndarray] = None
        self._latest_focal: Optional[float] = None  # focal used for the latest depth map
        self._fresh = False
        self._busy = False
        self._last_submit_t: float = -999.0
        self._infer_exc_logged = 0
        self._infer_ok_count = 0

        self._trigger = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="depth-worker")
        self._thread.start()

    def submit(self, rgb_hwc: np.ndarray) -> None:
        """Queue a frame for inference — drops the frame if too soon or busy."""
        now = time.monotonic()
        with self._lock:
            if self._busy:
                return
            if now - self._last_submit_t < self._min_interval:
                return
            self._inbox = rgb_hwc.copy()
            self._last_submit_t = now
        self._trigger.set()

    def get(self) -> Tuple[Optional[np.ndarray], bool, Optional[float]]:
        """Return ``(depth_map, fresh, focal_px_used)``; ``fresh`` flips to False after first read."""
        with self._lock:
            fresh = self._fresh
            self._fresh = False
            return self._latest, fresh, self._latest_focal

    def _depth_is_usable(self, depth: np.ndarray) -> bool:
        """Reject only truly invalid maps (NaNs everywhere, constant glitch)."""
        if depth is None or depth.size == 0:
            return False
        if not np.any(np.isfinite(depth)):
            return False
        flat = depth[np.isfinite(depth)].astype(np.float64).ravel()
        if flat.size < 64:
            return False
        spread = float(np.percentile(flat, 95) - np.percentile(flat, 5))
        mx = float(np.percentile(np.abs(flat), 99))
        # Some HF disparity maps cluster in [0, 0.01]; old check (p95 > 0.03) falsely rejected those.
        return spread > max(1e-9 * max(mx, 1.0), 1e-12)

    def _run(self) -> None:
        while True:
            self._trigger.wait()
            self._trigger.clear()

            with self._lock:
                frame = self._inbox
                self._inbox = None
                self._busy = True

            if frame is None:
                with self._lock:
                    self._busy = False
                continue

            try:
                depth, inferred_f = self._backend.infer_depth(frame, self._focal_px)
                if depth is not None and np.all(np.isfinite(depth)) and self._depth_is_usable(depth):
                    focal_used = (
                        inferred_f if inferred_f is not None
                        else (self._focal_px if self._focal_px > 1e-3 else None)
                    )
                    with self._lock:
                        self._latest = depth
                        self._latest_focal = focal_used
                        self._fresh = True
                        self._infer_ok_count += 1
            except Exception as e:  # noqa: BLE001
                if self._infer_exc_logged < 5:
                    self._infer_exc_logged += 1
                    print(f"[depth-worker] inference failed ({self._infer_exc_logged}x): {e!r}", flush=True)
                    if self._infer_exc_logged == 1:
                        import traceback
                        traceback.print_exc()
            finally:
                with self._lock:
                    self._busy = False


# ---------------------------------------------------------------------------
# High-level pipeline
# ---------------------------------------------------------------------------

class DepthEstimatePipeline:
    """
    Camera capture + optional async depth + frame buffering.

    ``snapshot()`` never blocks on depth — inference runs in a daemon thread
    and results are consumed non-blocking on the next available frame.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        device_str: Optional[str] = None,
        depth_model: str = "depth_anything",
        depth_anything_hub_id: Optional[str] = None,
        depth_anything_local_pth: Optional[Path] = None,
        depth_da_encoder: str = "vits",
        pose_hw: Tuple[int, int] = (320, 240),
        depth_infer_hw: Tuple[int, int] = (256, 256),
        buffer_len: int = 4,
        depth_every_nth_frame: int = 1,
        focal_px: float = 600.0,
        camera_device: "int | str" = 0,
        depth_min_interval_s: float = 2.0,
        loop_video: bool = True,
        capture_fps: Optional[int] = None,
    ) -> None:
        self.pose_wh = pose_hw
        self.depth_infer_wh = depth_infer_hw
        self._camera = DepthCamera(
            device_index=camera_device,
            capture_size=pose_hw,
            loop_video=loop_video,
            capture_fps=capture_fps,
        )
        # Expose video info for callers
        self.is_video_file: bool = self._camera._is_file  # noqa: SLF001
        self.video_fps: float = self._camera.video_fps
        self.video_frame_count: int = self._camera.video_frame_count
        self._buffer = DepthCircularBuffer(maxlen=buffer_len)
        self._focal_px = float(focal_px)
        self._frame_ix = 0
        self._last_valid_depth: Optional[np.ndarray] = None
        self._last_blend_depth: Optional[np.ndarray] = None
        self._blend_alpha = 0.5
        self.depth_model = str(depth_model).lower().strip()

        # Async worker — None when depth disabled or backends missing
        self._worker: Optional[_AsyncDepthWorker] = None
        dm = self.depth_model
        if dm == "depth_pro":
            if checkpoint_path is not None:
                ck = Path(checkpoint_path)
                if ck.is_file():
                    backend = get_or_create_depth_backend(ck, device_str)
                    self._worker = _AsyncDepthWorker(
                        backend, focal_px, min_interval_s=depth_min_interval_s
                    )
        elif dm == "depth_anything":
            local_pth = Path(depth_anything_local_pth) if depth_anything_local_pth else None
            if local_pth is not None and local_pth.is_file():
                backend = get_or_create_depth_anything_torch_backend(
                    local_pth,
                    device_str,
                    depth_da_encoder,
                )
            else:
                hub = (depth_anything_hub_id or "").strip() or _DEFAULT_DA_HUB
                backend = get_or_create_depth_anything_backend(hub, device_str)
            self._worker = _AsyncDepthWorker(
                backend, focal_px, min_interval_s=depth_min_interval_s
            )
        elif dm in ("none", "off", ""):
            pass
        else:
            raise ValueError(f"unsupported depth_model: {depth_model!r}")

    def focal_px_override(self, fx: float) -> None:
        self._focal_px = float(fx)
        if self._worker is not None:
            self._worker._focal_px = float(fx)  # noqa: SLF001

    def close(self) -> None:
        self._camera.release()

    def snapshot(
        self,
    ) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray], bool, Optional[np.ndarray]]:
        """
        Capture + (non-blocking) depth result.

        Returns
        -------
        t_mono, rgb_pose_HWC, rgb_depth_HWC, depth_fresh, depth_metres_HW
        """
        t_mono, bgr = self._camera.read_bgr()
        rgb_pose, rgb_depth = self._camera.rgb_resized_pair(bgr, self.pose_wh, self.depth_infer_wh)
        if rgb_pose is None or rgb_depth is None:
            return t_mono, None, None, False, None

        self._frame_ix += 1

        # Submit this frame to background worker (non-blocking; may be dropped if too soon)
        if self._worker is not None and rgb_depth is not None:
            self._worker.submit(rgb_depth)

        # Check whether a new depth result is ready
        fresh_measurement = False
        depth_small: Optional[np.ndarray] = None
        _fresh_focal: Optional[float] = None
        if self._worker is not None:
            result, fresh, focal_hint = self._worker.get()
            if fresh and result is not None:
                fresh_measurement = True
                depth_small = result
                _fresh_focal = focal_hint
                self._last_valid_depth = result.copy()

        # Hold / blend when no new measurement
        if not fresh_measurement or depth_small is None:
            last = self._last_valid_depth if self._last_valid_depth is not None else np.zeros(
                (self.depth_infer_wh[1], self.depth_infer_wh[0]), dtype=np.float32
            )
            scaled = cv2.resize(last, self.pose_wh, interpolation=cv2.INTER_LINEAR)
            blended = (
                scaled if self._last_blend_depth is None
                else self._blend_alpha * self._last_blend_depth + (1 - self._blend_alpha) * scaled
            )
            self._last_blend_depth = blended
            pose_depth = blended.astype(np.float32)
        else:
            pose_depth = cv2.resize(depth_small, self.pose_wh, interpolation=cv2.INTER_LINEAR).astype(np.float32)
            self._last_blend_depth = pose_depth.copy()

        # focal_length_px_used: the focal at depth_infer_hw resolution (not pose_hw).
        # Callers that do back-projection must scale by (pose_w / depth_infer_w).
        focal_record = _fresh_focal if (fresh_measurement and _fresh_focal is not None) else None
        self._buffer.push(DepthFrameRecord(
            t_mono=float(t_mono),
            depth_metres=pose_depth.copy(),
            focal_length_px_used=focal_record,
        ))

        return t_mono, rgb_pose, rgb_depth, fresh_measurement, pose_depth


def none_depth_like(scalar_gray: np.ndarray) -> np.ndarray:
    return np.zeros(scalar_gray.shape, dtype=np.float32)


__all__ = [
    "DepthAnythingBackend",
    "DepthAnythingTorchBackend",
    "DepthCamera",
    "DepthCircularBuffer",
    "DepthFrameRecord",
    "DepthEstimatePipeline",
    "DepthProBackend",
    "get_or_create_depth_anything_torch_backend",
    "rgb_resized_pair_from_bgr",
]
