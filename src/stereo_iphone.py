"""
stereo_iphone.py — receive synchronized wide + telephoto frames from the
MultiCamStreamer iOS app and produce a metric depth map via stereo SGBM.

Requirements
------------
    pip install opencv-python numpy

Usage (standalone test)
-----------------------
    python -m src.stereo_iphone --iphone-ip 192.168.1.42 --show

Integration with main.py
------------------------
    mjpython -m src.main --mode sim \\
        --stereo-iphone --iphone-ip 192.168.1.42 \\
        --stereo-calib config/stereo_iphone_calib.npz \\
        --show-cv

Wire protocol (from MultiCamSession.swift)
------------------------------------------
  [4B magic  "STRO"]
  [4B wide_jpeg_len  – uint32 big-endian]
  [4B tele_jpeg_len  – uint32 big-endian]
  [36B wide intrinsics – 9 × float32 LE, column-major simd_float3x3]
  [36B tele intrinsics – same]
  [8B timestamp – float64 LE, seconds since 2001-01-01 (Apple reference date)]
  [wide_len bytes: wide JPEG]
  [tele_len bytes:  tele JPEG]

Intrinsics layout (column-major = iOS convention):
  data = [fx, 0, 0,  0, fy, 0,  cx, cy, 1]
  → np.array(data, float32).reshape(3,3, order='F')
    gives standard camera matrix K = [[fx,0,cx],[0,fy,cy],[0,0,1]]
"""

from __future__ import annotations

import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------

_MAGIC = b"STRO"
_HEADER_SIZE = 4 + 4 + 4 + 36 + 36 + 8   # = 96 bytes after the magic
_APPLE_REF_OFFSET = 978307200.0           # seconds between 1970-01-01 and 2001-01-01


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class StereoFrame:
    wide_bgr:   np.ndarray       # H×W×3 BGR
    tele_bgr:   np.ndarray       # H×W×3 BGR
    K_wide:     np.ndarray       # 3×3 intrinsics
    K_tele:     np.ndarray       # 3×3 intrinsics
    t_unix:     float            # Unix timestamp
    depth_m:    Optional[np.ndarray] = None   # filled by StereoProcessor
    focal_rect: float = 0.0      # focal length of the rectified frame (pixels)


# ---------------------------------------------------------------------------
# TCP receiver
# ---------------------------------------------------------------------------

class _TCPReceiver:
    """Receives stereo packets from the iOS app (outbound client or inbound server mode)."""

    def __init__(
        self,
        host: str = "",
        port: int = 8080,
        *,
        listen: bool = False,
    ) -> None:
        self._host = host.strip()
        self._port = port
        self._listen = listen
        self._sock: Optional[socket.socket] = None
        self._latest: Optional[StereoFrame] = None
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._route_help_printed = False

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="stereo-tcp")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass

    def get_latest(self, timeout: float = 0.5) -> Optional[StereoFrame]:
        self._event.wait(timeout=timeout)
        self._event.clear()
        with self._lock:
            return self._latest

    # ------------------------------------------------------------------

    def _accept_one(self) -> bool:
        """Mac listens; iPhone connects (works when hotspot blocks inbound TCP to the phone)."""
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind(("0.0.0.0", self._port))
            srv.listen(1)
        except OSError as e:
            print(f"[StereoIPhone] listen bind failed: {e}", flush=True)
            try:
                srv.close()
            except OSError:
                pass
            return False

        print(
            f"[StereoIPhone] listening on 0.0.0.0:{self._port} — on iPhone set «Mac IP (reverse)» to this Mac's IP "
            f"(hotspot: often 172.20.10.2) and tap Start Streaming.",
            flush=True,
        )
        srv.settimeout(1.0)
        while self._running:
            try:
                conn, addr = srv.accept()
                conn.settimeout(None)
                self._sock = conn
                srv.close()
                print(f"[StereoIPhone] accepted connection from {addr}", flush=True)
                return True
            except socket.timeout:
                continue
            except OSError as e:
                if not self._running:
                    break
                print(f"[StereoIPhone] accept: {e}", flush=True)
                time.sleep(0.5)
        try:
            srv.close()
        except OSError:
            pass
        return False

    def _connect(self) -> bool:
        while self._running:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(5.0)
                s.connect((self._host, self._port))
                s.settimeout(None)
                self._sock = s
                print(f"[StereoIPhone] connected to {self._host}:{self._port}", flush=True)
                return True
            except OSError as e:
                errno = getattr(e, "errno", None)
                if errno in (65, 64) and not self._route_help_printed:
                    self._route_help_printed = True
                    print(
                        "[StereoIPhone] errno 65 = no IP route to the phone (Mac cannot reach that address).\n"
                        "  • Same Wi‑Fi: Mac and iPhone must join the **same** SSID (not guest vs main).\n"
                        "  • On the Mac run:  ping " + self._host + "\n"
                        "    If ping fails, use another address from the app's «All IPv4 interfaces» list.\n"
                        "  • Router «AP / client isolation» blocks phone↔Mac — disable it for your LAN.\n"
                        "  • Personal Hotspot: join Wi‑Fi from the iPhone, then use the **172.20.x.x** IP shown.\n"
                        "  • If ping works but TCP times out, try **--mac-listen** (iPhone reverse mode).\n"
                        "  • macOS Firewall: allow incoming/outbound for Python if prompted.",
                        flush=True,
                    )
                print(f"[StereoIPhone] connecting … ({e})", flush=True)
                time.sleep(2.0)
        return False

    def _loop(self) -> None:
        while self._running:
            if self._listen:
                if not self._accept_one():
                    return
            else:
                if not self._connect():
                    return
            try:
                self._read_loop()
            except OSError:
                print("[StereoIPhone] connection lost, reconnecting …", flush=True)
            self._sock = None

    def _read_loop(self) -> None:
        buf = bytearray()
        while self._running:
            chunk = self._sock.recv(65536)
            if not chunk:
                raise OSError("disconnected")
            buf.extend(chunk)

            # Parse as many complete packets as possible
            while True:
                frame, buf = _try_parse_packet(buf)
                if frame is None:
                    break
                with self._lock:
                    self._latest = frame
                self._event.set()


def _try_parse_packet(buf: bytearray) -> Tuple[Optional[StereoFrame], bytearray]:
    """Return (StereoFrame, remaining_buf) or (None, buf) if incomplete."""
    # Find magic
    idx = buf.find(_MAGIC)
    if idx < 0:
        return None, buf[-3:]   # keep last 3 bytes (partial magic match)
    if idx > 0:
        buf = buf[idx:]         # discard leading junk

    total_header = len(_MAGIC) + _HEADER_SIZE
    if len(buf) < total_header:
        return None, buf

    off = len(_MAGIC)
    wide_len = struct.unpack_from(">I", buf, off)[0];  off += 4
    tele_len = struct.unpack_from(">I", buf, off)[0];  off += 4
    wide_k_raw = struct.unpack_from("<9f", buf, off);  off += 36
    tele_k_raw = struct.unpack_from("<9f", buf, off);  off += 36
    ts_apple   = struct.unpack_from("<d",  buf, off)[0]; off += 8

    payload_len = wide_len + tele_len
    if len(buf) < total_header + payload_len:
        return None, buf

    wide_jpg = bytes(buf[off : off + wide_len]);  off += wide_len
    tele_jpg = bytes(buf[off : off + tele_len]);  off += tele_len
    remaining = buf[off:]

    # Decode JPEGs
    wide_bgr = cv2.imdecode(np.frombuffer(wide_jpg, np.uint8), cv2.IMREAD_COLOR)
    tele_bgr = cv2.imdecode(np.frombuffer(tele_jpg, np.uint8), cv2.IMREAD_COLOR)
    if wide_bgr is None or tele_bgr is None:
        return None, bytearray(remaining)

    # iOS intrinsics: column-major simd_float3x3 → standard K
    def _to_K(raw: tuple) -> np.ndarray:
        return np.array(raw, dtype=np.float64).reshape(3, 3, order='F')

    K_wide = _to_K(wide_k_raw)
    K_tele = _to_K(tele_k_raw)
    t_unix = ts_apple + _APPLE_REF_OFFSET

    frame = StereoFrame(wide_bgr=wide_bgr, tele_bgr=tele_bgr,
                         K_wide=K_wide, K_tele=K_tele, t_unix=t_unix)
    return frame, bytearray(remaining)


# ---------------------------------------------------------------------------
# Stereo processor
# ---------------------------------------------------------------------------

class StereoProcessor:
    """
    Rectifies wide + tele image pairs and computes a disparity → depth map.

    Parameters
    ----------
    calib_path : Path or None
        Path to an .npz file produced by ``calib_stereo_iphone.py``.
        If None, the processor uses the per-frame Apple intrinsics alone
        (no distortion correction, no extrinsics — coarse mode).
    output_size : (W, H)
        Resolution of the output depth map.
    sgbm_min_disp : int
        Minimum disparity (pixels). Negative values handle cameras where
        the telephoto image is shifted left w.r.t. the wide image.
    sgbm_num_disp : int
        Number of disparity levels (must be divisible by 16).
    sgbm_block : int
        Semi-global block size (odd, 5–15 is typical).
    """

    def __init__(
        self,
        calib_path: Optional[Path] = None,
        output_size: Tuple[int, int] = (320, 240),
        sgbm_min_disp: int = -48,
        sgbm_num_disp: int = 96,
        sgbm_block: int = 7,
    ) -> None:
        self._output_wh = output_size
        self._calib: Optional[dict] = None
        self._rect_maps: Optional[tuple] = None   # cached rectification maps
        self._last_K_wide: Optional[np.ndarray] = None

        if calib_path and calib_path.is_file():
            data = np.load(calib_path)
            self._calib = {k: data[k] for k in data.files}
            print(f"[StereoProcessor] loaded calibration from {calib_path}", flush=True)
            self._build_rect_maps_from_calib()
        else:
            print("[StereoProcessor] no calibration file — using per-frame intrinsics (coarse mode)", flush=True)

        P1 = 8 * 3 * sgbm_block ** 2
        P2 = 4 * P1
        self._sgbm = cv2.StereoSGBM.create(
            minDisparity=sgbm_min_disp,
            numDisparities=sgbm_num_disp,
            blockSize=sgbm_block,
            P1=P1, P2=P2,
            disp12MaxDiff=2,
            uniquenessRatio=10,
            speckleWindowSize=80,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        self._wls = cv2.ximgproc.createDisparityWLSFilter(self._sgbm)
        self._sgbm_right = cv2.ximgproc.createRightMatcher(self._sgbm)
        self._wls.setLambda(8000)
        self._wls.setSigmaColor(1.5)

    # ------------------------------------------------------------------

    def process(self, frame: StereoFrame) -> StereoFrame:
        """Fill frame.depth_m and frame.focal_rect in-place; return frame."""
        W, H = self._output_wh

        if self._calib is not None:
            # Full rectification from calibration file
            map1w, map2w, map1t, map2t = self._rect_maps
            rect_w = cv2.remap(frame.wide_bgr, map1w, map2w, cv2.INTER_LINEAR)
            rect_t = cv2.remap(frame.tele_bgr, map1t, map2t, cv2.INTER_LINEAR)
            baseline = float(self._calib["baseline_m"])
            focal_rect = float(self._calib["focal_rect_px"])
        else:
            # Coarse mode: assume cameras roughly aligned, crop tele to wide FOV
            rect_w, rect_t, baseline, focal_rect = self._coarse_align(frame)

        # Resize to output size
        gw = cv2.cvtColor(cv2.resize(rect_w, (W, H)), cv2.COLOR_BGR2GRAY)
        gt = cv2.cvtColor(cv2.resize(rect_t, (W, H)), cv2.COLOR_BGR2GRAY)

        # Scale focal to output size (W / original_W)
        orig_W = rect_w.shape[1]
        focal_out = focal_rect * W / orig_W

        # SGBM disparity (left + right for WLS filtering)
        disp_l = self._sgbm.compute(gw, gt)
        disp_r = self._sgbm_right.compute(gt, gw)
        disp_filtered = self._wls.filter(disp_l, gw, disparity_map_right=disp_r)

        # Convert disparity → metric depth
        # depth = baseline * focal / disparity   (disparity in SGBM is fixed-point ×16)
        disp_f = disp_filtered.astype(np.float32) / 16.0
        with np.errstate(divide='ignore', invalid='ignore'):
            depth = np.where(disp_f > 0.5,
                             baseline * focal_out / disp_f,
                             0.0).astype(np.float32)

        # Clip to plausible arm-tracking range [0.15 m, 3.0 m]
        depth = np.clip(depth, 0.15, 3.0)
        depth[disp_f <= 0.5] = 0.0

        frame.depth_m = depth
        frame.focal_rect = focal_out
        return frame

    # ------------------------------------------------------------------

    def _build_rect_maps_from_calib(self) -> None:
        c = self._calib
        img_size = tuple(c["image_size"].tolist())   # (W, H)
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            c["K_wide"], c["D_wide"],
            c["K_tele"], c["D_tele"],
            img_size,
            c["R_tele_from_wide"], c["T_tele_from_wide"],
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0,
        )
        map1w, map2w = cv2.initUndistortRectifyMap(
            c["K_wide"], c["D_wide"], R1, P1, img_size, cv2.CV_16SC2)
        map1t, map2t = cv2.initUndistortRectifyMap(
            c["K_tele"], c["D_tele"], R2, P2, img_size, cv2.CV_16SC2)
        self._rect_maps = (map1w, map2w, map1t, map2t)
        # Focal length of rectified image (from P1[0,0])
        self._calib["focal_rect_px"] = float(P1[0, 0])
        # Baseline from Q matrix: baseline = 1 / Q[3,2]
        self._calib["baseline_m"] = abs(float(1.0 / Q[3, 2])) if abs(Q[3, 2]) > 1e-9 else 0.012

    def _coarse_align(
        self,
        frame: StereoFrame,
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Without calibration: scale the telephoto to approximately match the
        wide camera's FOV (based on focal-length ratio), then center-crop so
        they cover the same scene area.
        """
        K_w, K_t = frame.K_wide, frame.K_tele
        fx_w = K_w[0, 0]; fx_t = K_t[0, 0]
        zoom = fx_t / fx_w if fx_w > 1 else 2.0   # typically ~2 for 2× telephoto

        Hw, Ww = frame.wide_bgr.shape[:2]
        Ht, Wt = frame.tele_bgr.shape[:2]

        # Downscale telephoto so it covers the same FOV as the wide camera
        new_Wt = int(round(Wt / zoom))
        new_Ht = int(round(Ht / zoom))
        tele_scaled = cv2.resize(frame.tele_bgr, (new_Wt, new_Ht), interpolation=cv2.INTER_AREA)

        # Center-crop both to the same size
        crop_W = min(Ww, new_Wt)
        crop_H = min(Hw, new_Ht)
        def crop_center(img, cw, ch):
            h, w = img.shape[:2]
            x = (w - cw) // 2; y = (h - ch) // 2
            return img[y:y+ch, x:x+cw]

        wide_c = crop_center(frame.wide_bgr, crop_W, crop_H)
        tele_c = crop_center(tele_scaled,    crop_W, crop_H)

        # Physical baseline ~12 mm; focal in output pixels
        baseline_m = 0.012
        focal_px = fx_w * crop_W / Ww
        return wide_c, tele_c, baseline_m, focal_px


# ---------------------------------------------------------------------------
# Main pipeline object (drop-in for DepthEstimatePipeline)
# ---------------------------------------------------------------------------

class StereoIPhonePipeline:
    """
    Drop-in replacement for DepthEstimatePipeline using stereo iPhone cameras.

    snapshot() returns:
        (t, rgb_pose_hwc, rgb_depth_hwc, depth_fresh, depth_m)

    The wide camera RGB is used for MediaPipe pose estimation.
    The stereo depth replaces the monocular depth model entirely.
    """

    def __init__(
        self,
        iphone_ip: str = "",
        calib_path: Optional[Path] = None,
        output_size: Tuple[int, int] = (320, 240),
        iphone_port: int = 8080,
        *,
        mac_listen: bool = False,
        listen_port: int = 9080,
        **sgbm_kwargs,
    ) -> None:
        if mac_listen:
            self._receiver = _TCPReceiver(listen=True, port=listen_port)
            self._processor = StereoProcessor(calib_path, output_size, **sgbm_kwargs)
            self.last_focal_px: Optional[float] = None
            self._camera = _LiveShim()
            self._receiver.start()
            print(
                f"[StereoIPhone] Mac listen mode on port {listen_port} "
                "(iPhone app: enter this Mac's IP under «Mac IP (reverse)»).",
                flush=True,
            )
            return

        self._receiver = _TCPReceiver(iphone_ip.strip(), iphone_port)
        self._processor = StereoProcessor(calib_path, output_size, **sgbm_kwargs)
        self.last_focal_px: Optional[float] = None
        self._camera = _LiveShim()
        self._receiver.start()
        print(
            f"[StereoIPhone] connecting to {iphone_ip}:{iphone_port} …\n"
            "  (Open MultiCamStreamer on iPhone; or use mac_listen if TCP times out.)",
            flush=True,
        )

    def snapshot(
        self,
    ) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray], bool, Optional[np.ndarray]]:
        raw = self._receiver.get_latest(timeout=0.25)
        if raw is None:
            return time.monotonic(), None, None, False, None

        frame = self._processor.process(raw)
        self.last_focal_px = frame.focal_rect if frame.focal_rect > 0 else frame.K_wide[0, 0]

        rgb_wide = cv2.cvtColor(frame.wide_bgr, cv2.COLOR_BGR2RGB)
        return frame.t_unix, rgb_wide, rgb_wide, True, frame.depth_m

    def release(self) -> None:
        self._receiver.stop()


class _LiveShim:
    @property
    def at_end(self) -> bool:
        return False
    @property
    def _cap(self) -> None:
        return None


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

def _main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Test stereo iPhone pipeline")
    ap.add_argument("--iphone-ip", default="", help="iPhone IP (forward mode: Python connects to phone :8080)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument(
        "--mac-listen",
        action="store_true",
        help="Listen on this Mac (reverse mode: iPhone connects here). Use when hotspot blocks inbound TCP to the phone.",
    )
    ap.add_argument("--listen-port", type=int, default=9080, help="Port for --mac-listen (default 9080).")
    ap.add_argument("--calib", default="", help="Path to stereo calib .npz")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    if not args.mac_listen and not str(args.iphone_ip).strip():
        ap.error("Provide --iphone-ip or --mac-listen")

    calib_path = Path(args.calib) if args.calib else None
    if args.mac_listen:
        pipe = StereoIPhonePipeline(mac_listen=True, listen_port=int(args.listen_port), calib_path=calib_path)
    else:
        pipe = StereoIPhonePipeline(args.iphone_ip, calib_path, iphone_port=args.port)

    print("Press q to quit …")
    while True:
        t, rgb, _, fresh, depth = pipe.snapshot()
        if rgb is None:
            continue
        if args.show:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if depth is not None:
                norm_d = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                depth_vis = cv2.applyColorMap(norm_d, cv2.COLORMAP_TURBO)
                bgr = np.hstack([bgr, depth_vis])
            cv2.imshow("StereoIPhone (wide | depth)", bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    pipe.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    _main()
