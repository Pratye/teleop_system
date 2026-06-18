"""
calib_stereo_iphone.py — stereo calibration for iPhone wide + telephoto cameras.

Run this ONCE before using --stereo-iphone in main.py.
It receives live stereo frames from the iOS app, detects a checkerboard
in both images simultaneously, and computes:
  • K_wide, D_wide   — wide-camera intrinsics + distortion
  • K_tele, D_tele   — telephoto intrinsics + distortion
  • R, T             — rotation and translation from wide → tele
  • baseline_m       — physical camera separation in metres

Output is saved as config/stereo_iphone_calib.npz.

Usage
-----
    python scripts/calib_stereo_iphone.py \\
        --iphone-ip 192.168.1.42 \\
        --board-cols 9 --board-rows 6 --square-mm 25

    # If TCP to the phone times out on Personal Hotspot (inbound blocked):
    python scripts/calib_stereo_iphone.py --mac-listen --listen-port 9080 --show
    # Then on iPhone enter this Mac's IP (e.g. 172.20.10.2) in «Mac IP (reverse)».

Hold a checkerboard in view of BOTH cameras, moving it to different
positions/angles. The script captures a pair when the board is found
in both frames at the same time. Aim for 20-30 good pairs.

Press 'q' to quit early and compute with whatever pairs were captured.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# This file lives in teleop_system/scripts/ — parent[1] is teleop_system/
teleop_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(teleop_root / "src"))

from stereo_iphone import _TCPReceiver  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Stereo calibration for iPhone wide+tele.")
    ap.add_argument("--iphone-ip", default="", help="iPhone IP (forward mode: Python connects to phone).")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument(
        "--mac-listen",
        action="store_true",
        help="Mac listens for reverse connection from iPhone (use when hotspot TCP to phone times out).",
    )
    ap.add_argument("--listen-port", type=int, default=9080, help="Listen port for --mac-listen.")
    ap.add_argument("--board-cols", type=int, default=9,
                    help="Number of inner corners along cols (default 9).")
    ap.add_argument("--board-rows", type=int, default=6,
                    help="Number of inner corners along rows (default 6).")
    ap.add_argument("--square-mm", type=float, default=25.0,
                    help="Checkerboard square size in mm (default 25).")
    ap.add_argument("--n-pairs", type=int, default=25,
                    help="Number of good frame pairs to collect (default 25).")
    ap.add_argument("--out", default=str(teleop_root / "config" / "stereo_iphone_calib.npz"),
                    help="Output .npz path.")
    ap.add_argument("--show", action="store_true", help="Show live preview while capturing.")
    args = ap.parse_args()
    if not args.mac_listen and not str(args.iphone_ip).strip():
        print("Provide --iphone-ip or --mac-listen", file=sys.stderr)
        return

    board = (args.board_cols, args.board_rows)
    square_m = args.square_mm / 1000.0
    # 3D object points for the checkerboard
    objp = np.zeros((board[0] * board[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board[0], 0:board[1]].T.reshape(-1, 2) * square_m

    obj_pts: list = []
    wide_pts: list = []
    tele_pts: list = []
    img_size_wide = None
    img_size_tele = None

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4)
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    if args.mac_listen:
        receiver = _TCPReceiver(listen=True, port=int(args.listen_port))
        receiver.start()
        print(f"Listening on 0.0.0.0:{args.listen_port} …  (set iPhone «Mac IP (reverse)» to this Mac, then Start Streaming)")
    else:
        receiver = _TCPReceiver(args.iphone_ip.strip(), args.port)
        receiver.start()
        print(f"Connecting to {args.iphone_ip}:{args.port} …  (waiting for frames)")
    print(f"Need {args.n_pairs} good pairs.  Move the checkerboard slowly.")
    print("Press 'q' to stop early and compute calibration.\n")

    last_capture = 0.0
    min_capture_interval = 1.5   # seconds between captures to get diverse views

    while len(obj_pts) < args.n_pairs:
        frame = receiver.get_latest(timeout=1.0)
        if frame is None:
            continue

        gray_w = cv2.cvtColor(frame.wide_bgr, cv2.COLOR_BGR2GRAY)
        gray_t = cv2.cvtColor(frame.tele_bgr, cv2.COLOR_BGR2GRAY)

        ok_w, corners_w = cv2.findChessboardCorners(gray_w, board, flags)
        ok_t, corners_t = cv2.findChessboardCorners(gray_t, board, flags)

        vis_w = frame.wide_bgr.copy()
        vis_t = frame.tele_bgr.copy()

        if ok_w:
            cv2.drawChessboardCorners(vis_w, board, corners_w, ok_w)
        if ok_t:
            cv2.drawChessboardCorners(vis_t, board, corners_t, ok_t)

        now = time.monotonic()
        if ok_w and ok_t and (now - last_capture) >= min_capture_interval:
            # Refine corner positions
            corners_w = cv2.cornerSubPix(gray_w, corners_w, (11,11), (-1,-1), criteria)
            corners_t = cv2.cornerSubPix(gray_t, corners_t, (11,11), (-1,-1), criteria)
            obj_pts.append(objp)
            wide_pts.append(corners_w)
            tele_pts.append(corners_t)
            img_size_wide = (gray_w.shape[1], gray_w.shape[0])
            img_size_tele = (gray_t.shape[1], gray_t.shape[0])
            last_capture = now
            n = len(obj_pts)
            print(f"  [{n}/{args.n_pairs}] captured pair ✓", flush=True)
            # Flash the images green briefly
            cv2.rectangle(vis_w, (0,0), (50,50), (0,200,0), -1)
            cv2.rectangle(vis_t, (0,0), (50,50), (0,200,0), -1)

        if args.show:
            # Scale down for display if large
            disp = np.hstack([
                cv2.resize(vis_w, (480, 360)),
                cv2.resize(vis_t, (480, 360)),
            ])
            cv2.putText(disp, f"Pairs: {len(obj_pts)}/{args.n_pairs}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("Stereo calib (wide | tele) — q to stop", disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    receiver.stop()
    cv2.destroyAllWindows()

    if len(obj_pts) < 6:
        print(f"Only {len(obj_pts)} pairs — need at least 6. Aborting.")
        return

    print(f"\nCalibrating with {len(obj_pts)} pairs …")

    # Calibrate wide camera
    rms_w, K_w, D_w, _, _ = cv2.calibrateCamera(obj_pts, wide_pts, img_size_wide, None, None)
    print(f"  Wide  RMS = {rms_w:.3f} px")

    # Calibrate tele / second camera (images may be different resolution)
    rms_t, K_t, D_t, _, _ = cv2.calibrateCamera(obj_pts, tele_pts, img_size_tele, None, None)
    print(f"  2nd cam RMS = {rms_t:.3f} px")

    # OpenCV 4.13+ rejects CALIB_USE_EXTRINSIC_GUESS for stereoCalibrate. We try:
    #   • FIX_INTRINSIC — fast, can land in a bad local minimum (nonsense baseline).
    #   • USE_INTRINSIC_GUESS (joint) — refines K,D together with R,T; usually fixes scale.

    # (tag, rms, R, T, K_wide, D_wide, K_tele, D_tele)
    candidates: list[tuple[str, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    def _baseline_m(tvec: np.ndarray) -> float:
        return float(np.linalg.norm(tvec.reshape(3)))

    def _add_fix(tag: str, ip1, ip2, k1, d1, k2, d2, sz: tuple[int, int]) -> None:
        try:
            rms, k1o, d1o, k2o, d2o, rmat, tvec, _, _ = cv2.stereoCalibrate(
                obj_pts, ip1, ip2,
                k1, d1, k2, d2,
                sz,
                criteria=criteria_stereo,
                flags=cv2.CALIB_FIX_INTRINSIC,
            )
            tcol = tvec.reshape(3, 1)
            candidates.append(
                (tag, float(rms), rmat, tcol, np.asarray(k1o), np.asarray(d1o), np.asarray(k2o), np.asarray(d2o))
            )
        except cv2.error as e:
            print(f"  [{tag}] failed: {e}")

    def _add_joint(tag: str, ip1, ip2, k1, d1, k2, d2, sz: tuple[int, int]) -> None:
        try:
            rms, k1o, d1o, k2o, d2o, rmat, tvec, _, _ = cv2.stereoCalibrate(
                obj_pts, ip1, ip2,
                np.asarray(k1, np.float64).copy(),
                np.asarray(d1, np.float64).copy(),
                np.asarray(k2, np.float64).copy(),
                np.asarray(d2, np.float64).copy(),
                sz,
                criteria=criteria_stereo,
                flags=cv2.CALIB_USE_INTRINSIC_GUESS,
            )
            tcol = tvec.reshape(3, 1)
            candidates.append(
                (tag, float(rms), rmat, tcol, np.asarray(k1o), np.asarray(d1o), np.asarray(k2o), np.asarray(d2o))
            )
        except cv2.error as e:
            print(f"  [{tag}] failed: {e}")

    _add_fix("stereo fix-K (wide→2nd)", wide_pts, tele_pts, K_w, D_w, K_t, D_t, img_size_wide)
    _add_joint("stereo joint-K (wide→2nd)", wide_pts, tele_pts, K_w, D_w, K_t, D_t, img_size_wide)

    # OpenCV cam1 = 2nd stream, cam2 = wide → x_wide = R_sw @ x_2nd + T_sw; convert to wide→2nd extrinsics.
    for joint in (False, True):
        tag = ("stereo joint-K (2nd→wide swapped)" if joint else "stereo fix-K (2nd→wide swapped)")
        try:
            if not joint:
                rms, k1o, d1o, k2o, d2o, r_sw, t_sw, _, _ = cv2.stereoCalibrate(
                    obj_pts, tele_pts, wide_pts,
                    K_t, D_t, K_w, D_w,
                    img_size_tele,
                    criteria=criteria_stereo,
                    flags=cv2.CALIB_FIX_INTRINSIC,
                )
            else:
                rms, k1o, d1o, k2o, d2o, r_sw, t_sw, _, _ = cv2.stereoCalibrate(
                    obj_pts, tele_pts, wide_pts,
                    np.asarray(K_t, np.float64).copy(),
                    np.asarray(D_t, np.float64).copy(),
                    np.asarray(K_w, np.float64).copy(),
                    np.asarray(D_w, np.float64).copy(),
                    img_size_tele,
                    criteria=criteria_stereo,
                    flags=cv2.CALIB_USE_INTRINSIC_GUESS,
                )
            rmat = r_sw.T
            tcol = (-r_sw.T @ t_sw.reshape(3, 1)).reshape(3, 1)
            # Returned K1,D1 are for cam1 (wire tele); K2,D2 for cam2 (wire wide)
            K_wide_o, D_wide_o = np.asarray(k2o), np.asarray(d2o)
            K_tele_o, D_tele_o = np.asarray(k1o), np.asarray(d1o)
            candidates.append((tag, float(rms), rmat, tcol, K_wide_o, D_wide_o, K_tele_o, D_tele_o))
        except cv2.error as e:
            print(f"  [{tag}] failed: {e}")

    if not candidates:
        print("  All stereoCalibrate attempts failed. Aborting.")
        return

    lo_b, hi_b = 0.004, 0.040  # 4–40 mm — physical phone stereo range
    plausible = [c for c in candidates if lo_b <= _baseline_m(c[3]) <= hi_b]

    def sort_key(c: tuple) -> tuple:
        _, rms, _, tvec, *_ = c
        b = _baseline_m(tvec)
        in_range = lo_b <= b <= hi_b
        return (0 if in_range else 1, rms)

    chosen = min(plausible, key=lambda c: c[1]) if plausible else min(candidates, key=sort_key)
    tag_s, rms_s, R, T, K_wide_o, D_wide_o, K_tele_o, D_tele_o = chosen
    baseline_m = _baseline_m(T)

    print(f"  Stereo RMS = {rms_s:.3f} px  [{tag_s}]")
    print(
        f"  Baseline = {baseline_m*1000:.2f} mm  "
        f"(wide↔tele often ~10–14 mm; wide↔ultra‑wide can be a few mm larger on some models)"
    )

    if not plausible or not (lo_b <= baseline_m <= hi_b):
        print(
            "\n  ⚠ Baseline still looks non‑physical. The .npz may give bad depth.\n"
            "    • Use exact inner-corner counts and printed square size (mm).\n"
            "    • Keep the board steady while a pair is captured; fill the frame in both views.\n"
            "    • Avoid motion blur; add more diverse tilt/distance samples.\n"
            "    • Delete this .npz and re-run calibration after checking the above.\n"
        )
    elif rms_s > 3.0:
        print(
            "\n  Note: stereo RMS is fairly high — depth will be noisier; add more tilt/distance diversity or sharper captures.\n"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(out_path),
        K_wide=K_wide_o,
        D_wide=D_wide_o,
        K_tele=K_tele_o,
        D_tele=D_tele_o,
        R_tele_from_wide=R,
        T_tele_from_wide=T,
        baseline_m=np.float64(baseline_m),
        image_size=np.array(img_size_wide),
        rms_wide=np.float64(rms_w),
        rms_tele=np.float64(rms_t),
        rms_stereo=np.float64(rms_s),
    )
    print(f"\nCalibration saved → {out_path}")
    print("Run main.py with:  --stereo-iphone --iphone-ip <ip> "
          f"--stereo-calib {out_path}")


if __name__ == "__main__":
    main()
