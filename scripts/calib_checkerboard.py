#!/usr/bin/env python3
"""
Camera intrinsics estimation with a chessboard (OpenCV).
Saves NPZ bundle for use by pose_estimator / main.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--squares-rows", type=int, default=9, help="Inner corners horizontally")
    ap.add_argument("--squares-cols", type=int, default=6, help="Inner corners vertically")
    ap.add_argument("--square-mm", type=float, default=25.0, help="Printed square edge length")
    ap.add_argument("--device", type=int, default=0, help="OpenCV VideoCapture device index")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config" / "camera_calib.npz",
        help="Output path (.npz)",
    )
    ap.add_argument("--frames", type=int, default=20, help="Captured frames to accumulate")
    return ap.parse_args()


def main() -> None:
    import cv2  # noqa: PLC0415

    args = parse_args()

    pattern_size = (args.squares_cols, args.squares_rows)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        60,
        0.001,
    )
    obj_pts = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    obj_pts[:, :2] = (
        np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2) * args.square_mm
    )

    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open VideoCapture({args.device}).")

    obj_accum = []
    img_accum = []
    grabbed = 0

    print("Hold the chessboard steadily; press SPACE to capture a sample (esc to quit).")

    win = "CheckerboardCalibration"
    cv2.namedWindow(win)

    while grabbed < args.frames:
        ok, img = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(
            gray, pattern_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        viz = img.copy()
        if ret:
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(viz, pattern_size, corners, ret)

        cv2.imshow(win, viz)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        if k != 32:  # space
            continue
        if not ret:
            print("Corners not detected; try again.")
            continue
        obj_accum.append(obj_pts.copy())
        img_accum.append(corners)
        grabbed += 1
        print(f"captured sample {grabbed}/{args.frames}")

    cap.release()
    cv2.destroyAllWindows()

    if len(obj_accum) < 10:
        raise SystemExit(f"Too few usable frames ({len(obj_accum)}). Need roughly >=10.")

    h, w = gray.shape
    reproj_rms, mtx, dist, _rv, _tv = cv2.calibrateCamera(
        obj_accum,
        img_accum,
        (w, h),
        None,
        None,
    )
    # First return value is RMS reprojection error (scalar px).
    reproj_rms = float(np.asarray(reproj_rms).ravel()[0])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        camera_matrix_K=mtx,
        distortion_coefficients=dist,
        image_width=w,
        image_height=h,
        reprojection_rmse_rmse_pixels=reproj_rms,
    )
    print(
        "Saved:",
        args.out,
        "| reproj_RMSE(px):",
        reproj_rms,
        "| K[0,0] fx:",
        mtx[0, 0],
    )


if __name__ == "__main__":
    main()
