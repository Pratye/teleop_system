"""
Offline bundle: monocular depth (Depth Pro OR Depth Anything V2 Torch) + MediaPipe pose → NPZ + meta JSON.

**Depth Anything V2** — clone upstream once::

  cd teleop_system && git clone --depth 1 \\
    https://github.com/DepthAnything/Depth-Anything-V2.git third_party/Depth-Anything-V2

Then::

  python -m src.precompute_assets --video videos/clip.mov \\
    --depth-da-pth ../depth_anything_v2_vits.pth --depth-da-encoder vits

**Depth Pro**::

  python -m src.precompute_assets --video videos/clip.mov --depth-checkpoint /path/to/depth_pro.pt

Replay::

  mjpython -m src.main --mode sim --replay-bundle out/bundle.npz --video <same clip>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, List, Tuple

_src = Path(__file__).resolve().parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import cv2
import numpy as np

from depth_estimator import DepthProBackend, rgb_resized_pair_from_bgr
from pose_estimator import PoseTracker


def _world_triplet_nan() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
    return (n.copy(), n.copy(), n.copy())


def precompute_bundle(
    *,
    video_path: Path,
    output_npz: Path,
    pose_hw: Tuple[int, int],
    depth_infer_hw: Tuple[int, int],
    device_str: str | None,
    focal_px_arg: float,
    no_hand: bool,
    checkpoint_path_pro: Path | None,
    checkpoint_path_da: Path | None,
    depth_da_encoder: str,
) -> None:
    w_pose, h_pose = pose_hw
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    use_pro = checkpoint_path_pro is not None
    pose = PoseTracker(load_hand_model=not no_hand)

    backend: Any
    if use_pro:
        backend = DepthProBackend(checkpoint_path_pro, device_str=device_str)  # type: ignore[arg-type]
        depth_backend_label = "depth_pro"
    else:
        from depth_estimator import get_or_create_depth_anything_torch_backend

        if checkpoint_path_da is None or not checkpoint_path_da.is_file():
            raise ValueError("--depth-da-pth must point to a valid .pth when not using Depth Pro.")
        backend = get_or_create_depth_anything_torch_backend(
            checkpoint_path_da,
            device_str,
            depth_da_encoder,
        )
        depth_backend_label = "depth_anything_v2_torch"

    depth_rows: List[np.ndarray] = []
    focal_px: List[float] = []
    wrist_uv: List[np.ndarray] = []
    elbow_uv: List[np.ndarray] = []
    shoulder_uv: List[np.ndarray] = []
    visibility: List[np.ndarray] = []
    landmarks_ok: List[bool] = []
    wrist_w: List[np.ndarray] = []
    elbow_w: List[np.ndarray] = []
    shoulder_w: List[np.ndarray] = []
    pose_kp_rows: List[np.ndarray] = []
    pinch_open: List[float] = []
    pinch_valid: List[bool] = []
    fist: List[bool] = []
    hand_present: List[bool] = []
    handedness: List[str] = []
    arm_ids: List[Tuple[int, int, int]] = []
    right_hand_rows: List[np.ndarray] = []

    frame_i = 0
    while True:
        ok, bgr = cap.read()
        if not ok or bgr is None:
            break

        rgb_pose, rgb_depth = rgb_resized_pair_from_bgr(bgr, pose_hw, depth_infer_hw)
        if rgb_pose is None or rgb_depth is None:
            break

        obs = pose.update_rgb(rgb_pose, frame_i * 33)

        d_small, inferred_f = backend.infer_depth(rgb_depth.astype(np.uint8), float(focal_px_arg))
        pose_depth = cv2.resize(
            d_small,
            (w_pose, h_pose),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)

        if use_pro:
            if inferred_f is not None and inferred_f > 1e-3:
                fz = float(inferred_f)
            elif focal_px_arg > 1e-3:
                fz = float(focal_px_arg)
            else:
                fz = float("nan")
        else:
            fz = float("nan")

        depth_rows.append(pose_depth)
        focal_px.append(fz)

        wrist_uv.append(np.array(obs.wrist_uv, dtype=np.float64))
        elbow_uv.append(np.array(obs.elbow_uv, dtype=np.float64))
        shoulder_uv.append(np.array(obs.shoulder_uv, dtype=np.float64))

        pv = obs.pose_visibility
        visibility.append(np.array([pv[0], pv[1], pv[2]], dtype=np.float64))
        landmarks_ok.append(bool(obs.landmarks_ok))

        if obs.wrist_world_m is None or obs.elbow_world_m is None or obs.shoulder_world_m is None:
            ww, ee, ss = _world_triplet_nan()
        else:
            ww, ee, ss = obs.wrist_world_m.copy(), obs.elbow_world_m.copy(), obs.shoulder_world_m.copy()

        wrist_w.append(ww)
        elbow_w.append(ee)
        shoulder_w.append(ss)

        kp = obs.pose_keypoints_xy
        if kp is None:
            pose_kp_rows.append(np.zeros((33, 2), dtype=np.float32))
        else:
            pose_kp_rows.append(kp.astype(np.float32))

        pinch_open.append(float(obs.pinch_open))
        pinch_valid.append(bool(obs.pinch_valid))
        fist.append(bool(obs.fist_gesture_active))
        hand_present.append(bool(obs.hand_landmarks_present))
        handedness.append(str(obs.handedness_hand))
        arm_ids.append(tuple(int(x) for x in obs.arm_landmark_ids))

        rh = obs.right_hand_xy
        z = np.full((21, 2), np.nan, dtype=np.float32)
        if rh is not None:
            ar = np.array(rh, dtype=np.float32)
            assert ar.shape == (21, 2)
            z = ar
        right_hand_rows.append(z)

        frame_i += 1

    cap.release()

    if frame_i == 0:
        raise RuntimeError(f"No frames read from {video_path}")

    hh_obj = np.empty(len(handedness), dtype=object)
    for jj, hh in enumerate(handedness):
        hh_obj[jj] = hh
    arm_arr = np.array(arm_ids, dtype=np.int32)

    kwargs = {
        "depth_metres_pose": np.stack(depth_rows, axis=0),
        "focal_px": np.asarray(focal_px, dtype=np.float32),
        "wrist_uv": np.stack(wrist_uv, axis=0),
        "elbow_uv": np.stack(elbow_uv, axis=0),
        "shoulder_uv": np.stack(shoulder_uv, axis=0),
        "visibility": np.stack(visibility, axis=0),
        "landmarks_ok": np.asarray(landmarks_ok),
        "wrist_world_m": np.stack(wrist_w, axis=0),
        "elbow_world_m": np.stack(elbow_w, axis=0),
        "shoulder_world_m": np.stack(shoulder_w, axis=0),
        "pose_kp_xy": np.stack(pose_kp_rows, axis=0),
        "pinch_open": np.asarray(pinch_open),
        "pinch_valid": np.asarray(pinch_valid),
        "fist": np.asarray(fist),
        "hand_present": np.asarray(hand_present),
        "handedness": hh_obj,
        "arm_landmark_ids": arm_arr,
        "right_hand_xy": np.stack(right_hand_rows, axis=0),
    }

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **kwargs)

    meta = {
        "video_path": str(video_path.resolve()),
        "pose_hw": [int(pose_hw[0]), int(pose_hw[1])],
        "depth_infer_hw": [int(depth_infer_hw[0]), int(depth_infer_hw[1])],
        "frame_count": int(frame_i),
        "depth_backend": depth_backend_label,
    }
    meta_path = output_npz.parent / f"{output_npz.stem}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_npz} + {meta_path}  ({frame_i} frames)  [{depth_backend_label}]", flush=True)


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(
        description="Bundle Depth Pro or Depth Anything V2 (Torch *.pth) + Mediapipe pose for replay.",
    )
    ap.add_argument("--video", required=True, help="Source video (MOV/MP4 …).")
    ap.add_argument(
        "-o",
        "--output",
        default="",
        help="Bundle path (default videos/<stem>_bundle.npz under teleop_system).",
    )
    ap.add_argument(
        "--depth-checkpoint",
        default="",
        help="Apple Depth Pro .pt checkpoint (mutually exclusive with --depth-da-pth).",
    )
    ap.add_argument(
        "--depth-da-pth",
        default="",
        help="Depth Anything V2 weights *.pth (e.g. depth_anything_v2_vits.pth). Requires "
        "third_party/Depth-Anything-V2 cloned from GitHub.",
    )
    ap.add_argument(
        "--depth-da-encoder",
        default="vits",
        choices=["vits", "vitb", "vitl", "vitg"],
        help="Encoder tag matching the Depth Anything *.pth file.",
    )
    ap.add_argument("--depth-device", default="", help="Torch device: cpu, mps, cuda, or omit for auto.")
    ap.add_argument(
        "--focal-px",
        type=float,
        default=0.0,
        help="Only Depth Pro: focal guess at depth-inference resolution (default 0 = autofocus).",
    )
    ap.add_argument("--pose-w", type=int, default=320)
    ap.add_argument("--pose-h", type=int, default=240)
    ap.add_argument("--depth-w", type=int, default=256)
    ap.add_argument("--depth-h", type=int, default=256)
    ap.add_argument("--no-hand", action="store_true", help="Same as teleop `--no-hand`.")
    args = ap.parse_args()

    dp = args.depth_checkpoint.strip()
    ds = args.depth_da_pth.strip()
    if bool(dp) + bool(ds) != 1:
        print(
            "Provide exactly one of: --depth-checkpoint (Depth Pro) or --depth-da-pth (Depth Anything V2).",
            file=sys.stderr,
        )
        return 1

    vid = Path(args.video)
    if not vid.is_file():
        print(f"Video not found: {vid}", file=sys.stderr)
        return 1

    ck_pro: Path | None = None
    ck_da: Path | None = None

    if dp:
        cq = Path(dp)
        ck_pro = (repo / cq).resolve() if not cq.is_absolute() else cq.resolve()
        if not ck_pro.is_file():
            print(f"Depth Pro checkpoint not found: {ck_pro}", file=sys.stderr)
            return 1

    if ds:
        dq = Path(ds)
        ck_da = (repo / dq).resolve() if not dq.is_absolute() else dq.resolve()
        if not ck_da.is_file():
            print(f"--depth-da-pth not found: {ck_da}", file=sys.stderr)
            return 1

    out_arg = args.output.strip()
    if out_arg:
        out = Path(out_arg)
    else:
        out_dir = repo / "videos"
        out = out_dir / f"{vid.stem}_bundle.npz"

    pose_wh = (int(args.pose_w), int(args.pose_h))
    depth_infer_hw = (int(args.depth_w), int(args.depth_h))
    dv = args.depth_device.strip() or None

    try:
        precompute_bundle(
            video_path=vid.resolve(),
            output_npz=out.resolve(),
            pose_hw=pose_wh,
            depth_infer_hw=depth_infer_hw,
            device_str=dv,
            focal_px_arg=float(args.focal_px),
            no_hand=bool(args.no_hand),
            checkpoint_path_pro=ck_pro,
            checkpoint_path_da=ck_da,
            depth_da_encoder=str(args.depth_da_encoder).strip().lower(),
        )
    except Exception as e:  # noqa: BLE001
        print(f"precompute failed: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
