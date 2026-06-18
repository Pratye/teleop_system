"""Unified entry: monocular camera → depth + pose → IK → sim or serial."""

from __future__ import annotations

import sys
from pathlib import Path

# Resolve sibling modules whether run as `python -m src.main` from `teleop_system/`
# or `python main.py` from `teleop_system/src/`.
_src = Path(__file__).resolve().parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import argparse
import gc
import multiprocessing as _mp
import queue as _queue_mod
import time
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import cv_display as _cv_display_mod
from ard_controller import ArduinoTeleopPort
from debug_visuals import annotate_lines, colorize_depth_metres, compose_h_resize
from depth_estimator import DepthEstimatePipeline
from ik_mapper import MapperBase, load_mapper
from pose_estimator import PoseTracker, draw_pose_arm_and_labels, get_3d_from_depth
from replay_bundle import ReplayDepthPose
from sim_controller import SimTeleopEnv
from gesture_model import TFLiteGestureClassifier

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore

try:
    import mujoco
    from mujoco import viewer as mj_viewer

    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bgr_display_ready(img: np.ndarray) -> np.ndarray:
    """Contiguous H×W×3 ``uint8`` BGR suitable for OpenCV imshow / tobytes()."""
    if cv2 is None:
        return img
    arr = np.ascontiguousarray(img)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)

    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            mx = float(np.nanmax(arr)) if arr.size else 0.0
            scale = 255.0 if mx <= 1.01 else 1.0
            arr = np.clip(arr * scale, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(arr)


def _intrinsics_px(w: int, h: int, fx: float) -> np.ndarray:
    return np.array(
        [[fx, 0.0, w * 0.5], [0.0, fx, h * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _camera_K_for_backproject(pipeline: Any, w0: int, h0: int) -> np.ndarray:
    """Intrinsics at pose resolution ``(w0,h0)`` using depth buffer focal or heuristics."""
    depth_infer_w = float(getattr(pipeline, "depth_infer_wh", (256, 256))[0])
    fz = getattr(pipeline, "last_focal_px", None)
    if fz is not None and float(fz) > 1e-3:
        fx = float(fz) * (w0 / depth_infer_w)
        return _intrinsics_px(w0, h0, fx)
    rec = None
    buf = getattr(pipeline, "_buffer", None)
    if buf is not None:
        try:
            rec = buf.latest()  # noqa: SLF001
        except Exception:
            pass
    if rec and rec.focal_length_px_used is not None and rec.focal_length_px_used > 1e-3:
        fx = float(rec.focal_length_px_used) * (w0 / depth_infer_w)
    elif float(getattr(pipeline, "_focal_px", 0.0)) > 1e-3:
        fx = float(pipeline._focal_px) * (w0 / depth_infer_w)  # noqa: SLF001
    else:
        fx = 0.65 * w0
    return _intrinsics_px(w0, h0, fx)


def _ema_point(state: Dict[str, Any], name: str, pt: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential moving average on a 3-vector. ``alpha`` = weight on the NEW sample."""
    v = np.asarray(pt, dtype=np.float64).reshape(3)
    key = f"_ema_{name}"
    prev = state.get(key)
    if prev is None:
        out = v.copy()
    else:
        out = alpha * v + (1.0 - alpha) * np.asarray(prev, dtype=np.float64)
    state[key] = out
    return out


def _clear_landmark_ema(state: Dict[str, Any]) -> None:
    for k in ("_ema_wrist_lm", "_ema_elbow_lm", "_ema_shoulder_lm"):
        state.pop(k, None)


def _map_hand_only_to_q(
    obs: Any,
    img_w: int,
    img_h: int,
    state: Dict[str, Any],
) -> Optional[List[float]]:
    """
    Build pseudo joint commands from hand landmarks only.

    Returns q_cmd in degrees-like range around 0 for:
      [base, shoulder, elbow, wrist_y, wrist_x]
    """
    hand = obs.right_hand_xy
    if hand is None or len(hand) < 21:
        return None

    # Landmark aliases (MediaPipe hand)
    wrist = hand[0]
    idx_tip = hand[8]
    mid_mcp = hand[9]
    pinky_tip = hand[20]
    thumb_tip = hand[4]

    # Hand center from stable anchor points
    pts = np.array([hand[0], hand[5], hand[9], hand[13], hand[17]], dtype=np.float64)
    cx = float(np.mean(pts[:, 0]))
    cy = float(np.mean(pts[:, 1]))

    xn = np.clip((cx / max(1.0, float(img_w)) - 0.5) * 2.0, -1.0, 1.0)
    yn = np.clip((0.5 - cy / max(1.0, float(img_h))) * 2.0, -1.0, 1.0)

    # Use hand span as a depth-like proxy for elbow bend.
    span = float(np.hypot(mid_mcp[0] - wrist[0], mid_mcp[1] - wrist[1]))
    span_ref = float(state.get("hand_span_ref", span))
    if span_ref <= 1e-6:
        span_ref = span
    # Update reference slowly for robustness.
    state["hand_span_ref"] = 0.98 * span_ref + 0.02 * span
    elbow_n = np.clip((span - span_ref) / max(12.0, span_ref), -1.0, 1.0)

    # Wrist X rotation proxy from palm orientation.
    vx = float(mid_mcp[0] - wrist[0])
    vy = float(mid_mcp[1] - wrist[1])
    roll_n = np.clip(np.arctan2(-vy, vx) / np.pi, -1.0, 1.0)

    # Wrist Y from fingertip offset from wrist.
    wy_n = np.clip((idx_tip[0] - wrist[0]) / max(40.0, span), -1.0, 1.0)

    # Base from horizontal hand position.
    base_n = xn
    shoulder_n = yn

    # Degrees-like outputs around zero; sender normalizes these.
    q_cmd = [
        float(base_n * 100.0),
        float(shoulder_n * 100.0),
        float(elbow_n * 100.0),
        float(wy_n * 100.0),
        float(roll_n * 100.0),
    ]
    return q_cmd


def _map_gesture_label_to_q_and_grip(
    label: str,
    prev_q: List[float],
    prev_grip: float,
) -> Tuple[List[float], float]:
    """
    Map gesture label text to robot pseudo-joint command.
    Designed to work with kinivi-style labels by keyword matching.
    """
    l = label.lower()
    q = [0.0, 0.0, 0.0, 0.0, 0.0]
    grip = float(prev_grip)
    step = 45.0

    # User-specified gesture mapping:
    # Forward -> end effector forward
    # Stop    -> gripper closes
    # Up      -> end effector up
    # Land    -> gripper opens
    # Down    -> end effector down
    # Back    -> end effector back
    # Left    -> end effector left
    # Right   -> end effector right
    if "forward" in l:
        q[2] = step
    elif "back" in l:
        q[2] = -step
    elif "up" in l:
        q[1] = step
    elif "down" in l:
        q[1] = -step
    elif "left" in l:
        q[0] = -step
    elif "right" in l:
        q[0] = step
    elif "stop" in l:
        grip = 0.0
    elif "land" in l:
        grip = 1.0

    return q, float(np.clip(grip, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Per-frame teleop logic
# ---------------------------------------------------------------------------

def _run_teleop_iteration(
    *,
    mapper: MapperBase,
    pose: Optional[PoseTracker],
    pipeline: Any,
    state: Dict[str, Any],
    env: Optional[SimTeleopEnv],
    ard: Optional[ArduinoTeleopPort],
    freeze_on_fist: bool,
    build_panel: bool,
    hand_only_control: bool = False,
    gesture_control: bool = False,
    gesture_classifier: Optional[TFLiteGestureClassifier] = None,
    gesture_min_conf: float = 0.55,
    panel_every_nth: int = 2,
    landmark_smooth: float = 0.38,
    wrist_depth_blend: float = 0.22,
) -> Tuple[List[float], Optional[np.ndarray]]:
    """Capture → depth + pose → IK → optional sim/serial; returns (q_cmd, panel_bgr|None)."""
    _, rgb_pose, _, depth_fresh, pose_depth_m = pipeline.snapshot()

    if rgb_pose is None:
        return list(state["last_q_deg"]), None

    state["frame_i"] = int(state.get("frame_i", 0)) + 1
    if isinstance(pipeline, ReplayDepthPose):
        idx = pipeline.bundle_frame_index
        if idx is None:
            return list(state["last_q_deg"]), None
        obs = pipeline.pose_observation_for_index(idx)
    else:
        assert pose is not None
        obs = pose.update_rgb(rgb_pose, int(state["frame_i"] * 33))

    # --- grip command ---
    grip_cmd = float(obs.pinch_open)
    if obs.fist_gesture_active:
        grip_cmd = 0.0
    if not obs.pinch_valid:
        grip_cmd = float(state.get("last_grip", grip_cmd))
    state["last_grip"] = grip_cmd

    h0, w0 = rgb_pose.shape[:2]

    if gesture_control:
        if (
            obs.right_hand_xy is not None
            and len(obs.right_hand_xy) >= 21
            and gesture_classifier is not None
        ):
            _, conf, label = gesture_classifier.predict(obs.right_hand_xy)
            state["gesture_last"] = f"{label}:{conf:.2f}"
            if conf >= float(gesture_min_conf):
                q_cmd, grip_override = _map_gesture_label_to_q_and_grip(
                    label,
                    list(state["last_q_deg"]),
                    float(state.get("last_grip", grip_cmd)),
                )
                grip_cmd = grip_override
                state["last_q_deg"] = list(q_cmd)
            else:
                q_cmd = list(state["last_q_deg"])
                q_cmd = [0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            q_cmd = list(state["last_q_deg"])
        using_world = False
        geom_valid = bool(obs.right_hand_xy is not None and len(obs.right_hand_xy) >= 21)
    elif hand_only_control:
        hand_q = _map_hand_only_to_q(obs, w0, h0, state)
        if obs.fist_gesture_active and freeze_on_fist:
            q_cmd = list(state["last_q_deg"])
        elif hand_q is None:
            q_cmd = list(state["last_q_deg"])
        else:
            q_cmd = hand_q
            state["last_q_deg"] = list(q_cmd)
        using_world = False
        geom_valid = bool(hand_q is not None)
    else:
        # -----------------------------------------------------------------------
        # 3-D joint positions for IK
        # -----------------------------------------------------------------------
        using_world = (
            obs.wrist_world_m is not None
            and obs.elbow_world_m is not None
            and obs.shoulder_world_m is not None
        )

        if using_world:
            wrist_xyz    = obs.wrist_world_m.copy()     # type: ignore[union-attr]
            elbow_xyz    = obs.elbow_world_m.copy()     # type: ignore[union-attr]
            shoulder_xyz = obs.shoulder_world_m.copy()  # type: ignore[union-attr]
        else:
            K_bp = _camera_K_for_backproject(pipeline, w0, h0)
            _DEFAULT_Z = 0.75

            def _bp(uv: Tuple[float, float], depth_map: Optional[np.ndarray]) -> np.ndarray:
                if depth_map is not None and np.any(depth_map > 1e-3):
                    xyz, ok = get_3d_from_depth(depth_map, uv[0], uv[1], K_bp)
                    if ok:
                        return xyz
                u, v = float(uv[0]), float(uv[1])
                fx_k, fy_k = float(K_bp[0, 0]), float(K_bp[1, 1])
                cx_k, cy_k = float(K_bp[0, 2]), float(K_bp[1, 2])
                return np.array(
                    [(u - cx_k) * _DEFAULT_Z / fx_k,
                     (v - cy_k) * _DEFAULT_Z / fy_k,
                     _DEFAULT_Z],
                    dtype=np.float64,
                )

            wrist_xyz = _bp(obs.wrist_uv, pose_depth_m)
            elbow_xyz = _bp(obs.elbow_uv, pose_depth_m)
            shoulder_xyz = _bp(obs.shoulder_uv, pose_depth_m)

        geom_valid = bool(obs.landmarks_ok)

        if obs.fist_gesture_active and freeze_on_fist:
            q_cmd = list(state["last_q_deg"])
        elif not geom_valid:
            _clear_landmark_ema(state)
            q_cmd = list(state["last_q_deg"])
        else:
            wb = float(wrist_depth_blend) if not using_world else 0.0
            if wb > 1e-6 and pose_depth_m is not None and np.any(pose_depth_m > 1e-4):
                K_bp = _camera_K_for_backproject(pipeline, w0, h0)
                wd_sample, wd_ok = get_3d_from_depth(
                    pose_depth_m, float(obs.wrist_uv[0]), float(obs.wrist_uv[1]), K_bp
                )
                if wd_ok:
                    wrist_xyz = (1.0 - wb) * np.asarray(wrist_xyz, dtype=np.float64) + wb * np.asarray(
                        wd_sample, dtype=np.float64
                    )

            sm = float(landmark_smooth)
            if sm > 1e-6:
                wrist_xyz = _ema_point(state, "wrist_lm", wrist_xyz, sm)
                elbow_xyz = _ema_point(state, "elbow_lm", elbow_xyz, sm)
                shoulder_xyz = _ema_point(state, "shoulder_lm", shoulder_xyz, sm)

            qs = mapper.map_observation(
                wrist_xyz,
                elbow_xyz,
                shoulder_xyz,
                right_hand_xy=obs.right_hand_xy or None,
                wrist_uv=(float(obs.wrist_uv[0]), float(obs.wrist_uv[1])) if obs.wrist_uv is not None else None,
                elbow_uv=(float(obs.elbow_uv[0]), float(obs.elbow_uv[1])) if obs.elbow_uv is not None else None,
            )
            q_cmd = [float(x) for x in np.asarray(qs).flatten()[:5]]
            state["last_q_deg"] = list(q_cmd)

    # --- apply to sim / real ---
    if env is not None:
        # set_joint_degrees_deg auto-syncs the goal ball to the FK end-effector,
        # so the red sphere always sits exactly at the arm tip.
        env.set_joint_degrees_deg(q_cmd)
        env.set_gripper_aperture_visual(grip_cmd)

    if ard is not None:
        ard.write_joints_and_grip(q_cmd, grip_cmd)

    # --- compose debug panel (throttled to every Nth frame) ---
    panel: Optional[np.ndarray] = None
    want_panel = build_panel and cv2 is not None and (int(state["frame_i"]) % max(1, panel_every_nth) == 0)
    if want_panel:
        vis = draw_pose_arm_and_labels(rgb_pose.copy(), obs, depth_fresh)
        dcol = (
            colorize_depth_metres(pose_depth_m)
            if pose_depth_m is not None
            else np.zeros((h0, w0, 3), dtype=np.uint8)
        )
        stale_tag = "n(held)" if not bool(depth_fresh) else "y"
        src_tag = "world_lm" if using_world else "depth_bp"
        dcol = annotate_lines(dcol, [
            f"geom={geom_valid} fist={obs.fist_gesture_active} pinch={grip_cmd:.2f}",
            (
                "CTRL:hand_only"
                if hand_only_control
                else (
                    f"CTRL:gesture {state.get('gesture_last', 'n/a')}"
                    if gesture_control
                    else (
                        f"IK:{src_tag} ema={landmark_smooth:.2f} d_wrist={wrist_depth_blend:.2f}"
                        if using_world
                        else f"IK:{src_tag}"
                    )
                )
            ),
            f"depth_panel stale={stale_tag}",
        ])
        panel = compose_h_resize(vis, dcol, target_h=480)
        panel = annotate_lines(
            panel,
            ["L: pose+skeleton  R: metric depth  [pinch=gripper  fist=freeze]"],
            origin=(12, panel.shape[0] - 28),
        )

    return q_cmd, panel


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    # multiprocessing spawn guard (needed on macOS)
    _mp.set_start_method("spawn", force=False)

    repo = Path(__file__).resolve().parents[1]

    ap = argparse.ArgumentParser(description="Monocular depth + pose -> IK -> sim/real.")
    ap.add_argument("--mode", choices=["sim", "real"], default="sim")
    ap.add_argument("--config", default=str(repo / "config" / "robot_kinematics.yaml"))
    ap.add_argument(
        "--camera",
        default="0",
        help=(
            "Camera source. Options:\n"
            "  0, 1, 2 …   macOS device index (0=built-in, 1=Continuity Camera / iPhone)\n"
            "  rtsp://…    RTSP stream (e.g. EpocCam, Camo, OBS Virtual Cam)\n"
            "  http://…    MJPEG/HTTP stream\n"
            "Ignored when --video is set."
        ),
    )
    ap.add_argument("--video", default="",
                    help="Path to a recorded video file to use instead of the live camera "
                         "(e.g. videos/IMG_2100.mov). Teleop predictions drive the sim "
                         "exactly as if frames came from the camera.")
    ap.add_argument(
        "--replay-bundle",
        default="",
        help="Run from a precomputed .npz (see precompute_assets): Apple Depth Pro depth "
             "plus MediaPipe pose, no live inference. --video must match the bundle meta "
             "(omit --video to use the path recorded in the bundle JSON).",
    )
    ap.add_argument("--video-loop", action="store_true", default=True,
                    help="Loop the video when it reaches the end (default: on).")
    ap.add_argument("--no-video-loop", dest="video_loop", action="store_false",
                    help="Stop when the video file ends.")
    ap.add_argument("--serial-pattern", default="cu.usbmodem*")
    ap.add_argument(
        "--hand-only-control",
        action="store_true",
        help="Use hand landmarks only for control (ignore full arm IK).",
    )
    ap.add_argument(
        "--gesture-control",
        action="store_true",
        help="Use TFLite hand-gesture classifier (kinivi-style keypoint model).",
    )
    ap.add_argument(
        "--gesture-model",
        default="",
        help="Path to keypoint_classifier.tflite model file.",
    )
    ap.add_argument(
        "--gesture-labels",
        default="",
        help="Path to keypoint_classifier_label.csv (one label per line).",
    )
    ap.add_argument(
        "--gesture-min-conf",
        type=float,
        default=0.55,
        help="Minimum confidence to accept gesture class (default 0.55).",
    )
    ap.add_argument(
        "--no-arduino",
        action="store_true",
        help="Run camera/pose pipeline without opening Arduino serial.",
    )
    ap.add_argument("--depth-stride", type=int, default=2)
    ap.add_argument("--depth-model", choices=["depth_anything", "depth_pro", "none"], default="depth_anything",
                    help="depth_anything = HF small model OR local --depth-da-pth (PyTorch *.pth); "
                         "depth_pro = Apple Depth Pro (--depth-checkpoint); none disables depth.")
    ap.add_argument("--depth-anything-model", default="depth-anything/Depth-Anything-V2-Small-hf",
                    help="HF model id when --depth-model depth_anything.")
    ap.add_argument("--depth-checkpoint", default="",
                    help="Path to Depth Pro .pt checkpoint (only for --depth-model depth_pro).")
    ap.add_argument(
        "--depth-da-pth",
        default="",
        help="Depth Anything V2 PyTorch weights (*.pth), e.g. depth_anything_v2_vits.pth. "
             "Uses teleop_system/third_party/Depth-Anything-V2 (clone from GitHub). "
             "When set with --depth-model depth_anything, runs native PyTorch (fast) instead of Hugging Face.",
    )
    ap.add_argument(
        "--depth-da-encoder",
        default="vits",
        choices=["vits", "vitb", "vitl", "vitg"],
        help="Encoder tag for --depth-da-pth (must match the checkpoint).",
    )
    ap.add_argument("--depth-device", default="",
                    help="Torch device for depth: 'cpu', 'mps', 'cuda'. "
                         "Default auto. Use 'cpu' to free the GPU for MuJoCo rendering.")
    ap.add_argument("--depth-interval", type=float, default=None,
                    help="Minimum seconds between depth inferences. "
                         "Default depends on --depth-model (Anything is faster = shorter interval).")
    ap.add_argument("--focal-px", type=float, default=0.0,
                    help="Focal-length guess at depth-inference resolution for Depth Pro only "
                         "(default 0 = auto-estimate). Ignored for Depth Anything.")
    ap.add_argument(
        "--iphone-depth",
        action="store_true",
        help=(
            "Use iPhone 16 Pro LiDAR depth via Record3D USB streaming. "
            "Requires: pip install record3d + Record3D app on iPhone with USB Streaming enabled. "
            "Replaces all monocular depth estimation (Depth Anything / Depth Pro). "
            "Only works on iPhone 16 Pro / Pro Max (has LiDAR Scanner)."
        ),
    )
    ap.add_argument(
        "--iphone-device",
        type=int,
        default=0,
        help="Record3D device index when multiple iPhones are connected (default 0).",
    )
    ap.add_argument(
        "--stereo-iphone",
        action="store_true",
        help=(
            "Use iPhone 16 wide + telephoto stereo for metric depth. "
            "Requires the MultiCamStreamer iOS app (teleop_system/ios_streamer/) "
            "and --iphone-ip. Optionally pair with --stereo-calib for full accuracy."
        ),
    )
    ap.add_argument(
        "--iphone-ip",
        default="",
        help="iPhone IP address shown in the MultiCamStreamer app (used with --stereo-iphone).",
    )
    ap.add_argument(
        "--stereo-calib",
        default="",
        help=(
            "Path to stereo calibration .npz produced by scripts/calib_stereo_iphone.py. "
            "Required for full accuracy; without it the pipeline uses a coarse approximation."
        ),
    )
    ap.add_argument(
        "--stereo-mac-listen",
        action="store_true",
        help=(
            "Stereo iPhone reverse mode: Mac listens on --stereo-listen-port; "
            "set MultiCamStreamer «Mac IP (reverse)» to this Mac. "
            "Use when Personal Hotspot blocks inbound TCP to the phone."
        ),
    )
    ap.add_argument(
        "--stereo-listen-port",
        type=int,
        default=9080,
        help="TCP port for --stereo-mac-listen (default 9080).",
    )
    ap.add_argument(
        "--capture-fps",
        type=int,
        default=None,
        help=(
            "Ask the camera driver to capture at this frame rate (e.g. 15 or 20). "
            "Useful to reduce Continuity Camera latency: lower fps = less wireless "
            "bandwidth = smaller buffer backlog. Default: let the driver decide (usually 30)."
        ),
    )
    ap.add_argument("--no-freeze-on-fist", action="store_true")
    ap.add_argument("--no-viewer", action="store_true",
                    help="Run MuJoCo physics headlessly (no GLFW window).")
    ap.add_argument("--show-cv", action="store_true",
                    help="Show pose + depth debug panel in a separate OpenCV window.")
    ap.add_argument("--panel-every", type=int, default=2,
                    help="Rebuild debug panel every N frames (default 2, saves CPU).")
    ap.add_argument("--jpeg-quality", type=int, default=70,
                    help="JPEG quality for IPC frames (default 70).")
    ap.add_argument("--target-fps", type=float, default=None,
                    help="Target processing rate in Hz. "
                         "Default: 5 for --video, 1 for live camera.")
    ap.add_argument("--no-hand", action="store_true",
                    help="Skip hand landmarker (saves ~300 MB + 1 GL context). "
                         "Arm still moves; pinch/fist gestures disabled.")
    ap.add_argument(
        "--landmark-smooth",
        type=float,
        default=0.25,
        help="Temporal EMA weight on each NEW shoulder/elbow/wrist sample before IK "
             "(0 = frozen, 1 = raw). Lower = steadier but more lag. Default 0.25.",
    )
    ap.add_argument(
        "--wrist-depth-blend",
        type=float,
        default=0.0,
        help="Blend monocular depth into the wrist 3-D position when NOT using world "
             "landmarks (0 = world-only, default). World landmarks are in a different "
             "metric frame than depth backprojection so blending is disabled for them.",
    )
    args = ap.parse_args()

    if args.show_cv and cv2 is None:
        print("opencv-python not installed; remove --show-cv.", file=sys.stderr)
        return 1

    replay_npz_raw = Path(args.replay_bundle.strip()) if args.replay_bundle.strip() else None
    replay_bundle_path = replay_npz_raw.resolve() if replay_npz_raw else None

    # --- resolve video / camera source ---
    video_path_str = args.video.strip() if args.video else ""
    if video_path_str and not Path(video_path_str).is_absolute():
        # Relative paths are resolved from teleop_system/ root
        video_path_str = str((repo / video_path_str).resolve())

    replay_obj: Optional[ReplayDepthPose] = None

    if replay_bundle_path is not None:
        if not replay_bundle_path.is_file():
            print(f"Replay bundle not found: {replay_bundle_path}", file=sys.stderr)
            return 1
        try:
            replay_obj = ReplayDepthPose(replay_bundle_path)
        except (OSError, FileNotFoundError, RuntimeError) as e:
            print(f"Failed to load replay bundle: {e}", file=sys.stderr)
            return 1
        vb = str(replay_obj.video_path.resolve())
        if video_path_str:
            if Path(video_path_str).resolve() != replay_obj.video_path.resolve():
                print(
                    f"--replay-bundle meta video is {replay_obj.video_path} but "
                    f"--video is {video_path_str}. Use the same clip or omit --video.",
                    file=sys.stderr,
                )
                replay_obj.close()
                return 1
        else:
            video_path_str = vb

    if replay_bundle_path is None and video_path_str and not Path(video_path_str).is_file():
        print(f"Video file not found: {video_path_str}", file=sys.stderr)
        return 1
    _cam_arg = str(args.camera).strip()
    camera_source: "int | str"
    if video_path_str:
        camera_source = video_path_str
    elif _cam_arg.isdigit():
        camera_source = int(_cam_arg)   # device index: 0, 1, 2 …
    else:
        camera_source = _cam_arg        # URL / stream string
    using_video = bool(video_path_str)

    # FPS default: 5 for video (step through at a watchable pace), 1 for live
    if args.target_fps is not None:
        target_fps = max(0.1, float(args.target_fps))
    else:
        target_fps = 5.0 if using_video else 1.0

    # Depth-interval default: Depth Anything is fast → refresh often; Depth Pro is heavy.
    depth_model = args.depth_model.strip().lower()
    if args.depth_interval is not None:
        depth_interval = float(args.depth_interval)
    else:
        if depth_model == "depth_anything":
            depth_interval = 0.12 if using_video else 0.35
        elif depth_model == "depth_pro":
            depth_interval = 0.5 if using_video else 3.0
        else:
            depth_interval = 9999.0  # none: effectively no depth worker submits (no worker anyway)

    if using_video:
        print(f"[video mode] source        : {video_path_str}")
        if replay_bundle_path is not None:
            print(f"[video mode] replay bundle : {replay_bundle_path}")
        print(f"[video mode] loop          : {args.video_loop}")
        print(f"[video mode] target fps    : {target_fps:.1f}")
        if replay_bundle_path is None:
            print(f"[video mode] depth model   : {depth_model}")
            print(f"[video mode] depth interval: {depth_interval:.3f} s")

    # --- build objects ---
    mapper = load_mapper(Path(args.config))

    ck_str = args.depth_checkpoint.strip()
    ck_path = Path(ck_str).resolve() if ck_str else None
    if ck_path is not None and not ck_path.is_file():
        print(f"Depth Pro checkpoint not found: {ck_path}", file=sys.stderr)
        ck_path = None

    if replay_bundle_path is None and depth_model == "depth_pro" and (
        ck_path is None or not ck_path.is_file()
    ):
        print(
            "--depth-model depth_pro requires a valid --depth-checkpoint .pt file.",
            file=sys.stderr,
        )
        return 1

    if args.depth_checkpoint.strip() and depth_model != "depth_pro":
        print(
            "[note] --depth-checkpoint is ignored unless you use --depth-model depth_pro "
            f"(current depth model: {depth_model}).",
            flush=True,
        )

    da_str = args.depth_da_pth.strip()
    depth_da_path: Optional[Path] = None
    if da_str:
        da_p = Path(da_str)
        if not da_p.is_absolute():
            da_p = (repo / da_p).resolve()
        else:
            da_p = da_p.resolve()
        if not da_p.is_file():
            print(f"--depth-da-pth not found: {da_p}", file=sys.stderr)
            return 1
        depth_da_path = da_p
        if depth_model != "depth_anything":
            print(
                "[note] --depth-da-pth is only used with --depth-model depth_anything; ignoring.",
                flush=True,
            )
            depth_da_path = None

    if replay_bundle_path is None and depth_model == "depth_anything" and depth_da_path is not None:
        print(
            f"[depth] Depth Anything V2 Torch: {depth_da_path}  encoder={args.depth_da_encoder}",
            flush=True,
        )

    frame_period = 1.0 / target_fps
    # At low FPS, run depth on every frame (stride=1) so we always have a fresh map
    depth_stride = args.depth_stride if target_fps > 5.0 else 1

    pose: Optional[PoseTracker] = (
        None if replay_obj is not None else PoseTracker(load_hand_model=not args.no_hand)
    )
    depth_device_str = args.depth_device.strip() or None
    pipeline: Any
    if replay_obj is not None:
        pipeline = replay_obj
        print(
            f"[replay] bundle={replay_bundle_path}  frames={replay_obj.n_frames}  "
            f"video={replay_obj.video_path}",
            flush=True,
        )
        print("[teleop] depth_model=replay (offline)  pose=replay", flush=True)
    elif getattr(args, "iphone_depth", False):
        from iphone_source import Record3DPipeline  # type: ignore[import]
        pipeline = Record3DPipeline(
            device_idx=int(args.iphone_device),
            pose_wh=(int(args.pose_size[0]), int(args.pose_size[1])) if hasattr(args, "pose_size") else (640, 480),
            depth_wh=(256, 192),
        )
        print("[teleop] depth_model=record3d_lidar  (iPhone 16 Pro LiDAR)", flush=True)
    elif getattr(args, "stereo_iphone", False):
        from stereo_iphone import StereoIPhonePipeline  # type: ignore[import]
        iphone_ip = str(args.iphone_ip).strip()
        mac_listen = bool(getattr(args, "stereo_mac_listen", False))
        if not mac_listen and not iphone_ip:
            print("--stereo-iphone requires --iphone-ip <address> or --stereo-mac-listen", file=sys.stderr)
            return 1
        calib_path = Path(args.stereo_calib).resolve() if args.stereo_calib else None
        listen_port = int(getattr(args, "stereo_listen_port", 9080) or 9080)
        pipeline = StereoIPhonePipeline(
            iphone_ip=iphone_ip,
            calib_path=calib_path if calib_path and calib_path.is_file() else None,
            output_size=(320, 240),
            mac_listen=mac_listen,
            listen_port=listen_port,
        )
        calib_label = str(calib_path) if (calib_path and calib_path.is_file()) else "none (coarse mode)"
        mode = f"mac_listen:{listen_port}" if mac_listen else f"forward:{iphone_ip}"
        print(f"[teleop] depth_model=stereo_iphone  transport={mode}  calib={calib_label}", flush=True)
    else:
        try:
            pipeline = DepthEstimatePipeline(
                checkpoint_path=ck_path,
                device_str=depth_device_str,
                depth_model=depth_model,
                depth_anything_hub_id=str(args.depth_anything_model).strip() or None,
                depth_anything_local_pth=depth_da_path,
                depth_da_encoder=str(args.depth_da_encoder).strip().lower(),
                camera_device=camera_source,
                depth_every_nth_frame=depth_stride,
                focal_px=args.focal_px,
                depth_min_interval_s=depth_interval,
                loop_video=bool(args.video_loop),
                capture_fps=int(args.capture_fps) if args.capture_fps else None,
            )
        except Exception as e:  # noqa: BLE001
            print(f"Depth pipeline init failed: {e}", file=sys.stderr)
            print(
                "For Hugging Face Depth Anything: pip install transformers pillow accelerate. "
                "For --depth-da-pth: clone Depth-Anything-V2 into teleop_system/third_party/.",
                file=sys.stderr,
            )
            return 1
        print(
            f"[teleop] depth_model={depth_model}  "
            f"worker={'on' if pipeline._worker else 'off'}  "  # noqa: SLF001
            f"interval={depth_interval:.3f}s",
            flush=True,
        )
    state: Dict[str, Any] = {
        "last_q_deg": [0.0, 0.0, 0.0, 0.0, 0.0],
        "last_grip": 0.5,
        "frame_i": 0,
        "gesture_last": "n/a",
    }

    gesture_classifier: Optional[TFLiteGestureClassifier] = None
    if args.gesture_control:
        if not args.gesture_model:
            print("--gesture-control requires --gesture-model <path/to/keypoint_classifier.tflite>", file=sys.stderr)
            return 1
        model_path = Path(args.gesture_model).resolve()
        labels_path = Path(args.gesture_labels).resolve() if args.gesture_labels else None
        if not model_path.is_file():
            print(f"Gesture model not found: {model_path}", file=sys.stderr)
            return 1
        try:
            gesture_classifier = TFLiteGestureClassifier(model_path=model_path, labels_path=labels_path)
            print(f"[gesture] model={model_path} labels={labels_path}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"Failed to init gesture classifier: {e}", file=sys.stderr)
            return 1

    ard: Optional[ArduinoTeleopPort] = None
    if args.mode == "real":
        if args.no_arduino:
            print("[real mode] --no-arduino enabled: skipping serial output.", flush=True)
        else:
            candidates = sorted(Path("/dev").glob(args.serial_pattern))
            if not candidates:
                print(f"No device at /dev/{args.serial_pattern}; connect Arduino.")
                pipeline.close()
                return 1
            ard = ArduinoTeleopPort(candidates[0])

    # --- MuJoCo sim ---
    env: Optional[SimTeleopEnv] = None
    viewer_enter = None
    if args.mode == "sim":
        if not HAS_MUJOCO:
            print("`pip install mujoco` required for --mode sim.", file=sys.stderr)
            pipeline.close()
            return 1
        env = SimTeleopEnv(mjcf_path=repo / "models" / "scene_teleop.xml")
        mujoco.mj_resetData(env.model, env.data)
        if not args.no_viewer:
            try:
                viewer_enter = mj_viewer.launch_passive(env.model, env.data)
            except RuntimeError as e:
                if "mjpython" in str(e).lower():
                    print(
                        "MuJoCo viewer needs mjpython on macOS.\n"
                        "  mjpython -m src.main --mode sim --show-cv\n"
                        "  or: python -m src.main --mode sim --no-viewer",
                        file=sys.stderr,
                    )
                    pipeline.close()
                    return 1
                raise

    # --- OpenCV display subprocess ---
    # mjpython owns the Cocoa main-thread run loop; cv2.imshow must live in its
    # own spawned process which gets a fresh Cocoa event loop.
    disp_proc: Optional[_mp.Process] = None
    frame_q: Optional["_mp.Queue[Optional[bytes]]"] = None
    jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)] if cv2 else []

    if args.show_cv:
        ctx = _mp.get_context("spawn")
        frame_q = ctx.Queue(maxsize=2)
        # Use cv_display.worker — a module that imports ONLY cv2 + numpy.
        # This prevents spawn from re-importing mediapipe/torch (~6-8 GB saved).
        disp_proc = ctx.Process(
            target=_cv_display_mod.worker,
            args=(frame_q,),
            daemon=True,
        )
        disp_proc.start()

    # --- main loop ---
    stop_requested = False
    ctx_mgr = viewer_enter if viewer_enter is not None else nullcontext()

    try:
        with ctx_mgr as viewer_obj:
            _t_last = time.monotonic() - frame_period  # fire immediately on first iteration
            while not stop_requested:
                if viewer_obj is not None and not viewer_obj.is_running():
                    break

                q_cmd, panel = _run_teleop_iteration(
                    mapper=mapper,
                    pose=pose,
                    pipeline=pipeline,
                    state=state,
                    env=env,
                    ard=ard,
                    freeze_on_fist=not args.no_freeze_on_fist,
                    build_panel=bool(args.show_cv),
                    hand_only_control=bool(args.hand_only_control),
                    gesture_control=bool(args.gesture_control),
                    gesture_classifier=gesture_classifier,
                    gesture_min_conf=float(args.gesture_min_conf),
                    panel_every_nth=int(args.panel_every),
                    landmark_smooth=float(args.landmark_smooth),
                    wrist_depth_blend=float(args.wrist_depth_blend),
                )

                # Video-file EOF: stop when not looping
                if using_video and not args.video_loop and pipeline._camera.at_end:  # noqa: SLF001
                    print("[video mode] end of file — stopping.")
                    stop_requested = True

                # JPEG-compress → send to display subprocess (non-blocking, drop oldest if full)
                if args.show_cv and panel is not None and frame_q is not None and cv2 is not None:
                    arr = _bgr_display_ready(panel)
                    ok, buf = cv2.imencode(".jpg", arr, jpeg_params)
                    if ok:
                        data = buf.tobytes()
                        try:
                            frame_q.put_nowait(data)
                        except _queue_mod.Full:
                            try:
                                frame_q.get_nowait()
                            except Exception:
                                pass
                            try:
                                frame_q.put_nowait(data)
                            except Exception:
                                pass

                # Periodically release Python-level cyclic garbage (every 30 frames)
                frame_i = int(state.get("frame_i", 0))
                if frame_i % 30 == 0:
                    gc.collect()

                # Print progress for video mode every 10 processed frames
                if using_video and frame_i > 0 and frame_i % 10 == 0:
                    total = pipeline.video_frame_count
                    _POS = 1  # cv2.CAP_PROP_POS_FRAMES
                    pos = int(pipeline._camera._cap.get(_POS) or 0)  # noqa: SLF001
                    pct = f"{100*pos//total}%" if total > 0 else "?"
                    print(f"[video] frame {frame_i} (file pos {pos}/{total} = {pct})", flush=True)

                # Check if the display subprocess exited (user pressed q)
                if disp_proc is not None and not disp_proc.is_alive():
                    stop_requested = True
                    break

                if viewer_obj is not None:
                    viewer_obj.sync()

                # Accurate wall-clock rate limiter
                _now = time.monotonic()
                _elapsed = _now - _t_last
                _t_last = _now
                _sleep = frame_period - _elapsed
                if _sleep > 0:
                    time.sleep(_sleep)

    except KeyboardInterrupt:
        pass
    finally:
        if frame_q is not None:
            try:
                frame_q.put_nowait(None)
            except Exception:
                pass
        if disp_proc is not None:
            disp_proc.join(timeout=2.0)
            if disp_proc.is_alive():
                disp_proc.terminate()
        if ard is not None:
            ard.close()
        pipeline.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
