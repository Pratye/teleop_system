"""MediaPipe pose + hand tracking with 3D wrist lift via depth sampling."""

from __future__ import annotations

import pathlib
import urllib.request
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Sequence, Tuple

import cv2
import numpy as np

_MEDIAPIPE_POSE_LITE = (
    "https://storage.googleapis.com/"
    "mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/"
    "1/pose_landmarker_lite.task"
)

_MEDIAPIPE_HAND_TASK = (
    "https://storage.googleapis.com/"
    "mediapipe-models/hand_landmarker/hand_landmarker/float16/"
    "1/hand_landmarker.task"
)

# Thumb tip / index tip (pinch aperture)
_MP_THUMB_TIP, _MP_IDX_TIP = 4, 8
_MP_WRIST_I, _MP_MIDDLE_MCP = 0, 9


def _ensure_file(url: str, dest: pathlib.Path) -> pathlib.Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        return dest
    urllib.request.urlretrieve(url, dest)  # noqa: S310 urllib for model weights
    return dest


def pinch_open_from_landmarks(norm_landmarks: Sequence[object]) -> float:
    """
    Thumb–index aperture normalized by hand span (roughly independent of depth).
    Returns ~0 pinched tight, ~1 fingers spread for pinch/release.
    """
    if len(norm_landmarks) < 21:
        return 0.5

    def dn(i: int) -> Tuple[float, float]:
        p = norm_landmarks[i]
        return float(p.x), float(p.y)

    ax, ay = dn(_MP_THUMB_TIP)
    bx, by = dn(_MP_IDX_TIP)
    thumb_index = float(np.hypot(ax - bx, ay - by))
    sx, sy = dn(_MP_WRIST_I)
    mx, my = dn(_MP_MIDDLE_MCP)
    span = float(np.hypot(sx - mx, sy - my))
    if span < 8e-3:
        return 0.5
    ratio = thumb_index / max(span, 1e-5)
    # Empirical range; tune via config if needed (~0.1 tight pinch … ~0.7+ separated)
    return float(np.clip((ratio - 0.10) / (0.52 - 0.10 + 1e-6), 0.0, 1.0))


def get_3d_from_depth(
    depth_metres_hw: np.ndarray,
    u_px: float,
    v_px: float,
    camera_matrix_k: np.ndarray,
    *,
    neighbourhood: int = 5,
    min_depth_metres: float = 0.05,
    max_depth_metres: float = 50.0,
) -> Tuple[np.ndarray, bool]:
    """
    Back-project pixel (u,v) to metres in OpenCV camera frame (x→right,y→down,z→forward).

    Returns (xyz_m, sample_ok).
    """
    h, w = depth_metres_hw.shape[:2]

    yi = int(np.clip(round(v_px), 0, h - 1))
    xi = int(np.clip(round(u_px), 0, w - 1))
    r = max(1, neighbourhood // 2)

    patch = depth_metres_hw[
        max(0, yi - r) : min(h, yi + r + 1),
        max(0, xi - r) : min(w, xi + r + 1),
    ]
    valid = patch[np.logical_and(patch > min_depth_metres, patch < max_depth_metres)]

    if valid.size >= 5:
        z_m = float(np.median(valid))
        sample_ok = True
    elif valid.size > 0:
        z_m = float(np.nanmedian(valid))
        sample_ok = bool(np.isfinite(z_m))
    else:
        z_m = float(depth_metres_hw[yi, xi])
        sample_ok = np.isfinite(z_m) and min_depth_metres < z_m < max_depth_metres

    fx, fy = float(camera_matrix_k[0, 0]), float(camera_matrix_k[1, 1])
    cx, cy = float(camera_matrix_k[0, 2]), float(camera_matrix_k[1, 2])

    x_m = (u_px - cx) * z_m / fx if fx > 1e-6 else 0.0
    y_m = (v_px - cy) * z_m / fy if fy > 1e-6 else 0.0
    return np.array([x_m, y_m, z_m], dtype=np.float64), sample_ok


@dataclass
class PoseObservation:
    wrist_uv: Tuple[float, float]
    elbow_uv: Tuple[float, float]
    shoulder_uv: Tuple[float, float]
    pose_visibility: Tuple[float, float, float]

    fist_gesture_active: bool
    hand_landmarks_present: bool
    handedness_hand: Literal["Right", "Left", "Unknown"]
    landmarks_ok: bool
    # 0 = pinched / closed, 1 = open — only meaningful when right hand detected
    pinch_open: float = 0.5
    pinch_valid: bool = False
    # Populated for debug drawing (after update_rgb)
    pose_keypoints_xy: Optional[np.ndarray] = None  # (33, 2) pixel coords
    right_hand_xy: Optional[List[Tuple[float, float]]] = field(default=None)
    # Landmark indices actually used (shoulder, elbow, wrist) — for debug drawing
    arm_landmark_ids: Tuple[int, int, int] = (12, 14, 16)  # default RIGHT_*

    # --- 3D world landmarks (metric, camera-OpenCV frame) ---
    # MediaPipe world frame: origin=hip-midpoint, x=viewer-left, y=down, z=backward.
    # Converted here to camera-OpenCV (x_cam=-x_w, y_cam=y_w, z_cam=z_w) so they
    # can be passed directly to the IK mapper's cam_to_robot_pt().
    # None when the model did not return world landmarks.
    wrist_world_m: Optional[np.ndarray] = None    # (3,) float64 metres
    elbow_world_m: Optional[np.ndarray] = None
    shoulder_world_m: Optional[np.ndarray] = None


def _dist_xy(ax: float, ay: float, bx: float, by: float) -> float:
    return float(np.hypot(ax - bx, ay - by))


def _fist_detected(norm_landmarks: Sequence[object]) -> bool:
    """
    Landmark indices — MediaPipe hand topology.
    Wrist = 0; tips index/middle/ring/pinky = 8,12,16,20; middle MCP ref = 9.
    """

    def xy(i: int) -> Tuple[float, float]:
        p = norm_landmarks[i]
        return float(p.x), float(p.y)

    wx, wy = xy(0)
    ref = _dist_xy(wx, wy, *xy(9))
    if ref < 5e-3:
        return False

    thresh = ref * 0.55

    tips_ok = [_dist_xy(wx, wy, *xy(i)) < thresh for i in (8, 12, 16, 20)]
    return sum(tips_ok) >= 3


# MediaPipe hand skeleton connections (21 keypoints)
_HAND_CONNECTIONS: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # index
    (9, 10), (10, 11), (11, 12),               # middle
    (13, 14), (14, 15), (15, 16),              # ring
    (17, 18), (18, 19), (19, 20),              # little
    (0, 17), (2, 5), (5, 9), (9, 13), (13, 17),  # palm
]
_FINGERTIP_IDS = frozenset({4, 8, 12, 16, 20})


def draw_pose_arm_and_labels(
    rgb_uint8: np.ndarray,
    obs: PoseObservation,
    depth_fresh: bool,
    gesture_label: str = "",
    gesture_conf: float = 0.0,
    fps: int = 0,
) -> np.ndarray:
    """RGB (H,W,3) uint8 -> BGR image with kinivi-style hand overlay + arm skeleton."""
    bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]

    # --- pose arm skeleton (teal) ---
    if obs.pose_keypoints_xy is not None and len(obs.pose_keypoints_xy) >= 17:
        px = obs.pose_keypoints_xy
        r_sh, r_el, r_wr = obs.arm_landmark_ids
        for pair in ((r_sh, r_el), (r_el, r_wr)):
            a = tuple(np.round(px[pair[0]]).astype(int))
            b = tuple(np.round(px[pair[1]]).astype(int))
            cv2.line(bgr, a, b, (80, 200, 255), 3, cv2.LINE_AA)
        wr = tuple(np.round(px[r_wr]).astype(int))
        cv2.circle(bgr, wr, 10, (0, 255, 0), -1)

    # --- kinivi-style hand overlay ---
    if obs.right_hand_xy and len(obs.right_hand_xy) >= 21:
        lms = obs.right_hand_xy

        # skeleton lines: thick black shadow then thin white
        for a_i, b_i in _HAND_CONNECTIONS:
            pa = (int(round(lms[a_i][0])), int(round(lms[a_i][1])))
            pb = (int(round(lms[b_i][0])), int(round(lms[b_i][1])))
            cv2.line(bgr, pa, pb, (0, 0, 0), 6, cv2.LINE_AA)
            cv2.line(bgr, pa, pb, (255, 255, 255), 2, cv2.LINE_AA)

        # landmark circles: white fill + black outline (fingertips larger)
        for idx, lm in enumerate(lms):
            pt = (int(round(lm[0])), int(round(lm[1])))
            r = 8 if idx in _FINGERTIP_IDS else 5
            cv2.circle(bgr, pt, r, (255, 255, 255), -1)
            cv2.circle(bgr, pt, r, (0, 0, 0), 1)

        # bounding rect
        xs = [int(round(lm[0])) for lm in lms]
        ys = [int(round(lm[1])) for lm in lms]
        bx1, by1 = max(0, min(xs) - 8), max(0, min(ys) - 8)
        bx2, by2 = min(w - 1, max(xs) + 8), min(h - 1, max(ys) + 8)
        cv2.rectangle(bgr, (bx1, by1), (bx2, by2), (0, 0, 0), 1)

        # label badge above bounding rect
        hand_char = obs.handedness_hand[0] if obs.handedness_hand else "R"
        badge = f"{hand_char}:{gesture_label}" if gesture_label else hand_char
        badge_y = max(22, by1)
        cv2.rectangle(bgr, (bx1, badge_y), (bx2, badge_y - 22), (0, 0, 0), -1)
        cv2.putText(
            bgr, badge, (bx1 + 5, badge_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
        )

    # --- FPS counter (kinivi style: black shadow + white text) ---
    fps_str = f"FPS:{fps}"
    cv2.putText(bgr, fps_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(bgr, fps_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    # --- depth-fresh indicator bottom-right ---
    stale = "" if depth_fresh else " [depth stale]"
    if stale:
        cv2.putText(
            bgr, stale.strip(), (w - 160, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 100, 255), 1, cv2.LINE_AA,
        )

    return bgr


class PoseTracker:
    """Mediapipe Tasks pose + right-hand fist + pinch heuristic.

    ``hand_every_nth`` — run hand landmarker only every Nth frame (default 3).
    Pose runs every frame; hand is the expensive GPU/CPU path.
    """

    def __init__(
        self,
        *,
        pose_model_path: Optional[pathlib.Path] = None,
        hand_model_path: Optional[pathlib.Path] = None,
        cache_dir: Optional[pathlib.Path] = None,
        hand_every_nth: int = 3,
        mirrored_input: bool = False,
        load_hand_model: bool = True,
    ) -> None:
        # MediaPipe labels by image orientation, not body anatomy.
        # Non-mirrored camera: person's right arm → image LEFT side → use LEFT_* pose landmarks
        #                       and look for "Left" handedness label in HandLandmarker.
        # Mirrored/selfie camera: person's right arm → image RIGHT side → use RIGHT_*.
        self._mirrored = bool(mirrored_input)
        self._hand_target = "Right" if mirrored_input else "Left"
        mp_cache = pathlib.Path(cache_dir) if cache_dir else (
            pathlib.Path(__file__).resolve().parents[1] / "models" / "mp_cache"
        )

        pose_p = pathlib.Path(pose_model_path) if pose_model_path else mp_cache / "pose_landmarker_lite.task"
        hand_p = pathlib.Path(hand_model_path) if hand_model_path else mp_cache / "hand_landmarker.task"

        pose_file = _ensure_file(_MEDIAPIPE_POSE_LITE, pathlib.Path(pose_p))
        hand_ok = False
        if load_hand_model:
            try:
                hand_file = _ensure_file(_MEDIAPIPE_HAND_TASK, pathlib.Path(hand_p))
                hand_ok = hand_file.is_file()
            except OSError:
                hand_ok = False

        from mediapipe.tasks import python as mp_python  # noqa: PLC0415
        from mediapipe.tasks.python import vision  # noqa: PLC0415
        from mediapipe.tasks.python.vision import pose_landmarker as pl  # noqa: PLC0415
        from mediapipe.tasks.python.vision.core import image as mp_img  # noqa: PLC0415

        pose_options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(pose_file)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.4,
            min_pose_presence_confidence=0.4,
        )
        self._pose_lm = vision.PoseLandmarker.create_from_options(pose_options)
        self._Pl = pl.PoseLandmark

        self._hand_lm = None
        if hand_ok:
            ho = vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(hand_file)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.4,
                min_hand_presence_confidence=0.4,
                min_tracking_confidence=0.35,
            )
            self._hand_lm = vision.HandLandmarker.create_from_options(ho)

        self._mp_img = mp_img.Image
        self._mp_fmt = mp_img.ImageFormat

        self._hand_every_nth = max(1, int(hand_every_nth))
        self._frame_count = 0
        # Cache last hand result so we can reuse on skipped frames
        self._last_hand_result: Optional[object] = None

    def update_rgb(self, rgb_hwc_uint8: np.ndarray, timestamp_ms: int) -> PoseObservation:
        h, w, _ = rgb_hwc_uint8.shape

        mpi = self._mp_img(self._mp_fmt.SRGB, np.ascontiguousarray(rgb_hwc_uint8))

        pose_res = self._pose_lm.detect_for_video(mpi, timestamp_ms)

        _arm_ids_default = (
            (int(self._Pl.RIGHT_SHOULDER), int(self._Pl.RIGHT_ELBOW), int(self._Pl.RIGHT_WRIST))
            if self._mirrored
            else (int(self._Pl.LEFT_SHOULDER), int(self._Pl.LEFT_ELBOW), int(self._Pl.LEFT_WRIST))
        )
        defaults = PoseObservation(
            wrist_uv=(w / 2, h / 2),
            elbow_uv=(w / 2, h / 2),
            shoulder_uv=(w / 2, h / 2),
            pose_visibility=(0.0, 0.0, 0.0),
            fist_gesture_active=False,
            hand_landmarks_present=False,
            handedness_hand="Unknown",
            landmarks_ok=False,
            pinch_open=0.5,
            pinch_valid=False,
            pose_keypoints_xy=None,
            right_hand_xy=None,
            arm_landmark_ids=_arm_ids_default,
        )

        if not pose_res.pose_landmarks:
            return defaults

        lm_pose = pose_res.pose_landmarks[0]

        pix = np.zeros((33, 2), dtype=np.float64)
        for ii, lm in enumerate(lm_pose):
            pix[ii] = [float(lm.x) * w, float(lm.y) * h]

        # MediaPipe labels landmarks by image-left/right, not by body anatomy.
        # Non-mirrored camera: person's anatomical right arm is on the IMAGE left
        #   → use LEFT_* indices (11=shoulder, 13=elbow, 15=wrist).
        # Mirrored/selfie camera: person's right arm is on the IMAGE right
        #   → use RIGHT_* indices (12=shoulder, 14=elbow, 16=wrist).
        if self._mirrored:
            rw = lm_pose[self._Pl.RIGHT_WRIST]
            re = lm_pose[self._Pl.RIGHT_ELBOW]
            rs = lm_pose[self._Pl.RIGHT_SHOULDER]
            arm_ids = (
                int(self._Pl.RIGHT_SHOULDER),
                int(self._Pl.RIGHT_ELBOW),
                int(self._Pl.RIGHT_WRIST),
            )
        else:
            rw = lm_pose[self._Pl.LEFT_WRIST]
            re = lm_pose[self._Pl.LEFT_ELBOW]
            rs = lm_pose[self._Pl.LEFT_SHOULDER]
            arm_ids = (
                int(self._Pl.LEFT_SHOULDER),
                int(self._Pl.LEFT_ELBOW),
                int(self._Pl.LEFT_WRIST),
            )

        def pix_one(lm) -> Tuple[float, float]:
            return float(lm.x) * w, float(lm.y) * h

        wx, wy = pix_one(rw)
        ex, ey = pix_one(re)
        sx, sy = pix_one(rs)

        # Require all three arm joints to be visible with reasonable confidence.
        # 0.45 filters out the jittery low-confidence readings that cause sudden jumps.
        landmarks_ok = min(rw.visibility, re.visibility, rs.visibility) > 0.45

        # --- 3D world landmarks (metric) ---
        # world frame: x=viewer-left, y=down, z=backward(away-from-camera)
        # → camera-OpenCV: x_cam = -x_w,  y_cam = y_w,  z_cam = z_w
        wrist_w = elbow_w = shoulder_w = None
        if pose_res.pose_world_landmarks:
            wl = pose_res.pose_world_landmarks[0]
            def _wl(lm) -> np.ndarray:
                return np.array([-float(lm.x), float(lm.y), float(lm.z)], dtype=np.float64)
            if self._mirrored:
                wrist_w    = _wl(wl[self._Pl.RIGHT_WRIST])
                elbow_w    = _wl(wl[self._Pl.RIGHT_ELBOW])
                shoulder_w = _wl(wl[self._Pl.RIGHT_SHOULDER])
            else:
                wrist_w    = _wl(wl[self._Pl.LEFT_WRIST])
                elbow_w    = _wl(wl[self._Pl.LEFT_ELBOW])
                shoulder_w = _wl(wl[self._Pl.LEFT_SHOULDER])

        obs = PoseObservation(
            wrist_uv=(wx, wy),
            elbow_uv=(ex, ey),
            shoulder_uv=(sx, sy),
            pose_visibility=(float(rw.visibility), float(re.visibility), float(rs.visibility)),
            fist_gesture_active=False,
            hand_landmarks_present=False,
            handedness_hand="Unknown",
            landmarks_ok=landmarks_ok,
            pinch_open=0.5,
            pinch_valid=False,
            pose_keypoints_xy=pix,
            right_hand_xy=None,
            arm_landmark_ids=arm_ids,
            wrist_world_m=wrist_w,
            elbow_world_m=elbow_w,
            shoulder_world_m=shoulder_w,
        )

        fist_hand = False
        pinch_o = 0.5
        pinch_ok = False
        right_px: Optional[List[Tuple[float, float]]] = None

        self._frame_count += 1
        run_hand = self._hand_lm is not None and (self._frame_count % self._hand_every_nth == 0)

        if run_hand:
            hres = self._hand_lm.detect_for_video(mpi, timestamp_ms)  # type: ignore[union-attr]
            self._last_hand_result = hres
        else:
            hres = self._last_hand_result  # reuse previous result on skipped frames

        if self._hand_lm is not None and hres is not None:
            if hres.hand_landmarks:
                obs.hand_landmarks_present = True
                idx_pick = 0
                for idx in range(len(hres.hand_landmarks)):
                    hh = hres.handedness[idx][0] if idx < len(hres.handedness) and hres.handedness[idx] else None
                    cand = getattr(hh, "category_name", "") if hh else ""
                    if isinstance(cand, str) and cand.startswith(self._hand_target):
                        idx_pick = idx
                        break

                landmarks = hres.hand_landmarks[idx_pick]

                fist_hand = _fist_detected(landmarks)

                pinch_ok = True
                pinch_o = pinch_open_from_landmarks(landmarks)
                right_px = [(float(landmarks[i].x) * w, float(landmarks[i].y) * h) for i in range(21)]

                if hres.handedness and idx_pick < len(hres.handedness) and hres.handedness[idx_pick]:
                    hh0 = hres.handedness[idx_pick][0]
                    cand = getattr(hh0, "category_name", "?")
                    obs.handedness_hand = cand if isinstance(cand, str) else str(cand)

                obs.right_hand_xy = right_px
                obs.pinch_open = float(pinch_o)
                obs.pinch_valid = bool(pinch_ok)

        obs.fist_gesture_active = bool(fist_hand)

        return obs


__all__ = [
    "PoseTracker",
    "PoseObservation",
    "get_3d_from_depth",
    "draw_pose_arm_and_labels",
    "pinch_open_from_landmarks",
]
