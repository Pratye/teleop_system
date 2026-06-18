"""Load precomputed Depth Pro + pose tensors; drive teleop without live inference."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

from pose_estimator import PoseObservation


class _ReplayCamShim:
    """Stand-in so ``pipeline._camera.at_end`` / ``._cap`` match DepthCamera."""

    __slots__ = ("_replay",)

    def __init__(self, replay: Any):
        object.__setattr__(self, "_replay", replay)

    @property
    def at_end(self) -> bool:
        return self._replay._idx_frame >= self._replay.n_frames  # noqa: SLF001

    @property
    def _cap(self):  # noqa: ANN001
        return self._replay._cap  # noqa: SLF001


def pose_observation_from_bundle_row(
    *,
    wrist_uv: np.ndarray,
    elbow_uv: np.ndarray,
    shoulder_uv: np.ndarray,
    visibility: np.ndarray,
    landmarks_ok: bool,
    wrist_w: np.ndarray,
    elbow_w: np.ndarray,
    shoulder_w: np.ndarray,
    pose_kp_xy: Optional[np.ndarray],
    pinch_open: float,
    pinch_valid: bool,
    fist: bool,
    hand_lm: bool,
    handedness_raw: str,
    right_hand_xy: Optional[List[Tuple[float, float]]],
    arm_landmark_ids: Tuple[int, int, int],
) -> PoseObservation:
    ww = np.all(np.isfinite(wrist_w)) and np.all(np.isfinite(elbow_w)) and np.all(np.isfinite(shoulder_w))
    hh_use = handedness_raw if handedness_raw in ("Right", "Left") else "Unknown"

    return PoseObservation(
        wrist_uv=(float(wrist_uv[0]), float(wrist_uv[1])),
        elbow_uv=(float(elbow_uv[0]), float(elbow_uv[1])),
        shoulder_uv=(float(shoulder_uv[0]), float(shoulder_uv[1])),
        pose_visibility=(float(visibility[0]), float(visibility[1]), float(visibility[2])),
        fist_gesture_active=bool(fist),
        hand_landmarks_present=bool(hand_lm),
        handedness_hand=hh_use,  # type: ignore[arg-type]
        landmarks_ok=bool(landmarks_ok),
        pinch_open=float(pinch_open),
        pinch_valid=bool(pinch_valid),
        pose_keypoints_xy=(pose_kp_xy.astype(np.float64) if pose_kp_xy is not None else None),
        right_hand_xy=right_hand_xy,
        arm_landmark_ids=arm_landmark_ids,
        wrist_world_m=wrist_w.astype(np.float64) if ww else None,
        elbow_world_m=elbow_w.astype(np.float64) if ww else None,
        shoulder_world_m=shoulder_w.astype(np.float64) if ww else None,
    )


class ReplayDepthPose:
    """
    Sequentially reads ``video_path`` and returns snapshots compatible with ``DepthEstimatePipeline``,

    using arrays produced by ``precompute_assets``.
    """

    #: Focal at depth-inference resolution (px), copied from Depth Pro for this frame.
    last_focal_px: Optional[float] = None
    #: NPZ slice used for the returned ``snapshot()`` (before advancing the stream).

    bundle_frame_index: Optional[int] = None

    def __init__(self, bundle_npz: Path, meta_json: Optional[Path] = None) -> None:

        """

        NPZ keys are produced by ``precompute_assets.precompute_bundle``.


        Meta defaults to ``<bundle_stem>_meta.json`` with ``video_path`` and resolutions.


        """

        self._bundle_path = Path(bundle_npz).resolve()

        sibling = self._bundle_path.parent / f"{self._bundle_path.stem}_meta.json"

        self._meta_path = Path(meta_json) if meta_json else sibling

        raw = dict(np.load(self._bundle_path, allow_pickle=True))

        self.depth_pose = raw["depth_metres_pose"].astype(np.float32)

        self.focal_px = raw["focal_px"].astype(np.float32)

        self.wrist_uv = raw["wrist_uv"].astype(np.float64)

        self.elbow_uv = raw["elbow_uv"].astype(np.float64)

        self.shoulder_uv = raw["shoulder_uv"].astype(np.float64)

        self.visibility = raw["visibility"].astype(np.float64)

        self.landmarks_ok = raw["landmarks_ok"].astype(bool)

        self.wrist_world = raw["wrist_world_m"].astype(np.float64)

        self.elbow_world = raw["elbow_world_m"].astype(np.float64)

        self.shoulder_world = raw["shoulder_world_m"].astype(np.float64)

        kp = raw.get("pose_kp_xy")

        self.pose_kp_xy = kp.astype(np.float32) if kp is not None else None

        self.pinch_open = raw["pinch_open"].astype(np.float64)

        self.pinch_valid = raw["pinch_valid"].astype(bool)

        self.fist = raw["fist"].astype(bool)

        self.hand_lm = raw["hand_present"].astype(bool)

        hh_arr = raw.get("handedness")

        self.handedness = self._decode_handedness(hh_arr)

        aids = raw.get("arm_landmark_ids")

        meta: dict = {}

        if self._meta_path.is_file():

            meta = json.loads(self._meta_path.read_text(encoding="utf-8"))

        vid = meta.get("video_path")

        self.video_path = Path(vid).resolve() if vid else None

        self.n_frames = int(self.depth_pose.shape[0])

        rp = meta.get("pose_hw", [320, 240])

        self.pose_wh = (int(rp[0]), int(rp[1]))

        dip = meta.get("depth_infer_hw", [256, 256])

        self.depth_infer_wh = (int(dip[0]), int(dip[1]))

        hp = raw.get("right_hand_xy")

        self._right_hand_xy = hp.astype(np.float32) if hp is not None else None

        self._cap: Optional[cv2.VideoCapture] = None

        self._replay_cam_shim = _ReplayCamShim(self)

        if aids is not None:

            aa = aids.astype(np.int32)

            self._arm_ids = [(int(aa[i, 0]), int(aa[i, 1]), int(aa[i, 2])) for i in range(aa.shape[0])]

        else:

            self._arm_ids = [(11, 13, 15)] * self.n_frames

        self._idx_frame = 0

        dm = meta.get("depth_backend")

        self.depth_model = str(dm or "depth_pro")

        if self.video_path is None or not self.video_path.is_file():

            raise FileNotFoundError(

                "Replay requires `video_path` in the meta JSON (same clip as precompute) "
                f"(got {vid!r}).",
            )

        self._cap = cv2.VideoCapture(str(self.video_path))

        if not self._cap.isOpened():

            raise RuntimeError(f"Could not open replay video {self.video_path}")

    def _decode_handedness(self, hh_arr: Any) -> List[str]:
        if hh_arr is None:
            return ["Unknown"] * self.n_frames


        out: List[str] = []
        for x in hh_arr:
            if isinstance(x, bytes):

                out.append(x.decode("utf-8", errors="replace"))

            elif isinstance(x, str):

                out.append(x)

            else:

                out.append(str(x))



        while len(out) < self.n_frames:


            out.append("Unknown")


        return out[: self.n_frames]


    @property


    def video_frame_count(self) -> int:


        return self.n_frames


    @property


    def is_video_file(self) -> bool:


        return True


    @property


    def video_fps(self) -> float:


        if self._cap is None:


            return 30.0


        fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 30.0)


        return fps if fps > 1e-3 else 30.0


    depth_model_runtime = "replay_bundle"


    _focal_px = 0.0




    def close(self) -> None:


        self.bundle_frame_index = None


        self.last_focal_px = None


        if self._cap is not None:


            self._cap.release()


            self._cap = None




    def focal_px_override(self, fx: float) -> None:

        """Replay ignores overrides; focal comes from the bundle."""

        pass






    @property


    def _camera(self) -> _ReplayCamShim:


        return self._replay_cam_shim




    def snapshot(


        self,


    ) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray], bool, Optional[np.ndarray]]:


        """Sequential ``VideoCapture.read`` paired with NPZ slice; depth marked fresh each frame."""

        self.last_focal_px = None


        self.bundle_frame_index = None


        if self._cap is None or self._idx_frame >= self.n_frames:


            return time.monotonic(), None, None, False, None


        ok, bgr = self._cap.read()


        if not ok or bgr is None:


            self._idx_frame = self.n_frames


            return time.monotonic(), None, None, False, None


        iw, ih = self.pose_wh


        rgb_pose = cv2.cvtColor(


            cv2.resize(bgr, (iw, ih), interpolation=cv2.INTER_LINEAR),


            cv2.COLOR_BGR2RGB,


        )


        i = self._idx_frame


        hd, wd = ih, iw


        dpose = self.depth_pose[i, :hd, :wd].copy()


        fz = float(self.focal_px[i])


        self.last_focal_px = fz if np.isfinite(fz) and fz > 1e-3 else None


        iw_d, ih_d = self.depth_infer_wh


        rgb_depth_infer = cv2.resize(


            rgb_pose,


            (iw_d, ih_d),


            interpolation=cv2.INTER_AREA,


        )


        self.bundle_frame_index = i


        self._idx_frame += 1


        return (


            time.monotonic(),


            rgb_pose.astype(np.uint8),


            rgb_depth_infer.astype(np.uint8),


            True,


            dpose,


        )




    def pose_observation_for_index(self, i: int) -> PoseObservation:


        kp = (
            None


            if self.pose_kp_xy is None


            else self.pose_kp_xy[i].copy().astype(np.float64)


        )


        rhs: Optional[List[Tuple[float, float]]] = None


        if (
            self.hand_lm[i]


            and self._right_hand_xy is not None


        ):


            hh = self._right_hand_xy[i]


            rhs = [(float(hh[j, 0]), float(hh[j, 1])) for j in range(hh.shape[0])]


        hh_raw = self.handedness[i] if i < len(self.handedness) else "Unknown"


        return pose_observation_from_bundle_row(


            wrist_uv=self.wrist_uv[i],


            elbow_uv=self.elbow_uv[i],


            shoulder_uv=self.shoulder_uv[i],


            visibility=self.visibility[i],


            landmarks_ok=bool(self.landmarks_ok[i]),


            wrist_w=self.wrist_world[i],


            elbow_w=self.elbow_world[i],


            shoulder_w=self.shoulder_world[i],


            pose_kp_xy=kp,


            pinch_open=float(self.pinch_open[i]),


            pinch_valid=bool(self.pinch_valid[i]),


            fist=bool(self.fist[i]),


            hand_lm=bool(self.hand_lm[i]),


            handedness_raw=hh_raw,


            right_hand_xy=rhs,


            arm_landmark_ids=self._arm_ids[i],


        )






__all__ = ["ReplayDepthPose", "pose_observation_from_bundle_row"]


