"""
5-DOF arm IK: anatomical angle extraction → joint angles (degrees).

Joint layout
------------
  J1  base_yaw   – rotation around Z (horizontal swivel)
  J2  shoulder_pitch – rotation around Y (arm elevation)
  J3  elbow_pitch    – rotation around Y (forearm bend)
  J4  wrist_pitch    – rotation around Y (wrist up/down)
  J5  gripper_roll   – rotation around X (forearm axial spin)

Angle extraction
----------------
All angles are derived from the *vectors* between body landmarks after the
3-point pose (shoulder, elbow, wrist) has been transformed to the robot-base
frame.  This is deliberately independent of absolute 3-D position so the same
motion maps identically regardless of how far the operator stands from the
camera.

Z-clamping
----------
After the angles are derived we run a forward-kinematics check to verify the
predicted wrist height.  If it would fall below the table surface we lift J2
(and optionally J3) until the constraint is satisfied, keeping the elbow
posture as natural as possible.

J4 – wrist pitch
-----------------
Three modes (set via YAML ``ik.j4_mode``):
  "human"   track the forearm elevation relative to J2+J3 (default)
  "down"    always point the end-effector straight down
  "neutral" leave J4 = 0

J5 – gripper roll
-----------------
Estimated from the 2-D hand-landmark plane (right_hand_xy) when available.
Falls back to the lateral component of the forearm vector.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------

def _deg(x_rad: float) -> float:
    return x_rad * 180.0 / math.pi


def _rad(x_deg: float) -> float:
    return float(x_deg) * math.pi / 180.0


def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _rz(angle_rad: float) -> np.ndarray:
    """3x3 rotation matrix around Z."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / max(n, 1e-9)


# ---------------------------------------------------------------------------
# Planar 2R IK (kept for backward-compatibility / fallback)
# ---------------------------------------------------------------------------

def planar_two_link(
    rx: float,
    rz: float,
    la: float,
    lb: float,
    elbow_above: bool,
) -> Tuple[float, float]:
    """2-link IK in XZ plane: rx = horizontal reach, rz = vertical (+Z up)."""
    rr = math.hypot(rx, rz)
    rr = clamp(rr, abs(la - lb) + 1e-5, la + lb - 1e-5)
    cos_th2 = (rr**2 - la**2 - lb**2) / (2.0 * la * lb)
    cos_th2 = clamp(cos_th2, -1.0 + 1e-6, 1.0 - 1e-6)
    th_elbow = math.acos(cos_th2)
    if not elbow_above:
        th_elbow = -th_elbow
    phi_tgt = math.atan2(rz, rx)
    psi_off = math.atan2(lb * math.sin(th_elbow), la + lb * math.cos(th_elbow))
    th_shoulder = phi_tgt - psi_off
    return th_shoulder, th_elbow


# ---------------------------------------------------------------------------
# Z-clamping function (standalone, as requested)
# ---------------------------------------------------------------------------

def clamp_wrist_z(
    wrist_robot_z: float,
    *,
    table_height: float = 0.0,
    margin: float = 0.02,
    max_reach: float = 0.45,
) -> float:
    """
    Clamp the robot end-effector Z coordinate to a safe workspace band.

    Parameters
    ----------
    wrist_robot_z : float
        Current commanded wrist height in the robot-base frame (+Z up, metres).
    table_height : float
        Height of the table surface in the robot-base frame (usually 0.0 because
        the robot is mounted *on* the table, so the surface is at the base).
    margin : float
        Safety clearance above the table surface (default 0.02 m).
    max_reach : float
        Maximum height above the table the wrist may reach (default 0.45 m).

    Returns
    -------
    float
        Clamped wrist height in metres.
    """
    z_min = table_height + margin
    z_max = table_height + max_reach
    return clamp(wrist_robot_z, z_min, z_max)


# ---------------------------------------------------------------------------
# Forearm roll from 2-D hand landmarks
# ---------------------------------------------------------------------------

def hand_roll_from_landmarks(
    right_hand_xy: List[Tuple[float, float]],
    wrist_uv: Tuple[float, float],
    elbow_uv: Tuple[float, float],
) -> float:
    """
    Estimate forearm axial roll from the 2-D hand skeleton.

    The vector from the wrist (landmark 0) to the middle-finger MCP
    (landmark 9) in the image plane defines the "back-of-hand up" direction.
    Rolling the forearm rotates this vector around the forearm axis.  We
    measure the angle between that vector and the direction perpendicular to
    the forearm projection in the image.

    Returns roll in radians, positive = hand tilts toward thumb side.
    """
    if len(right_hand_xy) < 10:
        return 0.0

    # Forearm direction in image (elbow → wrist)
    fa_img = np.array(
        [wrist_uv[0] - elbow_uv[0], wrist_uv[1] - elbow_uv[1]], dtype=np.float64
    )
    if np.linalg.norm(fa_img) < 1e-6:
        return 0.0

    # Perpendicular to forearm (90° CCW)
    fa_perp = np.array([-fa_img[1], fa_img[0]])
    fa_perp = fa_perp / np.linalg.norm(fa_perp)

    # Hand "up" direction (wrist → middle-finger MCP)
    mcp = right_hand_xy[9]
    hand_up = np.array(
        [mcp[0] - wrist_uv[0], mcp[1] - wrist_uv[1]], dtype=np.float64
    )
    if np.linalg.norm(hand_up) < 1e-6:
        return 0.0
    hand_up = hand_up / np.linalg.norm(hand_up)

    # Signed angle between fa_perp and hand_up
    cos_r = float(np.clip(np.dot(fa_perp, hand_up), -1.0, 1.0))
    sin_r = float(np.cross(fa_perp, hand_up))
    return math.atan2(sin_r, cos_r)


# ---------------------------------------------------------------------------
# Core 5-DOF IK function (standalone)
# ---------------------------------------------------------------------------

def ik_5dof(
    target_pos: np.ndarray,
    link_lengths: Tuple[float, float, float],
    joint_limits: Tuple[
        Tuple[float, float],
        Tuple[float, float],
        Tuple[float, float],
        Tuple[float, float],
        Tuple[float, float],
    ],
    *,
    shoulder_pos: Optional[np.ndarray] = None,
    elbow_pos: Optional[np.ndarray] = None,
    j5_rad: float = 0.0,
    j4_mode: str = "human",
    elbow_above: bool = False,
    table_height: float = 0.0,
    z_margin: float = 0.02,
    max_reach: float = 0.45,
    robot_shoulder_z: Optional[float] = None,
    j1_scale: float = 1.0,
) -> np.ndarray:
    """
    Analytical 5-DOF IK returning joint angles in *degrees*.

    If ``shoulder_pos`` and ``elbow_pos`` are provided the function extracts
    J2 and J3 directly from the human-arm segment angles (anatomical mode).
    Otherwise it falls back to pure positional planar IK to reach
    ``target_pos``.

    Parameters
    ----------
    target_pos       : (3,) wrist position in the robot-base frame.
    link_lengths     : (L2, L3, L4) – upper-arm, forearm, wrist lengths (m).
    joint_limits     : five (lo_deg, hi_deg) pairs for J1-J5.
    shoulder_pos     : (3,) shoulder position in robot-base frame (angle mode).
    elbow_pos        : (3,) elbow position in robot-base frame (angle mode).
    j5_rad           : gripper roll (radians, pre-computed).
    j4_mode          : "human" | "down" | "neutral" (see module docstring).
    elbow_above      : elbow-above hint for positional fallback.
    table_height     : table surface height in robot frame (default 0.0).
    z_margin         : safety clearance above table (default 0.02 m).
    max_reach        : maximum wrist Z above table (default 0.45 m).
    robot_shoulder_z : height of the robot's J2 pivot above the table (= L1).
                       Used for FK Z-clamping.  If None, falls back to
                       shoulder_pos[2] which is WRONG for world landmarks.
    j1_scale         : scale factor applied to the raw J1 angle (default 1.0).
                       Reduce below 1 to tame over-rotation when arm reach is short.

    Returns
    -------
    np.ndarray, shape (5,), dtype float64 – joint angles in **degrees**.
    """
    L2, L3, L4 = float(link_lengths[0]), float(link_lengths[1]), float(link_lengths[2])
    L345 = L3 + L4

    w = np.asarray(target_pos, dtype=np.float64).flatten()[:3].copy()

    # ---- J1: base yaw from horizontal azimuth of the arm ------------------
    if shoulder_pos is not None:
        s = np.asarray(shoulder_pos, dtype=np.float64).flatten()[:3]
        arm_h = w - s
    else:
        # Positional fallback: clamp wrist z before computing J1 and 2R IK.
        # In angle-based mode (shoulder_pos provided) we do NOT clamp here
        # because w[2] is the camera-mapped position and can be >max_reach when
        # the arm is raised.  The FK check below enforces the Z floor instead.
        w[2] = clamp_wrist_z(float(w[2]), table_height=table_height, margin=z_margin, max_reach=max_reach)
        s = np.zeros(3)
        arm_h = w.copy()

    # Damp J1 when the arm has little forward reach to avoid over-rotation
    # on small lateral movements. weight → 1 at full extension, → 0.2 when close.
    horiz_reach = math.hypot(float(arm_h[0]), float(arm_h[1]))
    j1_raw = math.atan2(float(arm_h[1]), float(arm_h[0]))
    # Apply user-configured scale (e.g. 0.7 halves the rotational sensitivity)
    j1 = j1_raw * float(j1_scale)

    # ---- Rotate into sagittal plane (remove J1) ---------------------------
    Rz_inv = _rz(-j1)

    if shoulder_pos is not None and elbow_pos is not None:
        e = np.asarray(elbow_pos, dtype=np.float64).flatten()[:3].copy()
        # Only clamp elbow below table when it would actually be below table;
        # leave it alone for raised-arm positions (clamping elbow down corrupts J3)

        ua = e - s          # upper-arm vector in robot frame
        fa = w - e          # forearm vector in robot frame

        ua_sag = Rz_inv @ ua  # upper arm projected to sagittal plane
        fa_sag = Rz_inv @ fa  # forearm projected to sagittal plane

        # J2: elevation angle of upper arm (positive = arm raised above horizontal)
        ua_horiz = math.hypot(float(ua_sag[0]), float(ua_sag[1]))
        j2 = math.atan2(float(ua_sag[2]), max(ua_horiz, 1e-6))

        # J3: elbow bend = forearm elevation minus upper-arm elevation
        fa_horiz = math.hypot(float(fa_sag[0]), float(fa_sag[1]))
        fa_elevation = math.atan2(float(fa_sag[2]), max(fa_horiz, 1e-6))
        # Normalize to [-π, π] to avoid wrap-around jumps when arm is near vertical
        j3 = ((fa_elevation - j2) + math.pi) % (2.0 * math.pi) - math.pi

    else:
        # Positional fallback: planar 2R from reach + height
        w_sag = Rz_inv @ (w - s)
        rx = math.hypot(float(w_sag[0]), float(w_sag[1]))
        rz = float(w_sag[2])
        j2, j3 = planar_two_link(rx, rz, L2, L345, elbow_above)
        fa_elevation = j2 + j3  # used for J4 below

    # ---- FK Z-clamping: lift J2 if wrist would be below table ---------------
    # IMPORTANT: use the ROBOT's actual shoulder mount height (L1), not the
    # person's shoulder position in robot frame.  The person's shoulder maps
    # to ~0.60 m in robot space due to the camera-to-base offset, but the
    # robot's J2 pivot is physically at L1 ≈ 0.12 m above the table.
    # Using s[2] (person shoulder) here would make the clamp never trigger.
    fk_shoulder_z = float(robot_shoulder_z) if robot_shoulder_z is not None else float(s[2])
    predicted_wrist_z = fk_shoulder_z + L2 * math.sin(j2) + L345 * math.sin(j2 + j3)
    z_min = table_height + z_margin
    if predicted_wrist_z < z_min:
        z_deficit = z_min - predicted_wrist_z
        # Gradient: d(wrist_z)/d(j2) = L2*cos(j2) + L345*cos(j2+j3)
        dz_dj2 = L2 * math.cos(j2) + L345 * math.cos(j2 + j3)
        if abs(dz_dj2) > 0.01:
            j2 = j2 + z_deficit / dz_dj2
        else:
            # Near singularity: raise j2 to its upper limit
            j2 = _rad(float(joint_limits[1][1]))

    # Recompute fa_elevation after possible j2 adjustment
    fa_elevation = j2 + j3

    # ---- J4: wrist pitch ---------------------------------------------------
    j4_mode = str(j4_mode).lower().strip()
    if j4_mode == "down":
        # Point end-effector straight down: total elevation = -π/2
        j4 = -math.pi / 2.0 - fa_elevation
    elif j4_mode == "neutral":
        j4 = 0.0
    else:
        # "human" mode: wrist pitch = deviation of forearm from expected direction
        # Here we let the forearm angle drive J4; J4 damps the tendency to droop
        j4 = clamp(0.25 * (-j3), math.radians(-120.0), math.radians(120.0))

    # ---- J5: gripper roll (passed in) -------------------------------------
    j5 = j5_rad  # pre-computed by the caller (hand_roll_from_landmarks or fallback)

    # ---- Assemble and clamp to joint limits --------------------------------
    q = np.array(
        [_deg(j1), _deg(j2), _deg(j3), _deg(j4), _deg(j5)],
        dtype=np.float64,
    )
    for i, (lo, hi) in enumerate(joint_limits):
        q[i] = clamp(float(q[i]), float(lo), float(hi))

    return q


# ---------------------------------------------------------------------------
# Mapper base
# ---------------------------------------------------------------------------

class MapperBase:
    """Pluggable mapper interface."""

    def map_points(
        self,
        wrist_xyz: np.ndarray,
        elbow_xyz: np.ndarray,
        shoulder_xyz: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError

    def map_observation(
        self,
        wrist_xyz: np.ndarray,
        elbow_xyz: np.ndarray,
        shoulder_xyz: np.ndarray,
        *,
        right_hand_xy: Optional[List[Tuple[float, float]]] = None,
        wrist_uv: Optional[Tuple[float, float]] = None,
        elbow_uv: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        """Extended entry point that accepts hand-orientation data for J5."""
        return self.map_points(wrist_xyz, elbow_xyz, shoulder_xyz)

    def cam_to_robot_pt(self, p_cam: Sequence[float]) -> np.ndarray:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Kinematic params dataclass
# ---------------------------------------------------------------------------

@dataclass
class RobotKinematicParams:
    L1: float           # base-plate → shoulder pivot height (vertical riser)
    L2: float           # shoulder → elbow (upper arm)
    L3: float           # elbow → wrist pitch (forearm)
    L4: float           # wrist pitch → tool (wrist extension)
    joint_limits_deg: Tuple[Tuple[float, float], ...]
    Tb: np.ndarray
    elbow_above_default: bool
    min_z_m: float = 0.03
    z_margin: float = 0.02
    max_reach_m: float = 0.50
    j4_mode: str = "human"
    smoothing_alpha: float = 0.35
    # Maximum joint-angle change allowed per call (degrees). Prevents instantaneous
    # jumps caused by landmark occlusion recovery or atan2 wrap-arounds.
    max_joint_vel_deg: float = 30.0
    # Scale J1 (base yaw) to reduce over-rotation for small forward reach.
    j1_scale: float = 0.7

    @property
    def L345(self) -> float:
        return self.L3 + self.L4


# ---------------------------------------------------------------------------
# Main mapper
# ---------------------------------------------------------------------------

class AnalyticalIKMapper(MapperBase):
    """
    Anatomical angle-based 5-DOF mapper.

    Extracts joint angles directly from the vectors between shoulder, elbow,
    and wrist rather than solving IK to an absolute target position.  This
    makes the mapping shoulder-relative and naturally handles the case where
    the human hand is below shoulder height.
    """

    def __init__(self, config_path: Path | str) -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        rob = cfg["robot"]
        j_keys = (
            "j1_base_z",
            "j2_shoulder_y",
            "j3_elbow_y",
            "j4_wrist_y",
            "j5_gripper_x",
        )
        # Fall back to the old key name for J5 if the new one isn't present
        j5_key = "j5_gripper_x" if "j5_gripper_x" in rob["joint_limits_deg"] else "j5_gripper_z"
        actual_keys = j_keys[:4] + (j5_key,)
        jlim = tuple(
            tuple(float(x) for x in rob["joint_limits_deg"][k]) for k in actual_keys
        )

        lengths = rob["lengths"]
        L1 = float(lengths.get("L1", 0.12))
        L2 = float(lengths["L2"])
        L3 = float(lengths["L3"])
        L4 = float(lengths["L4"])

        ik_cfg = cfg.get("ik", {})
        ep_raw = str(ik_cfg.get("elbow_preference", "negative")).lower()
        elbow_above = ep_raw.startswith("pos") or ep_raw in ("+", "above")
        j4_mode = str(ik_cfg.get("j4_mode", "human")).lower()
        smoothing = float(ik_cfg.get("smoothing_alpha", 0.35))

        ws = cfg.get("workspace", {})
        min_z = float(ws.get("min_wrist_z_m", 0.03))
        z_margin = float(ws.get("z_margin_m", 0.02))
        max_reach = float(ws.get("max_reach_m", L2 + L3 + L4 - 0.01))

        max_vel = float(ik_cfg.get("max_joint_vel_deg", 30.0))
        j1_scale = float(ik_cfg.get("j1_scale", 0.7))

        self.params = RobotKinematicParams(
            L1=L1,
            L2=L2,
            L3=L3,
            L4=L4,
            joint_limits_deg=jlim,  # type: ignore[arg-type]
            Tb=np.asarray(cfg["camera_to_base_matrix"], dtype=np.float64),
            elbow_above_default=elbow_above,
            min_z_m=min_z,
            z_margin=z_margin,
            max_reach_m=max_reach,
            j4_mode=j4_mode,
            smoothing_alpha=smoothing,
            max_joint_vel_deg=max_vel,
            j1_scale=j1_scale,
        )

        self._q_prev: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def cam_to_robot_pt(self, p_cam: Sequence[float]) -> np.ndarray:
        v = np.array([float(p_cam[0]), float(p_cam[1]), float(p_cam[2]), 1.0])
        pw = self.params.Tb @ v
        return pw[:3] / max(float(pw[3]), 1e-12)

    # ------------------------------------------------------------------
    def map_observation(
        self,
        wrist_xyz: np.ndarray,
        elbow_xyz: np.ndarray,
        shoulder_xyz: np.ndarray,
        *,
        right_hand_xy: Optional[List[Tuple[float, float]]] = None,
        wrist_uv: Optional[Tuple[float, float]] = None,
        elbow_uv: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        """Full mapping with optional hand-orientation data for J5 (roll)."""
        p = self.params

        # Transform all points to robot-base frame
        w = self.cam_to_robot_pt(np.asarray(wrist_xyz).flatten()[:3])
        e = self.cam_to_robot_pt(np.asarray(elbow_xyz).flatten()[:3])
        s = self.cam_to_robot_pt(np.asarray(shoulder_xyz).flatten()[:3])

        # ---- J5: forearm roll ----------------------------------------------
        if (
            right_hand_xy is not None
            and wrist_uv is not None
            and elbow_uv is not None
            and len(right_hand_xy) >= 10
        ):
            j5_rad = hand_roll_from_landmarks(right_hand_xy, wrist_uv, elbow_uv)
        else:
            # No hand data: hold J5 at 0 (gripper roll = neutral).
            # Using the forearm's lateral vector as a proxy is too noisy and
            # produces erratic roll commands when the elbow is tracked loosely.
            j5_rad = 0.0

        # ---- Delegate to core IK ------------------------------------------
        qs_deg = ik_5dof(
            target_pos=w,
            link_lengths=(p.L2, p.L3, p.L4),
            joint_limits=p.joint_limits_deg,  # type: ignore[arg-type]
            shoulder_pos=s,
            elbow_pos=e,
            j5_rad=j5_rad,
            j4_mode=p.j4_mode,
            elbow_above=p.elbow_above_default,
            table_height=0.0,
            z_margin=p.z_margin,
            max_reach=p.max_reach_m,
            # Use L1 (physical robot shoulder height) for FK check, not the
            # person's shoulder position which is at ~0.60 m due to the
            # camera-to-base transform offset and would prevent the clamp
            # from ever triggering.
            robot_shoulder_z=p.L1,
            j1_scale=p.j1_scale,
        )

        # ---- Per-joint velocity limiting -----------------------------------
        # Applied BEFORE smoothing so a single noisy spike never propagates.
        max_vel = float(p.max_joint_vel_deg)
        if max_vel > 0.0 and self._q_prev is not None and self._q_prev.shape == qs_deg.shape:
            delta = qs_deg - self._q_prev
            delta = np.clip(delta, -max_vel, max_vel)
            qs_deg = self._q_prev + delta

        # ---- Temporal smoothing --------------------------------------------
        alpha = clamp(p.smoothing_alpha, 0.0, 1.0)
        if alpha < 1.0 and self._q_prev is not None and self._q_prev.shape == qs_deg.shape:
            qs_deg = alpha * qs_deg + (1.0 - alpha) * self._q_prev
        self._q_prev = qs_deg.copy()

        return qs_deg

    def map_points(
        self,
        wrist_xyz: np.ndarray,
        elbow_xyz: np.ndarray,
        shoulder_xyz: np.ndarray,
    ) -> np.ndarray:
        """Backward-compatible wrapper (no hand orientation data)."""
        return self.map_observation(wrist_xyz, elbow_xyz, shoulder_xyz)


# ---------------------------------------------------------------------------
# RBF stub
# ---------------------------------------------------------------------------

class RBFMapperStub(MapperBase):
    """Placeholder for calibrated RBF teleop — forwards to analytic."""

    def __init__(self, analytic: MapperBase):
        self._analytic = analytic

    def map_points(
        self,
        wrist_xyz: np.ndarray,
        elbow_xyz: np.ndarray,
        shoulder_xyz: np.ndarray,
    ) -> np.ndarray:
        return self._analytic.map_points(wrist_xyz, elbow_xyz, shoulder_xyz)

    def map_observation(
        self,
        wrist_xyz: np.ndarray,
        elbow_xyz: np.ndarray,
        shoulder_xyz: np.ndarray,
        **kwargs: object,
    ) -> np.ndarray:
        return self._analytic.map_observation(wrist_xyz, elbow_xyz, shoulder_xyz, **kwargs)  # type: ignore[arg-type]

    def cam_to_robot_pt(self, p_cam: Sequence[float]) -> np.ndarray:
        return self._analytic.cam_to_robot_pt(p_cam)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def load_mapper(path: Path | str, use_rbf: bool = False) -> MapperBase:
    base = AnalyticalIKMapper(path)
    if use_rbf:
        return RBFMapperStub(base)
    return base


__all__ = [
    "AnalyticalIKMapper",
    "MapperBase",
    "RBFMapperStub",
    "clamp",
    "clamp_wrist_z",
    "hand_roll_from_landmarks",
    "ik_5dof",
    "load_mapper",
    "planar_two_link",
]
