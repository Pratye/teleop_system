"""MuJoCo teleoperation scene: table + five-DoF arm + visual wrist goal."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

try:
    import mujoco

    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False


JOINT_NAME_ORDER_DEG_KEYS = (
    "jnt1_base_z",
    "jnt2_shoulder_y",
    "jnt3_elbow_y",
    "jnt4_wrist_y",
    "jnt5_gripper_z",
)

# This must match the robot_root body pos in scene_teleop.xml (0.52, 0, 0.810).
# The table top is at world z = 0.810; the arm is mounted on the table surface.
ROOT_POS_WORLD = np.array([0.52, 0.0, 0.810], dtype=np.float64)


class SimTeleopEnv:
    """MuJoCo helper: five joint angles (deg) plus optional finger colour from pinch."""

    def __init__(self, mjcf_path: Optional[Path | str] = None) -> None:
        if not HAS_MUJOCO:
            raise ImportError("`mujoco` required (`pip install mujoco`).")

        path = (
            Path(mjcf_path)
            if mjcf_path
            else Path(__file__).resolve().parents[1] / "models" / "scene_teleop.xml"
        )
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)

        self._qadr = [int(self.model.joint(n).qposadr[0]) for n in JOINT_NAME_ORDER_DEG_KEYS]

        body_id_goal = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "wrist_goal")
        self._m_cid = int(self.model.body_mocapid[body_id_goal])
        if self._m_cid < 0:
            raise ValueError("MJCF missing mocap `wrist_goal` body")

        self._finger_geom_ids: List[int] = []
        self._finger_rgba0: Dict[int, np.ndarray] = {}
        self._finger_pos0: Dict[int, np.ndarray] = {}  # original local-frame positions
        for gn in ("finger_l", "finger_r"):
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, gn)
            if gid >= 0:
                self._finger_geom_ids.append(int(gid))
                self._finger_rgba0[int(gid)] = np.array(self.model.geom_rgba[gid], copy=True)
                self._finger_pos0[int(gid)] = np.array(self.model.geom_pos[gid], copy=True)

        # End-effector site for FK-based goal ball positioning
        self._eef_site_id = int(mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "eef_tip"
        ))

        mujoco.mj_forward(self.model, self.data)

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def set_goal_xyz_metres_world(self, p_xyz: Sequence[float]) -> None:
        """Absolute world XYZ (metres) for the translucent wrist goal sphere."""
        self.data.mocap_pos[self._m_cid][:] = np.asarray(p_xyz, dtype=np.float64).reshape(3)
        mujoco.mj_forward(self.model, self.data)

    def set_goal_from_robot_base_xyz(self, p_robot: Sequence[float]) -> None:
        """Offsets robot-base coordinates by approximate `robot_root` world pose."""
        self.set_goal_xyz_metres_world(ROOT_POS_WORLD + np.asarray(p_robot, dtype=np.float64).reshape(3))

    def set_joint_degrees_deg(self, q_deg: Sequence[float]) -> None:
        """Applies joint commanded angles [deg], order `JOINT_NAME_ORDER_DEG_KEYS`."""
        if len(q_deg) != len(self._qadr):
            raise ValueError(f"need {len(self._qadr)} joint values")
        qr = np.deg2rad(np.asarray(q_deg, dtype=np.float64))
        for adr, qi in zip(self._qadr, qr):
            self.data.qpos[adr] = float(qi)
        mujoco.mj_forward(self.model, self.data)
        # Auto-sync goal ball to FK end-effector position so the ball always
        # sits exactly at the arm tip with zero visual mismatch.
        self._sync_goal_to_eef()

    def _sync_goal_to_eef(self) -> None:
        """Move the goal ball to the FK end-effector site world position."""
        if self._eef_site_id >= 0:
            self.set_goal_xyz_metres_world(self.data.site_xpos[self._eef_site_id])

    def set_gripper_aperture_visual(self, grip_open01: float) -> None:
        """
        Drive the visual gripper fingers.

        `grip_open01` in [0, 1]:
          0 = closed (dark, fingers together)
          1 = open   (green, fingers spread)

        Moves finger geoms in their local frame so the gripper visibly opens/closes,
        and blends colour from dark-grey (closed) → green (open).
        """
        if not self._finger_geom_ids:
            return

        t = float(np.clip(grip_open01, 0.0, 1.0))

        # --- colour blend: dark grey → green ---
        closed_rgb = np.array([0.08, 0.08, 0.08], dtype=np.float64)
        open_rgb = np.array([0.20, 0.85, 0.35], dtype=np.float64)
        colour_target = closed_rgb * (1.0 - t) + open_rgb * t

        # --- Y-axis spread (local frame): closed ≈ 0.012 m, open ≈ 0.048 m ---
        Y_CLOSED, Y_OPEN = 0.012, 0.048
        y_sep = Y_CLOSED + (Y_OPEN - Y_CLOSED) * t

        for gid in self._finger_geom_ids:
            # colour
            base = self._finger_rgba0[gid][:3]
            rgba = np.zeros(4, dtype=np.float64)
            rgba[:3] = 0.45 * base + 0.55 * colour_target
            rgba[3] = 1.0
            self.model.geom_rgba[gid] = rgba

            # position — preserve x/z from rest pose, vary only |y|
            p0 = self._finger_pos0[gid]
            new_pos = p0.copy()
            new_pos[1] = float(np.sign(p0[1])) * y_sep  # keep original sign (L=+, R=-)
            self.model.geom_pos[gid] = new_pos

        mujoco.mj_forward(self.model, self.data)


__all__ = ["SimTeleopEnv", "JOINT_NAME_ORDER_DEG_KEYS", "ROOT_POS_WORLD"]
