"""
Serial sender for handpose-driven Arduino sketch.

Input from teleop pipeline remains:
  q_deg: [base, shoulder, elbow, wrist_y, wrist_x] in degrees
  grip_open01: scalar in [0,1]

Output line sent to Arduino (115200 baud):
  H hx hy hz roll pinch base
where all motion fields except pinch are normalized to [-1,1].
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import serial


class ArduinoTeleopPort:
    # Sender-side gains to avoid saturation and over-aggressive commands.
    G_HX = 1.0
    G_HY = 1.0
    G_HZ = 1.0
    G_ROLL = 1.0
    G_BASE = 1.0

    """Thin pyserial wrapper."""

    def __init__(self, port: str | Path, baud: int = 115200, timeout: float = 0.02) -> None:
        self._ser = serial.Serial(str(port), baud, timeout=timeout)

    def close(self) -> None:
        if self._ser.is_open:
            self._ser.close()

    @staticmethod
    def _norm_from_deg(v_deg: float, neutral: float = 0.0, span: float = 120.0) -> float:
        x = (float(v_deg) - neutral) / span
        if x < -1.0:
            return -1.0
        if x > 1.0:
            return 1.0
        return x

    @staticmethod
    def format_command_line(q_deg: Sequence[float], grip_open01: float) -> bytes:
        if len(q_deg) != 5:
            raise ValueError("expected 5 joint angles in degrees")
        base_deg, sh_deg, el_deg, wy_deg, wx_deg = [float(x) for x in q_deg]

        # Inverse of Arduino mapping in arduino_robot_arm_handpose.ino:
        # sh = -hy, el = hz, wy = hx, wx = roll, base = base
        hx = ArduinoTeleopPort._norm_from_deg(wy_deg) * ArduinoTeleopPort.G_HX
        hy = -ArduinoTeleopPort._norm_from_deg(sh_deg) * ArduinoTeleopPort.G_HY
        hz = ArduinoTeleopPort._norm_from_deg(el_deg) * ArduinoTeleopPort.G_HZ
        roll = ArduinoTeleopPort._norm_from_deg(wx_deg) * ArduinoTeleopPort.G_ROLL
        base = ArduinoTeleopPort._norm_from_deg(base_deg) * ArduinoTeleopPort.G_BASE

        grip = float(grip_open01)
        if grip < 0.0:
            grip = 0.0
        elif grip > 1.0:
            grip = 1.0

        return f"H {hx:.4f} {hy:.4f} {hz:.4f} {roll:.4f} {grip:.4f} {base:.4f}\n".encode("ascii")

    def write_joints_and_grip(self, q_deg: Sequence[float], grip_open01: float) -> None:
        payload = self.format_command_line(q_deg, grip_open01)
        # Debug visibility: print exact line sent to Arduino.
        print(f"[arduino tx] {payload.decode('ascii').strip()}", flush=True)
        self._ser.write(payload)
        self._ser.flush()


__all__ = ["ArduinoTeleopPort"]
