"""
Lightweight OpenCV display worker — intentionally imports ONLY cv2 and numpy.

This module is the target of the multiprocessing `spawn` call in main.py.
Keeping it self-contained means the spawned child process does NOT re-import
mediapipe, torch, or any other heavy library.
"""

from __future__ import annotations

from typing import Optional


def worker(q: "object") -> None:  # q is a multiprocessing.Queue
    """
    Receive JPEG-encoded ``bytes`` from *q* and call ``cv2.imshow``.

    Runs in its own process so it gets a fresh macOS Cocoa event loop,
    independent of mjpython's viewer.  Send ``None`` to exit cleanly.
    """
    import sys
    import numpy as np

    try:
        import cv2
    except ImportError:
        print("[cv_display] opencv-python not found; install it.", file=sys.stderr)
        return

    cv2.namedWindow("teleop_pose_depth", cv2.WINDOW_NORMAL)

    while True:
        try:
            data = q.get(timeout=0.5)
        except Exception:
            cv2.waitKey(1)
            continue

        if data is None:
            break

        buf = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            continue

        cv2.imshow("teleop_pose_depth", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
