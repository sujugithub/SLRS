"""
CameraWorker — QThread that captures frames and runs MediaPipe detection
completely off the main (Qt) thread.

Emits frame_ready(annotated_frame, hand_lms_list, pose_lms, face_lms,
holistic_feats) on every processed frame.  The main thread only receives
the finished result and updates the UI — it never blocks on camera I/O or
MediaPipe inference.
"""
from __future__ import annotations

import numpy as np
import cv2
from PyQt6.QtCore import QThread, pyqtSignal

# Holistic drawing constants — loaded lazily so this module can be imported
# before holistic_detector initialises MediaPipe.
_HAND_CONN: list | None = None
_FACE_CONN: list | None = None
_POSE_CONN: list | None = None
_POSE_VIS: float = 0.5
_KEY_FACE: list = []


def _load_draw_consts() -> None:
    global _HAND_CONN, _FACE_CONN, _POSE_CONN, _POSE_VIS, _KEY_FACE
    if _HAND_CONN is not None:
        return
    try:
        from core.holistic_detector import (          # noqa: PLC0415
            HAND_CONNECTIONS, FACE_CONNECTIONS, POSE_CONNECTIONS,
            POSE_VIS_THRESHOLD, KEY_FACE_INDICES,
        )
        _HAND_CONN = HAND_CONNECTIONS
        _FACE_CONN = FACE_CONNECTIONS
        _POSE_CONN = POSE_CONNECTIONS
        _POSE_VIS  = POSE_VIS_THRESHOLD
        _KEY_FACE  = KEY_FACE_INDICES
    except Exception as exc:
        print(f"[CameraWorker] Could not load holistic draw constants: {exc}")
        _HAND_CONN = []
        _FACE_CONN = []
        _POSE_CONN = []


class CameraWorker(QThread):
    """
    Background thread responsible for:
    1. Reading frames from OpenCV camera
    2. Running MediaPipe (holistic or hand+pose) detection
    3. Drawing landmark overlays on the frame
    4. Emitting the annotated frame + raw detection data to the main thread

    The main thread connects to ``frame_ready`` and performs only lightweight
    UI updates (QLabel.setPixmap, progress bars, text labels).
    """

    # annotated BGR frame (np.ndarray), hand_lms_list (list), pose_lms,
    # face_lms, holistic_feats (np.ndarray shape (171,))
    frame_ready = pyqtSignal(object, object, object, object, object)

    def __init__(self, camera, detector, holistic=None,
                 pose_detector=None, parent=None):
        super().__init__(parent)
        self._camera        = camera
        self._detector      = detector
        self._holistic      = holistic
        self._pose_detector = pose_detector
        self._active        = False

    # ── public API ───────────────────────────────────────────────────────────

    def start_capture(self) -> None:
        """Start the background capture loop."""
        self._active = True
        self.start()
        print("[CameraWorker] Started")

    def stop_capture(self) -> None:
        """Signal the loop to exit and block until the thread finishes."""
        print("[CameraWorker] Stopping…")
        self._active = False
        # Give the thread up to 3 s to finish its current detection pass.
        if not self.wait(3000):
            print("[CameraWorker] WARNING: thread did not stop cleanly")

    # ── QThread entry point ──────────────────────────────────────────────────

    def run(self) -> None:
        _load_draw_consts()
        feat_len = 171  # HOLISTIC_FEATURE_LENGTH from config

        while self._active:
            frame = self._camera.get_frame()
            if frame is None:
                self.msleep(10)
                continue

            annotated      = frame.copy()
            hand_lms_list  = []
            pose_lms       = None
            face_lms       = None
            holistic_feats = np.zeros(feat_len, dtype=np.float32)

            try:
                if self._holistic is not None:
                    hand_lms_list, pose_lms, face_lms = \
                        self._holistic.detect_all(annotated)
                    self._draw_landmarks(
                        annotated, hand_lms_list, pose_lms, face_lms,
                        handedness=getattr(self._holistic, "_last_handedness",
                                          None))
                    holistic_feats = self._holistic.extract_holistic_features(
                        hand_lms_list, pose_lms, face_lms)
                else:
                    if self._pose_detector is not None:
                        p = self._pose_detector.detect(annotated)
                        if p:
                            self._pose_detector.draw_face_and_shoulders(
                                annotated, p)
                            pose_lms = p
                    raw = self._detector.detect(annotated)
                    if raw:
                        for lms in raw:
                            self._detector.draw_landmarks(annotated, lms)
                        hand_lms_list = raw

            except Exception as exc:
                print(f"[CameraWorker] Detection error: {exc}")

            # Emit a copy so the main thread owns its buffer.
            self.frame_ready.emit(
                annotated.copy(), hand_lms_list, pose_lms,
                face_lms, holistic_feats,
            )

        print("[CameraWorker] Stopped")

    # ── landmark drawing (runs in worker thread — OpenCV is thread-safe) ─────

    def _draw_landmarks(self, frame: np.ndarray, hand_lms_list: list,
                         pose_lms, face_lms, handedness=None) -> None:
        h, w = frame.shape[:2]

        # Face mesh (teal)
        if face_lms and _FACE_CONN:
            _TEAL = (0, 210, 190)
            for i, j in _FACE_CONN:
                if i < len(face_lms) and j < len(face_lms):
                    pi = (int(face_lms[i].x * w), int(face_lms[i].y * h))
                    pj = (int(face_lms[j].x * w), int(face_lms[j].y * h))
                    cv2.line(frame, pi, pj, _TEAL, 1)
            for idx in _KEY_FACE:
                if idx < len(face_lms):
                    cv2.circle(frame,
                               (int(face_lms[idx].x * w),
                                int(face_lms[idx].y * h)),
                               2, _TEAL, -1)

        # Upper-body pose (orange)
        if pose_lms and _POSE_CONN:
            _ORANGE = (0, 165, 255)
            pts: dict[int, tuple[int, int]] = {}
            for idx, lm in enumerate(pose_lms):
                if idx <= 10:
                    continue
                vis = getattr(lm, "visibility", 1.0)
                if vis >= _POSE_VIS:
                    pts[idx] = (int(lm.x * w), int(lm.y * h))
            for i, j in _POSE_CONN:
                if i > 10 and j > 10 and i in pts and j in pts:
                    cv2.line(frame, pts[i], pts[j], _ORANGE, 2)
            for pt in pts.values():
                cv2.circle(frame, pt, 4, _ORANGE, -1)

        # Hands (green = right, purple = left)
        for idx, hand_lms in enumerate(hand_lms_list):
            is_right = (
                handedness
                and idx < len(handedness)
                and handedness[idx]
                and handedness[idx][0].category_name.lower() == "right"
            )
            dot_col  = (0, 220, 0)   if is_right else (200, 0, 220)
            line_col = (0, 160, 0)   if is_right else (140, 0, 160)
            hpts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
            if _HAND_CONN:
                for a, b in _HAND_CONN:
                    if a < len(hpts) and b < len(hpts):
                        cv2.line(frame, hpts[a], hpts[b], line_col, 2)
            for pt in hpts:
                cv2.circle(frame, pt, 4, dot_col, -1)
