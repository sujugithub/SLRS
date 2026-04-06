"""
Holistic detector for the Sign Language Recognition App.
Uses MediaPipe Tasks API (required for mediapipe >= 0.10.18 / Python 3.14).

Visualization matches the notebook color scheme:
  - Face mesh  : teal contours + dots
  - Right hand : dark-brown dots  + slate-blue lines
  - Left hand  : dark-purple dots + bright-purple lines
  - Pose       : orange dots      + magenta lines  (visibility-filtered)

Feature vector layout (171 values total):
  [0:63]    - Right/first hand  (21 pts × 3, wrist-origin normalized)
  [63:126]  - Left/second hand  (21 pts × 3, zeros if absent)
  [126:144] - Upper-body pose   (6 pts × 3, shoulder-midpoint normalized)
               Landmarks: L-shoulder(11) R-shoulder(12) L-elbow(13)
                          R-elbow(14)   L-wrist(15)    R-wrist(16)
  [144:171] - Essential face    (9 pts × 3, nose-tip normalized)
               Landmarks: nose-tip(1) L-brow-out(70) L-brow-in(107)
                          R-brow-out(300) R-brow-in(336) mouth-L(61)
                          mouth-R(291) upper-lip(13) lower-lip(14)
"""

import os
import sys

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    HandLandmarker,
    HandLandmarkerOptions,
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    MAX_NUM_HANDS,
    MODELS_DIR,
)

# ─── Model paths ──────────────────────────────────────────────────────────────
HAND_MODEL = os.path.join(MODELS_DIR, "hand_landmarker.task")
POSE_MODEL = os.path.join(MODELS_DIR, "pose_landmarker_lite.task")
FACE_MODEL = os.path.join(MODELS_DIR, "face_landmarker.task")

# ─── Feature dimensions ───────────────────────────────────────────────────────
NUM_HAND_LMS         = 21
NUM_COORDS           = 3
SINGLE_HAND_FEATURES = NUM_HAND_LMS * NUM_COORDS          # 63
BOTH_HANDS_FEATURES  = SINGLE_HAND_FEATURES * 2           # 126

KEY_POSE_INDICES = [11, 12, 13, 14, 15, 16]              # shoulders→elbows→wrists
POSE_FEATURES    = len(KEY_POSE_INDICES) * NUM_COORDS     # 18

KEY_FACE_INDICES = [1, 70, 107, 300, 336, 61, 291, 13, 14]   # nose+brows+mouth
FACE_FEATURES    = len(KEY_FACE_INDICES) * NUM_COORDS     # 27

HOLISTIC_FEATURE_LENGTH = BOTH_HANDS_FEATURES + POSE_FEATURES + FACE_FEATURES  # 171

# ─── Notebook-style drawing colours ───────────────────────────────────────────
# Face mesh
_FACE_DOT   = (0, 210, 190)      # teal
_FACE_LINE  = (0, 210, 190)
# Right hand (notebook: dark brown dots, slate-blue lines)
_R_DOT      = (10,  22,  80)
_R_LINE     = (121, 44,  80)
# Left hand (notebook: dark purple dots, bright purple lines)
_L_DOT      = (76,  22, 121)
_L_LINE     = (250, 44, 121)
# Pose (notebook: orange dots, magenta lines)
_P_DOT      = (66, 117, 245)
_P_LINE     = (230, 66, 245)

DOT_R_SM  = 2    # face dot radius
DOT_R_MD  = 4    # hand / pose dot radius
LINE_W_TH = 1    # face line width
LINE_W_MD = 2    # hand / pose line width

POSE_VIS_THRESHOLD = 0.5   # only draw pose landmarks with visibility above this

# ─── Hand skeleton (21 landmarks) ────────────────────────────────────────────
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

# ─── Full pose skeleton (33 landmarks, notebook-identical) ───────────────────
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (15, 17), (15, 19), (15, 21), (17, 19),
    (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]

# ─── Full face mesh (478-point, notebook-identical contours) ─────────────────
_FACE_OVAL = [
    (10, 338), (338, 297), (297, 332), (332, 284), (284, 251),
    (251, 389), (389, 356), (356, 454), (454, 323), (323, 361),
    (361, 288), (288, 397), (397, 365), (365, 379), (379, 378),
    (378, 400), (400, 377), (377, 152), (152, 148), (148, 176),
    (176, 149), (149, 150), (150, 136), (136, 172), (172, 58),
    (58, 132), (132, 93), (93, 234), (234, 127), (127, 162),
    (162, 21), (21, 54), (54, 103), (103, 67), (67, 109), (109, 10),
]
_LEFT_EYE  = [(33, 160), (160, 158), (158, 133), (133, 153), (153, 144), (144, 33)]
_RIGHT_EYE = [(362, 385), (385, 387), (387, 263), (263, 373), (373, 380), (380, 362)]
_L_BROW    = [(70, 63), (63, 105), (105, 66), (66, 107)]
_R_BROW    = [(300, 293), (293, 334), (334, 296), (296, 336)]
_LIPS      = [(61, 37), (37, 0), (0, 267), (267, 291),
              (61, 84), (84, 17), (17, 314), (314, 291)]
_NOSE      = [(168, 6), (6, 197), (197, 195), (195, 5), (5, 4)]
FACE_CONNECTIONS = (_FACE_OVAL + _LEFT_EYE + _RIGHT_EYE +
                    _L_BROW + _R_BROW + _LIPS + _NOSE)

_FACE_MAX_IDX = max(max(p) for p in FACE_CONNECTIONS)


class HolisticDetector:
    """Hand + pose + face detector with notebook-style visualization."""

    def __init__(self):
        hand_opts = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=HAND_MODEL),
            running_mode=RunningMode.IMAGE,
            num_hands=MAX_NUM_HANDS,
            min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )
        pose_opts = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=POSE_MODEL),
            running_mode=RunningMode.IMAGE,
            num_poses=1,
        )
        face_opts = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=FACE_MODEL),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
        )
        self._hand_lmk = HandLandmarker.create_from_options(hand_opts)
        self._pose_lmk = PoseLandmarker.create_from_options(pose_opts)
        self._face_lmk = FaceLandmarker.create_from_options(face_opts)
        self._last_handedness: list = []   # stored for left/right color coding

    # ─── Detection ────────────────────────────────────────────────────────────
    def detect_all(self, frame):
        """Run all three detectors on a BGR frame.

        Returns:
            (hand_lms_list, pose_lms, face_lms)
        """
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        hand_res = self._hand_lmk.detect(mp_img)
        pose_res = self._pose_lmk.detect(mp_img)
        face_res = self._face_lmk.detect(mp_img)

        self._last_handedness = hand_res.handedness
        hand_lms_list = hand_res.hand_landmarks
        pose_lms = pose_res.pose_landmarks[0] if pose_res.pose_landmarks else None
        face_lms = face_res.face_landmarks[0] if face_res.face_landmarks else None

        return hand_lms_list, pose_lms, face_lms

    # ─── Feature extraction ───────────────────────────────────────────────────
    def extract_holistic_features(self, hand_lms_list, pose_lms, face_lms):
        """Return a fixed-length (171,) float64 feature vector."""
        features = np.zeros(HOLISTIC_FEATURE_LENGTH, dtype=np.float64)

        # Both hands (0-125)
        for i, hand_lms in enumerate(hand_lms_list[:2]):
            raw = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms],
                           dtype=np.float64)
            raw -= raw[0]
            m = np.max(np.abs(raw))
            if m > 0:
                raw /= m
            start = i * SINGLE_HAND_FEATURES
            features[start : start + SINGLE_HAND_FEATURES] = raw.flatten()

        # Upper-body pose (126-143)
        if pose_lms:
            raw = np.array(
                [[pose_lms[idx].x, pose_lms[idx].y, pose_lms[idx].z]
                 for idx in KEY_POSE_INDICES],
                dtype=np.float64,
            )
            origin = (raw[0] + raw[1]) / 2.0    # midpoint of shoulders
            raw -= origin
            m = np.max(np.abs(raw))
            if m > 0:
                raw /= m
            features[BOTH_HANDS_FEATURES : BOTH_HANDS_FEATURES + POSE_FEATURES] = (
                raw.flatten()
            )

        # Essential face (144-170)
        if face_lms and len(face_lms) > max(KEY_FACE_INDICES):
            raw = np.array(
                [[face_lms[idx].x, face_lms[idx].y, face_lms[idx].z]
                 for idx in KEY_FACE_INDICES],
                dtype=np.float64,
            )
            raw -= raw[0]                        # nose-tip at origin
            m = np.max(np.abs(raw))
            if m > 0:
                raw /= m
            start = BOTH_HANDS_FEATURES + POSE_FEATURES
            features[start : start + FACE_FEATURES] = raw.flatten()

        return features

    # ─── Drawing ──────────────────────────────────────────────────────────────
    def draw_all(self, frame, hand_lms_list, pose_lms, face_lms):
        """Draw all landmarks with notebook-style colors, in-place."""
        h, w = frame.shape[:2]

        # 1. Full face mesh contours
        if face_lms and len(face_lms) > _FACE_MAX_IDX:
            all_idx = {i for pair in FACE_CONNECTIONS for i in pair}
            fp = {idx: (int(face_lms[idx].x * w), int(face_lms[idx].y * h))
                  for idx in all_idx}
            for i, j in FACE_CONNECTIONS:
                cv2.line(frame, fp[i], fp[j], _FACE_LINE, LINE_W_TH)
            for idx in KEY_FACE_INDICES:
                if idx in fp:
                    cv2.circle(frame, fp[idx], DOT_R_SM + 1, _FACE_DOT, -1)

        # 2. Full pose skeleton (visibility-filtered)
        if pose_lms:
            pts = {}
            for idx in range(len(pose_lms)):
                lm = pose_lms[idx]
                vis = getattr(lm, 'visibility', 1.0)
                if vis >= POSE_VIS_THRESHOLD:
                    pts[idx] = (int(lm.x * w), int(lm.y * h))
            for i, j in POSE_CONNECTIONS:
                if i in pts and j in pts:
                    cv2.line(frame, pts[i], pts[j], _P_LINE, LINE_W_MD)
            for pt in pts.values():
                cv2.circle(frame, pt, DOT_R_MD, _P_DOT, -1)

        # 3. Hands — right=red-brown/blue, left=purple (notebook colors)
        for i, hand_lms in enumerate(hand_lms_list):
            is_right = (
                i < len(self._last_handedness)
                and self._last_handedness[i]
                and self._last_handedness[i][0].category_name.lower() == "right"
            )
            dot_col  = _R_DOT  if is_right else _L_DOT
            line_col = _R_LINE if is_right else _L_LINE

            hpts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, hpts[a], hpts[b], line_col, LINE_W_MD)
            for pt in hpts:
                cv2.circle(frame, pt, DOT_R_MD, dot_col, -1)

        return frame

    # ─── Cleanup ──────────────────────────────────────────────────────────────
    def close(self):
        for attr in ("_hand_lmk", "_pose_lmk", "_face_lmk"):
            lmk = getattr(self, attr, None)
            if lmk is not None:
                try:
                    lmk.close()
                except Exception:
                    pass
                setattr(self, attr, None)
