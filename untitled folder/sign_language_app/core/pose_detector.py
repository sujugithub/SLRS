"""
Pose detector using MediaPipe Tasks API.

Detects 33 body landmarks per frame. Used to draw face and shoulder
overlays on the camera preview. Only a subset of landmark indices are
drawn (face + shoulders) — the full body landmark set is not needed
for sign language recognition.

MediaPipe Pose landmark indices (relevant subset):
    0  = nose
    1  = left eye inner,  2 = left eye,   3 = left eye outer
    4  = right eye inner, 5 = right eye,  6 = right eye outer
    7  = left ear,        8 = right ear
    9  = mouth left,      10 = mouth right
    11 = left shoulder,   12 = right shoulder
"""

import os
import sys

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import MODELS_DIR

MODEL_PATH = os.path.join(MODELS_DIR, "pose_landmarker_lite.task")

# Landmark indices to draw as dots (face + shoulders only)
FACE_INDICES = list(range(0, 11))   # nose, eyes, ears, mouth
SHOULDER_INDICES = [11, 12]         # left and right shoulder

# Skeleton connections for face + shoulder region
POSE_CONNECTIONS = [
    # Left eye path: inner → eye → outer → ear
    (1, 2), (2, 3), (3, 7),
    # Right eye path: inner → eye → outer → ear
    (4, 5), (5, 6), (6, 8),
    # Nose to eye inners
    (0, 1), (0, 4),
    # Mouth
    (9, 10),
    # Ear to shoulder
    (7, 11), (8, 12),
    # Shoulder bar
    (11, 12),
]

LANDMARK_COLOR = (66, 117, 245)    # orange dots  (notebook style)
CONNECTION_COLOR = (230, 66, 245)  # magenta lines (notebook style)
LANDMARK_RADIUS = 4
CONNECTION_THICKNESS = 2


class PoseDetector:
    """Detects body pose landmarks and draws face + shoulder overlays."""

    def __init__(self, model_path=MODEL_PATH):
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            num_poses=1,
        )
        self.landmarker = PoseLandmarker.create_from_options(options)

    def detect(self, frame):
        """Detect pose landmarks in a BGR frame.

        Returns:
            A list of 33 NormalizedLandmark objects, or None if no pose found.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)

        if result.pose_landmarks:
            return result.pose_landmarks[0]
        return None

    def draw_face_and_shoulders(self, frame, landmarks):
        """Draw face and shoulder landmarks + connections onto the frame in-place.

        Args:
            frame: BGR numpy array — modified in-place.
            landmarks: List of 33 NormalizedLandmark objects.

        Returns:
            The same frame (for convenience).
        """
        h, w = frame.shape[:2]

        # Convert all 33 normalized coords to pixel positions
        points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        # Draw skeleton connections (face + shoulder area only)
        for i, j in POSE_CONNECTIONS:
            if i < len(points) and j < len(points):
                cv2.line(frame, points[i], points[j],
                         CONNECTION_COLOR, CONNECTION_THICKNESS)

        # Draw face landmark dots
        for idx in FACE_INDICES:
            if idx < len(points):
                cv2.circle(frame, points[idx], LANDMARK_RADIUS,
                           LANDMARK_COLOR, -1)

        # Draw shoulder dots (slightly larger to stand out)
        for idx in SHOULDER_INDICES:
            if idx < len(points):
                cv2.circle(frame, points[idx], LANDMARK_RADIUS + 2,
                           LANDMARK_COLOR, -1)

        return frame

    def close(self):
        """Release MediaPipe resources."""
        if hasattr(self, "landmarker") and self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None
