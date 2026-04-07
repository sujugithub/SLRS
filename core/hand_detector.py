"""
Hand detector using MediaPipe Tasks API for the Sign Language Recognition App.

Responsible for:
- Initializing the MediaPipe HandLandmarker
- Detecting hand landmarks from camera frames
- Extracting and normalizing landmark coordinates into feature vectors
- Drawing hand landmarks on frames for visual feedback
"""

import sys
import os
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

# Allow imports from project root when running this file directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    MAX_NUM_HANDS,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    CAMERA_INDEX,
    MODELS_DIR,
)

NUM_LANDMARKS = 21
NUM_COORDS = 3  # x, y, z per landmark
FEATURE_LENGTH = NUM_LANDMARKS * NUM_COORDS  # 63

MODEL_PATH = os.path.join(MODELS_DIR, "hand_landmarker.task")

# MediaPipe hand skeleton connections (21 landmarks)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # index
    (0, 9), (9, 10), (10, 11), (11, 12),  # middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (5, 9), (9, 13), (13, 17),             # palm
]

LANDMARK_COLOR = (10, 22, 80)    # dark-brown dots  (notebook right-hand style)
CONNECTION_COLOR = (121, 44, 80) # slate-blue lines (notebook right-hand style)
LANDMARK_RADIUS = 4
CONNECTION_THICKNESS = 2


class HandDetector:
    """Detects hands and extracts normalized landmark features using MediaPipe."""

    def __init__(
        self,
        model_path=MODEL_PATH,
        max_hands=MAX_NUM_HANDS,
        detection_confidence=MIN_DETECTION_CONFIDENCE,
        tracking_confidence=MIN_TRACKING_CONFIDENCE,
    ):
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.landmarker = HandLandmarker.create_from_options(options)

    def detect(self, frame):
        """Process a BGR frame and return landmarks for all detected hands.

        Args:
            frame: A BGR image (numpy array) from OpenCV.

        Returns:
            A list of hands, where each hand is a list of 21
            NormalizedLandmark objects. Returns an empty list if no
            hand is found.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)

        return result.hand_landmarks  # list of hands (may be empty)

    def extract_features(self, landmarks):
        """Convert 21 landmarks to a normalized flat array of 63 values.

        Normalization: all coordinates are shifted so the wrist (landmark 0)
        is at the origin, then scaled by the maximum absolute value across
        all axes so the range falls roughly within [-1, 1].

        Args:
            landmarks: A list of 21 NormalizedLandmark objects.

        Returns:
            A numpy array of shape (63,) with dtype float64.
        """
        raw = np.array(
            [[lm.x, lm.y, lm.z] for lm in landmarks],
            dtype=np.float64,
        )  # shape (21, 3)

        # Shift origin to wrist
        raw -= raw[0]

        # Scale by max absolute value (avoid division by zero)
        max_val = np.max(np.abs(raw))
        if max_val > 0:
            raw /= max_val

        return raw.flatten()  # (63,)

    def draw_landmarks(self, frame, landmarks):
        """Draw the hand skeleton and landmarks onto the frame in-place.

        Args:
            frame: A BGR image (numpy array) — modified in-place.
            landmarks: A list of 21 NormalizedLandmark objects.

        Returns:
            The same frame (for convenience).
        """
        h, w = frame.shape[:2]

        # Convert normalized coords to pixel positions
        points = []
        for lm in landmarks:
            px, py = int(lm.x * w), int(lm.y * h)
            points.append((px, py))

        # Draw connections
        for i, j in HAND_CONNECTIONS:
            cv2.line(frame, points[i], points[j], CONNECTION_COLOR, CONNECTION_THICKNESS)

        # Draw landmark dots
        for px, py in points:
            cv2.circle(frame, (px, py), LANDMARK_RADIUS, LANDMARK_COLOR, -1)

        return frame

    def close(self):
        """Release MediaPipe resources."""
        if hasattr(self, "landmarker") and self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None


# ----------------------------------------------------------------- test
if __name__ == "__main__":
    from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

    print("Initializing HandDetector …")
    detector = HandDetector()

    # --- Offline unit test with fabricated landmarks ---
    print("\n--- Offline unit test ---")
    fake_landmarks = [
        NormalizedLandmark(x=i * 0.01, y=i * 0.02, z=i * 0.001)
        for i in range(NUM_LANDMARKS)
    ]

    features = detector.extract_features(fake_landmarks)
    print(f"Feature shape : {features.shape}")
    assert features.shape == (FEATURE_LENGTH,), f"Expected ({FEATURE_LENGTH},)"
    assert features[0] == 0.0 and features[1] == 0.0 and features[2] == 0.0, \
        "Wrist should be at origin after normalization"
    print(f"Feature range : [{features.min():.4f}, {features.max():.4f}]")
    print("Offline test PASSED")

    # --- Draw test on blank frame ---
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    detector.draw_landmarks(blank, fake_landmarks)
    print(f"Draw test     : frame modified = {blank.any()}")
    print("Draw test PASSED")

    # --- Live webcam test (if camera is available) ---
    print("\n--- Live webcam test ---")
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("No camera available — skipping live test.")
    else:
        print("Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            landmarks = detector.detect(frame)
            if landmarks:
                features = detector.extract_features(landmarks)
                print(f"Features: shape={features.shape}  "
                      f"range=[{features.min():.3f}, {features.max():.3f}]")
                detector.draw_landmarks(frame, landmarks)
            else:
                print("No hand detected")

            cv2.imshow("Hand Detector Test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("Live test complete.")

    detector.close()
