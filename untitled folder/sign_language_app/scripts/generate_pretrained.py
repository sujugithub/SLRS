"""
Interactive script to capture pretrained sign language data via webcam.

Opens the camera, guides the user through each sign with a countdown,
captures 20 samples per sign using HandDetector, and saves the feature
arrays to data/pretrained/.

Usage:
    cd sign_language_app
    python scripts/generate_pretrained.py

Press 'q' at any time during capture to abort.
"""

import sys
import os
import json
import time

import cv2
import numpy as np

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import CAMERA_INDEX, PRETRAINED_DATA_DIR, CAMERA_WIDTH, CAMERA_HEIGHT
from core.hand_detector import HandDetector, FEATURE_LENGTH

# ------------------------------------------------------------ settings
SIGNS = [
    {
        "name": "peace",
        "description": "Hold up your INDEX and MIDDLE fingers in a V shape. "
                       "Curl the other fingers down.",
    },
    {
        "name": "hello",
        "description": "Show an OPEN PALM facing the camera with all five "
                       "fingers spread apart.",
    },
    {
        "name": "thumbs_up",
        "description": "Make a FIST and extend your THUMB straight up.",
    },
    {
        "name": "stop",
        "description": "Hold up a FLAT PALM facing the camera, fingers together "
                       "and pointing up.",
    },
    {
        "name": "point",
        "description": "Extend your INDEX FINGER forward/up. Curl all other "
                       "fingers into a fist.",
    },
]

SAMPLES_PER_SIGN = 20
COUNTDOWN_SECONDS = 3
DELAY_BETWEEN_CAPTURES = 0.4  # seconds between each sample

# OSD colours
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
BG_DARK = (30, 30, 46)


# --------------------------------------------------------------- helpers
def draw_text_centered(frame, text, y, color=WHITE, scale=1.0, thickness=2):
    """Draw horizontally centered text on the frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (frame.shape[1] - tw) // 2
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_overlay(frame, sign_name, description, captured, total, status=""):
    """Draw an informational overlay on the camera frame."""
    overlay = frame.copy()
    h, w = frame.shape[:2]

    # Dark banner at top
    cv2.rectangle(overlay, (0, 0), (w, 90), BG_DARK, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    draw_text_centered(frame, f'Sign: "{sign_name}"', 35, YELLOW, 1.0, 2)
    draw_text_centered(frame, description[:80], 65, WHITE, 0.5, 1)

    # Progress in top-right
    progress_text = f"{captured}/{total}"
    cv2.putText(frame, progress_text, (w - 100, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, GREEN, 2, cv2.LINE_AA)

    # Status message at bottom
    if status:
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, h - 50), (w, h), BG_DARK, -1)
        cv2.addWeighted(overlay2, 0.75, frame, 0.25, 0, frame)
        draw_text_centered(frame, status, h - 18, GREEN, 0.7, 2)


def draw_countdown(frame, seconds_left):
    """Draw a large countdown number in the center of the frame."""
    h, w = frame.shape[:2]
    text = str(seconds_left)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 5.0
    thickness = 8
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (w - tw) // 2
    y = (h + th) // 2

    # Shadow
    cv2.putText(frame, text, (x + 3, y + 3), font, scale, (0, 0, 0), thickness + 4, cv2.LINE_AA)
    # Number
    color = GREEN if seconds_left > 1 else RED
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


# ----------------------------------------------------------------- main
def main():
    print("=" * 60)
    print("  SIGN LANGUAGE DATA CAPTURE TOOL")
    print("=" * 60)
    print()
    print(f"Signs to capture : {', '.join(s['name'] for s in SIGNS)}")
    print(f"Samples per sign : {SAMPLES_PER_SIGN}")
    print(f"Camera index     : {CAMERA_INDEX}")
    print(f"Output directory : {PRETRAINED_DATA_DIR}")
    print()
    print("Instructions:")
    print("  1. A camera window will open.")
    print("  2. For each sign you'll see a description and a countdown.")
    print("  3. Hold the sign STEADY during capture.")
    print("  4. Move your hand slightly between samples for variety.")
    print("  5. Press 'q' at any time to abort.")
    print("  6. Press 's' to skip a sign.")
    print()
    input("Press ENTER to start …")

    # --- Open camera ---
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera at index {CAMERA_INDEX}.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    detector = HandDetector()
    labels = {}  # will become labels.json
    aborted = False

    print("\nCamera opened. Starting capture …\n")

    for sign_idx, sign_info in enumerate(SIGNS):
        sign_name = sign_info["name"]
        description = sign_info["description"]
        sign_dir = os.path.join(PRETRAINED_DATA_DIR, sign_name)
        os.makedirs(sign_dir, exist_ok=True)

        features_list = []
        skip = False

        print(f"[{sign_idx + 1}/{len(SIGNS)}] {sign_name}")
        print(f"    {description}")

        # ---- Wait screen: show instructions until user presses SPACE ----
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)  # mirror

            landmarks = detector.detect(frame)
            if landmarks:
                detector.draw_landmarks(frame, landmarks)

            draw_overlay(frame, sign_name, description, 0, SAMPLES_PER_SIGN,
                         "Press SPACE to begin  |  S to skip  |  Q to quit")
            cv2.imshow("Sign Language Capture", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                break
            elif key == ord("s"):
                skip = True
                print(f"    Skipped.")
                break
            elif key == ord("q"):
                aborted = True
                break

        if aborted:
            break
        if skip:
            continue

        # ---- Countdown ----
        print(f"    Countdown …", end="", flush=True)
        for sec in range(COUNTDOWN_SECONDS, 0, -1):
            t_end = time.time() + 1.0
            while time.time() < t_end:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)

                landmarks = detector.detect(frame)
                if landmarks:
                    detector.draw_landmarks(frame, landmarks)

                draw_overlay(frame, sign_name, description, 0, SAMPLES_PER_SIGN,
                             f"Get ready! Starting in {sec} …")
                draw_countdown(frame, sec)
                cv2.imshow("Sign Language Capture", frame)
                cv2.waitKey(1)
            print(f" {sec}", end="", flush=True)
        print()

        # ---- Capture loop ----
        captured = 0
        no_hand_streak = 0

        while captured < SAMPLES_PER_SIGN:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            landmarks = detector.detect(frame)

            if landmarks:
                no_hand_streak = 0
                features = detector.extract_features(landmarks)
                features_list.append(features)
                captured += 1
                detector.draw_landmarks(frame, landmarks)

                status = f"Captured {captured}/{SAMPLES_PER_SIGN} — hold steady!"
                draw_overlay(frame, sign_name, description, captured, SAMPLES_PER_SIGN, status)

                # Brief flash to show capture happened
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 5), GREEN, -1)

                print(f"    Sample {captured}/{SAMPLES_PER_SIGN}", end="\r")
            else:
                no_hand_streak += 1
                status = "No hand detected — show your hand to the camera"
                draw_overlay(frame, sign_name, description, captured, SAMPLES_PER_SIGN, status)

            cv2.imshow("Sign Language Capture", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                aborted = True
                break

            # Delay between captures so user can adjust slightly
            if landmarks:
                time.sleep(DELAY_BETWEEN_CAPTURES)

        if aborted:
            break

        # ---- Save features ----
        if features_list:
            features_array = np.array(features_list, dtype=np.float64)
            npy_path = os.path.join(sign_dir, "features.npy")
            np.save(npy_path, features_array)

            labels[sign_name] = {
                "samples": len(features_list),
                "features_file": os.path.relpath(npy_path, PRETRAINED_DATA_DIR),
                "feature_length": FEATURE_LENGTH,
            }

            print(f"    Saved {features_array.shape[0]} samples → {npy_path}")
        print()

    # ---- Cleanup ----
    cap.release()
    cv2.destroyAllWindows()
    detector.close()

    if aborted:
        print("\nCapture aborted by user.")
        print("Any fully completed signs have still been saved.\n")

    # ---- Save labels.json ----
    if labels:
        labels_path = os.path.join(PRETRAINED_DATA_DIR, "labels.json")
        with open(labels_path, "w") as f:
            json.dump(labels, f, indent=2)
        print(f"Labels written to: {labels_path}")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("  CAPTURE SUMMARY")
    print("=" * 60)
    for name, info in labels.items():
        print(f"  {name:15s}  {info['samples']:3d} samples")
    total = sum(info["samples"] for info in labels.values())
    print(f"  {'TOTAL':15s}  {total:3d} samples")
    if len(labels) < len(SIGNS):
        missing = [s["name"] for s in SIGNS if s["name"] not in labels]
        print(f"\n  Missing/skipped: {', '.join(missing)}")
    print("=" * 60)
    print("\nDone! You can now train the model with:")
    print("  python -c \"from core.model_trainer import SignModel; "
          "m = SignModel(); m.load_pretrained(); m.save_model(); "
          "print('Trained:', m.get_all_signs())\"")
    print()


if __name__ == "__main__":
    main()
