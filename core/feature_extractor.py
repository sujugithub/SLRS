"""
Feature extraction for sign language recognition.

Two extractors are available:

1. SpatialFeatureExtractor  (NEW — recommended for all new training data)
   ─────────────────────────────────────────────────────────────────────
   Face/body-relative features that are invariant to:
     • Distance from camera  (normalised by head_scale)
     • Head height variation between users  (nose anchor)
     • Absolute screen position  (all coordinates relative to nose)

   Feature vector layout  (65 values total):
   ┌─────────────────────────────────────────────────────────────┬──────┐
   │ Group B — hand position + zones  (left + right)            │  20  │
   │   per hand: wrist_rel_x, wrist_rel_y,                      │      │
   │             tip_rel_x,   tip_rel_y,                        │      │
   │             at_forehead, at_chin, at_cheek,                │      │
   │             at_chest,    at_mouth, hand_visible            │      │
   ├─────────────────────────────────────────────────────────────┼──────┤
   │ Group C — finger joint angles  (left + right)              │  20  │
   │   10 angles per hand (2 joints × 5 fingers)                │      │
   ├─────────────────────────────────────────────────────────────┼──────┤
   │ Group D — arm position  (left + right)                     │   8  │
   │   per arm: elbow_rel_x, elbow_rel_y,                       │      │
   │            shoulder_to_wrist_angle, elbow_bend_angle       │      │
   ├─────────────────────────────────────────────────────────────┼──────┤
   │ Group E — dominant hand indicator                          │   1  │
   ├─────────────────────────────────────────────────────────────┼──────┤
   │ Group F — relative hand-to-hand position                   │   3  │
   │   hands_touching, left_above_right, horizontal_separation  │      │
   ├─────────────────────────────────────────────────────────────┼──────┤
   │ Group G — contact features  (left + right)                 │  13  │
   │   per hand (6): mouth_dist, chin_dist, forehead_dist,      │      │
   │                 touching_mouth, touching_chin,             │      │
   │                 touching_forehead                          │      │
   │   shared (1): face_visible                                 │      │
   └─────────────────────────────────────────────────────────────┴──────┘
   Total                                                           65

2. extract_features()  (LEGACY — single hand, 93 features)
   ─────────────────────────────────────────────────────────────────────
   Raw wrist-normalised coordinates + joint angles + inter-landmark
   distances.  Kept for backward compatibility; not body-relative.

⚠  Existing .npy training data captured with the legacy extractor is
   incompatible with the new 52-feature format.  Delete old .npy files
   and recollect training data before retraining.
"""

from __future__ import annotations

import numpy as np
import cv2

# ─── Spatial extractor dimensions ─────────────────────────────────────────────
_B_PER_HAND = 10   # wrist_rx, wrist_ry, tip_rx, tip_ry, 5 zones, visible
_C_PER_HAND = 10   # 10 finger joint angles
_D_PER_ARM  = 4    # elbow_rx, elbow_ry, sw_angle, eb_angle
_G_PER_HAND = 6    # mouth_dist, chin_dist, forehead_dist, 3 contact flags

SPATIAL_FEATURE_LENGTH = (
    _B_PER_HAND * 2    # Group B: both hands      [0:20]
    + _C_PER_HAND * 2  # Group C: both hands      [20:40]
    + _D_PER_ARM  * 2  # Group D: both arms       [40:48]
    + 1                # Group E: dominant hand   [48]
    + 3                # Group F: hand-to-hand    [49:52]
    + _G_PER_HAND * 2  # Group G: contact (hands) [52:64]
    + 1                # Group G: face_visible    [64]
)  # = 65

# Slice offsets into the 65-element vector
_B_L = 0    # left  hand position + zones   [0:10]
_B_R = 10   # right hand position + zones   [10:20]
_C_L = 20   # left  hand joint angles       [20:30]
_C_R = 30   # right hand joint angles       [30:40]
_D_L = 40   # left  arm                     [40:44]
_D_R = 44   # right arm                     [44:48]
_E   = 48   # dominant hand indicator       [48]
_F   = 49   # hand-to-hand block            [49:52]
_G_L = 52   # left  hand contact features   [52:58]
_G_R = 58   # right hand contact features   [58:64]
_G_FACE_VIS = 64  # face_visible flag        [64]

# Face mesh landmark indices used for contact detection
_FACE_MOUTH_IDX    = 13   # upper lip center
_FACE_CHIN_IDX     = 152  # chin
_FACE_FOREHEAD_IDX = 10   # forehead

# Hand fingertip landmark indices
_FINGERTIPS = (8, 12, 4)  # index_tip, middle_tip, thumb_tip

# Contact distance threshold (normalised by head_scale)
_CONTACT_THRESHOLD = 0.15

# ─── Legacy extractor dimensions ──────────────────────────────────────────────
NUM_BASE       = 63
NUM_ANGLES     = 15
NUM_DISTS      = 15
FEATURE_LENGTH = NUM_BASE + NUM_ANGLES + NUM_DISTS  # 93

# ─── Finger joint triplets (a-b-c → angle at b) ───────────────────────────────
# Used by SpatialFeatureExtractor (group C): 2 joints × 5 fingers = 10 per hand
_SPATIAL_JOINTS = [
    # Thumb
    (1, 2, 3), (2, 3, 4),
    # Index
    (5, 6, 7), (6, 7, 8),
    # Middle
    (9, 10, 11), (10, 11, 12),
    # Ring
    (13, 14, 15), (14, 15, 16),
    # Pinky
    (17, 18, 19), (18, 19, 20),
]

# ─── Legacy joint angle triplets (15 joints) ──────────────────────────────────
_JOINT_TRIPLETS = [
    (0, 1, 2), (1, 2, 3), (2, 3, 4),
    (5, 6, 7), (6, 7, 8),
    (9, 10, 11), (10, 11, 12),
    (13, 14, 15), (14, 15, 16),
    (17, 18, 19), (18, 19, 20),
    (5, 9, 13), (9, 13, 17),
    (1, 0, 5), (4, 0, 8),
]

_DIST_PAIRS = [
    (4, 8), (4, 12), (4, 16), (4, 20),
    (8, 12), (12, 16), (16, 20),
    (8, 0), (12, 0), (4, 0),
    (8, 16), (8, 20),
    (5, 17), (0, 9), (4, 17),
]


# ─── Geometry helpers ─────────────────────────────────────────────────────────

def _angle_at(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle in radians at vertex b formed by rays b→a and b→c."""
    ba, bc = a - b, c - b
    n_ba, n_bc = np.linalg.norm(ba), np.linalg.norm(bc)
    if n_ba < 1e-9 or n_bc < 1e-9:
        return 0.0
    return float(np.arccos(np.clip(np.dot(ba, bc) / (n_ba * n_bc), -1.0, 1.0)))


def _lm3(lm) -> np.ndarray:
    """Extract (x, y, z) as float64 array from a NormalizedLandmark."""
    return np.array([lm.x, lm.y, getattr(lm, "z", 0.0)], dtype=np.float64)


# ─── Hand assignment helper ────────────────────────────────────────────────────

def _split_hands(hand_lms_list: list, pose_lms) -> tuple:
    """
    Assign detected hands to left/right using pose wrist proximity.

    MediaPipe Pose landmark indices:
        15 = left wrist,  16 = right wrist

    Each detected hand is matched to whichever pose wrist is closest.
    If pose is unavailable, the first hand is treated as the right hand.

    Returns:
        (left_hand_lms, right_hand_lms) — each may be None.
    """
    if not hand_lms_list:
        return None, None

    if pose_lms is None or len(pose_lms) < 17:
        right = hand_lms_list[0]
        left  = hand_lms_list[1] if len(hand_lms_list) > 1 else None
        return left, right

    l_pose = pose_lms[15]
    r_pose = pose_lms[16]
    left_hand = right_hand = None

    for hand in hand_lms_list:
        wrist = hand[0]
        d_l = (wrist.x - l_pose.x) ** 2 + (wrist.y - l_pose.y) ** 2
        d_r = (wrist.x - r_pose.x) ** 2 + (wrist.y - r_pose.y) ** 2
        if d_l <= d_r:
            if left_hand is None:
                left_hand = hand
        else:
            if right_hand is None:
                right_hand = hand

    return left_hand, right_hand


# ─── SpatialFeatureExtractor ──────────────────────────────────────────────────

class SpatialFeatureExtractor:
    """
    Extracts a 52-dimensional spatially-normalised feature vector from
    MediaPipe landmark detections.

    All hand/arm positions are expressed relative to the nose anchor and
    scaled by head_scale, making features invariant to:
      • camera distance / zoom
      • absolute head position on screen
      • user height variation

    Usage
    ─────
    Basic (you manage L/R splitting):
        feats = SpatialFeatureExtractor().extract(
            pose_lms, left_hand_lms, right_hand_lms, face_lms)

    Convenience (auto-splits using pose wrist proximity):
        feats = SpatialFeatureExtractor().extract_from_holistic(
            hand_lms_list, pose_lms, face_lms)
    """

    # ── Public API ─────────────────────────────────────────────────────────────

    def extract(self,
                pose_landmarks,
                left_hand_landmarks,
                right_hand_landmarks,
                face_landmarks) -> np.ndarray:
        """
        Build the 52-feature spatial vector.

        Args:
            pose_landmarks       : list of ≥17 NormalizedLandmark (Pose), or None.
            left_hand_landmarks  : list of 21 NormalizedLandmark, or None.
            right_hand_landmarks : list of 21 NormalizedLandmark, or None.
            face_landmarks       : list of ≥153 NormalizedLandmark (FaceMesh), or None.

        Returns:
            np.ndarray of shape (52,), dtype float64.
        """
        out = np.zeros(SPATIAL_FEATURE_LENGTH, dtype=np.float64)

        # ── A. Anchor + head_scale ─────────────────────────────────────────
        anchor_x, anchor_y, head_scale = self._compute_anchor(
            pose_landmarks, face_landmarks)

        # ── B. Hand position relative to face ─────────────────────────────
        # Track absolute wrist positions for Groups E and F
        lw_x = lw_y = rw_x = rw_y = 0.0
        l_visible = r_visible = False

        for hand_lms, b_off, side in (
            (left_hand_landmarks,  _B_L, "L"),
            (right_hand_landmarks, _B_R, "R"),
        ):
            if hand_lms is None or len(hand_lms) < 21:
                continue   # hand_visible stays 0.0 (already zeroed)

            wrist = hand_lms[0]
            tip   = hand_lms[12]   # middle finger tip

            rel_x  = (wrist.x - anchor_x) / head_scale
            rel_y  = (wrist.y - anchor_y) / head_scale
            tip_rx = (tip.x   - anchor_x) / head_scale
            tip_ry = (tip.y   - anchor_y) / head_scale

            out[b_off + 0] = rel_x
            out[b_off + 1] = rel_y
            out[b_off + 2] = tip_rx
            out[b_off + 3] = tip_ry
            # Zone flags
            out[b_off + 4] = 1.0 if rel_y < -0.8 else 0.0
            out[b_off + 5] = 1.0 if -0.3 < rel_y < 0.1 else 0.0
            out[b_off + 6] = 1.0 if abs(rel_x) > 0.4 and -0.5 < rel_y < 0.1 else 0.0
            out[b_off + 7] = 1.0 if rel_y > 0.5 else 0.0
            out[b_off + 8] = 1.0 if abs(rel_x) < 0.3 and -0.1 < rel_y < 0.2 else 0.0
            out[b_off + 9] = 1.0  # hand_visible

            if side == "L":
                lw_x, lw_y, l_visible = wrist.x, wrist.y, True
            else:
                rw_x, rw_y, r_visible = wrist.x, wrist.y, True

        # ── C. Finger joint angles ─────────────────────────────────────────
        for hand_lms, c_off in (
            (left_hand_landmarks,  _C_L),
            (right_hand_landmarks, _C_R),
        ):
            if hand_lms is None or len(hand_lms) < 21:
                continue
            pts = np.array([_lm3(lm) for lm in hand_lms], dtype=np.float64)
            for i, (a, b, c) in enumerate(_SPATIAL_JOINTS):
                out[c_off + i] = _angle_at(pts[a], pts[b], pts[c])

        # ── D. Arm position ────────────────────────────────────────────────
        if pose_landmarks is not None and len(pose_landmarks) >= 17:
            for sh_idx, el_idx, wr_idx, d_off in (
                (11, 13, 15, _D_L),   # left arm
                (12, 14, 16, _D_R),   # right arm
            ):
                sh = pose_landmarks[sh_idx]
                el = pose_landmarks[el_idx]
                wr = pose_landmarks[wr_idx]

                out[d_off + 0] = (el.x - anchor_x) / head_scale
                out[d_off + 1] = (el.y - anchor_y) / head_scale
                out[d_off + 2] = float(np.arctan2(wr.y - sh.y, wr.x - sh.x))
                out[d_off + 3] = _angle_at(_lm3(sh), _lm3(el), _lm3(wr))

        # ── E. Dominant hand indicator ─────────────────────────────────────
        if l_visible and r_visible:
            l_d = ((lw_x - anchor_x) ** 2 + (lw_y - anchor_y) ** 2) ** 0.5
            r_d = ((rw_x - anchor_x) ** 2 + (rw_y - anchor_y) ** 2) ** 0.5
            out[_E] = 1.0 if r_d < l_d else 0.0
        elif r_visible:
            out[_E] = 1.0

        # ── F. Relative hand-to-hand position ─────────────────────────────
        if l_visible and r_visible:
            dist = ((lw_x - rw_x) ** 2 + (lw_y - rw_y) ** 2) ** 0.5
            out[_F + 0] = 1.0 if dist < 0.15 * head_scale else 0.0
            out[_F + 1] = 1.0 if lw_y < rw_y else 0.0
            out[_F + 2] = (rw_x - lw_x) / head_scale

        # ── G. Contact features (fingertip-to-face distances + flags) ─────
        out[_G_FACE_VIS] = 1.0 if face_landmarks is not None else 0.0

        if face_landmarks is not None and len(face_landmarks) > _FACE_CHIN_IDX:
            mouth_lm    = face_landmarks[_FACE_MOUTH_IDX]
            chin_lm     = face_landmarks[_FACE_CHIN_IDX]
            forehead_lm = face_landmarks[_FACE_FOREHEAD_IDX]

            mouth_pt    = np.array([mouth_lm.x,    mouth_lm.y],    dtype=np.float64)
            chin_pt     = np.array([chin_lm.x,     chin_lm.y],     dtype=np.float64)
            forehead_pt = np.array([forehead_lm.x, forehead_lm.y], dtype=np.float64)

            for hand_lms, g_off in (
                (left_hand_landmarks,  _G_L),
                (right_hand_landmarks, _G_R),
            ):
                if hand_lms is None or len(hand_lms) < 13:
                    continue  # stays 0.0

                tips = np.array(
                    [[hand_lms[i].x, hand_lms[i].y] for i in _FINGERTIPS],
                    dtype=np.float64,
                )

                d_mouth    = float(np.min(np.linalg.norm(tips - mouth_pt,    axis=1))) / head_scale
                d_chin     = float(np.min(np.linalg.norm(tips - chin_pt,     axis=1))) / head_scale
                d_forehead = float(np.min(np.linalg.norm(tips - forehead_pt, axis=1))) / head_scale

                out[g_off + 0] = d_mouth
                out[g_off + 1] = d_chin
                out[g_off + 2] = d_forehead
                out[g_off + 3] = 1.0 if d_mouth    < _CONTACT_THRESHOLD else 0.0
                out[g_off + 4] = 1.0 if d_chin     < _CONTACT_THRESHOLD else 0.0
                out[g_off + 5] = 1.0 if d_forehead < _CONTACT_THRESHOLD else 0.0

        return out

    def extract_from_holistic(self,
                               hand_lms_list: list,
                               pose_landmarks,
                               face_landmarks) -> np.ndarray:
        """
        Convenience wrapper: auto-assigns hands to left/right slots using
        pose wrist proximity, then calls extract().

        Args:
            hand_lms_list : list of 0-2 detected hand landmark lists (unordered
                            as returned by CameraWorker / HolisticDetector).
            pose_landmarks: MediaPipe Pose landmarks (33 points), or None.
            face_landmarks: MediaPipe FaceMesh landmarks, or None.

        Returns:
            np.ndarray of shape (52,), dtype float64.
        """
        left_hand, right_hand = _split_hands(hand_lms_list, pose_landmarks)
        return self.extract(pose_landmarks, left_hand, right_hand, face_landmarks)

    def draw_debug(self, frame: np.ndarray, features: np.ndarray,
                   pose_landmarks, face_lms=None,
                   hand_lms_list=None) -> np.ndarray:
        """
        Draw spatial feature debug overlay onto ``frame`` (in-place).

        Renders:
        - White cross/dot at the nose anchor
        - Line from nose to each detected wrist, coloured by zone:
            green  = hand_at_mouth
            blue   = hand_at_forehead
            yellow = hand_at_chin
            white  = none of the above
        - Text showing normalised (rel_x, rel_y) per hand
        - Group G contact: line from index fingertip to nearest face anchor
            red  = contact detected (any touching flag active)
            gray = no contact
        - "CONTACT" text in red when any touching flag is active
        - "mouth_dist: X.XX" showing the raw distance value

        Args:
            frame         : BGR numpy array — modified in-place.
            features      : 65-element array returned by extract().
            pose_landmarks: MediaPipe Pose landmarks, or None.
            face_lms      : MediaPipe Face landmarks, or None.
            hand_lms_list : list of raw hand landmark lists (unordered), or None.

        Returns:
            The same frame.
        """
        if pose_landmarks is None or len(pose_landmarks) == 0:
            return frame

        h, w = frame.shape[:2]
        nose = pose_landmarks[0]
        ax, ay = int(nose.x * w), int(nose.y * h)

        # Nose anchor marker
        cv2.drawMarker(
            frame, (ax, ay), (255, 255, 255),
            cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA,
        )

        for side_label, b_off, pose_wrist_idx in (
            ("L", _B_L, 15),
            ("R", _B_R, 16),
        ):
            if len(features) <= b_off + 9:
                continue
            if features[b_off + 9] < 0.5:   # hand_visible flag
                continue
            if len(pose_landmarks) <= pose_wrist_idx:
                continue

            pw = pose_landmarks[pose_wrist_idx]
            wx, wy = int(pw.x * w), int(pw.y * h)

            rel_x = features[b_off + 0]
            rel_y = features[b_off + 1]

            # Zone-based color (BGR)
            if features[b_off + 8] > 0.5:    # mouth  → green
                color = (0, 220, 0)
            elif features[b_off + 4] > 0.5:  # forehead → blue
                color = (255, 80, 0)
            elif features[b_off + 5] > 0.5:  # chin   → yellow
                color = (0, 220, 220)
            else:                             # other  → white
                color = (200, 200, 200)

            cv2.line(frame, (ax, ay), (wx, wy), color, 2, cv2.LINE_AA)
            cv2.circle(frame, (wx, wy), 5, color, -1)
            cv2.putText(
                frame,
                f"{side_label}: ({rel_x:.2f}, {rel_y:.2f})",
                (wx + 6, max(wy - 8, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA,
            )

        # ── Group G: contact debug overlay ────────────────────────────────
        if (face_lms is not None and len(face_lms) > _FACE_CHIN_IDX
                and len(features) > _G_FACE_VIS):

            mouth_pt    = (int(face_lms[_FACE_MOUTH_IDX].x    * w),
                           int(face_lms[_FACE_MOUTH_IDX].y    * h))
            chin_pt     = (int(face_lms[_FACE_CHIN_IDX].x     * w),
                           int(face_lms[_FACE_CHIN_IDX].y     * h))
            forehead_pt = (int(face_lms[_FACE_FOREHEAD_IDX].x * w),
                           int(face_lms[_FACE_FOREHEAD_IDX].y * h))
            face_anchor_pts = [mouth_pt, chin_pt, forehead_pt]

            left_lms, right_lms = _split_hands(hand_lms_list or [], pose_landmarks)

            any_contact   = False
            mouth_dist_lbl: float | None = None

            for hand_side_lms, g_off in (
                (left_lms,  _G_L),
                (right_lms, _G_R),
            ):
                if hand_side_lms is None or len(hand_side_lms) < 9:
                    continue

                touching = (
                    features[g_off + 3] > 0.5
                    or features[g_off + 4] > 0.5
                    or features[g_off + 5] > 0.5
                )
                line_color = (0, 0, 220) if touching else (110, 110, 110)
                if touching:
                    any_contact = True
                if mouth_dist_lbl is None:
                    mouth_dist_lbl = features[g_off + 0]

                # Index fingertip → nearest face anchor
                idx_tip = (int(hand_side_lms[8].x * w),
                           int(hand_side_lms[8].y * h))
                nearest = min(
                    face_anchor_pts,
                    key=lambda p: (p[0] - idx_tip[0]) ** 2 + (p[1] - idx_tip[1]) ** 2,
                )
                cv2.line(frame, idx_tip, nearest, line_color, 1, cv2.LINE_AA)
                cv2.circle(frame, idx_tip, 4, line_color, -1)

            if any_contact:
                cv2.putText(
                    frame, "CONTACT",
                    (ax + 8, ay - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 220), 2, cv2.LINE_AA,
                )
            if mouth_dist_lbl is not None:
                cv2.putText(
                    frame, f"mouth_dist: {mouth_dist_lbl:.2f}",
                    (ax + 8, ay - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA,
                )

        return frame

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _compute_anchor(self, pose_lms, face_lms) -> tuple:
        """
        Compute (anchor_x, anchor_y, head_scale).

        Anchor  : nose tip (pose landmark 0).
        Scale   : Euclidean distance nose → chin.
                  Preferred source: face mesh landmark 152 (chin).
                  Fallback: nose (0) → mouth_left (9) from pose.
        """
        if pose_lms is None or len(pose_lms) == 0:
            return 0.5, 0.5, 1.0

        nose = pose_lms[0]
        ax, ay = nose.x, nose.y

        if face_lms is not None and len(face_lms) > 152:
            chin  = face_lms[152]
            scale = ((chin.x - ax) ** 2 + (chin.y - ay) ** 2) ** 0.5
        elif len(pose_lms) > 9:
            ml    = pose_lms[9]
            scale = ((ml.x - ax) ** 2 + (ml.y - ay) ** 2) ** 0.5
        else:
            scale = 0.0

        return ax, ay, (scale if scale > 1e-6 else 1.0)


# ─── Legacy single-hand extractor (93 features) ───────────────────────────────

def extract_features(landmarks) -> np.ndarray:
    """
    LEGACY: Convert 21 MediaPipe hand landmarks into a 93-feature vector.

    Features: raw wrist-origin coords (63) + joint angles (15) +
              inter-landmark distances (15).

    ⚠  This extractor is NOT body-relative — it is sensitive to camera
       distance and absolute hand position on screen.
       Use SpatialFeatureExtractor for new training data.

    Returns:
        np.ndarray of shape (93,), dtype float64.
    """
    raw = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float64)
    raw -= raw[0]
    scale = np.max(np.abs(raw))
    if scale > 0:
        raw /= scale
    base = raw.flatten()

    angles = np.array(
        [_angle_at(raw[a], raw[b], raw[c]) / np.pi for a, b, c in _JOINT_TRIPLETS],
        dtype=np.float64,
    )

    palm_size = np.linalg.norm(raw[9] - raw[0])
    if palm_size < 1e-9:
        palm_size = 1.0
    dists = np.array(
        [np.linalg.norm(raw[i] - raw[j]) / palm_size for i, j in _DIST_PAIRS],
        dtype=np.float64,
    )

    return np.concatenate([base, angles, dists])


def extract_features_batch(landmarks_batch) -> np.ndarray:
    """Vectorized legacy extraction for a list of landmark sets."""
    return np.stack([extract_features(lms) for lms in landmarks_batch])
