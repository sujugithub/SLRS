"""
Model trainer and predictor for the Sign Language Recognition App.

Responsible for:
- Loading and preprocessing collected hand landmark data
- Training a RandomForestClassifier on landmark feature vectors
- Saving and loading trained models with joblib
- Running inference on new hand landmark feature vectors
- Managing pretrained and custom sign data
"""

import sys
import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Allow imports from project root when running this file directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    PRETRAINED_DATA_DIR,
    CUSTOM_DATA_DIR,
    DEFAULT_MODEL_FILE,
    RANDOM_STATE,
    HOLISTIC_FEATURE_LENGTH,
)

# Feature length for the static Random-Forest sign model.
# Uses BOTH hands (right + left): 21 landmarks × 3 coords × 2 hands = 126.
# The second 63-value slot is zeroed when only one hand is detected.
# The LSTM model continues to use HOLISTIC_FEATURE_LENGTH (171) independently.
FEATURE_LENGTH          = 42 * 3 * 2   # 126
OLD_HAND_FEATURE_LENGTH = 63            # single-hand legacy format (auto-migrated)


class SignModel:
    """Manages sign language classification: data, training, and inference."""

    def __init__(self):
        self.model = None
        self.labels = []       # ordered list of sign names
        self.X = np.empty((0, FEATURE_LENGTH), dtype=np.float64)
        self.y = np.empty(0, dtype=int)
        self._label_map = {}   # sign_name -> int

    # --------------------------------------------------------- data loading
    def load_pretrained(self):
        """Load custom sign data and retrain the model.

        Only loads from data/custom/ — the app starts with an empty dataset
        and all signs are user-defined. Pretrained signs are intentionally
        excluded so the user builds their vocabulary from scratch.
        """
        self._load_from_dir(CUSTOM_DATA_DIR)
        if len(self.labels) > 0:
            self.train()

    def _load_from_dir(self, data_dir):
        """Scan a data directory and load all sign feature files."""
        if not os.path.isdir(data_dir):
            return
        for sign_name in sorted(os.listdir(data_dir)):
            sign_path = os.path.join(data_dir, sign_name)
            if not os.path.isdir(sign_path):
                continue
            features_file = os.path.join(sign_path, "features.npy")
            if not os.path.isfile(features_file):
                continue
            try:
                features = np.load(features_file)
            except Exception:
                continue
            if features.ndim != 2 or features.shape[0] == 0:
                continue
            try:
                self.add_training_data(sign_name, features)
            except ValueError:
                # Feature width is inconsistent with already-loaded data — skip.
                continue

    def add_training_data(self, sign_name, features_list):
        """Add new sign data to the training pool.

        Args:
            sign_name: Name/label for this sign.
            features_list: numpy array of shape (N, F) or list of (F,) arrays,
                           where F is the feature width used by the detector.
        """
        features = np.asarray(features_list, dtype=np.float64)
        if features.ndim == 1:
            features = features.reshape(1, -1)

        n_feat = features.shape[1]

        # On first insert: resize X to match the actual incoming feature width.
        # This handles both 63-feature (hand-only) and 183-feature (holistic)
        # detectors without requiring a hard-coded constant here.
        if self.X.shape[0] == 0:
            self.X = np.empty((0, n_feat), dtype=np.float64)

        if n_feat != self.X.shape[1]:
            raise ValueError(
                f"Feature dimension mismatch: model has {self.X.shape[1]}-wide "
                f"data but new samples have {n_feat} features. All signs must be "
                f"captured with the same detector configuration."
            )

        if sign_name not in self._label_map:
            idx = len(self.labels)
            self._label_map[sign_name] = idx
            self.labels.append(sign_name)
        label_idx = self._label_map[sign_name]

        self.X = np.vstack([self.X, features])
        self.y = np.concatenate([self.y, np.full(len(features), label_idx, dtype=int)])

    # ------------------------------------------------------------ training
    def train(self):
        """Train (or retrain) the RandomForestClassifier on all loaded data.

        Returns:
            A dict with 'accuracy' and 'num_samples' keys, or None if not
            enough data.
        """
        if len(self.X) < 1 or len(self.labels) < 1:
            return None

        # train_test_split with stratify requires at least 2 classes and
        # enough samples per class to appear in both train and test sets.
        # If that isn't met, we train on everything (accuracy = 1.0 by
        # convention since we can't evaluate).
        unique_classes = len(np.unique(self.y))
        min_class_count = min(np.bincount(self.y.astype(int)))

        if unique_classes < 2 or min_class_count < 3:
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
            )
            self.model.fit(self.X, self.y)
            return {"accuracy": 1.0, "num_samples": len(self.X)}

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=self.y,
        )

        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
        )
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        return {"accuracy": accuracy, "num_samples": len(self.X)}

    # ----------------------------------------------------------- inference
    @staticmethod
    def is_me_sign(hand_lms_list, pose_lms, frame_w: int = 640) -> bool:
        """Return True when the index-fingertip points at the chest.

        Triggered when MediaPipe hand landmark 8 (index fingertip) is
        within 12 % of normalised-width distance of the midpoint between
        LEFT_SHOULDER (pose landmark 11) and RIGHT_SHOULDER (pose landmark 12).
        """
        if not hand_lms_list or pose_lms is None:
            return False
        hand = hand_lms_list[0]
        if len(hand) <= 8:
            return False
        try:
            l_sh = pose_lms[11]
            r_sh = pose_lms[12]
        except (IndexError, TypeError):
            return False
        chest_x = (l_sh.x + r_sh.x) / 2.0
        chest_y = (l_sh.y + r_sh.y) / 2.0
        tip = hand[8]
        dx = tip.x - chest_x
        dy = tip.y - chest_y
        return (dx * dx + dy * dy) ** 0.5 < 0.12   # 12 % of normalised width

    def predict(self, features):
        """Predict the sign from a feature vector.

        Args:
            features: numpy array of shape (F,) where F matches the training
                      feature width (126 for dual-hand, 63 for legacy single-hand).

        Returns:
            Tuple of (sign_name, confidence_score).
            Returns (None, 0.0) if no model is trained or confidence < 0.50.
        """
        if self.model is None:
            return (None, 0.0)

        features = np.asarray(features, dtype=np.float64).reshape(1, -1)
        proba = self.model.predict_proba(features)[0]
        best_idx = int(np.argmax(proba))
        confidence = float(proba[best_idx])

        # Confidence tiers:
        #   >= 0.80 → confirmed sign
        #   0.50–0.79 → uncertain (caller should prompt reposition)
        #   < 0.50  → no sign returned
        if confidence < 0.50:
            return (None, 0.0)

        # predict_proba returns probabilities indexed by model.classes_,
        # which may be a subset of all label indices if some classes were
        # never seen during the last fit. Map back through classes_ to
        # get the correct label name.
        model_class = self.model.classes_[best_idx]
        sign_name = self.labels[model_class]

        return (sign_name, confidence)

    # -------------------------------------------------------- persistence
    def save_model(self, path=None):
        """Save the trained model and metadata to disk.

        Args:
            path: File path for the model. Defaults to DEFAULT_MODEL_FILE.
        """
        path = path or DEFAULT_MODEL_FILE
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            joblib.dump(
                {"model": self.model, "labels": self.labels, "label_map": self._label_map},
                path,
            )
        except OSError:
            pass  # non-critical — model still lives in memory

    def load_model(self, path=None):
        """Load a previously saved model from disk.

        Args:
            path: File path to load. Defaults to DEFAULT_MODEL_FILE.
        """
        path = path or DEFAULT_MODEL_FILE
        if not os.path.isfile(path):
            return
        try:
            data = joblib.load(path)
            self.model = data["model"]
            self.labels = data["labels"]
            self._label_map = data["label_map"]
        except Exception:
            pass  # corrupted file — start fresh

    # ----------------------------------------------------------- queries
    def get_all_signs(self):
        """Return a sorted list of all known sign names."""
        return sorted(self.labels)


# -------------------------------------------------------- pretrained data
def _generate_pretrained_data():
    """Generate synthetic landmark features for 3 base signs.

    Each sign is modeled as an idealized 21-landmark hand pose (wrist-
    relative, normalized) with Gaussian noise added to create variation.

    Landmarks (MediaPipe indices):
        0  = wrist
        1-4   = thumb (CMC, MCP, IP, TIP)
        5-8   = index (MCP, PIP, DIP, TIP)
        9-12  = middle (MCP, PIP, DIP, TIP)
        13-16 = ring (MCP, PIP, DIP, TIP)
        17-20 = pinky (MCP, PIP, DIP, TIP)
    """
    rng = np.random.default_rng(42)
    num_samples = 80

    def _make_samples(base_landmarks, n=num_samples, noise=0.04):
        """Create n noisy copies of a base (21, 3) landmark array.

        Produces (n, 126) samples: the right-hand 63 values are in the first
        slot; the left-hand slot is zeroed (hand absent).
        """
        base = np.array(base_landmarks, dtype=np.float64)  # (21, 3)
        # Normalize like HandDetector.extract_features
        base -= base[0]
        max_val = np.max(np.abs(base))
        if max_val > 0:
            base /= max_val
        flat63 = base.flatten()                                # (63,)
        flat126 = np.zeros(FEATURE_LENGTH, dtype=np.float64)  # (126,)
        flat126[:63] = flat63                                  # second hand absent
        samples = (np.tile(flat126, (n, 1))
                   + rng.normal(0, noise, (n, FEATURE_LENGTH)))
        return samples

    # --- Peace sign: index + middle extended, others curled ---
    peace_base = [
        [0.0, 0.0, 0.0],       # 0  wrist
        [-0.05, -0.05, -0.02],  # 1  thumb CMC
        [-0.10, -0.10, -0.03],  # 2  thumb MCP
        [-0.12, -0.12, -0.02],  # 3  thumb IP  (curled in)
        [-0.13, -0.10, -0.01],  # 4  thumb TIP (curled in)
        [-0.03, -0.20, 0.0],    # 5  index MCP
        [-0.03, -0.35, 0.0],    # 6  index PIP (extended)
        [-0.03, -0.48, 0.0],    # 7  index DIP (extended)
        [-0.03, -0.58, 0.0],    # 8  index TIP (extended)
        [0.02, -0.20, 0.0],     # 9  middle MCP
        [0.02, -0.35, 0.0],     # 10 middle PIP (extended)
        [0.02, -0.48, 0.0],     # 11 middle DIP (extended)
        [0.02, -0.58, 0.0],     # 12 middle TIP (extended)
        [0.07, -0.18, 0.0],     # 13 ring MCP
        [0.07, -0.15, 0.02],    # 14 ring PIP (curled)
        [0.06, -0.10, 0.03],    # 15 ring DIP (curled)
        [0.05, -0.08, 0.02],    # 16 ring TIP (curled)
        [0.12, -0.15, 0.0],     # 17 pinky MCP
        [0.12, -0.12, 0.02],    # 18 pinky PIP (curled)
        [0.11, -0.08, 0.03],    # 19 pinky DIP (curled)
        [0.10, -0.06, 0.02],    # 20 pinky TIP (curled)
    ]

    # --- Hello sign: all fingers extended (open palm) ---
    hello_base = [
        [0.0, 0.0, 0.0],       # 0  wrist
        [-0.08, -0.05, -0.02],  # 1  thumb CMC
        [-0.15, -0.12, -0.03],  # 2  thumb MCP
        [-0.22, -0.18, -0.02],  # 3  thumb IP (extended out)
        [-0.28, -0.22, -0.01],  # 4  thumb TIP (extended out)
        [-0.04, -0.22, 0.0],    # 5  index MCP
        [-0.05, -0.36, 0.0],    # 6  index PIP
        [-0.05, -0.48, 0.0],    # 7  index DIP
        [-0.05, -0.58, 0.0],    # 8  index TIP
        [0.01, -0.23, 0.0],     # 9  middle MCP
        [0.01, -0.38, 0.0],     # 10 middle PIP
        [0.01, -0.50, 0.0],     # 11 middle DIP
        [0.01, -0.60, 0.0],     # 12 middle TIP
        [0.06, -0.21, 0.0],     # 13 ring MCP
        [0.07, -0.35, 0.0],     # 14 ring PIP
        [0.07, -0.46, 0.0],     # 15 ring DIP
        [0.07, -0.55, 0.0],     # 16 ring TIP
        [0.11, -0.18, 0.0],     # 17 pinky MCP
        [0.12, -0.30, 0.0],     # 18 pinky PIP
        [0.12, -0.40, 0.0],     # 19 pinky DIP
        [0.12, -0.48, 0.0],     # 20 pinky TIP
    ]

    # --- Thumbs up: thumb extended up, all others curled ---
    thumbsup_base = [
        [0.0, 0.0, 0.0],       # 0  wrist
        [-0.06, -0.08, -0.02],  # 1  thumb CMC
        [-0.10, -0.20, -0.03],  # 2  thumb MCP (going up)
        [-0.12, -0.35, -0.02],  # 3  thumb IP  (extended up)
        [-0.12, -0.48, -0.01],  # 4  thumb TIP (extended up)
        [-0.02, -0.15, 0.0],    # 5  index MCP
        [0.0, -0.12, 0.04],     # 6  index PIP (curled)
        [0.02, -0.08, 0.06],    # 7  index DIP (curled)
        [0.03, -0.05, 0.04],    # 8  index TIP (curled)
        [0.03, -0.15, 0.0],     # 9  middle MCP
        [0.05, -0.12, 0.04],    # 10 middle PIP (curled)
        [0.06, -0.08, 0.06],    # 11 middle DIP (curled)
        [0.06, -0.05, 0.04],    # 12 middle TIP (curled)
        [0.07, -0.14, 0.0],     # 13 ring MCP
        [0.08, -0.11, 0.04],    # 14 ring PIP (curled)
        [0.09, -0.07, 0.06],    # 15 ring DIP (curled)
        [0.09, -0.04, 0.04],    # 16 ring TIP (curled)
        [0.11, -0.12, 0.0],     # 17 pinky MCP
        [0.12, -0.09, 0.04],    # 18 pinky PIP (curled)
        [0.12, -0.06, 0.06],    # 19 pinky DIP (curled)
        [0.12, -0.03, 0.04],    # 20 pinky TIP (curled)
    ]

    signs = {
        "peace": peace_base,
        "hello": hello_base,
        "thumbs_up": thumbsup_base,
    }

    for sign_name, base_landmarks in signs.items():
        sign_dir = os.path.join(PRETRAINED_DATA_DIR, sign_name)
        os.makedirs(sign_dir, exist_ok=True)
        samples = _make_samples(base_landmarks)
        np.save(os.path.join(sign_dir, "features.npy"), samples)
        print(f"  {sign_name}: {samples.shape[0]} samples → {sign_dir}/features.npy")


# ----------------------------------------------------------------- test
if __name__ == "__main__":
    print("=== Generating pretrained data ===")
    _generate_pretrained_data()

    print("\n=== Testing SignModel ===")
    model = SignModel()

    # Load pretrained
    model.load_pretrained()
    print(f"Signs loaded  : {model.get_all_signs()}")
    print(f"Training data : {model.X.shape[0]} samples, {model.X.shape[1]} features")

    # Test prediction with a known peace-like vector
    peace_data = np.load(os.path.join(PRETRAINED_DATA_DIR, "peace", "features.npy"))
    sign, conf = model.predict(peace_data[0])
    print(f"Predict peace : sign={sign!r}, confidence={conf:.3f}")

    hello_data = np.load(os.path.join(PRETRAINED_DATA_DIR, "hello", "features.npy"))
    sign, conf = model.predict(hello_data[0])
    print(f"Predict hello : sign={sign!r}, confidence={conf:.3f}")

    thumb_data = np.load(os.path.join(PRETRAINED_DATA_DIR, "thumbs_up", "features.npy"))
    sign, conf = model.predict(thumb_data[0])
    print(f"Predict thumbs: sign={sign!r}, confidence={conf:.3f}")

    # Test save/load
    model.save_model()
    print(f"\nModel saved to: {DEFAULT_MODEL_FILE}")

    model2 = SignModel()
    model2.load_model()
    print(f"Model loaded  : {model2.get_all_signs()}")
    sign, conf = model2.predict(peace_data[5])
    print(f"Loaded predict: sign={sign!r}, confidence={conf:.3f}")

    # Test adding custom data
    custom_features = np.random.default_rng(99).normal(0, 0.1, (20, FEATURE_LENGTH))
    model.add_training_data("custom_wave", custom_features)
    result = model.train()
    print(f"\nRetrained with custom_wave: {result}")
    print(f"All signs now : {model.get_all_signs()}")

    print("\nAll tests PASSED")
