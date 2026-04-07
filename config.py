"""
Application configuration: paths, constants, and default settings
for the Sign Language Recognition App.
"""

import os
import sys
import shutil


# ── Path helpers ───────────────────────────────────────────────────────────────

def resource_path(relative: str) -> str:
    """Absolute path to a bundled read-only resource.

    In development : resolved relative to this config.py file.
    When frozen    : resolved relative to sys._MEIPASS (PyInstaller temp dir).
    """
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, relative)


def user_data_path(relative: str = "") -> str:
    """Writable per-user data directory for SLRS.

    macOS  : ~/Library/Application Support/SLRS/
    Windows: %APPDATA%/SLRS/
    Linux  : ~/.local/share/SLRS/
    """
    if sys.platform == "darwin":
        base = os.path.join(
            os.path.expanduser("~"), "Library", "Application Support", "SLRS"
        )
    elif sys.platform == "win32":
        base = os.path.join(
            os.environ.get("APPDATA", os.path.expanduser("~")), "SLRS"
        )
    else:
        base = os.path.join(os.path.expanduser("~"), ".local", "share", "SLRS")
    return os.path.join(base, relative) if relative else base


_FROZEN = getattr(sys, "frozen", False)


def _seed_user_data() -> None:
    """Copy bundled training data into the writable user directory on first launch.

    Only runs when the app is frozen (packaged).  A sentinel file prevents
    re-seeding on every subsequent launch.
    """
    user_root = user_data_path()
    sentinel  = os.path.join(user_root, ".initialized")
    if os.path.exists(sentinel):
        return

    os.makedirs(user_root, exist_ok=True)

    # Seed data/ (pretrained signs, custom signs, sequences)
    bundle_data = resource_path("data")
    user_data   = user_data_path("data")
    if os.path.isdir(bundle_data):
        shutil.copytree(bundle_data, user_data, dirs_exist_ok=True)

    # Seed any pre-built model files (.pkl / .keras)
    bundle_models = resource_path("models")
    user_models   = user_data_path("models")
    os.makedirs(user_models, exist_ok=True)
    for fname in (
        "sign_language_model.pkl",
        "lstm_sign_model.keras",
        "lstm_sign_model.keras.labels.json",
    ):
        src = os.path.join(bundle_models, fname)
        dst = os.path.join(user_models, fname)
        if os.path.isfile(src) and not os.path.isfile(dst):
            shutil.copy2(src, dst)

    # Write sentinel so we only seed once
    with open(sentinel, "w") as fh:
        fh.write("1")


# Seed on first frozen launch (no-op in development)
if _FROZEN:
    _seed_user_data()


# ── Read-only bundle paths (MediaPipe .task files) ─────────────────────────────
# resource_path() → sys._MEIPASS/models  when frozen
#                 → BASE_DIR/models      in development
MODELS_DIR = resource_path("models")

# ── Writable data / model paths ────────────────────────────────────────────────
# Frozen : ~/Library/Application Support/SLRS/...  (read-write, persists between runs)
# Dev    : BASE_DIR/data  and  BASE_DIR/models     (same as original behaviour)
if _FROZEN:
    _data_base   = user_data_path("data")
    _models_base = user_data_path("models")
else:
    _base        = os.path.dirname(os.path.abspath(__file__))
    _data_base   = os.path.join(_base, "data")
    _models_base = os.path.join(_base, "models")

DATA_DIR            = _data_base
PRETRAINED_DATA_DIR = os.path.join(_data_base, "pretrained")
CUSTOM_DATA_DIR     = os.path.join(_data_base, "custom")
SEQUENCE_DATA_DIR   = os.path.join(_data_base, "sequences")

DEFAULT_MODEL_FILE  = os.path.join(_models_base, "sign_language_model.pkl")
LSTM_MODEL_FILE     = os.path.join(_models_base, "lstm_sign_model.keras")
LSTM_LABELS_FILE    = os.path.join(_models_base, "lstm_sign_model.keras.labels.json")

# ── User-editable config files (settings, phrases, training metadata) ────────
SETTINGS_FILE       = os.path.join(_data_base, "settings.json")
PHRASES_FILE        = os.path.join(_data_base, "phrases.json")
TRAINING_META_FILE  = os.path.join(_data_base, "training_meta.json")

# ── Camera settings ────────────────────────────────────────────────────────────
CAMERA_INDEX = 2
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# ── MediaPipe hand detection settings ─────────────────────────────────────────
MAX_NUM_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5

# ── Training settings ─────────────────────────────────────────────────────────
NUM_CLASSES = 26          # A-Z alphabet signs
SAMPLES_PER_CLASS = 100
RANDOM_STATE = 42

# Combined feature vector: both hands (126) + upper-body pose (18) + essential face (27)
HOLISTIC_FEATURE_LENGTH = 171

# ── Dynamic (LSTM) settings ────────────────────────────────────────────────────
SEQ_LENGTH = 30                # consecutive frames per gesture window
MIN_SEQUENCES_PER_SIGN = 5     # minimum sequences before LSTM training is allowed

# ── GUI settings ───────────────────────────────────────────────────────────────
APP_TITLE = "Sign Language Recognition App"
WINDOW_WIDTH = 900
WINDOW_HEIGHT = 700
