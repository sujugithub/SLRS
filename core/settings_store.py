"""
Settings store for SignBridge.

Loads and saves user-editable runtime settings to ``data/settings.json``.
The file is read once at app startup and written whenever the user clicks
"Save" in the Settings dialog.

All callers should go through ``load_settings()`` / ``save_settings()`` —
do not read the file directly.
"""
from __future__ import annotations

import json
import os
import tempfile

from config import SETTINGS_FILE


# Single source of truth for the schema. New keys can be added here without
# breaking existing settings.json files — load_settings() merges over these.
DEFAULTS: dict = {
    "confidence_threshold": 0.6,    # float in [0.0, 1.0]
    "tts_enabled":          True,   # bool
    "tts_voice":            "",     # str — empty = system default
    "camera_index":         2,      # int
    "mediapipe_complexity": 0,      # 0=Light, 1=Full, 2=Heavy
    "smoothing_window":     5,      # int 1..15
}


def settings_path() -> str:
    """Return the absolute path to settings.json."""
    return SETTINGS_FILE


def load_settings() -> dict:
    """Load settings from disk, merging missing keys from DEFAULTS.

    Returns a fresh dict — callers may mutate it freely.
    """
    merged = dict(DEFAULTS)
    try:
        if os.path.isfile(SETTINGS_FILE):
            with open(SETTINGS_FILE, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                for k, v in data.items():
                    if k in DEFAULTS:
                        merged[k] = v
    except Exception as exc:
        print(f"[settings_store] Failed to read settings.json: {exc}")
    return merged


def save_settings(settings: dict) -> None:
    """Atomically write settings.json. Unknown keys are dropped."""
    cleaned = {k: settings.get(k, DEFAULTS[k]) for k in DEFAULTS}
    try:
        os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
        # Atomic write: tempfile in the same dir + os.replace
        fd, tmp_path = tempfile.mkstemp(
            prefix=".settings_", suffix=".json",
            dir=os.path.dirname(SETTINGS_FILE),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(cleaned, fh, indent=2)
            os.replace(tmp_path, SETTINGS_FILE)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
    except Exception as exc:
        print(f"[settings_store] Failed to write settings.json: {exc}")
