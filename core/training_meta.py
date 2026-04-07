"""
Training metadata store for SignBridge.

Reads and writes ``data/training_meta.json``, which records the most
recent retrain summary:

    {
        "trained_at":       "2026-04-07T15:30:21",
        "overall_accuracy": 0.94,
        "num_samples":      920,
        "per_sign": {
            "hello":     {"accuracy": 0.95, "support": 60},
            "thank_you": {"accuracy": 0.92, "support": 45}
        }
    }

Written by ``RetrainWorker`` after a successful train; read by the
View dialog to display per-sign accuracy chips.
"""
from __future__ import annotations

import json
import os
import tempfile

from config import TRAINING_META_FILE


def meta_path() -> str:
    return TRAINING_META_FILE


def load_meta() -> dict:
    """Read training_meta.json. Returns {} if missing or malformed."""
    try:
        if not os.path.isfile(TRAINING_META_FILE):
            return {}
        with open(TRAINING_META_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            return {}
        return data
    except Exception as exc:
        print(f"[training_meta] Failed to read training_meta.json: {exc}")
        return {}


def save_meta(meta: dict) -> None:
    """Atomically write training_meta.json."""
    try:
        os.makedirs(os.path.dirname(TRAINING_META_FILE), exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=".training_meta_", suffix=".json",
            dir=os.path.dirname(TRAINING_META_FILE),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(meta, fh, indent=2)
            os.replace(tmp_path, TRAINING_META_FILE)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
    except Exception as exc:
        print(f"[training_meta] Failed to write training_meta.json: {exc}")
