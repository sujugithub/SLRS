"""
Temporal dynamic sign classifier — scikit-learn MLP backend.

Replaces the Keras/TensorFlow LSTM with a scikit-learn MLPClassifier trained on
hand-crafted temporal features extracted from holistic gesture sequences.  The
external interface is identical to the original Keras version so the rest of the
application (training worker, prediction screen, main window) requires no changes.

Feature engineering
-------------------
For each sequence of shape ``(seq_len, feature_len)`` four statistical descriptors
are computed along the time axis and concatenated:

  ┌──────────────────────────────┬──────────────────┐
  │ descriptor                   │ shape            │
  ├──────────────────────────────┼──────────────────┤
  │ mean   (temporal average)    │ (feature_len,)   │
  │ std    (temporal variance)   │ (feature_len,)   │
  │ disp   (last − first frame)  │ (feature_len,)   │
  │ vel    (mean |Δ| over time)  │ (feature_len,)   │
  └──────────────────────────────┴──────────────────┘

  Total input to MLP: feature_len × 4  (e.g. 183 × 4 = 732 floats)

Architecture
------------
  MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation="relu",
                solver="adam", max_iter=400, early_stopping=True,
                n_iter_no_change=20)

Persistence
-----------
  Model:  joblib-serialised ``MLPClassifier``  saved to ``model_path``
  Labels: JSON metadata file saved to ``labels_path``
  (Same file-path conventions as the original Keras version.)
"""

from __future__ import annotations

import json
import os

import joblib
import numpy as np


_DEFAULT_SEQ_LEN     = 30
_DEFAULT_FEATURE_LEN = 183   # HOLISTIC_FEATURE_LENGTH


def _extract_temporal_features(sequence: np.ndarray) -> np.ndarray:
    """Convert a (seq_len, F) array into a flat (4*F,) feature vector."""
    seq = np.asarray(sequence, dtype=np.float32)   # (T, F)
    mean = seq.mean(axis=0)
    std  = seq.std(axis=0)
    disp = seq[-1] - seq[0]
    vel  = np.abs(np.diff(seq, axis=0)).mean(axis=0)
    return np.concatenate([mean, std, disp, vel])  # (4F,)


class LSTMSignModel:
    """Train and run inference with a temporal gesture classifier.

    Drop-in replacement for the Keras-based version — identical public API.
    """

    def __init__(self, seq_len: int = _DEFAULT_SEQ_LEN,
                 feature_len: int = _DEFAULT_FEATURE_LEN):
        self.seq_len     = seq_len
        self.feature_len = feature_len

        self.labels:     list = []
        self._label_map: dict = {}
        self._model           = None   # sklearn MLPClassifier (None until trained/loaded)

    # ------------------------------------------------------------------ state

    def is_trained(self) -> bool:
        return self._model is not None and bool(self.labels)

    def get_all_signs(self) -> list:
        return sorted(self.labels)

    # ---------------------------------------------------------------- training

    def train(self, sequences_by_sign: dict) -> dict:
        """Fit the MLP on collected gesture sequences.

        Args:
            sequences_by_sign: ``{sign_name: ndarray(N, seq_len, feature_len)}``
                A single sample ``ndarray(seq_len, F)`` is promoted to (1, T, F).

        Returns:
            dict with keys ``accuracy``, ``num_sequences``, ``epochs_run``.
        """
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        self.labels     = sorted(sequences_by_sign.keys())
        self._label_map = {name: i for i, name in enumerate(self.labels)}

        X_parts: list = []
        y_parts: list = []
        for sign_name, seqs in sequences_by_sign.items():
            seqs = np.asarray(seqs, dtype=np.float32)
            if seqs.ndim == 2:
                seqs = seqs[np.newaxis]           # (1, T, F)
            for seq in seqs:
                X_parts.append(_extract_temporal_features(seq))
                y_parts.append(self._label_map[sign_name])

        X = np.stack(X_parts, axis=0)            # (N, 4F)
        y = np.array(y_parts)                    # (N,)

        # early_stopping needs enough samples to form a stratified val split
        # (at least 1 sample per class in the val set at validation_fraction=0.1)
        use_early_stopping = len(X) >= len(self.labels) * 10
        mlp = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            solver="adam",
            max_iter=400,
            early_stopping=use_early_stopping,
            n_iter_no_change=20,
            random_state=42,
            verbose=False,
        )
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp",    mlp),
        ])
        pipeline.fit(X, y)

        preds   = pipeline.predict(X)
        acc     = float((preds == y).mean())
        epochs  = int(pipeline.named_steps["mlp"].n_iter_)

        self._model = pipeline
        return {
            "accuracy":      acc,
            "num_sequences": len(X),
            "epochs_run":    epochs,
        }

    # --------------------------------------------------------------- inference

    def predict(self, sequence: np.ndarray):
        """Classify a single gesture sequence.

        Args:
            sequence: Array of shape ``(seq_len, feature_len)``.

        Returns:
            ``(sign_name, confidence)`` or ``(None, 0.0)`` if not yet trained.
        """
        if self._model is None or not self.labels:
            return (None, 0.0)

        feats  = _extract_temporal_features(sequence).reshape(1, -1)
        proba  = self._model.predict_proba(feats)[0]
        best   = int(np.argmax(proba))
        return (self.labels[best], float(proba[best]))

    # --------------------------------------------------------------- persistence

    def save(self, model_path: str, labels_path: str | None = None) -> None:
        if self._model is None:
            return

        dir_ = os.path.dirname(model_path)
        if dir_:
            os.makedirs(dir_, exist_ok=True)

        joblib.dump(self._model, model_path)

        if labels_path is None:
            labels_path = model_path + ".labels.json"

        with open(labels_path, "w") as f:
            json.dump({
                "labels":      self.labels,
                "label_map":   self._label_map,
                "seq_len":     self.seq_len,
                "feature_len": self.feature_len,
            }, f, indent=2)

    def load(self, model_path: str, labels_path: str | None = None) -> bool:
        if labels_path is None:
            labels_path = model_path + ".labels.json"

        if not os.path.isfile(model_path) or not os.path.isfile(labels_path):
            return False

        try:
            self._model = joblib.load(model_path)
            with open(labels_path) as f:
                data = json.load(f)
            self.labels      = data["labels"]
            self._label_map  = data["label_map"]
            self.seq_len     = data.get("seq_len",     self.seq_len)
            self.feature_len = data.get("feature_len", self.feature_len)
            return True
        except Exception:
            return False
