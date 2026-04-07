"""
Rolling-window frame buffer for temporal gesture sequence collection.

Keeps the last *seq_len* holistic feature vectors in a circular deque.
Once the buffer is full, ``is_ready()`` returns True and a (seq_len, F)
numpy array can be retrieved for LSTM inference or sequence saving.

Usage::

    col = SequenceCollector(seq_len=30)
    # inside the camera loop:
    col.add(holistic_features)          # shape (183,)
    if col.is_ready():
        seq = col.get_sequence()        # shape (30, 183)
        sign, conf = lstm_model.predict(seq)
"""

from collections import deque
import numpy as np


class SequenceCollector:
    """Fixed-length circular buffer of feature vectors."""

    def __init__(self, seq_len: int = 30):
        """
        Args:
            seq_len: Number of frames to hold.  The deque is automatically
                     capped — oldest frames drop off as new ones arrive.
        """
        self.seq_len = seq_len
        self._buffer: deque = deque(maxlen=seq_len)

    # ------------------------------------------------------------------ writes

    def add(self, feature_vector) -> None:
        """Append one frame's feature vector to the rolling window.

        Args:
            feature_vector: 1-D array-like of shape (F,).
        """
        self._buffer.append(np.asarray(feature_vector, dtype=np.float32))

    def clear(self) -> None:
        """Discard all buffered frames."""
        self._buffer.clear()

    # ------------------------------------------------------------------ reads

    def is_ready(self) -> bool:
        """Return True when exactly *seq_len* frames have been collected."""
        return len(self._buffer) == self.seq_len

    def fill_ratio(self) -> float:
        """Fraction of the buffer filled (0.0 – 1.0)."""
        return len(self._buffer) / self.seq_len

    def get_sequence(self) -> np.ndarray:
        """Return the current window as a ``(seq_len, F)`` float32 array.

        Call only after ``is_ready()`` returns True; otherwise the leading
        rows of the returned array will be zero-padded.
        """
        if len(self._buffer) == self.seq_len:
            return np.array(self._buffer, dtype=np.float32)

        # Partial buffer — zero-pad at the front
        pad_len = self.seq_len - len(self._buffer)
        feat_len = self._buffer[0].shape[0] if self._buffer else 1
        pad = np.zeros((pad_len, feat_len), dtype=np.float32)
        data = np.array(self._buffer, dtype=np.float32) if self._buffer else pad
        return np.concatenate([pad, data], axis=0)

    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:  # pragma: no cover
        return f"SequenceCollector(seq_len={self.seq_len}, filled={len(self._buffer)})"
