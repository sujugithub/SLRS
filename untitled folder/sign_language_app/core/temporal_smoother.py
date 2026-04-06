"""
Temporal smoothing for real-time sign predictions.

The core problem with single-frame Random Forest prediction:
  - A 30 fps camera produces 30 predictions/second
  - Any individual frame may misclassify (lighting, motion blur)
  - This causes flickering and false word additions

This module provides two complementary strategies:

  TemporalSmoother   — sliding-window majority vote (primary filter)
  GestureDebouncer   — hold-and-commit FSM (auto-add controller)

USAGE
─────
    smoother  = TemporalSmoother(window=15)
    debouncer = GestureDebouncer(hold_frames=40, cooldown_frames=55)

    # In your camera loop (called every frame):
    smoother.update(raw_sign, raw_conf)
    stable_sign, stable_conf = smoother.best()

    event = debouncer.update(stable_sign, stable_conf)
    if event == GestureEvent.COMMIT:
        sentence_buffer.add_word(debouncer.committed_sign)
"""

from collections import deque
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional


# ─── TemporalSmoother ─────────────────────────────────────────────────────────

class TemporalSmoother:
    """
    Confidence-weighted majority vote over a sliding window of frames.

    Instead of trusting every raw prediction, accumulates the last
    ``window`` frames and returns the sign that holds a dominant
    vote share, weighted by per-frame confidence.

    Example::

        smoother = TemporalSmoother(window=15, min_vote_share=0.55)
        for sign, conf in live_predictions:
            smoother.update(sign, conf)
            best_sign, best_conf = smoother.best()
    """

    def __init__(self, window: int = 15, min_vote_share: float = 0.55):
        """
        Args:
            window:         Rolling window size in frames.
            min_vote_share: Weighted vote fraction required to confirm
                            a sign (0.55 = 55% of weighted votes).
                            Higher = less sensitive but more stable.
        """
        self._window    = window
        self._threshold = min_vote_share
        self._buf: deque = deque(maxlen=window)

    # ── public interface ──────────────────────────────────────────────────
    def update(self, sign: Optional[str], confidence: float) -> None:
        """Push one frame's prediction into the window."""
        self._buf.append((sign, max(0.0, float(confidence))))

    def best(self) -> tuple[Optional[str], float]:
        """
        Return (winning_sign, mean_confidence) if a sign clears the
        vote threshold, otherwise (None, 0.0).
        """
        if not self._buf:
            return (None, 0.0)

        # Accumulate confidence mass per sign
        votes: dict[str, float] = {}
        total_conf = 0.0
        for sign, conf in self._buf:
            if sign is not None:
                votes[sign] = votes.get(sign, 0.0) + conf
            total_conf += conf

        if not votes or total_conf < 1e-9:
            return (None, 0.0)

        winner = max(votes, key=votes.__getitem__)
        share  = votes[winner] / total_conf

        if share < self._threshold:
            return (None, 0.0)

        # Mean confidence for the winning sign's frames only
        winner_confs = [c for s, c in self._buf if s == winner]
        mean_conf    = sum(winner_confs) / len(winner_confs)
        return (winner, mean_conf)

    def is_full(self) -> bool:
        """True once the window has been populated with ``window`` frames."""
        return len(self._buf) == self._window

    def reset(self) -> None:
        """Flush all buffered predictions."""
        self._buf.clear()

    @property
    def dominant_sign(self) -> Optional[str]:
        """Convenience: winning sign name only, or None."""
        return self.best()[0]


# ─── GestureDebouncer ─────────────────────────────────────────────────────────

class GestureEvent(Enum):
    """Events emitted by GestureDebouncer.update()."""
    NONE     = auto()   # No action this frame
    PROGRESS = auto()   # Hold in progress — update the progress bar
    COMMIT   = auto()   # Sign held long enough — add to sentence
    COOLDOWN = auto()   # Post-commit cooldown active


@dataclass
class GestureDebouncer:
    """
    Finite-state machine that converts a stable stream of sign predictions
    into explicit COMMIT events.

    Flow::

        [IDLE] ──sign appears──► [HOLDING] ──hold_frames reached──► [COMMIT]
                                      │                                  │
                                   sign changes                    cooldown_frames
                                      │                                  │
                                    reset                            [COOLDOWN]
                                                                         │
                                                               cooldown expires
                                                                         │
                                                                      [IDLE]

    Example::

        deb = GestureDebouncer(hold_frames=40, cooldown_frames=55)
        ev  = deb.update("hello", 0.91)
        if ev == GestureEvent.COMMIT:
            add_word(deb.committed_sign)
        elif ev == GestureEvent.PROGRESS:
            draw_progress_bar(deb.hold_progress)
    """

    hold_frames:     int   = 40    # frames to hold before committing (~1.3 s @ 30 fps)
    cooldown_frames: int   = 55    # frames to wait after commit     (~1.8 s @ 30 fps)
    min_confidence:  float = 0.50  # minimum confidence to start tracking

    # ── internal state (not constructor args) ─────────────────────────────
    _stable_sign:   Optional[str] = field(default=None, init=False, repr=False)
    _stable_count:  int           = field(default=0,    init=False, repr=False)
    _cooldown_left: int           = field(default=0,    init=False, repr=False)
    committed_sign: Optional[str] = field(default=None, init=False, repr=False)

    def update(self, sign: Optional[str], confidence: float) -> GestureEvent:
        """
        Feed one frame's smoothed prediction.

        Args:
            sign:       The stable sign name (from TemporalSmoother), or None.
            confidence: Associated confidence in [0, 1].

        Returns:
            GestureEvent indicating what happened this frame.
        """
        # ── cooldown phase ────────────────────────────────────────────────
        if self._cooldown_left > 0:
            self._cooldown_left -= 1
            self._reset_hold()
            return GestureEvent.COOLDOWN

        # ── no sign or low confidence ─────────────────────────────────────
        if sign is None or confidence < self.min_confidence:
            self._reset_hold()
            return GestureEvent.NONE

        # ── accumulate hold ───────────────────────────────────────────────
        if sign == self._stable_sign:
            self._stable_count += 1
        else:
            self._stable_sign  = sign
            self._stable_count = 1

        # ── commit? ───────────────────────────────────────────────────────
        if self._stable_count >= self.hold_frames:
            self.committed_sign  = self._stable_sign
            self._cooldown_left  = self.cooldown_frames
            self._reset_hold()
            return GestureEvent.COMMIT

        return GestureEvent.PROGRESS

    # ── public helpers ────────────────────────────────────────────────────
    @property
    def hold_progress(self) -> float:
        """Progress toward commit in [0.0, 1.0]."""
        if self._stable_sign is None or self.hold_frames == 0:
            return 0.0
        return min(1.0, self._stable_count / self.hold_frames)

    @property
    def tracking_sign(self) -> Optional[str]:
        """The sign currently being held, or None."""
        return self._stable_sign

    @property
    def in_cooldown(self) -> bool:
        return self._cooldown_left > 0

    def force_commit(self, sign: str) -> None:
        """Manually commit a sign (for the 'Add Word' button)."""
        self.committed_sign = sign
        self._cooldown_left = self.cooldown_frames
        self._reset_hold()

    def reset(self) -> None:
        """Full reset — clears hold, cooldown, and last committed sign."""
        self._reset_hold()
        self._cooldown_left = 0
        self.committed_sign = None

    # ── private ───────────────────────────────────────────────────────────
    def _reset_hold(self) -> None:
        self._stable_sign  = None
        self._stable_count = 0
