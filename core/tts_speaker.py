"""
Text-to-speech speaker for the Sign Language Recognition App.

Design goals
────────────
1. Non-blocking  — speech runs on a background thread; the UI never freezes.
2. Sentence-level — per-sign announcements are optional; sentences are primary.
3. Quality        — uses macOS AVFoundation (`say` command) when available for
                    noticeably better voice quality; falls back to pyttsx3.
4. Smart dedup    — won't repeat the same sign announcement on every frame;
                    will always speak a requested sentence even if identical.

Architecture
────────────
  Main thread  ──put()──►  priority queue  ──►  TTS thread  ──►  speaker
                                                    │
                              (SENTENCE items jump ahead of SIGN items)
"""

import queue
import subprocess
import platform
import threading
from dataclasses import dataclass
from enum import IntEnum


# ─── Priority levels (lower = higher priority) ────────────────────────────────
class _P(IntEnum):
    SENTENCE = 0   # user-requested sentence — always spoken first
    SIGN     = 1   # per-sign announcement   — skipped if stale


@dataclass(order=True)
class _Item:
    priority: int
    text: str = ""   # field excluded from ordering
    counter: int = 0  # tie-break: later items lose


class TTSSpeaker:
    """
    Non-blocking, sentence-aware text-to-speech engine.

    Usage::

        speaker = TTSSpeaker()
        speaker.say("hello")           # announce detected sign (de-duped)
        speaker.speak_sentence("Hello, how are you?")  # always spoken
        speaker.stop()                 # clean shutdown
    """

    # Rate and voice tuned for clarity in a noisy environment
    _RATE_SLOW    = 145   # sentences (easier to understand)
    _RATE_FAST    = 175   # sign announcements (brief)
    _MACOS_VOICE  = "Samantha"   # best built-in macOS voice for English

    def __init__(self):
        self._pq: queue.PriorityQueue = queue.PriorityQueue()
        self._last_sign: str | None   = None
        self._enabled: bool           = True
        self._counter: int            = 0     # monotonic sequence for tie-break
        self._use_macos: bool         = (platform.system() == "Darwin"
                                         and self._macos_say_available())
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # ─── Public API ───────────────────────────────────────────────────────────
    def say(self, sign_name: str) -> None:
        """
        Announce a detected sign — spoken only when it changes.
        Stale queued announcements are replaced (not stacked).

        Args:
            sign_name: Display name of the detected sign (e.g. "Hello").
        """
        if not self._enabled or sign_name == self._last_sign:
            return
        self._last_sign = sign_name
        self._flush_signs()   # drop stale sign items
        self._enqueue(sign_name, _P.SIGN)

    def speak_sentence(self, sentence: str) -> None:
        """
        Speak a complete sentence — always queued, even if repeated.
        Sentence items jump ahead of any pending sign announcements.

        Args:
            sentence: The NLP-corrected sentence to read aloud.
        """
        if not self._enabled or not sentence:
            return
        self._flush_signs()                # clear pending sign noise
        self._last_sign = sentence         # prevent immediate re-announcement
        self._enqueue(sentence, _P.SENTENCE)

    def reset(self) -> None:
        """Clear de-dup memory so the next sign is always announced."""
        self._last_sign = None

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable all speech output."""
        self._enabled = enabled
        if not enabled:
            self._last_sign = None
            self._flush_all()

    def stop(self) -> None:
        """Signal the background thread to exit cleanly."""
        self._pq.put(_Item(priority=-1, text="__STOP__", counter=-1))

    # ─── Internal helpers ─────────────────────────────────────────────────────
    def _enqueue(self, text: str, priority: _P) -> None:
        self._counter += 1
        self._pq.put(_Item(priority=int(priority), text=text,
                           counter=self._counter))

    def _flush_signs(self) -> None:
        """Remove all pending SIGN-priority items from the queue."""
        surviving = []
        while True:
            try:
                item = self._pq.get_nowait()
                if item.priority != int(_P.SIGN):
                    surviving.append(item)
            except queue.Empty:
                break
        for item in surviving:
            self._pq.put(item)

    def _flush_all(self) -> None:
        while True:
            try:
                self._pq.get_nowait()
            except queue.Empty:
                break

    @staticmethod
    def _macos_say_available() -> bool:
        try:
            result = subprocess.run(["say", "--version"],
                                    capture_output=True, timeout=2)
            return result.returncode == 0
        except Exception:
            return False

    # ─── Background thread ────────────────────────────────────────────────────
    def _run(self) -> None:
        """TTS thread: owns the engine, drains the priority queue."""
        engine = self._init_engine()

        while True:
            item = self._pq.get()
            if item.text == "__STOP__" or item.priority == -1:
                break
            if not item.text:
                continue
            try:
                is_sentence = item.priority == int(_P.SENTENCE)
                self._speak(engine, item.text, sentence=is_sentence)
            except Exception:
                pass  # never crash the TTS thread

    def _init_engine(self):
        """Initialize pyttsx3 or return None if unavailable (macOS path used)."""
        if self._use_macos:
            return None
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", self._RATE_FAST)
            # Pick a clearer voice if multiple are available
            voices = engine.getProperty("voices")
            for v in voices:
                if "female" in v.name.lower() or "zira" in v.id.lower():
                    engine.setProperty("voice", v.id)
                    break
            return engine
        except Exception:
            return None

    def _speak(self, engine, text: str, sentence: bool = False) -> None:
        """Dispatch to macOS `say` or pyttsx3."""
        if self._use_macos:
            rate = self._RATE_SLOW if sentence else self._RATE_FAST
            subprocess.run(
                ["say", "--voice", self._MACOS_VOICE,
                 "--rate", str(rate), text],
                capture_output=True,
                timeout=30,
            )
        elif engine is not None:
            rate = self._RATE_SLOW if sentence else self._RATE_FAST
            engine.setProperty("rate", rate)
            engine.say(text)
            engine.runAndWait()
