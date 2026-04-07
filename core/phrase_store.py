"""
Phrase store + matcher for SignBridge.

Persists user-recorded multi-word phrases to ``data/phrases.json`` and
provides longest-tail matching against the rolling word buffer in the
prediction screen.

Schema (one entry per phrase):

    {
        "sequence": ["nice", "to", "meet", "you"],
        "output":   "nice to meet you"
    }

Matching is case-insensitive — both incoming words and stored sequences
are compared in lowercase.
"""
from __future__ import annotations

import json
import os
import tempfile
from typing import Optional

from config import PHRASES_FILE


def _phrases_path() -> str:
    return PHRASES_FILE


def load_phrases() -> list[dict]:
    """Read phrases.json and return the list of phrase entries.

    Returns an empty list if the file is missing or malformed.
    """
    try:
        if not os.path.isfile(PHRASES_FILE):
            return []
        with open(PHRASES_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            return []
        cleaned: list[dict] = []
        for entry in data:
            if not isinstance(entry, dict):
                continue
            seq = entry.get("sequence")
            out = entry.get("output")
            if (isinstance(seq, list) and seq
                    and all(isinstance(w, str) for w in seq)
                    and isinstance(out, str) and out):
                cleaned.append({"sequence": [w.lower() for w in seq],
                                 "output":  out})
        return cleaned
    except Exception as exc:
        print(f"[phrase_store] Failed to read phrases.json: {exc}")
        return []


def save_phrases(phrases: list[dict]) -> None:
    """Atomically write phrases.json."""
    try:
        os.makedirs(os.path.dirname(PHRASES_FILE), exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=".phrases_", suffix=".json",
            dir=os.path.dirname(PHRASES_FILE),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(phrases, fh, indent=2)
            os.replace(tmp_path, PHRASES_FILE)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
    except Exception as exc:
        print(f"[phrase_store] Failed to write phrases.json: {exc}")


def add_phrase(sequence: list[str], output: str) -> None:
    """Append a phrase entry to phrases.json (idempotent on identical sequence)."""
    if not sequence or not output:
        return
    seq_lower = [w.lower() for w in sequence]
    existing = load_phrases()
    for entry in existing:
        if entry["sequence"] == seq_lower:
            entry["output"] = output  # update existing
            save_phrases(existing)
            return
    existing.append({"sequence": seq_lower, "output": output})
    save_phrases(existing)


class PhraseMatcher:
    """Matches the tail of a rolling word buffer against stored phrases.

    The matcher prefers the LONGEST matching tail so phrases like
    "nice to meet you" win over "meet you".
    """

    def __init__(self, phrases: Optional[list[dict]] = None):
        self._phrases: list[dict] = phrases if phrases is not None else []

    def reload(self) -> None:
        """Re-read phrases.json from disk."""
        self._phrases = load_phrases()

    def set_phrases(self, phrases: list[dict]) -> None:
        self._phrases = phrases

    def match_tail(self, words: list[str]) -> Optional[tuple[int, str]]:
        """Return (matched_count, output_phrase) for the longest tail match.

        Args:
            words: list of words from SentenceBuffer (uppercase or any case).

        Returns:
            None if no phrase matches the tail, otherwise a tuple of
            (number of words consumed from the tail, phrase output string).
        """
        if not words or not self._phrases:
            return None

        words_lower = [w.lower() for w in words]
        best: Optional[tuple[int, str]] = None
        for entry in self._phrases:
            seq = entry["sequence"]
            n = len(seq)
            if n > len(words_lower):
                continue
            if words_lower[-n:] == seq:
                if best is None or n > best[0]:
                    best = (n, entry["output"])
        return best
