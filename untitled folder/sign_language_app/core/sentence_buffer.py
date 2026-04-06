"""
Sentence buffer for the Sign Language Recognition App.

Manages a dynamic word queue that builds up a full sentence from
sequential gesture predictions. Supports undo, clear, and raw-text
access so the GUI can display the current state at any time.
"""


class SentenceBuffer:
    """Stores predicted words and builds up a sentence incrementally.

    Words are stored in uppercase (the native form of sign labels) and
    joined with spaces when retrieved as a sentence.

    Example usage::

        buf = SentenceBuffer()
        buf.add_word("HELLO")
        buf.add_word("YOU")
        buf.add_word("WANT")
        print(buf.get_raw_sentence())   # "HELLO YOU WANT"
        buf.undo()
        print(buf.get_raw_sentence())   # "HELLO YOU"
    """

    def __init__(self, max_words: int = 25):
        """
        Args:
            max_words: Hard cap on the number of words in the buffer.
                       Prevents unbounded growth during long sessions.
        """
        self._words: list = []
        self._max_words = max_words

    # ------------------------------------------------------------------ writes

    def add_word(self, word: str) -> bool:
        """Append a word to the buffer.

        Args:
            word: Sign label to add; stored as uppercase.

        Returns:
            True if the word was added, False if the buffer is full.
        """
        if len(self._words) >= self._max_words:
            return False
        self._words.append(word.strip().upper())
        return True

    def undo(self):
        """Remove and return the last word.

        Returns:
            The removed word string, or None if the buffer was empty.
        """
        return self._words.pop() if self._words else None

    def clear(self) -> None:
        """Remove all words from the buffer."""
        self._words.clear()

    # ------------------------------------------------------------------ reads

    def get_words(self) -> list:
        """Return a shallow copy of the word list (uppercase strings)."""
        return list(self._words)

    def get_raw_sentence(self) -> str:
        """Return all words joined by spaces, e.g. ``'HELLO YOU WANT'``."""
        return " ".join(self._words)

    def is_empty(self) -> bool:
        """Return True when no words are stored."""
        return len(self._words) == 0

    def __len__(self) -> int:
        return len(self._words)

    def __repr__(self) -> str:  # pragma: no cover
        return f"SentenceBuffer({self._words!r})"
