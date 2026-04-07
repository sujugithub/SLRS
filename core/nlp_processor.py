"""
NLP processor for the Sign Language Recognition App.

Converts raw signed-word sequences (e.g. ["YOU", "WANT", "FOOD"]) into
natural English sentences ("You want food.") using a lightweight,
dependency-free rule-based pipeline.

──────────────────────────────────────────────────────────────────────────────
APPROACH A — Rule-Based (default, zero extra dependencies)
──────────────────────────────────────────────────────────────────────────────
Pipeline:
  1. Multi-word sign expansion  ("I LOVE YOU" → single phrase)
  2. Single-word normalization  (ME → I, WANNA → want to, …)
  3. Consecutive-duplicate removal
  4. BE-verb insertion          (I HAPPY → I am happy)
  5. Capitalization + punctuation

──────────────────────────────────────────────────────────────────────────────
APPROACH B — AI-based grammar correction (optional, needs extra packages)
──────────────────────────────────────────────────────────────────────────────
Install:  pip install transformers torch
Usage:
    from core.nlp_processor import AIGrammarCorrector
    corrector = AIGrammarCorrector()            # loads T5-small first time
    sentence  = corrector.process(["YOU", "GO", "STORE"])
    # → "You go to the store."

The rule-based class (RuleBasedNLP) is recommended for local, real-time use
because it runs instantly with no GPU or internet connection required.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Vocabulary tables
# ──────────────────────────────────────────────────────────────────────────────

# Known multi-word sign combinations that map directly to English phrases.
# Checked before single-word substitution (longest match wins).
_PHRASE_MAP: dict = {
    "I LOVE YOU":  "I love you",
    "THUMBS UP":   "thumbs up",
    "GOOD MORNING": "good morning",
    "GOOD NIGHT":  "good night",
    "THANK YOU":   "thank you",
    "EXCUSE ME":   "excuse me",
    "I AM SORRY":  "I am sorry",
}

# Single-word substitutions applied after phrase expansion.
# NOTE: "ME" and "MYSELF" are intentionally NOT remapped here so that
# small joining words like "me" are preserved in the output sentence.
_WORD_MAP: dict = {
    "AINT":     "am not",
    "WANNA":    "want to",
    "GONNA":    "going to",
    "GOTTA":    "have to",
    "STORE":    "the store",
    "THUMBS_UP": "thumbs up",
    "I_LOVE_YOU": "I love you",
    "PEACE":    "peace",
    "HELLO":    "hello",
}

# Predicate adjectives: if a known subject directly precedes one of these,
# the appropriate form of "to be" is automatically inserted.
_ADJECTIVES: frozenset = frozenset({
    "HAPPY", "SAD", "TIRED", "HUNGRY", "THIRSTY", "SICK", "FINE", "OKAY",
    "OK", "GOOD", "BAD", "ANGRY", "SCARED", "EXCITED", "BORED", "BUSY",
    "READY", "HOT", "COLD", "BIG", "SMALL", "OLD", "YOUNG", "BEAUTIFUL",
    "NICE", "SORRY", "THANKFUL", "GRATEFUL", "LATE", "EARLY", "LUCKY",
    "RIGHT", "WRONG", "STRONG", "WEAK", "FAST", "SLOW", "QUIET", "LOUD",
    "FREE", "LOST", "SAFE", "HURT", "SICK", "WELL", "FULL", "EMPTY",
})

# First word determines terminal punctuation.
_QUESTION_WORDS: frozenset = frozenset({
    "WHO", "WHAT", "WHERE", "WHEN", "WHY", "HOW", "WHICH",
})
_GREETING_WORDS: frozenset = frozenset({
    "HELLO", "HI", "HEY",
})

# Subject → "to be" conjugation table
_BE_FORMS: dict = {
    "I":    "am",
    "YOU":  "are",
    "WE":   "are",
    "THEY": "are",
    "HE":   "is",
    "SHE":  "is",
    "IT":   "is",
}

# ──────────────────────────────────────────────────────────────────────────────
# Approach A — Rule-based NLP
# ──────────────────────────────────────────────────────────────────────────────


class RuleBasedNLP:
    """Convert a list of uppercase sign labels into natural English.

    No external dependencies — runs instantly on every frame.

    Example::

        nlp = RuleBasedNLP()
        nlp.process(["YOU", "HAPPY"])      # → "You are happy."
        nlp.process(["I", "WANT", "FOOD"]) # → "I want food."
        nlp.process(["WHERE", "GO"])        # → "Where go?"
        nlp.process(["HELLO"])              # → "Hello!"
    """

    def process(self, words: list) -> str:
        """Convert a raw word list to a natural English sentence.

        Args:
            words: List of sign labels (case-insensitive).

        Returns:
            Corrected, punctuated English sentence string, or ``""`` if
            *words* is empty.
        """
        if not words:
            return ""

        tokens = [w.strip().upper() for w in words if w.strip()]
        if not tokens:
            return ""

        tokens = self._expand_phrases(tokens)
        tokens = self._normalize_words(tokens)
        tokens = self._remove_consecutive_duplicates(tokens)
        tokens = self._insert_be_verbs(tokens)
        return self._finalize(tokens)

    # ------------------------------------------------------------------ steps

    def _expand_phrases(self, tokens: list) -> list:
        """Replace known multi-word sign names with their English phrases."""
        result = []
        i = 0
        while i < len(tokens):
            matched = False
            for length in (3, 2):           # longest match first
                end = i + length
                if end <= len(tokens):
                    key = " ".join(tokens[i:end])
                    if key in _PHRASE_MAP:
                        result.append(_PHRASE_MAP[key])
                        i = end
                        matched = True
                        break
            if not matched:
                result.append(tokens[i])
                i += 1
        return result

    def _normalize_words(self, tokens: list) -> list:
        """Apply single-word substitutions from the vocabulary table."""
        return [_WORD_MAP.get(t, t) for t in tokens]

    def _remove_consecutive_duplicates(self, tokens: list) -> list:
        """Collapse immediately repeated words (case-insensitive)."""
        if not tokens:
            return tokens
        result = [tokens[0]]
        for token in tokens[1:]:
            if token.upper() != result[-1].upper():
                result.append(token)
        return result

    def _insert_be_verbs(self, tokens: list) -> list:
        """Insert 'am/are/is' between a known subject and an adjective."""
        if len(tokens) < 2:
            return tokens
        result = []
        i = 0
        while i < len(tokens):
            result.append(tokens[i])
            if i + 1 < len(tokens):
                curr = tokens[i].upper()
                nxt  = tokens[i + 1].upper()
                if curr in _BE_FORMS and nxt in _ADJECTIVES:
                    result.append(_BE_FORMS[curr])
            i += 1
        return result

    def _finalize(self, tokens: list) -> str:
        """Lower-case all words (keeping bare 'I' capitalised) then punctuate."""
        lowered = []
        for t in tokens:
            if t.upper() == "I":
                lowered.append("I")
            else:
                lowered.append(t.lower())

        if lowered:
            lowered[0] = lowered[0].capitalize()

        sentence = " ".join(lowered)

        first = tokens[0].upper() if tokens else ""
        if first in _QUESTION_WORDS:
            sentence += "?"
        elif first in _GREETING_WORDS:
            sentence += "!"
        else:
            sentence += "."

        return sentence


# ──────────────────────────────────────────────────────────────────────────────
# Approach B — AI grammar correction with T5 (optional)
# ──────────────────────────────────────────────────────────────────────────────

class AIGrammarCorrector:
    """Grammar correction using a local T5-small transformer model.

    Requires:  ``pip install transformers torch``

    The model is downloaded on first use (~240 MB) and cached locally by
    the Hugging Face library.  After the first download it runs offline.

    Recommended model: ``grammarly/coedit-small``  (50 MB, very accurate)
    Fallback model:    ``vennify/t5-base-grammar-correction``

    Example::

        corrector = AIGrammarCorrector()
        corrector.process(["YOU", "GO", "STORE"])
        # → "You go to the store."
    """

    _MODEL_NAME = "grammarly/coedit-small"

    def __init__(self):
        try:
            from transformers import T5ForConditionalGeneration, T5Tokenizer  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "transformers and torch are required for AIGrammarCorrector.\n"
                "Install them with:  pip install transformers torch"
            ) from exc

        self._tokenizer = T5Tokenizer.from_pretrained(self._MODEL_NAME)
        self._model     = T5ForConditionalGeneration.from_pretrained(self._MODEL_NAME)

        # Pre-build rule-based processor for initial normalisation
        self._rules = RuleBasedNLP()

    def process(self, words: list) -> str:
        """Convert raw sign words to natural English via the T5 model.

        First runs the rule-based normaliser, then feeds the result to T5
        for grammar correction.

        Args:
            words: List of sign labels.

        Returns:
            Grammar-corrected English sentence.
        """
        if not words:
            return ""

        # Rule-based pre-processing gives T5 a cleaner input
        rough = self._rules.process(words)

        # CoEdit expects a task prefix
        prompt = f"Fix grammatical errors: {rough}"

        inputs  = self._tokenizer(prompt, return_tensors="pt", max_length=128, truncation=True)
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=64,
            num_beams=4,
            early_stopping=True,
        )
        return self._tokenizer.decode(outputs[0], skip_special_tokens=True)
