"""Spot a model that is repeating itself *while it's still generating*.

Local reasoning models sometimes fall into a loop inside a single reply:
the same few sentences of "thinking" over and over until they hit the
output-token cap, which can take many minutes. The SDK's stuck detector
only looks at finished actions, so it never sees this. The supervisor feeds
the streamed text here and cuts the generation off when it loops.

Pure and incremental, so it's cheap to call on every streamed update.
"""

from __future__ import annotations

import re
from collections import Counter

WINDOW = 8000          # chars of recent output to look at
MIN_SENTENCE = 30      # shorter sentences ("Let me check.") repeat legitimately
SENTENCE_REPEATS = 5   # same sentence this many times in the window = loop
BLOCK = 160            # a verbatim block of this many chars...
BLOCK_REPEATS = 3      # ...this many times in the window = loop
MIN_TEXT = 1200        # don't judge very short output

_SPLIT_RE = re.compile(r"(?<=[.!?:;])\s+|\n+")
_NUM_RE = re.compile(r"\d+")
_WS_RE = re.compile(r"\s+")


def _norm(text: str) -> str:
    # Loops often differ only in numbers ("step 3", "line 41") or spacing.
    return _WS_RE.sub(" ", _NUM_RE.sub("0", text.lower())).strip()


class RepetitionDetector:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._text = ""

    def feed(self, text: str | None) -> None:
        if text:
            self._text = (self._text + text)[-WINDOW * 2:]

    def check(self) -> str | None:
        """The repeated passage if the recent output is looping, else None."""
        window = _norm(self._text[-WINDOW:])
        if len(window) < MIN_TEXT:
            return None
        # 1. The same sentence again and again.
        sentences = [s.strip() for s in _SPLIT_RE.split(window) if len(s.strip()) >= MIN_SENTENCE]
        if sentences:
            sentence, count = Counter(sentences).most_common(1)[0]
            if count >= SENTENCE_REPEATS:
                return f"{sentence[:200]} (x{count})"
        # 2. A long block repeated verbatim, whatever the punctuation.
        probe = window[-BLOCK:]
        if len(window) >= BLOCK * BLOCK_REPEATS and window.count(probe) >= BLOCK_REPEATS:
            return f"{probe.strip()[:200]} (x{window.count(probe)})"
        return None
