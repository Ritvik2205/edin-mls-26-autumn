"""Word-level transcript comparison (no ground-truth query text exists, so WER is vs V1)."""
from __future__ import annotations

import re


def _words(s: str) -> list[str]:
    return re.sub(r"[^\w\s]", "", s.upper()).split()


def word_errors(ref: str, hyp: str) -> int:
    """Levenshtein distance between the word sequences of ref and hyp."""
    r, h = _words(ref), _words(hyp)
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, start=1):
        cur = [i] + [0] * len(h)
        for j, hw in enumerate(h, start=1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw))
        prev = cur
    return prev[-1]


def wer(ref: str, hyp: str) -> float:
    """word_errors / max(1, words in ref), so an empty reference stays finite."""
    return word_errors(ref, hyp) / max(1, len(_words(ref)))
