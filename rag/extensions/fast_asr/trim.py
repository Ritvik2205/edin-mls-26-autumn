"""Encoder input length for trimmed (V3) runs: never cuts real audio."""
from __future__ import annotations

import math

FRAMES_PER_SECOND = 100  # mel hop of 10 ms
ALIGN = 8                # conv stride 2 x projector pool 4


def trimmed_frames(real_frames: int, margin_s: float | None = None,
                   bucket_s: float | None = None, total: int = 3000) -> int:
    """Mel frames to keep: a multiple of 8, >= real_frames, <= total.

    margin_s: keep real_frames plus this many seconds of padding.
    bucket_s: keep at least this many seconds, but never fewer than real_frames.
    """
    if (margin_s is None) == (bucket_s is None):
        raise ValueError("give exactly one of margin_s or bucket_s")
    if real_frames < 0 or real_frames > total:
        raise ValueError(f"real_frames must be in 0..{total}, got {real_frames}")
    if margin_s is not None:
        if margin_s < 0:
            raise ValueError("margin_s must be >= 0")
        wanted = real_frames + round(margin_s * FRAMES_PER_SECOND)
    else:
        if bucket_s <= 0:
            raise ValueError("bucket_s must be > 0")
        wanted = max(round(bucket_s * FRAMES_PER_SECOND), real_frames)
    aligned = math.ceil(wanted / ALIGN) * ALIGN
    return min(total, aligned)
