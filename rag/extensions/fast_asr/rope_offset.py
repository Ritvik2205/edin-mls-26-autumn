"""RoPE at positions pos_offset .. pos_offset+S-1, for KV-cache decoding.

Fallback implementation (spec §7): the supplied launch_rope only rotates positions
0..S-1, so q/k are copied into the last S rows of a zero scratch tensor of length
pos_offset+S, rotated with ops.rope_ (the student's kernel, or the PyTorch reference when
"rope" is off), and copied back. RoPE is independent per row, so the result is bitwise
what a full-length rotation gives those rows. Cost: pos_offset extra rows of work per call.
"""
from __future__ import annotations

from . import _paths  # noqa: F401
from glm_asr_cuda import ops


def rope_at_(q, k, inv_freq, rotary_dim: int, pos_offset: int) -> None:
    """Rotate q [B, S, Hq, D] and k [B, S, Hk, D] in place at positions pos_offset + (0..S-1)."""
    if pos_offset < 0:
        raise ValueError("pos_offset must be >= 0")
    if pos_offset == 0:
        ops.rope_(q, k, inv_freq, rotary_dim)
        return
    B, S, Hq, D = q.shape
    Hk = k.shape[2]
    qs = q.new_zeros(B, pos_offset + S, Hq, D)
    ks = k.new_zeros(B, pos_offset + S, Hk, D)
    qs[:, pos_offset:].copy_(q)
    ks[:, pos_offset:].copy_(k)
    ops.rope_(qs, ks, inv_freq, rotary_dim)
    q.copy_(qs[:, pos_offset:])
    k.copy_(ks[:, pos_offset:])
