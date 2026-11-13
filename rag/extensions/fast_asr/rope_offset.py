"""RoPE at positions pos_offset .. pos_offset+S-1, for KV-cache decoding.

Two implementations, chosen by `choose_impl`:

- "kernel": `launch_rope_at` in hw1-asr/csrc/task1_rope.cu, an added launcher that shifts the
  token position by pos_offset. Work is proportional to S. Used when the rope group runs on CUDA
  and the built library exports the symbol.
- "scratch" (spec §7 fallback): q/k are copied into the last S rows of a zero scratch tensor of
  length pos_offset+S, rotated with ops.rope_ (the student's kernel, or the PyTorch reference
  when "rope" is off), and copied back. Cost: pos_offset extra rows of work per call.

Both compute angle = float(pos) * inv_freq[i] with the same float operations, so they agree
bitwise. FAST_ASR_ROPE=kernel|scratch overrides the automatic choice (for the cost comparison).
"""
from __future__ import annotations

import ctypes
import os

from . import _paths  # noqa: F401

IMPLS = ("kernel", "scratch")
_SYMBOL = "launch_rope_at"
_ARGTYPES = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 7 + [ctypes.c_void_p]
_configured = False


def choose_impl(impl, *, rope_on_cuda: bool, has_symbol: bool, env) -> str:
    """Explicit impl beats FAST_ASR_ROPE beats auto ("kernel" when available)."""
    choice = impl or env
    if choice is None:
        return "kernel" if (rope_on_cuda and has_symbol) else "scratch"
    if choice not in IMPLS:
        raise ValueError(f"unknown RoPE impl {choice!r}; expected one of {IMPLS}")
    if choice == "kernel" and not (rope_on_cuda and has_symbol):
        raise ValueError("impl='kernel' needs the rope group on CUDA and a library built with "
                         "launch_rope_at (run `make -C hw1-asr lib` on this node)")
    return choice


def _ops():
    from glm_asr_cuda import ops
    return ops


def _has_symbol(ops) -> bool:
    global _configured
    try:
        fn = getattr(ops._library(), _SYMBOL)
    except (AttributeError, RuntimeError, OSError):
        return False
    if not _configured:
        fn.argtypes = _ARGTYPES
        fn.restype = ctypes.c_int
        _configured = True
    return True


def rope_at_(q, k, inv_freq, rotary_dim: int, pos_offset: int, impl: str | None = None) -> None:
    """Rotate q [B, S, Hq, D] and k [B, S, Hk, D] in place at positions pos_offset + (0..S-1)."""
    if impl is not None and impl not in IMPLS:
        raise ValueError(f"unknown RoPE impl {impl!r}; expected one of {IMPLS}")
    if pos_offset < 0:
        raise ValueError("pos_offset must be >= 0")
    ops = _ops()
    if pos_offset == 0 and impl is None:
        ops.rope_(q, k, inv_freq, rotary_dim)
        return
    on_cuda = "rope" in ops.USE_CUDA
    chosen = choose_impl(impl, rope_on_cuda=on_cuda,
                         has_symbol=on_cuda and _has_symbol(ops),
                         env=os.environ.get("FAST_ASR_ROPE"))
    if chosen == "kernel":
        _rope_kernel(ops, q, k, inv_freq, rotary_dim, pos_offset)
    else:
        _rope_scratch(ops, q, k, inv_freq, rotary_dim, pos_offset)


def _rope_kernel(ops, q, k, inv_freq, rotary_dim, pos_offset):
    B, S, Hq, D = q.shape
    Hk = k.shape[2]
    with ops._timed("rope", "rope"):
        ops._check_float32("rope", q, k, inv_freq)
        ops._launch("rope", _SYMBOL, ops._ptr(q), ops._ptr(k), ops._ptr(inv_freq),
                    B, S, Hq, Hk, D, rotary_dim, pos_offset)


def _rope_scratch(ops, q, k, inv_freq, rotary_dim, pos_offset):
    B, S, Hq, D = q.shape
    Hk = k.shape[2]
    qs = q.new_zeros(B, pos_offset + S, Hq, D)
    ks = k.new_zeros(B, pos_offset + S, Hk, D)
    qs[:, pos_offset:].copy_(q)
    ks[:, pos_offset:].copy_(k)
    ops.rope_(qs, ks, inv_freq, rotary_dim)
    q.copy_(qs[:, pos_offset:])
    k.copy_(ks[:, pos_offset:])
