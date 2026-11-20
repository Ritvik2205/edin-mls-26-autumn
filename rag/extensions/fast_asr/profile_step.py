#!/usr/bin/env python3
"""Where does one KV-cache transcription spend its time: inside kernels, or between them?

Run in the mls-asr env on a GPU node, from the repository root:

    python rag/extensions/fast_asr/profile_step.py [--cuda all|none] [--audio hw1-asr/test_audio.wav]

It transcribes one clip with V2 (KV cache) twice after a warm-up:
  1. unprofiled, timed with the variant's own CUDA events (encode_ms, prefill_ms, decode_ms);
  2. with the supplied ops profiler on, which brackets every operation with CUDA events.

"In kernels" is the sum of the per-operation event times. The rest of the decode time is spent
between operations: Python, ctypes, tensor allocation and launch overhead. If that remainder is
a large share of a decode step, the step is not bound by memory bandwidth alone.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):  # run as a file: make `extensions.*` importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REPO = Path(__file__).resolve().parents[3]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cuda", default="all")
    ap.add_argument("--audio", type=Path, default=REPO / "hw1-asr" / "test_audio.wav")
    args = ap.parse_args()

    import torch
    from extensions.fast_asr import _paths  # noqa: F401
    from glm_asr_cuda import ops
    from extensions.fast_asr.audio import read_audio
    from extensions.fast_asr.kv_model import load_fast, prepare_inputs
    if not torch.cuda.is_available():
        sys.exit("No GPU visible. Run on a GPU node.")
    ops.set_cuda_groups(args.cuda)
    fast, processor = load_fast()
    inp = prepare_inputs(processor, read_audio(args.audio))
    call = lambda: fast.generate_kv(inp["features"], inp["input_ids"], inp["features_mask"])

    call()  # warm-up
    plain = call()
    steps = max(len(plain.tokens) - 1, 1)
    step_ms = (plain.decode_ms - plain.prefill_ms) / steps

    ops.profile_start()
    prof = call()
    table = ops.profile_stop()
    kernels = sum(ms for _, ms in table.values())
    calls = sum(c for c, _ in table.values())

    print(f"GPU: {torch.cuda.get_device_name(0)}  kernels: {args.cuda}  audio: {args.audio.name}")
    print(f"prompt {plain.prompt_len} positions, {len(plain.tokens)} tokens")
    print(f"unprofiled: encode {plain.encode_ms:.1f} ms, prefill {plain.prefill_ms:.1f} ms, "
          f"decode {plain.decode_ms:.1f} ms, {step_ms:.2f} ms per step after the first token")
    print(f"profiled:   encode {prof.encode_ms:.1f} ms, decode {prof.decode_ms:.1f} ms")
    print(f"\nper operation, whole transcription ({calls} calls, {kernels:.1f} ms in kernels):")
    print(f"  {'operation':<12} {'runs on':<8} {'calls':>7} {'total ms':>10} {'us/call':>9}")
    for (op, backend), (c, ms) in sorted(table.items(), key=lambda kv: -kv[1][1]):
        print(f"  {op:<12} {backend:<8} {c:>7} {ms:>10.1f} {1000 * ms / c:>9.1f}")
    total = prof.encode_ms + prof.decode_ms
    print(f"\nprofiled encode+decode {total:.1f} ms = {kernels:.1f} ms in kernels "
          f"+ {total - kernels:.1f} ms between them ({100 * (total - kernels) / total:.1f}%)")


if __name__ == "__main__":
    main()
