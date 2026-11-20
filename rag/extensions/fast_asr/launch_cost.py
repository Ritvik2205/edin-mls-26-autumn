#!/usr/bin/env python3
"""Is a KV-cache decode step bound by the GPU or by the host that issues its launches?

Run in the mls-asr env on a GPU node, from the repository root:

    [ASR_GEMV=off|seq|warp] python rag/extensions/fast_asr/launch_cost.py

Kernel launches are asynchronous, so one decode step has two parts on the host clock:

  issue  from the start of the step until the last operation of the step has been launched
         (Python, ctypes, the launch calls, and PyTorch's small operations in between);
  wait   from there until torch.cuda.synchronize() returns, i.e. GPU work still queued.

If wait is near zero the GPU keeps up with the host and the step is bound by the host issuing
launches; a faster kernel cannot shorten it. If wait is large the step is bound by GPU work.

The second part times the launch path alone: the same calls on a 1 x 16 weight, where the kernel
has almost nothing to do, through ops.linear and through the ctypes function directly.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

if __package__ in (None, ""):  # run as a file: make `extensions.*` importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REPO = Path(__file__).resolve().parents[3]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", type=Path, default=REPO / "hw1-asr" / "test_audio.wav")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--calls", type=int, default=340, help="launches per batch in the launch-path test")
    args = ap.parse_args()

    import torch
    from extensions.fast_asr import _paths  # noqa: F401
    from glm_asr_cuda import ops
    from extensions.fast_asr.audio import read_audio
    from extensions.fast_asr.kv_model import KVCache, load_fast, prepare_inputs
    if not torch.cuda.is_available():
        sys.exit("No GPU visible. Run on a GPU node.")
    ops.set_cuda_groups("all")
    fast, processor = load_fast()
    b = fast.base
    inp = prepare_inputs(processor, read_audio(args.audio))
    now, sync = time.perf_counter, torch.cuda.synchronize

    @torch.no_grad()
    def decode():
        """generate_kv's loop with the clock read before and after a synchronize in every step."""
        audio = b.encode_audio(inp["features"], inp["features_mask"])
        embeds = fast._prompt(audio, inp["input_ids"])
        P = embeds.shape[1]
        cache = KVCache.allocate(b.c.text_layers, P + 100, b.c.text_kv_heads, b.text_head_dim, embeds.device)
        p = fast._head(fast._decoder_kv(embeds, cache, 0))
        tokens, issue, wait = [], [], []
        for step in range(100):
            best = torch.argmax(p, dim=-1)
            tokens.append(int(best[0]))  # reads the token back, so the GPU is idle from here
            if tokens[-1] in b.c.eos_token_ids:
                break
            sync()
            t0 = now()
            p = fast._head(fast._decoder_kv(fast._embed_token(best), cache, P + step))
            t1 = now()
            sync()
            t2 = now()
            issue.append((t1 - t0) * 1e3)
            wait.append((t2 - t1) * 1e3)
        return tokens, issue, wait

    decode()  # warm-up
    issue, wait, tokens = [], [], []
    for _ in range(args.repeats):
        tokens, i, w = decode()
        issue += i
        wait += w
    mean = lambda xs: sum(xs) / len(xs)
    step = mean(issue) + mean(wait)
    print(f"GPU: {torch.cuda.get_device_name(0)}  ASR_GEMV: {os.environ.get('ASR_GEMV', 'off')}  "
          f"audio: {args.audio.name}")
    print(f"{len(tokens)} tokens, {len(issue) // args.repeats} steps after the first token, {args.repeats} repeats")
    print(f"  step {step:.2f} ms = issue {mean(issue):.2f} ms (min {min(issue):.2f}, max {max(issue):.2f}) "
          f"+ wait {mean(wait):.2f} ms (min {min(wait):.2f}, max {max(wait):.2f}); "
          f"wait is {100 * mean(wait) / step:.0f}% of the step")

    # The launch path alone: M = 1 on a 1 x 16 weight.
    x = torch.randn(1, 16, device="cuda")
    wt = torch.randn(1, 16, device="cuda")
    y = torch.empty(1, 1, device="cuda")
    lib, stream = ops._library(), ops._stream()
    px, pw, py, pnone = ops._ptr(x), ops._ptr(wt), ops._ptr(y), ops._ptr(None)

    def batch(fn):
        best = float("inf")
        for _ in range(20):
            sync()
            t0 = now()
            for _ in range(args.calls):
                fn()
            t1 = now()
            sync()
            best = min(best, (t1 - t0) * 1e6 / args.calls)
        return best

    raw = batch(lambda: lib.launch_linear(px, pw, pnone, py, 1, 1, 16, 0, stream))
    wrapped = batch(lambda: ops.linear(x, wt))
    add = batch(lambda: x + x)
    print(f"  launch path, 1 x 16 weight, best of 20 batches of {args.calls} calls, us per call: "
          f"ctypes launch_linear {raw:.1f}, ops.linear {wrapped:.1f}, torch x + x {add:.1f}")
    print(f"  {args.calls} calls through ops.linear: {wrapped * args.calls / 1e3:.2f} ms")


if __name__ == "__main__":
    main()
