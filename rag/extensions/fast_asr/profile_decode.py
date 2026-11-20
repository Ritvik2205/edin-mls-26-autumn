#!/usr/bin/env python3
"""What one KV-cache decode step is made of, operation by operation.

Run in the mls-asr env on a GPU node, from the repository root:

    [ASR_GEMV=off|seq|warp] python rag/extensions/fast_asr/profile_decode.py [--cuda all|none]

profile_step.py sums the supplied ops profiler over a whole transcription, where the encoder
dominates. This script reads the profiler's per-call events instead and keeps only the decode
steps after the first token. A step starts at the embedding of the previous token and holds, per
decoder layer, seven linear calls at M = 1 (q, k, v, o, gate, up, down), and ends with lm_head.

For each operation it prints the GPU time per step (CUDA events around the operation, mean over
steps). The step's wall time minus that sum is time in which no profiled operation was running:
Python and ctypes between launches, and the small PyTorch operations the profiler does not see
(residual adds, cache writes, argmax). The weight-bandwidth floor of each linear is its weight
bytes over the device bandwidth measured by HW1's Task 2 test (--bandwidth, GB/s).
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):  # run as a file: make `extensions.*` importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REPO = Path(__file__).resolve().parents[3]
LAYER_LINEARS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def split_steps(entries, n_steps):
    """entries: [(op, ms)] in call order. The last n_steps segments that start at an embedding."""
    starts = [i for i, (op, _) in enumerate(entries) if op == "embedding"][-n_steps:]
    return [entries[a:b] for a, b in zip(starts, starts[1:] + [len(entries)])]


def label_step(step, layers):
    """[(op, ms)] for one step -> {label: ms}; linear calls are named by their place in the layer."""
    out, n_lin = defaultdict(float), 0
    linears = sum(1 for op, _ in step if op == "linear")
    if linears != layers * len(LAYER_LINEARS) + 1:
        raise RuntimeError(f"expected {layers * len(LAYER_LINEARS) + 1} linear calls in a step, saw {linears}")
    for op, ms in step:
        if op == "linear":
            name = "lm_head" if n_lin == linears - 1 else LAYER_LINEARS[n_lin % len(LAYER_LINEARS)]
            out["linear " + name] += ms
            n_lin += 1
        else:
            out[op] += ms
    return dict(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cuda", default="all")
    ap.add_argument("--audio", type=Path, default=REPO / "hw1-asr" / "test_audio.wav")
    ap.add_argument("--bandwidth", type=float, default=616.0, help="device bandwidth, GB/s")
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    import os
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
    c, w = fast.base.c, fast.base.w

    call()  # warm-up
    plain_ms, prof_ms, per_label, counts = [], [], defaultdict(list), {}
    for _ in range(args.repeats):
        plain = call()
        n_steps = len(plain.tokens) - 1
        plain_ms.append((plain.decode_ms - plain.prefill_ms) / n_steps)
        ops.profile_start()
        prof = call()
        torch.cuda.synchronize()
        entries = [(op, start.elapsed_time(end)) for op, _, start, end in ops._profile]
        ops.profile_stop()
        prof_ms.append((prof.decode_ms - prof.prefill_ms) / n_steps)
        steps = split_steps(entries, n_steps)
        counts = defaultdict(int)
        for op, _ in steps[0]:
            counts[op] += 1
        labelled = [label_step(s, c.text_layers) for s in steps]
        for label in labelled[0]:
            per_label[label].append(sum(s[label] for s in labelled) / n_steps)

    mean = lambda xs: sum(xs) / len(xs)
    p = "language_model.model.layers.0."
    weights = {"linear " + n: w[p + ("self_attn." if n[0] in "qkvo" else "mlp.") + n + ".weight"] for n in LAYER_LINEARS}
    weights["linear lm_head"] = w["language_model.lm_head.weight"]
    print(f"GPU: {torch.cuda.get_device_name(0)}  kernels: {args.cuda}  ASR_GEMV: {os.environ.get('ASR_GEMV', 'off')}  "
          f"audio: {args.audio.name}")
    print(f"prompt {plain.prompt_len} positions, {len(plain.tokens)} tokens, {n_steps} steps after the first token, "
          f"{args.repeats} repeats; {sum(counts.values())} profiled calls per step "
          f"({', '.join(f'{k} {v}' for k, v in sorted(counts.items()))})")
    print(f"\n  {'operation':<18} {'calls':>5} {'shape N x K':>13} {'ms/step':>8} {'us/call':>8} {'floor ms':>9} {'x floor':>8}")
    gpu_total = floor_total = 0.0
    for label, vals in sorted(per_label.items(), key=lambda kv: -mean(kv[1])):
        ms = mean(vals)
        gpu_total += ms
        if label in weights:
            wt = weights[label]
            calls = 1 if label.endswith("lm_head") else c.text_layers
            floor = calls * wt.numel() * wt.element_size() / (args.bandwidth * 1e9) * 1e3
            floor_total += floor
            shape = f"{wt.shape[0]} x {wt.shape[1]}"
            print(f"  {label:<18} {calls:>5} {shape:>13} {ms:>8.2f} {1000 * ms / calls:>8.1f} {floor:>9.2f} {ms / floor:>8.1f}")
        else:
            calls = counts[label]
            print(f"  {label:<18} {calls:>5} {'':>13} {ms:>8.2f} {1000 * ms / calls:>8.1f}")
    wall, wall_prof = mean(plain_ms), mean(prof_ms)
    print(f"\n  step, unprofiled: {wall:.2f} ms (min {min(plain_ms):.2f}, max {max(plain_ms):.2f}); "
          f"profiled: {wall_prof:.2f} ms")
    print(f"  GPU time in profiled operations: {gpu_total:.2f} ms per step; weight-bandwidth floor of the "
          f"linear calls: {floor_total:.2f} ms")
    print(f"  outside profiled operations: {wall - gpu_total:.2f} ms per step "
          f"({100 * (wall - gpu_total) / wall:.0f}% of the unprofiled step)")


if __name__ == "__main__":
    main()
