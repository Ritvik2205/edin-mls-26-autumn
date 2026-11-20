#!/usr/bin/env python3
"""Why trimming the encoder input changes some transcripts, measured per query.

Run in the mls-asr env on a GPU node, from the repository root:

    python rag/extensions/fast_asr/trim_effect.py --split dev --out-csv results/trim_effect/dev.csv

The encoder has no padding mask, so with the full 30 s input the frames of real audio attend to
padding frames. Trimming removes those keys. For every query and every setting this script
compares the audio embeddings of the real frames (the only ones the decoder sees) between the
full and the trimmed input, then decodes both and looks at the first token where they differ:

  rel_l2, mean_cos, min_cos   trimmed vs full audio embeddings (same shape: real frames only)
  first_diff                  index of the first differing token (-1: same tokens)
  p_full_own, p_full_alt      at that step, with the full encoder: probability of the token the
                              full run chose, and of the token the trimmed run chose
  p_trim_own, p_trim_alt      the same two with the trimmed encoder (own: the trimmed choice)
  min_top1                    lowest top-1 probability over the full run's tokens

The prefix is equal up to first_diff, so the two distributions at that step differ only through
the audio embeddings. A change where p_full_own and p_full_alt are close is a near-tie that a
small perturbation of the embeddings flips, not a different reading of the audio.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

if __package__ in (None, ""):  # run as a file: make `extensions.*` importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

SETTINGS = (("m0", 0.0, None), ("m2", 2.0, None), ("m5", 5.0, None), ("b10", None, 10.0))
FIELDS = ["query_id", "setting", "real_frames", "enc_frames", "rel_l2", "mean_cos", "min_cos", "n_tokens",
          "min_top1", "first_diff", "p_full_own", "p_full_alt", "p_trim_own", "p_trim_alt", "tok_full", "tok_trim"]


def first_difference(a, b) -> int:
    """Index of the first position where two token lists differ; -1 if they are equal."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return -1 if len(a) == len(b) else min(len(a), len(b))


def summarise(rows) -> list[str]:
    out = []
    for name, _, _ in SETTINGS:
        rs = [r for r in rows if r["setting"] == name]
        if not rs:
            continue
        ch = [r for r in rs if int(r["first_diff"]) >= 0]
        same = [r for r in rs if int(r["first_diff"]) < 0]
        med = lambda xs: statistics.median(xs) if xs else float("nan")
        f = lambda r, k: float(r[k])
        out.append(
            f"{name:<4} n={len(rs)}  rel_l2 median {med([f(r, 'rel_l2') for r in rs]):.4f} "
            f"(max {max(f(r, 'rel_l2') for r in rs):.4f})  mean cos {med([f(r, 'mean_cos') for r in rs]):.5f}  "
            f"changed {len(ch)}")
        out.append(
            f"       rel_l2 median, changed vs same: {med([f(r, 'rel_l2') for r in ch]):.4f} vs "
            f"{med([f(r, 'rel_l2') for r in same]):.4f};  min top-1 prob of the full run, changed vs same: "
            f"{med([f(r, 'min_top1') for r in ch]):.3f} vs {med([f(r, 'min_top1') for r in same]):.3f}")
        for r in ch:
            out.append(f"       {r['query_id']} step {r['first_diff']}: full {r['tok_full']!r} "
                       f"p={f(r, 'p_full_own'):.3f} (alt {f(r, 'p_full_alt'):.3f}); trimmed {r['tok_trim']!r} "
                       f"p={f(r, 'p_trim_own'):.3f} (alt {f(r, 'p_trim_alt'):.3f})")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["practice", "dev"], default="dev")
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--summary-only", action="store_true", help="print the summary of an existing CSV")
    args = ap.parse_args()
    if args.summary_only:
        print("\n".join(summarise(list(csv.DictReader(open(args.out_csv, newline=""))))))
        return

    import torch
    import torch.nn.functional as F
    from extensions.fast_asr import _paths  # noqa: F401
    from glm_asr_cuda import ops
    from extensions.fast_asr.audio import read_audio
    from extensions.fast_asr.bench_asr import CORPUS, load_items
    from extensions.fast_asr.kv_model import KVCache, load_fast, prepare_inputs, real_frames
    from extensions.fast_asr.trim import trimmed_frames
    if not torch.cuda.is_available():
        sys.exit("No GPU visible. Run on a GPU node.")
    ops.set_cuda_groups("all")
    fast, processor = load_fast()
    b = fast.base
    tok = processor.tokenizer

    @torch.no_grad()
    def run(inp, frames=None):
        """Audio embeddings, tokens and the probability vector of every step."""
        feats, mask = inp["features"], inp["features_mask"]
        if frames is not None:
            feats, mask = feats[:, :, :frames].contiguous(), mask[:, :frames].contiguous()
        audio = b.encode_audio(feats, mask)
        embeds = fast._prompt(audio, inp["input_ids"])
        P = embeds.shape[1]
        cache = KVCache.allocate(b.c.text_layers, P + 100, b.c.text_kv_heads, b.text_head_dim, embeds.device)
        p = fast._head(fast._decoder_kv(embeds, cache, 0))
        tokens, dists = [], []
        for step in range(100):
            best = torch.argmax(p, dim=-1)
            tokens.append(int(best[0]))
            dists.append(p[0].clone())
            if tokens[-1] in b.c.eos_token_ids or step == 99:
                break
            p = fast._head(fast._decoder_kv(fast._embed_token(best), cache, P + step))
        return audio, tokens, dists

    rows = []
    items = load_items(args.split, limit=args.limit)
    for n, item in enumerate(items):
        inp = prepare_inputs(processor, read_audio(CORPUS / item["audio_file"]))
        inp.pop("feat_ms")
        rf = real_frames(inp["features_mask"])
        a0, t0, d0 = run(inp)
        min_top1 = min(float(d.max()) for d in d0)
        for name, margin, bucket in SETTINGS:
            T = trimmed_frames(rf, margin_s=margin, bucket_s=bucket)
            a1, t1, d1 = run(inp, T)
            if a1.shape != a0.shape:
                raise RuntimeError(f"{item['query_id']} {name}: embeddings {tuple(a1.shape)} vs {tuple(a0.shape)}")
            cos = F.cosine_similarity(a0, a1, dim=-1)
            j = first_difference(t0, t1)
            row = {"query_id": item["query_id"], "setting": name, "real_frames": rf, "enc_frames": T,
                   "rel_l2": float((a1 - a0).norm() / a0.norm()), "mean_cos": float(cos.mean()),
                   "min_cos": float(cos.min()), "n_tokens": len(t0), "min_top1": min_top1, "first_diff": j}
            if j >= 0:
                row.update(p_full_own=float(d0[j][t0[j]]), p_full_alt=float(d0[j][t1[j]]),
                           p_trim_own=float(d1[j][t1[j]]), p_trim_alt=float(d1[j][t0[j]]),
                           tok_full=tok.decode([t0[j]]), tok_trim=tok.decode([t1[j]]))
            rows.append(row)
        if n % 25 == 0:
            print(f"[{n + 1}/{len(items)}] {item['query_id']}", flush=True)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.out_csv}")
    print("\n".join(summarise([{k: ("" if v is None else str(v)) for k, v in r.items()} for r in rows])))


if __name__ == "__main__":
    main()
