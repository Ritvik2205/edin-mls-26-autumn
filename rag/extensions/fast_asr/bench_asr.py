#!/usr/bin/env python3
"""Per-query ASR timing for one variant; writes a CSV row per query and a transcript cache.

Run in the mls-asr env on a GPU node, from the repository root:

    python rag/extensions/fast_asr/bench_asr.py --split practice --variant v2 \
        --out-csv results/asr/practice_v2.csv --cache rag/.cache/practice_v2.jsonl

Timing boundary: warm model; asr_ms = feat_ms (CPU mel) + encode_ms + decode_ms (CUDA events).
Model load is excluded and printed once. Re-running the same command resumes.
"""
from __future__ import annotations

import argparse
import csv
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

if __package__ in (None, ""):  # run as a file: make `extensions.*` importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from extensions.fast_asr.cachefile import append_cache, read_cache  # noqa: E402

COLUMNS = ["query_id", "split", "variant", "setting", "cuda", "gpu", "node", "commit", "pass",
           "duration_s", "real_frames", "enc_frames", "prompt_len", "n_tokens", "feat_ms",
           "encode_ms", "decode_ms", "asr_ms", "ms_per_token", "peak_mem_bytes", "kv_bytes",
           "empty", "transcript"]
REPO = Path(__file__).resolve().parents[3]
CORPUS = REPO / "rag" / "data" / "rag_corpus"


CONFIG_KEYS = ("split", "cuda", "variant", "setting")


def _existing(out_csv: Path) -> list[dict]:
    if not out_csv.exists():
        return []
    with out_csv.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_bench(model, items, variant, setting, out_csv, cache_path, warmup, repeat_ids, meta) -> int:
    """model: object with transcribe(item) -> dict of measurements and after_warmup().

    Writes pass 0 for every item, then pass 1 for each id in repeat_ids. Rows already in
    out_csv are skipped (resume). Returns the number of rows written by this call.
    """
    out_csv, cache_path = Path(out_csv), Path(cache_path)
    existing = _existing(out_csv)
    current = {**meta, "variant": variant, "setting": setting}
    for key in CONFIG_KEYS:
        found = {r[key] for r in existing} - {str(current[key])}
        if found:
            raise ValueError(f"{out_csv} already holds rows with {key}={sorted(found)}, this run has "
                             f"{key}={current[key]}; use a new --out-csv/--cache for a new configuration")
    done = {(r["query_id"], int(r["pass"])) for r in existing}
    cache = read_cache(cache_path)
    for r in existing:  # a run killed between the CSV row and the cache line: backfill from the CSV
        if r["pass"] == "0" and r["query_id"] not in cache:
            append_cache(cache_path, r["query_id"], r["transcript"])
            cache[r["query_id"]] = r["transcript"]
    for item in items[:warmup]:
        model.transcribe(item)
    model.after_warmup()

    by_id = {it["query_id"]: it for it in items}
    work = [(it, 0) for it in items] + [(by_id[q], 1) for q in repeat_ids]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out_csv.exists()
    written = 0
    with out_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            w.writeheader()
        for item, p in work:
            qid = item["query_id"]
            if (qid, p) in done:
                continue
            m = model.transcribe(item)
            asr_ms = m["feat_ms"] + m["encode_ms"] + m["decode_ms"]
            row = {**meta, **{k: m[k] for k in m if k in COLUMNS},
                   "query_id": qid, "variant": variant, "setting": setting, "pass": p,
                   "asr_ms": round(asr_ms, 3),
                   "ms_per_token": round(m["decode_ms"] / max(m["n_tokens"], 1), 3),
                   "empty": int(not m["transcript"].strip())}
            if qid not in cache:  # cache first, so a kill between the two never leaves a cache hole
                append_cache(cache_path, qid, m["transcript"])
                cache[qid] = m["transcript"]
            w.writerow(row)
            f.flush()
            done.add((qid, p))
            written += 1
            print(f"[{variant}/{setting} p{p}] {qid} asr={asr_ms:.1f} ms tokens={m['n_tokens']}", flush=True)
    return written


class GpuTranscriber:
    """Real transcriber: FastGlmAsr + processor for one variant/setting."""

    def __init__(self, fast, processor, variant, margin_s=None, bucket_s=None, max_new_tokens=100):
        import torch
        from extensions.fast_asr.audio import read_audio
        from extensions.fast_asr.kv_model import prepare_inputs, real_frames
        from extensions.fast_asr.trim import trimmed_frames
        self.torch, self.read_audio, self.prepare, self.real = torch, read_audio, prepare_inputs, real_frames
        self.trimmed_frames = trimmed_frames
        self.fast, self.processor, self.variant = fast, processor, variant
        self.margin_s, self.bucket_s, self.max_new_tokens = margin_s, bucket_s, max_new_tokens

    def after_warmup(self):
        self.torch.cuda.synchronize()
        self.torch.cuda.reset_peak_memory_stats()

    def transcribe(self, item):
        audio = self.read_audio(CORPUS / item["audio_file"] if "audio_file" in item else item["path"])
        inp = self.prepare(self.processor, audio)
        feat_ms = inp.pop("feat_ms")
        rf = self.real(inp["features_mask"])
        if self.variant == "v1":
            r = self.fast.generate_recompute(**inp, max_new_tokens=self.max_new_tokens)
        elif self.variant == "v2":
            r = self.fast.generate_kv(**inp, max_new_tokens=self.max_new_tokens)
        else:
            T = self.trimmed_frames(rf, margin_s=self.margin_s, bucket_s=self.bucket_s)
            r = self.fast.generate_kv(**inp, max_new_tokens=self.max_new_tokens, encoder_frames=T)
        text = self.processor.tokenizer.decode(r.tokens, skip_special_tokens=True).strip()
        return {"duration_s": round(len(audio) / 16000, 3), "real_frames": rf, "enc_frames": r.enc_frames,
                "prompt_len": r.prompt_len, "n_tokens": len(r.tokens), "feat_ms": round(feat_ms, 3),
                "encode_ms": round(r.encode_ms, 3), "decode_ms": round(r.decode_ms, 3),
                "peak_mem_bytes": self.torch.cuda.max_memory_allocated(), "kv_bytes": r.kv_bytes,
                "transcript": text}


def load_items(split, ids_file=None, limit=None):
    rows = [json.loads(l) for l in (CORPUS / "student_audio_manifest.jsonl").read_text().splitlines() if l.strip()]
    items = [r for r in rows if r["split"] == split]
    if ids_file:
        wanted = [l.strip() for l in Path(ids_file).read_text().splitlines() if l.strip()]
        by_id = {r["query_id"]: r for r in items}
        missing = [q for q in wanted if q not in by_id]
        if missing:
            raise SystemExit(f"{len(missing)} ids not in {split}: {missing[:3]}")
        items = [by_id[q] for q in wanted]
    return items[:limit] if limit else items


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["practice", "dev"], required=True)
    ap.add_argument("--variant", choices=["v1", "v2", "v3"], required=True)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--margin", type=float, help="v3: seconds of padding kept after the audio")
    g.add_argument("--bucket", type=float, help="v3: encode at least this many seconds")
    ap.add_argument("--cuda", default="all")
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--ids", type=Path)
    ap.add_argument("--repeat-subset", type=int, default=0, help="repeat the first N queries once")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-new-tokens", type=int, default=100)
    args = ap.parse_args()
    if args.variant == "v3" and args.margin is None and args.bucket is None:
        ap.error("v3 needs --margin or --bucket")
    if args.variant != "v3" and (args.margin is not None or args.bucket is not None):
        ap.error("--margin/--bucket only apply to v3")
    setting = "full" if args.variant != "v3" else (f"m{args.margin:g}" if args.margin is not None
                                                     else f"b{args.bucket:g}")

    import torch
    from extensions.fast_asr import _paths  # noqa: F401
    from glm_asr_cuda import ops
    from extensions.fast_asr.kv_model import load_fast
    if not torch.cuda.is_available():
        sys.exit("No GPU visible. Run on a GPU node.")
    ops.set_cuda_groups(args.cuda)
    t0 = time.perf_counter()
    fast, processor = load_fast()
    print(f"Model loaded in {time.perf_counter() - t0:.1f} s (excluded from timings)", flush=True)

    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    meta = {"split": args.split, "cuda": args.cuda, "gpu": torch.cuda.get_device_name(),
            "node": socket.gethostname(), "commit": commit}
    items = load_items(args.split, args.ids, args.limit)
    repeat_ids = [it["query_id"] for it in items[:args.repeat_subset]]
    tr = GpuTranscriber(fast, processor, args.variant, args.margin, args.bucket, args.max_new_tokens)
    n = run_bench(tr, items, args.variant, setting, args.out_csv, args.cache, args.warmup, repeat_ids, meta)
    print(f"Wrote {n} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
