#!/usr/bin/env python3
"""Turn bench CSVs and evaluate_rag outputs into report tables and figures (runs locally).

    python rag/extensions/analysis/summarise.py --split dev --out report

Inputs:  results/asr/{split}_*.csv            (bench_asr.py)
         results/retrieval/{split}_<name>.txt (tee'd evaluate_rag.py output; <name> matches
                                               the CSV stem after the split, e.g. dev_v3_m0)
Outputs: <out>/tables/asr_{split}.tex, <out>/figures/{breakdown,tradeoff,speedup}_{split}.pdf
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from extensions.analysis.textmetrics import wer  # noqa: E402

NUMERIC = {"pass", "duration_s", "real_frames", "enc_frames", "prompt_len", "n_tokens", "feat_ms",
           "encode_ms", "prefill_ms", "decode_ms", "asr_ms", "ms_per_token", "peak_mem_bytes", "kv_bytes", "empty"}
REPO = Path(__file__).resolve().parents[3]


def load_runs(paths) -> list[dict]:
    rows, gpus = [], set()
    for p in paths:
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                for k in NUMERIC & r.keys():
                    r[k] = float(r[k]) if r[k] != "" else float("nan")
                rows.append(r)
                gpus.add(r["gpu"])
    if len(gpus) > 1:
        raise ValueError(f"results from more than one GPU type: {', '.join(sorted(gpus))}")
    return rows


def main_rows(rows) -> list[dict]:
    return [r for r in rows if r["pass"] == 0]


def stats(xs) -> dict:
    a = np.asarray(xs, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std()), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)), "n": int(a.size)}


def paired_bootstrap_ci(a, b, iters=10000, seed=0):
    """Mean of (b - a) with a 95% percentile bootstrap interval over paired queries."""
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, d.size, size=(iters, d.size))].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def parse_retrieval(text: str) -> dict:
    per_query = {m.group(1): float(m.group(2))
                 for m in re.finditer(r"^\[\s*\d+/\d+\]\s+(\S+)\s+R@10=([\d.]+)", text, re.M)}
    agg = {m.group(1): float(m.group(2))
           for m in re.finditer(r"^\s+((?:Recall|MRR|NDCG)@\d+)\s+([\d.]+)", text, re.M)}
    return {"per_query": per_query, "aggregate": agg}


def variant_table(rows) -> list[dict]:
    """One summary per (variant, setting); transcript agreement is measured against v1."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["variant"], r["setting"])].append(r)
    v1 = {r["query_id"]: r["transcript"] for r in groups.get(("v1", "full"), [])}
    out = []
    for (variant, setting), rs in sorted(groups.items()):
        paired = [r for r in rs if r["query_id"] in v1]
        t = {"variant": variant, "setting": setting, "n": len(rs),
             "asr_ms": stats([r["asr_ms"] for r in rs]),
             "encode_ms": stats([r["encode_ms"] for r in rs]),
             "prefill_ms": stats([r["prefill_ms"] for r in rs]),
             "decode_ms": stats([r["decode_ms"] for r in rs]),
             "ms_per_token": stats([r["ms_per_token"] for r in rs]),
             "peak_gib": max(r["peak_mem_bytes"] for r in rs) / 2**30,
             "empty": int(sum(r["empty"] for r in rs))}
        if paired:
            t["match_pct"] = 100.0 * sum(v1[r["query_id"]] == r["transcript"] for r in paired) / len(paired)
            t["wer_vs_v1"] = float(np.mean([wer(v1[r["query_id"]], r["transcript"]) for r in paired]))
        out.append(t)
    return out


def _label(t):
    return t["variant"].upper() + ("" if t["setting"] == "full" else f" {t['setting']}")


def write_table(tables, retrieval, gpu, split, path: Path):
    v1r = retrieval.get(("v1", "full"))
    lines = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
             r"Variant & ASR ms (mean) & p95 & enc ms & dec ms & ms/token & match \% & WER & R@10 \\",
             r"\midrule"]
    for t in tables:
        r = retrieval.get((t["variant"], t["setting"]))
        r10 = f"{r['aggregate'].get('Recall@10', float('nan')):.3f}" if r else "--"
        if r and v1r and t["variant"] != "v1":
            q = sorted(set(r["per_query"]) & set(v1r["per_query"]))
            d, lo, hi = paired_bootstrap_ci([v1r["per_query"][x] for x in q], [r["per_query"][x] for x in q])
            r10 += f" ({d:+.3f} [{lo:+.3f}, {hi:+.3f}])"
        lines.append(f"{_label(t)} & {t['asr_ms']['mean']:.1f} & {t['asr_ms']['p95']:.1f} & "
                     f"{t['encode_ms']['mean']:.1f} & {t['decode_ms']['mean']:.1f} & "
                     f"{t['ms_per_token']['mean']:.1f} & {t.get('match_pct', float('nan')):.1f} & "
                     f"{t.get('wer_vs_v1', float('nan')):.3f} & {r10} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              f"% split={split}, n per variant={[t['n'] for t in tables]}, GPU={gpu}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def write_figures(tables, rows, retrieval, gpu, split, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    n = tables[0]["n"] if tables else 0
    note = f"{split}, n={n}, {gpu}"

    fig, ax = plt.subplots(figsize=(6, 3))
    labels = [_label(t) for t in tables]
    feat = [t["asr_ms"]["mean"] - t["encode_ms"]["mean"] - t["decode_ms"]["mean"] for t in tables]
    enc = [t["encode_ms"]["mean"] for t in tables]
    dec = [t["decode_ms"]["mean"] for t in tables]
    ax.bar(labels, feat, label="CPU mel features")
    ax.bar(labels, enc, bottom=feat, label="encoder (GPU)")
    ax.bar(labels, dec, bottom=np.add(feat, enc), label="decoder (GPU)")
    ax.set_ylabel("mean ASR latency per query (ms)")
    ax.set_title(note, fontsize=8)
    ax.legend(fontsize=7)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out / f"breakdown_{split}.pdf")
    plt.close(fig)

    pts = [(t["asr_ms"]["mean"], retrieval[(t["variant"], t["setting"])]["aggregate"]["Recall@10"], _label(t))
           for t in tables if (t["variant"], t["setting"]) in retrieval]
    if pts:
        fig, ax = plt.subplots(figsize=(5, 3))
        for x, y, lab in pts:
            ax.scatter(x, y)
            ax.annotate(lab, (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("mean ASR latency per query (ms)")
        ax.set_ylabel("Recall@10")
        ax.set_title(note, fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"tradeoff_{split}.pdf")
        plt.close(fig)

    v1 = {r["query_id"]: r for r in rows if r["variant"] == "v1"}
    if v1:
        fig, ax = plt.subplots(figsize=(5, 3))
        for t in tables:
            if t["variant"] == "v1":
                continue
            rs = [r for r in rows if (r["variant"], r["setting"]) == (t["variant"], t["setting"])
                  and r["query_id"] in v1]
            ax.scatter([r["duration_s"] for r in rs], [v1[r["query_id"]]["asr_ms"] / r["asr_ms"] for r in rs],
                       s=6, label=_label(t))
        ax.set_xlabel("query audio duration (s)")
        ax.set_ylabel("ASR speed-up vs V1 (x)")
        ax.set_title(note, fontsize=8)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / f"speedup_{split}.pdf")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["practice", "dev"], required=True)
    ap.add_argument("--results", type=Path, default=REPO / "results")
    ap.add_argument("--out", type=Path, default=REPO / "report")
    args = ap.parse_args()
    csvs = sorted((args.results / "asr").glob(f"{args.split}_*.csv"))
    if not csvs:
        sys.exit(f"no results/asr/{args.split}_*.csv yet")
    rows = main_rows(load_runs(csvs))
    gpu = rows[0]["gpu"]
    tables = variant_table(rows)
    retrieval = {}
    for t in tables:
        name = t["variant"] if t["setting"] == "full" else f"{t['variant']}_{t['setting']}"
        p = args.results / "retrieval" / f"{args.split}_{name}.txt"
        if p.exists():
            retrieval[(t["variant"], t["setting"])] = parse_retrieval(p.read_text())
    write_table(tables, retrieval, gpu, args.split, args.out / "tables" / f"asr_{args.split}.tex")
    write_figures(tables, rows, retrieval, gpu, args.split, args.out / "figures")
    for t in tables:
        print(f"{_label(t):<10} asr {t['asr_ms']['mean']:8.1f} ms  match {t.get('match_pct', float('nan')):5.1f}%")


if __name__ == "__main__":
    main()
