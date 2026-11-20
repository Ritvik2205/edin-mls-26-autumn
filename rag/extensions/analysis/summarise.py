#!/usr/bin/env python3
"""Turn bench CSVs and evaluate_rag outputs into report tables and figures (runs locally).

    python rag/extensions/analysis/summarise.py --split dev --out report

Inputs:  results/asr/{split}_*.csv            (bench_asr.py)
         results/asr_torch/{split}_*.csv      (optional: the same variants with ASR_CUDA=none)
         results/asr_rep/{split}_*.csv        (optional: a same-node replicate of results/asr)
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


STALL_MS = 1000.0  # feature-extraction time above this marks a host stall


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
    for (variant, setting), rs in sorted(groups.items(), key=lambda kv: _order(*kv[0])):
        paired = [r for r in rs if r["query_id"] in v1]
        # A query whose CPU feature extraction took over STALL_MS (median: about 20 ms) hit a host
        # stall on a shared node; it is left out of the timing statistics, not of the quality ones.
        timed = [r for r in rs if not r["feat_ms"] > STALL_MS]
        t = {"variant": variant, "setting": setting, "n": len(rs), "stalled": len(rs) - len(timed),
             "asr_ms": stats([r["asr_ms"] for r in timed]),
             "encode_ms": stats([r["encode_ms"] for r in timed]),
             "prefill_ms": stats([r["prefill_ms"] for r in timed]),
             "decode_ms": stats([r["decode_ms"] for r in timed]),
             "ms_per_token": stats([r["ms_per_token"] for r in timed]),
             "peak_gib": max(r["peak_mem_bytes"] for r in rs) / 2**30,
             "empty": int(sum(r["empty"] for r in rs))}
        if paired:
            t["changed"] = sum(v1[r["query_id"]] != r["transcript"] for r in paired)
            t["match_pct"] = 100.0 * sum(v1[r["query_id"]] == r["transcript"] for r in paired) / len(paired)
            t["wer_vs_v1"] = float(np.mean([wer(v1[r["query_id"]], r["transcript"]) for r in paired]))
        out.append(t)
    return out


def _order(variant, setting):
    """V1, V2, then V3 by margin (m0, m2, ...), then buckets."""
    base, _, gemv = setting.partition("+")  # "m0+seq": V3-m0 with the seq GEMV kernel
    return (variant, base[:1] != "m", float(base[1:]) if base != "full" else 0.0, gemv)


def _label(t):
    base, _, gemv = t["setting"].partition("+")
    return t["variant"].upper() + ("" if base == "full" else f"-{base}") + (f"+{gemv}" if gemv else "")


def _transcripts(rows, variant, setting):
    return {r["query_id"]: r["transcript"] for r in rows if (r["variant"], r["setting"]) == (variant, setting)}


def _row(label, t, r10, rerun=None):
    cells = [label, f"{t['asr_ms']['mean']:.0f}"]
    if rerun is not None:
        cells.append(rerun)
    cells += [f"{t['asr_ms']['p95']:.0f}", f"{t['encode_ms']['mean']:.0f}", f"{t['decode_ms']['mean']:.0f}",
              f"{t['ms_per_token']['mean']:.1f}",
              f"{t['changed']}" if "changed" in t else "--",
              f"{100 * t['wer_vs_v1']:.1f}" if "wer_vs_v1" in t else "--", r10]
    return " & ".join(cells) + r" \\"


def write_table(tables, retrieval, gpu, split, path: Path, rows=(), torch_rows=(), rep_tables=(),
                gemv_rows=()):
    """One row per variant. torch_rows: the same variants on the PyTorch path, listed under a rule
    and compared with the kernel-path V1 transcripts. rep_tables: a same-node replicate, shown as
    one extra column of mean ASR ms. gemv_rows: runs with a GEMV kernel (setting "full+seq",
    "m0+warp", ...), made on the replicate's node, so their mean goes in the rerun column."""
    v1r = retrieval.get(("v1", "full"))
    rep = {(t["variant"], t["setting"]): t for t in rep_tables}

    def r10_cell(key):
        r = retrieval.get(key)
        if not r:
            return "--"
        if key == ("v1", "full") or not v1r:
            return f"{r['aggregate'].get('Recall@10', float('nan')):.3f}"
        q = sorted(set(r["per_query"]) & set(v1r["per_query"]))
        d, lo, hi = paired_bootstrap_ci([v1r["per_query"][x] for x in q], [r["per_query"][x] for x in q])
        if lo == hi == 0.0:  # no query's Recall@10 differs from V1
            return "identical"
        return f"${d:+.3f}$ [${lo:+.3f}$, ${hi:+.3f}$]"

    def rerun_cell(key):
        if not rep:
            return None
        return f"{rep[key]['asr_ms']['mean']:.0f}" if key in rep else "--"

    head = ["Variant", "ASR"] + (["rerun"] if rep else []) + \
           ["p95", "enc", "dec", "step", "chg.", "WER", r"R@10; $\Delta$ [95\% CI]"]
    lines = [r"\begin{tabular}{l" + "r" * (len(head) - 2) + "l}", r"\toprule", " & ".join(head) + r" \\", r"\midrule"]
    for t in tables:
        key = (t["variant"], t["setting"])
        lines.append(_row(_label(t), t, r10_cell(key), rerun_cell(key)))
    v1 = _transcripts(rows, "v1", "full")
    if gemv_rows:
        lines.append(r"\midrule")
        for t in variant_table(list(gemv_rows)):
            key = (t["variant"], t["setting"])
            base = (t["variant"], t["setting"].partition("+")[0])
            mine = _transcripts(gemv_rows, *key)
            same = mine == _transcripts(rows, *base)  # then the base variant's retrieval score holds
            t["changed"] = sum(v1[q] != mine[q] for q in mine if q in v1)
            t["wer_vs_v1"] = float(np.mean([wer(v1[q], mine[q]) for q in mine if q in v1]))
            cells = _row(_label(t), t, r10_cell(base) if same else "--", f"{t['asr_ms']['mean']:.0f}" if rep else None)
            if rep:  # no run on the first table's nodes
                cells = cells.replace(f"& {t['asr_ms']['mean']:.0f} &", "& -- &", 1)
            lines.append(cells)
    if torch_rows:
        lines.append(r"\midrule")
        for t in variant_table(list(torch_rows)):
            key = (t["variant"], t["setting"])
            mine = _transcripts(torch_rows, *key)
            # Retrieval was scored on the kernel-path transcripts; it carries over only if this
            # path produced the same transcript for every query.
            same = mine == _transcripts(rows, *key)
            t["changed"] = sum(v1[q] != mine[q] for q in mine if q in v1)
            t["wer_vs_v1"] = float(np.mean([wer(v1[q], mine[q]) for q in mine if q in v1]))
            label = "V0 (torch V1)" if key == ("v1", "full") else "torch " + _label(t)
            lines.append(_row(label, t, r10_cell(key) if same else "--", "--" if rep else None))
    lines += [r"\bottomrule", r"\end{tabular}",
              f"% split={split}, n per variant={[t['n'] for t in tables]}, GPU={gpu}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def write_figures(tables, rows, retrieval, gpu, split, out: Path):
    """Figures are drawn at their printed size (about half a text column) so 7-8 pt text stays legible."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "legend.fontsize": 7, "axes.titlesize": 7})
    out.mkdir(parents=True, exist_ok=True)
    n = tables[0]["n"] if tables else 0
    note = f"{split}, n={n}"
    colours = {_label(t): f"C{i}" for i, t in enumerate(tables)}

    fig, ax = plt.subplots(figsize=(2.35, 2.3))
    labels = [_label(t) for t in tables]
    feat = [t["asr_ms"]["mean"] - t["encode_ms"]["mean"] - t["decode_ms"]["mean"] for t in tables]
    enc = [t["encode_ms"]["mean"] for t in tables]
    dec = [t["decode_ms"]["mean"] for t in tables]
    ax.bar(labels, feat, label="CPU mel features")
    ax.bar(labels, enc, bottom=feat, label="encoder (GPU)")
    ax.bar(labels, dec, bottom=np.add(feat, enc), label="decoder (GPU)")
    ax.set_ylabel("mean ASR latency per query (ms)")
    ax.set_title(note)
    ax.legend(handlelength=1.0)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out / f"breakdown_{split}.pdf")
    plt.close(fig)

    # Latency against the paired Recall@10 difference from V1, with its bootstrap interval. The
    # absolute Recall@10 values coincide to three decimals, so plotting them hides the comparison.
    v1r = retrieval.get(("v1", "full"))
    pts = [(t, retrieval[(t["variant"], t["setting"])]) for t in tables if (t["variant"], t["setting"]) in retrieval]
    if pts and v1r:
        fig, ax = plt.subplots(figsize=(3.3, 2.5))
        markers = ["o", "s", "^", "D", "v", "P", "X"]
        for i, (t, r) in enumerate(pts):
            q = sorted(set(r["per_query"]) & set(v1r["per_query"]))
            d, lo, hi = paired_bootstrap_ci([v1r["per_query"][x] for x in q], [r["per_query"][x] for x in q])
            ax.errorbar(t["asr_ms"]["mean"], d, yerr=[[d - lo], [hi - d]], fmt=markers[i % len(markers)],
                        ms=4, capsize=3, color=colours[_label(t)], label=_label(t))
        ax.axhline(0.0, lw=0.6, color="grey")
        ax.set_ylim(-0.03, 0.03)
        ax.set_xlabel("mean ASR latency per query (ms)")
        ax.set_ylabel("Recall@10 change vs V1 (95% CI)")
        ax.set_title(note)
        ax.legend(ncol=2)
        fig.tight_layout()
        fig.savefig(out / f"tradeoff_{split}.pdf")
        plt.close(fig)

    v1 = {r["query_id"]: r for r in rows if r["variant"] == "v1"}
    if v1:
        fig, ax = plt.subplots(figsize=(3.15, 2.3))
        for t in tables:
            if t["variant"] == "v1":
                continue
            rs = [r for r in rows if (r["variant"], r["setting"]) == (t["variant"], t["setting"])
                  and r["query_id"] in v1]
            ax.scatter([r["duration_s"] for r in rs], [v1[r["query_id"]]["asr_ms"] / r["asr_ms"] for r in rs],
                       s=5, color=colours[_label(t)], label=_label(t))
        ax.set_xlabel("query audio duration (s)")
        ax.set_ylabel("ASR speed-up over V1 (x)")
        ax.set_title(note)
        ax.set_ylim(top=ax.get_ylim()[1] * 1.18)  # headroom so the legend covers no points
        ax.legend(ncol=4, loc="upper center", columnspacing=0.6, handletextpad=0.1, borderpad=0.3)
        fig.tight_layout()
        fig.savefig(out / f"speedup_{split}.pdf")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["practice", "dev"], required=True)
    ap.add_argument("--results", type=Path, default=REPO / "results")
    ap.add_argument("--out", type=Path, default=REPO / "report" / "asr-redundancy")
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
    def optional(sub):
        found = sorted((args.results / sub).glob(f"{args.split}_*.csv"))
        return main_rows(load_runs(found)) if found else []
    write_table(tables, retrieval, gpu, args.split, args.out / "tables" / f"asr_{args.split}.tex",
                rows=rows, torch_rows=optional("asr_torch"), rep_tables=variant_table(optional("asr_rep")),
                gemv_rows=optional("asr_gemv"))
    write_figures(tables, rows, retrieval, gpu, args.split, args.out / "figures")
    for t in tables:
        print(f"{_label(t):<10} asr {t['asr_ms']['mean']:8.1f} ms  match {t.get('match_pct', float('nan')):5.1f}%")


if __name__ == "__main__":
    main()
