#!/usr/bin/env python3
"""Paired answer-score differences from the LLM judge, changed transcripts and controls apart.

Reads what scripts/judge_changed.sh wrote for one split:
    results/judge/{split}_{name}.jsonl                         answers, transcripts, timings
    results/judge/{split}_{name}/answer_judge_per_query.jsonl  judge scores (mean of the repeats)
for name in v1, v1b, v3_m0, v3_m2, v3_b10. v1b is a second generation from the V1 transcripts,
so v1b - v1 is the noise of generation plus judging with the input held fixed.

For each variant it prints the mean of (variant - v1) in the normalised answer score with a 95%
paired bootstrap interval, over (a) the queries whose transcript differs from V1 under that
variant and (b) the rest of the judged queries, and the generation and retrieval latency.

    python rag/extensions/analysis/judge_summary.py --split dev [--tex report/.../judge_dev.tex]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

RAG = Path(__file__).resolve().parents[2]
if str(RAG) not in sys.path:
    sys.path.insert(0, str(RAG))

from extensions.analysis.summarise import paired_bootstrap_ci  # noqa: E402

REPO = RAG.parent
NAMES = ("v1b", "v3_m0", "v3_m2", "v3_b10")
LABELS = {"v1b": "V1 again", "v3_m0": "V3-m0", "v3_m2": "V3-m2", "v3_b10": "V3-b10"}


def read_jsonl(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return {r["query_id"]: r for r in map(json.loads, f) if r}


def load(root: Path, split: str, name: str):
    answers = read_jsonl(root / f"{split}_{name}.jsonl")
    scores = read_jsonl(root / f"{split}_{name}" / "answer_judge_per_query.jsonl")
    return answers, {q: s["normalised_answer_score"] for q, s in scores.items()}


def diff(base: dict, other: dict, ids) -> dict | None:
    ids = sorted(ids)
    if not ids:
        return None
    a, b = [base[q] for q in ids], [other[q] for q in ids]
    d, lo, hi = paired_bootstrap_ci(a, b)
    return {"n": len(ids), "base": statistics.mean(a), "other": statistics.mean(b),
            "delta": d, "lo": lo, "hi": hi, "lower": sum(y < x for x, y in zip(a, b)),
            "higher": sum(y > x for x, y in zip(a, b))}


def summarise(root: Path, split: str) -> dict:
    """split "both" pools the dev and practice queries (query ids do not overlap)."""
    splits = ("dev", "practice") if split == "both" else (split,)

    def pooled(name):
        answers, scores = {}, {}
        for sp in splits:
            a, s = load(root, sp, name)
            answers.update(a)
            scores.update(s)
        return answers, scores

    v1_answers, v1_scores = pooled("v1")
    out = {"n": len(v1_scores), "v1_mean": statistics.mean(v1_scores.values()), "variants": {},
           "generation_ms": [], "retrieval_ms": []}
    for name in ("v1",) + NAMES:
        answers, scores = pooled(name)
        assert set(scores) == set(v1_scores), name
        out["generation_ms"] += [r["timing"]["generation_ms"] for r in answers.values()]
        out["retrieval_ms"] += [r["timing"]["retrieval_ms"] for r in answers.values()]
        if name == "v1":
            continue
        changed = {q for q, r in answers.items() if r["asr_transcript"] != v1_answers[q]["asr_transcript"]}
        out["variants"][name] = {"changed": diff(v1_scores, scores, changed),
                                 "same": diff(v1_scores, scores, set(scores) - changed)}
    return out


def _cell(d) -> str:
    if d is None:
        return "--"
    return f"{d['delta']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"


def write_tex(s: dict, path: Path):
    lines = [r"\begin{tabular}{lrlrl}", r"\toprule",
             r" & \multicolumn{2}{c}{transcript changed} & \multicolumn{2}{c}{transcript same} \\",
             r"Variant & $n$ & $\Delta$ score [95\% CI] & $n$ & $\Delta$ score [95\% CI] \\", r"\midrule"]
    for name, v in s["variants"].items():
        c, u = v["changed"], v["same"]
        lines.append(f"{LABELS[name]} & {c['n'] if c else 0} & {_cell(c)} & {u['n'] if u else 0} & {_cell(u)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["practice", "dev", "both"], required=True)
    ap.add_argument("--root", type=Path, default=REPO / "results" / "judge")
    ap.add_argument("--tex", type=Path)
    args = ap.parse_args()
    s = summarise(args.root, args.split)
    print(f"{args.split}: {s['n']} judged queries, V1 mean score {s['v1_mean']:.3f}")
    for name, v in s["variants"].items():
        for kind in ("changed", "same"):
            d = v[kind]
            if d:
                print(f"  {LABELS[name]:<9} {kind:<8} n={d['n']:<3} v1 {d['base']:.3f} -> {d['other']:.3f}  "
                      f"delta {_cell(d)}  lower {d['lower']}, higher {d['higher']}")
    g, r = s["generation_ms"], s["retrieval_ms"]
    q = statistics.quantiles(g, n=20)
    print(f"  generation: median {statistics.median(g):.0f} ms, p5 {q[0]:.0f}, p95 {q[-1]:.0f} (n={len(g)}); "
          f"retrieval: median {statistics.median(r):.1f} ms")
    if args.tex:
        write_tex(s, args.tex)
        print(f"wrote {args.tex}")


if __name__ == "__main__":
    main()
