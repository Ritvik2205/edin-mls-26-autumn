#!/usr/bin/env python3
"""Word-level transcript changes under trimming, checked against the relevant passages.

    python rag/extensions/analysis/changed_words.py

There are no reference transcripts. For every query whose trimmed transcript differs from V1's
in more than spacing, case or punctuation, this prints both transcripts, the words only one of
them has, and whether each such word occurs in a passage the relevance labels mark as relevant
to that query. Reads results/asr/{split}_{v1,v3_*}.csv and rag/data/rag_corpus/.
"""
from __future__ import annotations

import collections
import csv
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CORPUS = REPO / "rag" / "data" / "rag_corpus"


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower())


def transcripts(path: Path) -> dict:
    with open(path, newline="", encoding="utf-8") as f:
        return {r["query_id"]: r["transcript"] for r in csv.DictReader(f) if r["pass"] == "0"}


def word_level(a: str, b: str) -> bool:
    """True if a and b differ in more than spacing, case or punctuation."""
    return norm(a).replace(" ", "") != norm(b).replace(" ", "")


def main():
    corpus = {}
    with open(CORPUS / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            corpus[r["record_id"]] = r["text"]
    total = trimmed_in = v1_in = 0
    for split in ("dev", "practice"):
        relevant = collections.defaultdict(list)
        for line in (CORPUS / f"qrels_{split}.tsv").read_text().splitlines()[1:]:
            q, rec, _ = line.split("\t")
            relevant[q].append(rec)
        v1 = transcripts(REPO / "results" / "asr" / f"{split}_v1.csv")
        seen = set()
        for path in sorted((REPO / "results" / "asr").glob(f"{split}_v3_*.csv")):
            for q, text in transcripts(path).items():
                if not word_level(v1[q], text) or (q, text) in seen:
                    continue
                seen.add((q, text))
                passage = set(norm(" ".join(corpus[r] for r in relevant[q])).split())
                a, b = norm(v1[q]).split(), norm(text).split()
                only_v1 = [w for w in a if w not in b]
                only_trim = [w for w in b if w not in a]
                hit_v1 = any(w in passage for w in only_v1)
                hit_trim = any(w in passage for w in only_trim)
                total += 1
                trimmed_in += hit_trim and not hit_v1
                v1_in += hit_v1 and not hit_trim
                print(f"{split} {path.stem.split('_', 1)[1]} {q}\n  V1:      {v1[q]}\n  trimmed: {text}\n"
                      f"  only in V1 {only_v1} (in a relevant passage: {hit_v1}); "
                      f"only in trimmed {only_trim} (in a relevant passage: {hit_trim})")
    print(f"\n{total} word-level changes; only the trimmed wording occurs in a relevant passage: {trimmed_in}; "
          f"only V1's: {v1_in}; both or neither: {total - trimmed_in - v1_in}")


if __name__ == "__main__":
    main()
