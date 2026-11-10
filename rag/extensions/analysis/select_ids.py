#!/usr/bin/env python3
"""Query IDs to judge: every query whose variant transcript differs from V1, plus an equal
(or given) number of unchanged controls drawn deterministically.

    python rag/extensions/analysis/select_ids.py --v1 rag/.cache/dev_v1.jsonl \
        --variant rag/.cache/dev_v3_m0.jsonl --out results/judge/dev_v3_m0_ids.txt
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from extensions.fast_asr.cachefile import read_cache  # noqa: E402


def select_ids(v1_cache: dict, var_cache: dict, n_controls: int, seed: int = 0) -> list[str]:
    shared = sorted(set(v1_cache) & set(var_cache))
    differing = [q for q in shared if v1_cache[q] != var_cache[q]]
    unchanged = [q for q in shared if v1_cache[q] == var_cache[q]]
    controls = random.Random(seed).sample(unchanged, min(n_controls, len(unchanged)))
    return sorted(differing + controls)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v1", type=Path, required=True)
    ap.add_argument("--variant", type=Path, required=True)
    ap.add_argument("--controls", type=int, help="default: as many as differing")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    v1, var = read_cache(args.v1), read_cache(args.variant)
    n_diff = sum(1 for q in set(v1) & set(var) if v1[q] != var[q])
    ids = select_ids(v1, var, args.controls if args.controls is not None else n_diff, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(q + "\n" for q in ids))
    print(f"{n_diff} differing + {len(ids) - n_diff} controls -> {args.out}")


if __name__ == "__main__":
    main()
