"""Transcript caches in the JSON-lines format rag/scripts/evaluate_rag.py reads."""
from __future__ import annotations

import json
from pathlib import Path


def read_cache(path: Path) -> dict[str, str]:
    path = Path(path)
    cache = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    cache[row["query_id"]] = row["transcript"]
    return cache


def append_cache(path: Path, query_id: str, transcript: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"query_id": query_id, "transcript": transcript},
                           ensure_ascii=False) + "\n")
