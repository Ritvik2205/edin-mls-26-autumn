#!/bin/bash
# Answer quality where trimming changed the transcript. No ASR, no GPU: transcripts come from the
# caches the benchmark runs wrote. Needs internet (ELM), so run on the head node, from the
# repository root, in the RAG venv:
#     nice -n 19 bash scripts/judge_changed.sh dev [limit]
# For one split:
#   1. ids = every query whose transcript differs from V1 under any V3 setting, plus the same
#      number of queries unchanged under all of them (seed 0).
#   2. For V1 (twice: v1 and v1b), V3-m0, V3-m2 and V3-b10, retrieve and generate an answer from
#      that variant's transcript for every id. v1 vs v1b is the generation + judge noise floor.
#   3. Judge every answer against the V1 transcript as the question, 3 judge repeats.
# Outputs: results/judge/{split}_ids.txt, results/judge/{split}_{variant}.jsonl (answers, with
# retrieval and generation times), results/judge/{split}_{variant}/ (judge scores).
# A limit (e.g. 5) writes to results/judge/smoke_* instead and is a connectivity test only.
set -euo pipefail
split=${1:?usage: judge_changed.sh <dev|practice> [limit]}
limit=${2:-}
PY=${RAG_PYTHON:-$HOME/mls-rag-env/bin/python}
out=results/judge
tag=$split
mkdir -p "$out"

"$PY" - "$split" "$out/${split}_ids.txt" <<'EOF'
import random, sys
from pathlib import Path
sys.path.insert(0, "rag")
from extensions.fast_asr.cachefile import read_cache
split, dest = sys.argv[1], Path(sys.argv[2])
v1 = read_cache(Path(f"rag/.cache/{split}_v1.jsonl"))
changed = set()
for name in ("v3_m0", "v3_m2", "v3_b10"):
    var = read_cache(Path(f"rag/.cache/{split}_{name}.jsonl"))
    assert set(var) == set(v1), name
    changed |= {q for q in v1 if var[q] != v1[q]}
unchanged = sorted(set(v1) - changed)
controls = random.Random(0).sample(unchanged, min(len(changed), len(unchanged)))
dest.write_text("".join(q + "\n" for q in sorted(changed | set(controls))))
print(f"{split}: {len(changed)} changed + {len(controls)} controls -> {dest}")
EOF

ids="$out/${split}_ids.txt"
if [ -n "$limit" ]; then
    tag="smoke_$split"
    head -n "$limit" "$ids" > "$out/${tag}_ids.txt"
    ids="$out/${tag}_ids.txt"
fi

for name in v1 v1b v3_m0 v3_m2 v3_b10; do
    cache="rag/.cache/${split}_${name/v1b/v1}.jsonl"
    cand="$out/${tag}_$name.jsonl"
    if [ ! -s "$cand" ]; then
        rm -f "$cand"
        "$PY" rag/extensions/analysis/generate_from_cache.py --split "$split" \
            --question-cache "rag/.cache/${split}_v1.jsonl" --transcript-cache "$cache" \
            --ids "$ids" --output "$cand" --device cpu | tail -1
    fi
    # The judge sometimes returns malformed JSON; finished scores are cached, so a retry resumes.
    for attempt in 1 2 3 4 5; do
        "$PY" rag/scripts/evaluate_answers.py --split "$split" --candidates "$cand" --allow-subset \
            --repeats 3 --api-retries 8 --out-dir "$out/${tag}_$name" | tail -1 && break
        echo "judge attempt $attempt failed: $tag $name"
        [ "$attempt" = 5 ] && exit 1
    done
    echo "done: $tag $name"
    [ -n "$limit" ] && [ "$name" = v1b ] && break
done
echo "=== finished $tag"
