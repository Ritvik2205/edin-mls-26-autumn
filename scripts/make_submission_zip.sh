#!/usr/bin/env bash
# Build supplementary_code.zip from tracked files only: the five HW1 task files, our
# extensions, the supplementary README and measured results. No .git, .env, data, models,
# caches or weights. Fails if anything identifying or secret slips in.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
OUT=${1:-supplementary_code.zip}
rm -f "$OUT"

files=$( {
  printf '%s\n' hw1-asr/csrc/task1_rope.cu hw1-asr/csrc/task2_reduce.cuh hw1-asr/csrc/task2_reduce.cu \
                hw1-asr/csrc/task3_gemm.cu hw1-asr/csrc/task4_attention.cu
  git ls-files rag/extensions supplementary
  git ls-files results | grep -E '\.(csv|txt|md)$' || true
} | grep -v '__pycache__' | sort -u )

# shellcheck disable=SC2086
zip -q -X "$OUT" $files

bad=$(unzip -Z1 "$OUT" | grep -E '(^|/)\.env|\.git/|rag/data/|rag/models/|\.cache/|\.npy$|\.safetensors$|_solution/' || true)
[ -z "$bad" ] || { echo "refusing: forbidden paths in zip:"; echo "$bad"; rm -f "$OUT"; exit 1; }

leak_pattern=${LEAK_PATTERN:-'ELM_API_KEY=[^[:space:]]+|/Users/|/home/s[0-9]{7}|/afs/'}
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
unzip -q "$OUT" -d "$tmp"
if grep -rInE "$leak_pattern" "$tmp" >/dev/null; then
  echo "refusing: identifying or secret text found:"; grep -rInE "$leak_pattern" "$tmp" | sed "s#$tmp/##" | head; rm -f "$OUT"; exit 1
fi
echo "wrote $OUT ($(unzip -Z1 "$OUT" | wc -l | tr -d ' ') files)"
