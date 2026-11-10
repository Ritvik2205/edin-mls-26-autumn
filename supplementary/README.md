# Supplementary code

Supplementary material for the report. Code is optional and isn't marked separately; the report
carries the argument and the evidence.

## What is ours vs supplied

| Path | Status |
|------|--------|
| `hw1-asr/csrc/task1_rope.cu`, `task2_reduce.cuh`, `task2_reduce.cu`, `task3_gemm.cu`, `task4_attention.cu` | Supplied skeletons; the blanks are ours (HW1) |
| `rag/extensions/fast_asr/` | Ours: KV-cache decoding (V2), encoder trimming (V3), timing harness, variant ASR worker |
| `rag/extensions/analysis/` | Ours: answer candidates for the judge, ID selection, tables and figures |
| `rag/extensions/run_audio_query_fast.py` | Ours: `rag/scripts/run_audio_query.py` with a selectable ASR variant |
| `results/` | Our measurements (CSV and text outputs) |
| Everything else in the course repository | Supplied and unchanged (`ops.py`, `model.py`, `benchmark.py`, `provided.cu`, the tests, `rag/src`, `rag/scripts`) |

These files are meant to be placed into a checkout of the course repository at the same relative
paths. The supplied model code is never modified: `FastGlmAsr` wraps the loaded `GlmAsr` object and
calls only `glm_asr_cuda.ops`, so `ASR_CUDA` chooses between our kernels and PyTorch as usual.

## Dependencies

- ASR side: the course `mls-asr` conda environment (PyTorch, transformers), CUDA 12.8, a GPU node.
  First run `make -C hw1-asr lib`.
- RAG side: the course venv from `rag/requirements.txt`; analysis also needs `matplotlib`.

## Commands (run from the repository root)

```bash
# tests (GPU tests skip without CUDA)
python -m unittest discover -s rag/extensions -t rag -p "test_*.py" -v

# per-query ASR timing + transcript cache for one variant (mls-asr env, GPU node)
python rag/extensions/fast_asr/bench_asr.py --split dev --variant v1 --out-csv results/asr/dev_v1.csv --cache rag/.cache/dev_v1.jsonl
python rag/extensions/fast_asr/bench_asr.py --split dev --variant v2 --out-csv results/asr/dev_v2.csv --cache rag/.cache/dev_v2.jsonl
python rag/extensions/fast_asr/bench_asr.py --split dev --variant v3 --margin 1 --out-csv results/asr/dev_v3_m1.csv --cache rag/.cache/dev_v3_m1.jsonl

# retrieval quality from a cache (supplied script, unchanged)
python rag/scripts/evaluate_rag.py --device cpu --split dev --transcript-cache rag/.cache/dev_v3_m1.jsonl

# judge: same V1 question for every variant
python rag/extensions/analysis/select_ids.py --v1 rag/.cache/dev_v1.jsonl --variant rag/.cache/dev_v3_m1.jsonl --out results/judge/dev_v3_m1_ids.txt
python rag/extensions/analysis/generate_from_cache.py --split dev --question-cache rag/.cache/dev_v1.jsonl --transcript-cache rag/.cache/dev_v3_m1.jsonl --ids results/judge/dev_v3_m1_ids.txt --output rag/results/judge-v3_m1-dev.jsonl --device cpu
python rag/scripts/evaluate_answers.py --split dev --candidates rag/results/judge-v3_m1-dev.jsonl --allow-subset --repeats 2 --out-dir results/judge/v3_m1

# end-to-end demo with per-stage timings
python rag/extensions/run_audio_query_fast.py --device cpu --variant v3 --margin 1

# report tables and figures
python rag/extensions/analysis/summarise.py --split dev --out report
```
