# Design: Removing Redundant Work from the ASR Stage of Audio RAG

Status: draft for approval · 2026-11-10 · solo submission

## 1. Intent

The coursework (INFR11269, 70% group report + video, 30% individual review) asks for
(a) four hand-written CUDA kernels that run GLM-ASR-Nano, and (b) one precise systems
investigation on the audio → ASR → retrieval → ELM answer pipeline, judged on technical depth,
experimental rigour, mechanism-level analysis, and a 6-page report.

The chosen investigation (Option A) targets two redundancies in the supplied ASR stage:

1. **Prefix recomputation.** `GlmAsr.generate` has no KV cache: every generated token re-runs all
   28 decoder layers over the whole prefix (59 → 71 positions for the test clip).
2. **Padding encoding.** Every clip is padded to 30 s, so the 32-layer encoder always processes
   1500 frames. Course queries are 2.2–10.4 s (practice mean 5.2 s, dev mean 5.5 s, none > 30 s),
   so roughly 80% of encoder frames are padding.

**Research question.** *How much of ASR latency in the audio-RAG pipeline is redundant computation,
and which redundancy can be removed without changing what downstream retrieval sees?*

Removing (1) should leave the output unchanged in principle. Removing (2) is an approximation,
because real frames attend to padding frames in the non-causal encoder. That gives the report an
exact-vs-approximate contrast and a genuine quality–cost trade-off.

## 2. Goals, non-goals, anti-goals

**Goals**
- G1. All four HW1 tasks pass `make test` and `./run.sh` on one pinned GPU type. Required HW1
  comparisons, one Try-this per task, and one guide question per task are measured.
- G2. A KV-cache decoding path (V2) built on the student's own kernels, with transcripts
  matching V1. Hypothesis: bit-identical token probabilities (see §5).
- G3. Encoder trimming (V3) with a sweep of trim margins, producing a latency-vs-quality curve.
- G4. Predict each speed-up from a cost model *before* measuring, then compare with the measurement.
- G5. Quality measured where ground truth exists (retrieval qrels), plus transcript agreement
  and an LLM-judge subset.
- G6. A report and video that meet every requirement in Submission Requirements.

**Non-goals**
- Batching multiple queries, quantisation, bf16, CUDA graphs, speculative decoding.
- Replacing the ELM, encoder (enc_a) or retriever.
- Beating cuBLAS / PyTorch end-to-end. A negative or partial result is acceptable if explained.

**Anti-goals (failure even if it "works")**
- Modifying `hw1-asr/glm_asr_cuda/{ops.py,model.py}`, `benchmark.py`, `provided.cu`, or the tests
  in the hand-in tree.
- Using cuBLAS/cuDNN/CUTLASS/Thrust/PyTorch math to do a task kernel's work on the CUDA path.
- Timing numbers taken on different GPU types, or numbers that include model load, presented as
  comparable.
- Claiming V2 is exact without testing it.

## 3. Constraints (verbatim from the course materials)

- Student work in exactly five files: `csrc/task1_rope.cu`, `csrc/task2_reduce.cuh`,
  `csrc/task2_reduce.cu`, `csrc/task3_gemm.cu`, `csrc/task4_attention.cu`. "Feel free to add any
  helpers, but do not change the original functions. Keep every given function's name and
  signature, including the launch_* entry points."
- "Hand in with ops.py, model.py, benchmark.py, provided.cu and the tests unchanged."
- Investigation code may live in `rag/src/` or `rag/extensions/`.
- Report: supplied template, ≤ 6 main-body pages, References before the appendix, one final
  declaration/student page, anonymous. Video ≤ 10 min, final 2 min for Q1/Q2.
- "Use the same kind of GPU for every number you compare (pin it with --gres) and report
  nvidia-smi -L."
- ELM key is personal; full-split generation or judging needs a deliberate budget.

## 4. Decided parts of the how

### 4.1 Variants

| ID | ASR configuration | Role |
|----|-------------------|------|
| V0 | Supplied model, `ASR_CUDA=none` (PyTorch/cuBLAS) | Supplied reference |
| V1 | Supplied model, `ASR_CUDA=all` (student kernels) | Investigation baseline |
| V2 | V1 + KV-cache decoding | Exact optimisation |
| V3-m | V2 + encoder input trimmed to real length + margin m | Approximate optimisation |
| V2-torch | V2 control flow on the PyTorch path (`ASR_CUDA=none`) | Secondary contrast: cuBLAS at M = 1 (dev subset of 50 + test clip only) |

V3 margin sweep: m ∈ {0 s, 1 s, 2 s, 5 s}, plus a 10 s bucket defined as
T' = max(1000, roundup8(L)). The bucket never cuts real audio; the longest query is 10.4 s.
*(Builder's call: never truncate real frames.)* The untrimmed 30 s case is V2.
The trimmed mel length is rounded up to a multiple of 8 frames (conv stride 2 × projector pool 4),
capped at 3000. Invariant: T' ≥ L in every setting, so the prompt length is unchanged. A unit test
asserts this.

**Timing V1 (builder's call).** `FastGlmAsr.generate_recompute()` drives the supplied
`encode_audio` / `next_token_probs` with the same control flow as `GlmAsr.generate`, but with CUDA
events between the encoder and decoder. A test asserts that its tokens and probabilities equal the
supplied `generate()` bit for bit. The report also quotes one wall-clock number from the untouched
`./run.sh --profile`, to tie V1 to the supplied baseline.

**M = 1 GEMV kernel (builder's call).** Out of scope for V2, which must stay a pure algorithmic
change so the exactness claim holds. A dedicated GEMV path is optional *V2g*, built only after the
main results are frozen, reported as a separate variant, and with its exactness checked separately.

### 4.2 Code placement

```
hw1-asr/csrc/task1_rope.cu          + launch_rope_at(...): RoPE with a position offset (new extern "C";
                                      launch_rope unchanged, shares a __device__ helper)
rag/extensions/fast_asr/
  __init__.py
  kv_model.py      FastGlmAsr(GlmAsr): generate_kv(), encode_audio_trimmed(); wraps the supplied
                   model object (same weights in memory), never edits model.py
  rope_offset.py   ctypes binding for launch_rope_at via ops._library(); PyTorch fallback when
                   "rope" is not in ops.USE_CUDA
  worker.py        JSON-lines ASR worker, same protocol as rag/src/asr_worker.py, plus --variant
  bench_asr.py     per-query timing harness (runs in the mls-asr env)
  adapter.py       FastASRAdapter(ASRAdapter): spawns worker.py instead of src/asr_worker.py
  tests/           equivalence + unit tests
rag/extensions/analysis/
  summarise.py     CSV → tables/figures for the report (runs locally, numpy/matplotlib)
```

`FastGlmAsr` reuses the supplied `ops.*` functions for every operation, so the CUDA/PyTorch group
switch (`ASR_CUDA`) still selects kernels exactly as for V0/V1.

### 4.3 KV cache

- Per decoder layer, preallocate `k_cache`, `v_cache` of shape `[1, S_max, 4, 128]` float32, where
  `S_max = prompt_len + max_new_tokens`. Prompt length is 15 + n_audio_tokens. For 10.4 s audio
  that is ≈ 145; with max_new_tokens = 100 the cache is ≈ 28 × 2 × 245 × 512 × 4 B ≈ 28 MB.
- Prefill: run the decoder once over the full prompt with `causal=True`, writing k/v rows
  [0, P) into the cache.
- Decode step t: one position. q/k/v projections with M = 1. RoPE at position P + t − 1 via
  `launch_rope_at`. Write k/v into cache row P + t − 1. Then
  `ops.attention(q[1,1,16,128], k_cache[:, :P+t], v_cache[:, :P+t], causal=True)`.
  The kernel's `shift = seq_k − seq_q` already gives the correct causal window. The slices
  `[:, :n]` of a batch-1 cache are contiguous, so no copy is needed.
- Greedy argmax and the EOS set are identical to `GlmAsr.generate`.

### 4.4 Encoder trimming

- Compute the real frame length L from `input_features_mask`, then T' = min(3000, roundup8(L + m·100)).
- Slice `features[:, :, :T']` and `features_mask[:, :T']`, then run the supplied encoder path
  (convs → 32 layers → norm → projector). The existing `n_tokens` logic keeps only real-audio
  tokens, so the decoder prompt length is unchanged for any m ≥ 0.
- RoPE positions of real frames are unchanged. What changes is that real frames no longer attend to
  trimmed padding frames. This is the mechanism to explain in the report.

### 4.5 Measurement protocol

- Hardware: one pinned type, default `nvidia_geforce_rtx_2080_ti`. Every result table records
  `nvidia-smi -L`, the node name, and the git commit.
- ASR timing boundary: warm model, from "16 kHz float array in" to "token list out".
  `processor.apply_transcription_request` (CPU mel) is timed separately as `feat_ms`. Encoder and
  decoder are timed separately with CUDA events. Model load is excluded and reported once.
- Warm-up: 5 queries, discarded. Then every query in the split once, plus a repeat pass over a
  fixed 50-query subset to estimate run-to-run variance.
- Report mean, p50, p95, std of ASR ms/query, decoder ms/token, and peak GPU memory
  (`torch.cuda.reset_peak_memory_stats` per variant).
- Development on practice (350). Final tables on dev (125), run once after the design is frozen.
- One transcript cache per variant (`rag/.cache/{split}_{variant}.jsonl`) in the
  `{query_id, transcript}` format, so `evaluate_rag.py --transcript-cache` scores retrieval without
  re-running ASR.

### 4.6 Quality metrics

- Transcript agreement with V1: exact-match rate, and word-level edit distance (WER relative to
  V1; there are no ground-truth query transcripts).
- Retrieval vs qrels: Recall@5/10/20/100, MRR@10, NDCG@10 per variant, plus a paired per-query
  difference against V1.
- Token-probability agreement V2 vs V1: max |Δp| over every generated token.
- LLM judge: only for variants whose transcripts differ from V1, on the subset of dev queries where
  they differ plus an equal number of unchanged controls, with `--repeats 2`.
  *(Builder's call.)* Answers are generated from each variant's own transcript, because that is the
  system's output. The judge is always given the **V1 transcript as the question**, so the variants
  are judged against the same question. The supplied `evaluate_answers.py` judges each row's
  `transcript` field, so one extension script is enough:
  `rag/extensions/analysis/generate_from_cache.py` writes candidate rows with
  `transcript` = V1 transcript (the judged question), `asr_transcript` = the variant's transcript,
  and `answer` generated from the variant's transcript. The supplied judge then runs unchanged.
- ELM budget *(builder's call)*: at most ~600 requests in total. The default model is one of the
  free locally-hosted ELM models. Dry-run every batch first.

### 4.7 Cost model (predictions written down before measuring)

From a V1 per-operation profile, split time into encoder E, decoder prefill D_P, per-token decode
recompute D_t, and lm_head+softmax H per token. Predict:

- V2 ≈ E + D_P + Σ_t (step_t + H). step_t is bounded below by decoder weight bytes / DRAM bandwidth:
  ≈ 1.35 B params × 4 B ≈ 5.4 GB per token, so ≈ 8.8 ms on a 616 GB/s RTX 2080 Ti.
- V3-m: encoder cost scales ~linearly in T' for the linear layers and ~quadratically for attention.
  Predict the E(T')/E(3000) ratio from FLOPs and compare it with the measurement.

Analysis threads for the report:
- After the KV cache, decode becomes weight-bandwidth-bound GEMV with M = 1. This links to HW1
  Task 3 "Try this" 1 (TILE rows wasted at M = 1) and lm_head GB/s.
- Attention with seq_q = 1 launches one block per head (16 blocks) with only 1 of ATTN_WARPS warps
  active, so occupancy is low. Measure it; a split-K fix is out of scope but can be discussed.
- The ASR share of end-to-end latency once the ELM call is included (from a small
  `run_audio_query` sample).

### 4.8 Integration with the RAG pipeline

- Retrieval evaluation uses the transcript caches and the unchanged `evaluate_rag.py`.
- The end-to-end demo (video) uses `FastASRAdapter` with the supplied `RAGPipeline` components,
  showing V1 vs V2/V3 per-stage timings for one query.

## 5. Correctness expectations

- With student kernels, every operation for a row is independent of M: the tiled GEMM accumulates
  over K in the same order for each output, and attention is per-query. **Hypothesis:** V2's token
  probabilities are bit-identical to V1's. If they are not, the report quantifies max |Δp| and
  explains where reassociation enters.
- With PyTorch (V0 path + KV cache), cuBLAS may choose different algorithms for M = 1, so small
  differences are expected. This contrast is a reportable observation.
- V3 is expected to change some transcripts at small m. The trade-off curve is the result.

## 6. Left to the builder

- Internal structure of `FastGlmAsr` (subclass vs composition) and helper names, as long as
  `ops.*` stays the only compute path.
- Plot styling, CSV column order, CLI flag names beyond those listed.
- Whether to add an optional split-K decode attention kernel. Only if time remains after the
  frozen results.

## 7. Open item needing staff confirmation

*(Fallback decided)* If the staff answer is still pending at the end of Phase 1, build the fallback
first. It needs no change outside the five files. Swap in `launch_rope_at` if it is approved.

- That a model variant kept in `rag/extensions/` (supplied `model.py` untouched) and an *added*
  `launch_rope_at` in `task1_rope.cu` comply with the HW1 rules. Ask on the course forum in Phase 0.
  Fallback if refused: implement the RoPE offset in `rope_offset.py` as a second launch of the
  existing `launch_rope` over a padded `[1, pos+1, …]` scratch tensor. This is correct but wasteful;
  measure and report its cost.
