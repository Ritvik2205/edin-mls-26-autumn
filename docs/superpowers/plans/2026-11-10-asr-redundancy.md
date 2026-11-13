# ASR Redundancy Investigation: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the HW1 CUDA kernels, then measure how much ASR latency in the audio-RAG pipeline is redundant work. Compare an exact KV cache (V2) and approximate encoder trimming (V3) against the student-kernel baseline (V1), with retrieval quality as the check.

**Architecture:** The kernels live only in the five `hw1-asr/csrc/task*` files. One added C entry point, `launch_rope_at`, goes in `task1_rope.cu`. Every investigation path lives in `rag/extensions/` and wraps the supplied `GlmAsr` object without editing it. It calls only `glm_asr_cuda.ops.*`, so `ASR_CUDA` still chooses student kernels or the PyTorch path. A timing harness writes one per-query CSV and one transcript cache per variant. The supplied `evaluate_rag.py` and `evaluate_answers.py` score those files unchanged. A local analysis script turns the CSVs into report tables and figures.

**Tech Stack:** CUDA 12.8 C++17 (nvcc), Python 3 + PyTorch in conda env `mls-asr`, numpy/sentence-transformers/openai in venv `~/mls-rag-env`, `unittest`, matplotlib (local analysis), LaTeX (supplied template), OBS/Shotcut.

**Spec:** `docs/superpowers/specs/2026-11-10-asr-redundancy-design.md`

## Schedule and priorities (deadline 20 Nov 2026)

Priorities P0/P1/P2 are defined in spec §8. Run P0 first; P1/P2 only if a slot is free.

| Day | Work |
|-----|------|
| Fri 13 | Cluster setup (`docs/cluster-setup.md`), `make test`, fix kernel failures (Tasks 0–4). Report skeleton. |
| Sat 14 | Kernels pass; GPU tests with `FAST_ASR_TEST_CUDA=none`, then all (Tasks 7–10 verification); HW1 numbers (Task 5, T3/T4 Try-this only). |
| Sun 15 | Task 14 P0 runs: V1, V2, V3 m0/m2/b10 on practice, then dev; repeat subset; RoPE kernel vs scratch. |
| Mon 16 | Retrieval scoring locally from caches; Task 15 tables/figures; ELM go/no-go (spec §10). |
| Tue 17 – Wed 18 | Task 16 report; appendix answers. |
| Thu 19 | Task 17 video, Task 18 zip, format check. |
| Fri 20 | Buffer, submit. |

## Global Constraints

- Edit only `csrc/task1_rope.cu`, `csrc/task2_reduce.cuh`, `csrc/task2_reduce.cu`, `csrc/task3_gemm.cu`, `csrc/task4_attention.cu` under `hw1-asr/`. Keep every given function's name and signature, including the `launch_*` entry points.
- `ops.py`, `model.py`, `benchmark.py`, `provided.cu` and `hw1-asr/tests/` stay unchanged. Check with `git diff --stat main -- hw1-asr/glm_asr_cuda hw1-asr/benchmark.py hw1-asr/csrc/provided.cu hw1-asr/tests` → empty.
- No cuBLAS, cuDNN, CUTLASS, Thrust or PyTorch math doing a task kernel's work on the CUDA path.
- Every compared number comes from one GPU type, pinned with `--gres=gpu:nvidia_geforce_rtx_2080_ti:1`. Record `nvidia-smi -L`.
- Never commit `hw1-asr/_solution/`, `rag/.env`, `rag/data/`, `rag/models/`, `rag/.cache/`.
- Each `generate_answers.py`-style output file is new: names follow `rag/results/<exp>-<variant>-<split>.jsonl`.
- ELM requests stay at or below 600 in total. Always run `--dry-run` before a judge run.
- Report: supplied template, ≤ 6 main-body pages, References before the appendix, one final declaration page, anonymous. Video ≤ 10:00, with the final 2:00 for Question 1 and Question 2.
- Test commands: GPU tests run in `mls-asr` on a GPU node. Local tests run with `python -m unittest discover -s rag/extensions -t rag -p "test_*.py"` from the repository root (GPU tests skip without CUDA).

## Review Focus

1. **The longest clip (10.4 s, ≈1040 mel frames) under every trim setting.** Expect T' ≥ L, so no real audio is lost and the prompt length is unchanged. Pinned in Task 6 (`test_bucket_never_truncates`) and Task 10 (`test_prompt_len_unchanged_longest_clip`).
2. **Generation that hits `max_new_tokens` without EOS, or stops at EOS on the first token.** Expect `generate_kv` never to write past `S_max` and to return the same tokens as the recompute path. Pinned in Task 9 (`test_kv_max_new_tokens_boundary`, `test_kv_first_token_eos`).
3. **A variant that produces an empty transcript.** Expect the harness to record `transcript=""` with `empty=1` and carry on, without crashing or skipping the row. Pinned in Task 11 (`test_empty_transcript_row`).
4. **A bench run killed by the 1-hour Slurm limit and restarted.** Expect a resume that skips finished query IDs, with no duplicate CSV or cache rows. Pinned in Task 11 (`test_resume_skips_done`).
5. **CSVs from two GPU types fed to the summariser.** Expect a refusal naming both GPUs, not a mixed table. Pinned in Task 15 (`test_refuses_mixed_gpus`).

---

### Task 0: Access, environment, baseline capture

**Files:**
- Create: `rag/extensions/__init__.py` (empty), `rag/extensions/fast_asr/__init__.py` (empty), `rag/extensions/analysis/__init__.py` (empty)
- Create: `results/hw1/baseline_torch.txt` (captured output)

**Interfaces:**
- Produces: working `mls-asr` env, `~/mls-rag-env`, installed assets, `rag/.env` with a key, a pinned GPU type.

- [ ] **Step 1:** Request an ELM key at `https://elm.edina.ac.uk/elm` using the slide's usage statement. Put it in `rag/.env` as `ELM_API_KEY=…`.
- [ ] **Step 2:** Post on the course forum: "May an investigation keep a model variant in `rag/extensions/` (supplied `model.py` untouched), and may `task1_rope.cu` gain an additional `launch_rope_at` entry point?" Record the answer in the spec, §7.
- [ ] **Step 3:** On the head node, run `git pull`, `bash hw1-asr/setup_env.sh`, `git lfs pull --include="rag-assets.zip"`, the RAG venv steps from `rag/README.md` Step 1, and `install_assets.py --from-dir ~/mls-rag-assets`. Expected: `Course assets verified`.
- [ ] **Step 4:** On a pinned 2080 Ti node, run `make -C hw1-asr && (cd hw1-asr && ./run.sh --cuda none --profile) | tee results/hw1/baseline_torch.txt`. Expected: `Status: PASS`, about 990 ms per transcription.
- [ ] **Step 5:** Run `python rag/scripts/check_setup.py`. Expected: every line PASS.
- [ ] **Step 6:** Commit.

```bash
git checkout -b coursework
git add rag/extensions/__init__.py rag/extensions/fast_asr/__init__.py rag/extensions/analysis/__init__.py results/hw1/baseline_torch.txt
git commit -m "chore: scaffold extensions and record PyTorch baseline"
```

### Task 1: HW1 Task 1, RoPE kernel

**Files:**
- Modify: `hw1-asr/csrc/task1_rope.cu` (BLANK 1.1–1.4)

**Interfaces:**
- Produces: `rope_kernel`, as called by the existing `launch_rope`, correct for the q-then-k pair layout.

- [ ] **Step 1:** Run `make -C hw1-asr test1`. Expected: FAIL lines (blanks empty).
- [ ] **Step 2:** Fill BLANK 1.1–1.4 following `hw1-asr/docs/task1-rope.md`. Read both pair elements before writing either.
- [ ] **Step 3:** Run `make -C hw1-asr test1`. Expected: `ALL PASS`.
- [ ] **Step 4:** Run `cd hw1-asr && ./run.sh --cuda task1`. Expected: `Status: PASS`.
- [ ] **Step 5:** Commit: `git commit -am "hw1: task1 rope kernel"`

### Task 2: HW1 Task 2, reductions, RMSNorm, softmax

**Files:**
- Modify: `hw1-asr/csrc/task2_reduce.cuh` (BLANK 2.1–2.4), `hw1-asr/csrc/task2_reduce.cu` (3 blanks)

**Interfaces:**
- Produces: `warp_reduce_sum`, `warp_reduce_max`, `block_reduce_max`, and the MaxSum merge in `task2_reduce.cuh`. Task 4 uses `warp_reduce_max`/`warp_reduce_sum`.

- [ ] **Step 1:** Run `make -C hw1-asr test2`. Expected: FAIL.
- [ ] **Step 2:** Fill the blanks following `docs/task2-reduce.md`. Missing warps contribute `-INFINITY` to the max. Merging two `{-INFINITY, 0}` summaries must give `{-INFINITY, 0}`, not NaN.
- [ ] **Step 3:** Run `make -C hw1-asr test2`. Expected: `ALL PASS`, including "large inputs".
- [ ] **Step 4:** Run `compute-sanitizer --tool racecheck hw1-asr/build/sm_75/test_task2_reduce small`. Expected: `0 hazards`.
- [ ] **Step 5:** Run `cd hw1-asr && ./run.sh --cuda task1,task2`. Expected: `Status: PASS`.
- [ ] **Step 6:** Commit: `git commit -am "hw1: task2 reductions"`

### Task 3: HW1 Task 3, tiled GEMM with fused GELU

**Files:**
- Modify: `hw1-asr/csrc/task3_gemm.cu` (4 blanks)

**Interfaces:**
- Produces: `linear` kernel behind `launch_linear`. Per-output accumulation order over K must not depend on M. Tasks 8–9 rely on this for bit-identity.

- [ ] **Step 1:** Run `make -C hw1-asr test3`. Expected: FAIL.
- [ ] **Step 2:** Fill the blanks following `docs/task3-gemm.md`. Consecutive `tx` reads consecutive W addresses. Out-of-range threads still reach both `__syncthreads()`.
- [ ] **Step 3:** Run `make -C hw1-asr test3`. Expected: `ALL PASS`, printing GFLOP/s for naive, yours and cuBLAS.
- [ ] **Step 4:** Run `compute-sanitizer hw1-asr/build/sm_75/test_task3_gemm small` and the same with `--tool racecheck`. Expected: `0 errors` and `0 hazards`.
- [ ] **Step 5:** Run `cd hw1-asr && ./run.sh --cuda task1,task2,task3`. Expected: `Status: PASS`.
- [ ] **Step 6:** Commit: `git commit -am "hw1: task3 tiled gemm"`

### Task 4: HW1 Task 4, streaming attention

**Files:**
- Modify: `hw1-asr/csrc/task4_attention.cu` (6 blanks)

**Interfaces:**
- Consumes: Task 2 warp reductions.
- Produces: `launch_attention`, correct for `seq_q ≠ seq_k` with `causal=1`. Query `qi` sees keys `≤ qi + (seq_k − seq_q)`. Task 9 calls it with `seq_q = 1`.

- [ ] **Step 1:** Run `make -C hw1-asr test4`. Expected: FAIL.
- [ ] **Step 2:** Fill the blanks following `docs/task4-attention.md`. Handle `m_new = -INFINITY`, so that `corr` is not NaN.
- [ ] **Step 3:** Run `make -C hw1-asr test4`. Expected: `ALL PASS`.
- [ ] **Step 4:** Run `cd hw1-asr && ./run.sh --profile`. Expected: `Status: PASS` and a profile table with every op `cuda`.
- [ ] **Step 5:** Commit: `git commit -am "hw1: task4 attention"`

### Task 5: HW1 measurement pack

**Files:**
- Create: `results/hw1/run_asr_<jobid>.out`, `results/hw1/profile_torch.txt`, `results/hw1/tryit.csv`, `results/hw1/questions.md`

**Interfaces:**
- Produces: `tryit.csv` with columns `task,knob,value,case,metric,mean,std,repeats,gpu`. Task 15 reads it.

- [ ] **Step 1:** Run `sbatch --gres=gpu:nvidia_geforce_rtx_2080_ti:1 hw1-asr/run_asr.sbatch` and copy `asr-<jobid>.out` to `results/hw1/`. Expected: four ALL PASS, `Status: PASS`, and an `nvidia-smi -L` line.
- [ ] **Step 2:** On the same node type, run `./run.sh --cuda none --profile > results/hw1/profile_torch.txt`.
- [ ] **Step 3:** Try-this runs, 3 repeats each, values into `tryit.csv`. These are *optional* per forum #20/#25. Do T3 and T4 first because they feed the analysis; T1/T2 only if time allows:
  - T1: `ROPE_THREADS` ∈ {32, 64, 128, 256, 512, 1024}, audio-encoder GB/s.
  - T2: `REDUCE_THREADS` sweep, layernorm-encoder vs softmax-vocab GB/s.
  - T3: `TILE` ∈ {8, 16, 32}, encoder GFLOP/s and lm_head ms.
  - T4: `ATTN_WARPS` ∈ {1, 2, 4, 8, 16}, encoder-case ms.
  - Restore the default constants afterwards. `git diff hw1-asr/csrc` should show only the blank fills.
- [ ] **Step 4:** In `questions.md`, draft one answer per guide: T1 Q1, T2 Q3, T3 Q2, T4 Q1. These four are required and go in one main-body subsection, together with the key `make test`/`--profile` lines and the GPU name. Full logs go in the appendix. Use the GPU's peak bandwidth (616 GB/s for the 2080 Ti) and FP32 peak.
- [ ] **Step 5:** Commit: `git add results/hw1 && git commit -m "hw1: measurement pack"`

### Task 6: Pure helpers (local, no GPU)

**Files:**
- Create: `rag/extensions/fast_asr/trim.py`, `rag/extensions/fast_asr/cachefile.py`, `rag/extensions/analysis/textmetrics.py`
- Test: `rag/extensions/fast_asr/tests/test_trim.py`, `rag/extensions/fast_asr/tests/test_cachefile.py`, `rag/extensions/analysis/tests/test_textmetrics.py` (add `__init__.py` to each `tests/`)

**Interfaces:**
- Produces:
  - `trimmed_frames(real_frames: int, margin_s: float | None = None, bucket_s: float | None = None, total: int = 3000) -> int`. Exactly one of margin or bucket is given. The result is a multiple of 8, ≥ real_frames, ≤ total.
  - `read_cache(path: Path) -> dict[str, str]`, `append_cache(path: Path, query_id: str, transcript: str) -> None`. JSON-lines `{"query_id","transcript"}`, byte-compatible with `evaluate_rag.py`.
  - `word_errors(ref: str, hyp: str) -> int`, `wer(ref: str, hyp: str) -> float`. Normalise by uppercasing, stripping punctuation and splitting on whitespace. `wer` of an empty ref with an empty hyp is 0.0.

- [ ] **Step 1: Write the failing tests**

```python
def test_margin_rounds_up_to_8(self):
    self.assertEqual(trimmed_frames(350, margin_s=0), 352)
    self.assertEqual(trimmed_frames(350, margin_s=1), 456)        # 350+100 → 456
def test_caps_at_total(self):
    self.assertEqual(trimmed_frames(2990, margin_s=5), 3000)
def test_bucket_never_truncates(self):
    self.assertEqual(trimmed_frames(1040, bucket_s=10), 1040)     # 10.4 s clip
    self.assertEqual(trimmed_frames(500, bucket_s=10), 1000)
def test_exactly_one_mode(self):
    with self.assertRaises(ValueError): trimmed_frames(100)
    with self.assertRaises(ValueError): trimmed_frames(100, margin_s=1, bucket_s=10)
def test_cache_roundtrip_matches_evaluate_rag_format(self): ...   # append 2 rows, read → dict, raw line == json.dumps({"query_id":..,"transcript":..}, ensure_ascii=False)
def test_wer(self):
    self.assertEqual(wer("Concord returned to its place", "concord returned to its place."), 0.0)
    self.assertAlmostEqual(wer("a b c d", "a x c"), 0.5)             # 1 sub + 1 del
    self.assertEqual(wer("", ""), 0.0)
```

- [ ] **Step 2:** Run `python -m unittest discover -s rag/extensions -t rag -p "test_*.py" -v`. Expected: FAIL (ImportError).
- [ ] **Step 3:** Implement the three modules. `wer` uses a Levenshtein DP over word lists.
- [ ] **Step 4:** Rerun. Expected: PASS.
- [ ] **Step 5:** Commit: `git commit -m "ext: trim, cache and WER helpers"`

### Task 7: RoPE with position offset

**Files:**
- Modify: `hw1-asr/csrc/task1_rope.cu`. Move the blank logic into a `__device__` helper taking `pos_offset`. `rope_kernel` calls it with 0. Add `rope_at_kernel` and `launch_rope_at`.
- Create: `rag/extensions/fast_asr/rope_offset.py`
- Test: `rag/extensions/fast_asr/tests/test_rope_offset.py` (GPU, skips without CUDA)

**Interfaces:**
- Produces:
  - C: `ASR_API int launch_rope_at(float* q, float* k, const float* inv_freq, int batch, int seq_len, int num_q_heads, int num_k_heads, int head_dim, int rotary_dim, int pos_offset, cudaStream_t stream)`. Positions are `pos_offset .. pos_offset+seq_len-1`.
  - Python: `rope_at_(q, k, inv_freq, rotary_dim, pos_offset: int) -> None`. Rotates in place. Uses `ops._library()` (with argtypes set on first use) when `"rope" in ops.USE_CUDA`, otherwise a PyTorch formula mirroring `ops.rope_` with `pos = arange(S) + pos_offset`. Wrapped in `ops._timed("rope", "rope")` so profiles count it.
  - Fallback if staff refuse the added entry point: `rope_at_` builds a zero `[1, pos_offset+S, H, D]` scratch tensor, copies q/k into its last S rows, calls `ops.rope_`, and copies back. Same signature.

- [ ] **Step 1: Write the failing tests**

```python
def test_offset_zero_bitwise_equals_launch_rope(self): ...        # q,k random [1,59,16,128]/[1,59,4,128]; rope_at_(…,0) vs ops.rope_ → torch.equal
def test_offset_matches_full_sequence_row(self):
    # rotate a full [1,P+1,…] with ops.rope_; rotate only row P with rope_at_(…, pos_offset=P)
    for P in (58, 144): self.assertTrue(torch.equal(full[:, P:P+1], single))
def test_audio_partial_rotary(self): ...                          # head_dim 64, rotary_dim 32: dims 32..63 unchanged
```

- [ ] **Step 2:** Run on a GPU node: `python -m unittest discover -s rag/extensions -t rag -p "test_rope_offset.py" -v`. Expected: FAIL (`launch_rope_at` undefined).
- [ ] **Step 3:** Implement the C refactor and the binding.
- [ ] **Step 4:** Run `make -C hw1-asr test1`, then the tests above. Expected: `ALL PASS` (the original test still passes) and 3 OK.
- [ ] **Step 5:** Commit: `git commit -am "ext: rope with position offset"`

### Task 8: Instrumented recompute path (V1)

**Files:**
- Create: `rag/extensions/fast_asr/kv_model.py`
- Test: `rag/extensions/fast_asr/tests/test_equivalence.py` (GPU)

**Interfaces:**
- Consumes: supplied `glm_asr_cuda.model.load_model()`, `GlmAsr.encode_audio`, `GlmAsr.next_token_probs`, `ops.embedding`.
- Produces:
  - `class FastGlmAsr: __init__(self, base: GlmAsr)`. Composition: it holds `base`, shares weights, and never mutates it.
  - `prepare_inputs(processor, audio: np.ndarray) -> dict` with keys `features, input_ids, features_mask, feat_ms`.
  - `generate_recompute(self, features, input_ids, features_mask, max_new_tokens=100) -> GenResult`.
  - `@dataclass GenResult: tokens: list[int]; probs: list[float]; encode_ms: float; decode_ms: float; prompt_len: int; enc_frames: int`. Timing uses CUDA events around the encoder and around the token loop.
  - `load_fast(device="cuda") -> tuple[FastGlmAsr, processor]`.

- [ ] **Step 1: Write the failing test**

```python
def test_recompute_equals_supplied_generate(self):
    tok_ref, p_ref = base.generate(**inp)                         # supplied, untouched
    r = fast.generate_recompute(**inp)
    self.assertEqual(r.tokens, tok_ref); self.assertEqual(r.probs, p_ref)   # exact floats
    self.assertEqual(r.prompt_len, 59)
```

Use `hw1-asr/test_audio.wav` with `ASR_CUDA=all`. Load the model once in `setUpClass`.

- [ ] **Step 2:** Run it. Expected: FAIL (ImportError).
- [ ] **Step 3:** Implement. The loop mirrors `GlmAsr.generate` line for line.
- [ ] **Step 4:** Rerun. Expected: OK.
- [ ] **Step 5:** Commit: `git commit -m "ext: instrumented recompute baseline"`

### Task 9: KV-cache decoding (V2)

**Files:**
- Modify: `rag/extensions/fast_asr/kv_model.py`
- Test: `rag/extensions/fast_asr/tests/test_equivalence.py` (add cases)

**Interfaces:**
- Consumes: `rope_at_` (Task 7), `ops.linear/rmsnorm/attention/silu_mul/embedding/softmax`, the `base.w` weight names used in `GlmAsr.decoder_layer`.
- Produces:
  - `generate_kv(self, features, input_ids, features_mask, max_new_tokens=100, encoder_frames: int | None = None) -> GenResult`. `encoder_frames=None` means untrimmed. Trimming is implemented in Task 10.
  - `decoder_layer_kv(self, h, i, cache: KVCache, start: int) -> Tensor`.
  - `@dataclass KVCache: k: list[Tensor]; v: list[Tensor]; s_max: int`, each `[1, s_max, 4, 128]` float32 contiguous.
  - `GenResult` gains `kv_bytes: int`.
- Fixed values: `s_max = prompt_len + max_new_tokens`. Prefill writes rows `[0, P)`. Step t writes row `P+t-1` and calls attention on `cache.k[i][:, :P+t]`.

- [ ] **Step 1: Write the failing tests**

```python
def test_kv_bit_identical_on_student_kernels(self):    # ASR_CUDA=all
    a = fast.generate_recompute(**inp); b = fast.generate_kv(**inp)
    self.assertEqual(b.tokens, a.tokens)
    self.assertEqual(max(abs(x - y) for x, y in zip(a.probs, b.probs)), 0.0)   # spec §5 hypothesis
def test_kv_torch_path_close(self):                    # ops.set_cuda_groups("none")
    self.assertEqual(b.tokens, a.tokens); self.assertLess(max_dp, 1e-4)
def test_kv_max_new_tokens_boundary(self):
    b = fast.generate_kv(**inp, max_new_tokens=3); a = fast.generate_recompute(**inp, max_new_tokens=3)
    self.assertEqual(b.tokens, a.tokens); self.assertEqual(len(b.tokens), 3)
def test_kv_first_token_eos(self): ...                 # monkeypatch base.c.eos_token_ids to {first token}; len(tokens)==1, no exception
def test_kv_bytes(self): self.assertEqual(b.kv_bytes, 28 * 2 * (59 + 100) * 4 * 128 * 4)
```

- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement. If the bit-identical test fails while tokens match, keep the test as `assertLess(max_dp, 1e-6)`, record the measured max |Δp| in `results/notes.md`, and find the op responsible by comparing per-layer outputs. This becomes a report finding.
- [ ] **Step 4:** Rerun. Expected: OK.
- [ ] **Step 5:** Commit: `git commit -m "ext: kv-cache decoding"`

### Task 10: Encoder trimming (V3)

**Files:**
- Modify: `rag/extensions/fast_asr/kv_model.py`
- Test: `rag/extensions/fast_asr/tests/test_equivalence.py` (add cases)

**Interfaces:**
- Consumes: `trimmed_frames` (Task 6), `base.encode_audio`.
- Produces: `generate_kv(..., encoder_frames=T)` slices `features[:, :, :T]` and `features_mask[:, :T]` before `base.encode_audio`. `GenResult.enc_frames = T`. `real_frames(features_mask) -> int`.

- [ ] **Step 1: Write the failing tests**

```python
def test_full_length_equals_v2(self):
    self.assertEqual(fast.generate_kv(**inp, encoder_frames=3000).probs, fast.generate_kv(**inp).probs)
def test_prompt_len_unchanged_longest_clip(self):     # longest practice clip (10.4 s) from the manifest
    for kw in ({"margin_s": 0}, {"bucket_s": 10}):
        T = trimmed_frames(real_frames(inp["features_mask"]), **kw)
        self.assertEqual(fast.generate_kv(**inp, encoder_frames=T).prompt_len, v2.prompt_len)
def test_rejects_cutting_real_audio(self):
    with self.assertRaises(ValueError): fast.generate_kv(**inp, encoder_frames=real - 8)
```

- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement. Raise `ValueError` when `T < real_frames` or `T % 8 != 0`.
- [ ] **Step 4:** Rerun. Expected: OK.
- [ ] **Step 5:** Commit: `git commit -m "ext: encoder trimming"`

### Task 11: Timing harness

**Files:**
- Create: `rag/extensions/fast_asr/bench_asr.py`
- Test: `rag/extensions/fast_asr/tests/test_bench.py` (local, using a fake model object, plus one GPU smoke test)

**Interfaces:**
- Consumes: `FastGlmAsr`, `trimmed_frames`, `append_cache/read_cache`, `CourseData`-style manifest (read the JSONL directly, so the harness has no RAG dependency).
- Produces:
  - CLI: `python rag/extensions/fast_asr/bench_asr.py --split {practice,dev} --variant {v1,v2,v3} [--margin S | --bucket S] [--cuda all|none] --out-csv PATH --cache PATH [--warmup 5] [--ids FILE] [--repeat-subset 50] [--limit N]`.
  - CSV columns, in this order: `query_id,split,variant,setting,cuda,gpu,node,commit,pass,duration_s,real_frames,enc_frames,prompt_len,n_tokens,feat_ms,encode_ms,decode_ms,asr_ms,ms_per_token,peak_mem_bytes,kv_bytes,empty,transcript`. `pass` is 0 for the main pass and 1+ for repeats. `asr_ms = feat_ms + encode_ms + decode_ms` (model load excluded).
  - `run_bench(model, items, variant, setting, out_csv, cache_path, warmup, repeat_ids) -> int` (rows written), importable for tests.
  - Resume: rows whose `(query_id, pass)` are already in `out_csv` are skipped. The cache is appended only when the query is not in it yet.
  - Peak memory: `torch.cuda.reset_peak_memory_stats()` once after warm-up. Each row records the running peak.

- [ ] **Step 1: Write the failing tests**

```python
def test_writes_one_row_per_query_and_cache(self): ...   # fake model, 3 items → 3 rows, cache has 3 ids, header == COLUMNS
def test_resume_skips_done(self):
    run_bench(fake, items[:2], ...); n = run_bench(fake, items, ...)
    self.assertEqual(n, 1); self.assertEqual(len(read_rows(csv)), 3)
def test_empty_transcript_row(self): ...                 # fake returns "" for one id → row empty=1, cache has "", no exception
def test_warmup_rows_not_written(self): ...              # warmup=2 → fake called 2 extra times, rows == len(items)
```

- [ ] **Step 2:** Run locally. Expected: FAIL.
- [ ] **Step 3:** Implement, injecting the model and transcription function so the tests use a fake.
- [ ] **Step 4:** Rerun locally (OK). On a GPU node, run `--split practice --variant v2 --limit 3`. Expected: 3 rows, non-zero timings.
- [ ] **Step 5:** Commit: `git commit -m "ext: asr timing harness"`

### Task 12: Worker, adapter, end-to-end demo

**Files:**
- Create: `rag/extensions/fast_asr/worker.py`, `rag/extensions/fast_asr/adapter.py`, `rag/extensions/run_audio_query_fast.py`
- Test: `rag/extensions/fast_asr/tests/test_adapter.py` (local, fake worker in the style of `rag/tests/test_asr_adapter.py`)

**Interfaces:**
- Consumes: `rag/src/asr_adapter.ASRAdapter`, `rag/src/rag.RAGPipeline` and the supplied components.
- Produces:
  - `worker.py --asr-root DIR --cuda SPEC --variant {v1,v2,v3} [--margin S|--bucket S] --max-new-tokens N`. Same JSON-lines protocol as `src/asr_worker.py` (`{"ready": true}`, `{"audio_path"}` → `{"transcript"}` or `{"error"}`).
  - `FastASRAdapter(ASRAdapter).__init__(self, variant="v2", margin_s=None, bucket_s=None, **kw)`. Overrides `prepare()` only, to spawn `worker.py` with the variant flags.
  - `run_audio_query_fast.py` takes the same flags as `rag/scripts/run_audio_query.py` plus `--variant/--margin/--bucket`, and prints the per-stage timing dict.

- [ ] **Step 1:** Write a failing test: `test_fast_adapter_spawns_variant_worker`. A fake worker echoes its argv, and the transcript contains `--variant v3 --margin 1`.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Rerun (OK). On a GPU node, run `python rag/extensions/run_audio_query_fast.py --device cpu --variant v2`. Expected: transcript, answer and timings, with `asr_ms` below the V1 run of the same query.
- [ ] **Step 5:** Commit: `git commit -m "ext: variant worker and end-to-end demo"`

### Task 13: Answer candidates from caches

**Files:**
- Create: `rag/extensions/analysis/generate_from_cache.py`, `rag/extensions/analysis/select_ids.py`
- Test: `rag/extensions/analysis/tests/test_candidates.py` (local)

**Interfaces:**
- Consumes: `read_cache`, `src.encoder.LocalEncoder`, `src.retriever.ExactDenseRetriever`, `src.context_builder.SimpleContextBuilder`, `src.generator.ELMGenerator`.
- Produces:
  - `select_ids(v1_cache: dict, var_cache: dict, n_controls: int, seed: int = 0) -> list[str]`: every ID whose transcript differs, plus `n_controls` unchanged IDs drawn deterministically. With `n_controls = len(differing)`, the list is sorted.
  - `build_candidate(qid, split, question, asr_transcript, answer, record_ids, timing) -> dict`. Keys: `query_id, split, transcript (=question), asr_transcript, answer, retrieved_record_ids, timing`.
  - CLI: `generate_from_cache.py --split dev --question-cache V1.jsonl --transcript-cache VAR.jsonl --ids ids.txt --output rag/results/<name>.jsonl [--k 5] [--device cpu]`. It refuses to overwrite an existing output, retrieves with the *variant* transcript, and writes rows that pass `evaluate_answers.py --dry-run`.

- [ ] **Step 1: Write the failing tests**

```python
def test_select_ids_differing_plus_controls(self):
    ids = select_ids({"a":"x","b":"y","c":"z"}, {"a":"x","b":"Y!","c":"z"}, n_controls=1)
    self.assertIn("b", ids); self.assertEqual(len(ids), 2)
def test_candidate_uses_v1_question(self):
    row = build_candidate("q1","dev","V1 text","variant text","ans",["r1"],{})
    self.assertEqual(row["transcript"], "V1 text"); self.assertEqual(row["asr_transcript"], "variant text")
def test_validates_with_supplied_checker(self): ...   # import validate_candidates from rag/scripts/evaluate_answers.py; no exception
```

- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Rerun. Expected: OK.
- [ ] **Step 5:** Commit: `git commit -m "ext: fixed-question answer candidates"`

### Task 14: Experiment runs (cluster)

**Files:**
- Create: `results/asr/{split}_{variant}[_{setting}].csv`, `rag/.cache/{split}_{variant}[_{setting}].jsonl` (not committed), `results/retrieval/*.txt`, `results/judge/*`, `results/notes.md`
- Create: `rag/extensions/fast_asr/run_all.sbatch` (one variant per job, pinned gres, `--time=01:00:00`)

**Interfaces:**
- Consumes: Tasks 11–13 CLIs.
- Produces: the CSVs and caches Task 15 reads.

- [ ] **Step 1: Write predictions first.** From `results/hw1/run_asr_*.out` profile shares, fill the cost-model table in `results/notes.md` (spec §4.7): predicted V2 ms/query, ms/token floor ≈ 8.8 ms, and predicted E(T')/E(3000) for each margin at the median clip of 5.0 s. Commit before measuring.
- [ ] **Step 2: Practice development runs.** V1, V2, V3 m ∈ {0, 1, 2, 5}, bucket 10, each `--repeat-subset 50`. Expected: 350 rows per CSV plus 50 repeats. V2 transcripts equal V1 in every row.
- [ ] **Step 3: Retrieval on practice.** Run `evaluate_rag.py --device cpu --split practice --transcript-cache rag/.cache/practice_<v>.jsonl | tee results/retrieval/practice_<v>.txt` for each variant.
- [ ] **Step 4: Freeze.** Record the commit hash in `notes.md`. No code changes after this point except bug fixes, which are noted.
- [ ] **Step 5: Dev final runs.** Repeat Steps 2–3 on dev. Add V2-torch on 50 dev IDs plus the test clip (`--cuda none --ids dev50.txt`).
- [ ] **Step 6: Judge.** For each V3 setting with changed transcripts: `select_ids` → `generate_from_cache.py` for V1 and the variant (same IDs, same V1 question) → `evaluate_answers.py --split dev --candidates … --allow-subset --dry-run` → the same command with `--repeats 2 --out-dir results/judge/<name>`. Keep a request tally in `notes.md`. Stop if it would exceed 600.
- [ ] **Step 7: End-to-end share.** Run `run_audio_query_fast.py` on 20 dev queries for V1 and for the best V3. Record per-stage timings, with ELM latency kept separate.
- [ ] **Step 8:** Commit: `git add results rag/extensions/fast_asr/run_all.sbatch && git commit -m "exp: asr redundancy results"`

### Task 15: Analysis, tables and figures (local)

**Files:**
- Create: `rag/extensions/analysis/summarise.py`, `report/figures/*.pdf`, `report/tables/*.tex`
- Test: `rag/extensions/analysis/tests/test_summarise.py`

**Interfaces:**
- Consumes: Task 11 CSV columns, Task 5 `tryit.csv`, `evaluate_rag` outputs, judge `summary.json`.
- Produces:
  - `load_runs(paths) -> list[dict]`, which raises `ValueError` naming the GPUs if more than one `gpu` value appears.
  - `stats(xs) -> dict(mean, std, p50, p95, n)`.
  - `paired_bootstrap_ci(a, b, iters=10000, seed=0) -> (mean_diff, lo, hi)`.
  - CLI `summarise.py --split dev --out report/` writes:
    - Table 1: variant × {ASR ms mean/p50/p95, ms/token, encode/decode split, peak GiB, transcript match %, WER vs V1, Recall@10, MRR@10, ΔRecall@10 CI}.
    - Figure 1: latency breakdown stacked bars.
    - Figure 2: latency vs Recall@10 curve over trim settings.
    - Figure 3: speed-up vs clip duration with the cost-model prediction overlaid.
    - Table 2: HW1 required comparisons.

- [ ] **Step 1: Write the failing tests**

```python
def test_refuses_mixed_gpus(self):
    with self.assertRaisesRegex(ValueError, "RTX 2080 Ti.*A6000"): load_runs([csv_2080, csv_a6000])
def test_stats_percentiles(self): self.assertEqual(stats([1,2,3,4,100])["p50"], 3)
def test_bootstrap_deterministic_and_brackets(self):
    d, lo, hi = paired_bootstrap_ci([1,1,1,1], [2,2,2,2]); self.assertEqual((d, lo, hi), (1.0, 1.0, 1.0))
def test_repeat_rows_excluded_from_main_stats(self): ...   # pass>0 rows used only for the variance column
```

- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement with numpy and matplotlib. Every axis label carries its unit, and every figure caption states the split, n and GPU.
- [ ] **Step 4:** Rerun (OK). Run the CLI on the real results and open the figures.
- [ ] **Step 5:** Commit: `git commit -m "analysis: report tables and figures"`

### Task 16: Report

**Files:**
- Create: `report/main/report.tex` (copied from `report/latex_template/`), `report/main/refs.bib`

**Interfaces:**
- Consumes: Task 15 figures and tables, `results/hw1/questions.md`, `results/notes.md`.

- [ ] **Step 1:** Outline with page budget:
  - Intro and question: 0.5 pages.
  - System, baseline and supplied-vs-own components: 0.75.
  - HW1 kernels, required comparisons and 4 guide answers: 1.25.
  - KV-cache and trimming design, with alternatives considered: 1.0.
  - Methodology (controls, boundaries, repeats, judge settings): 0.75.
  - Results and mechanism analysis, predictions vs measured: 1.25.
  - Limitations and conclusion: 0.5.
- [ ] **Step 2:** Write it. Every number cites its table or figure. Every timing states its boundary. Negative or partial results are reported as found.
- [ ] **Step 3:** Appendix: full `make test`, `./run.sh --profile` and `nvidia-smi -L` outputs, plus extra guide answers. References come before the appendix. The declaration page is last, with the AI-use statement covering this planning and coding assistance.
- [ ] **Step 4:** Run `python report/format_checker/check_report_format.py report/main/report.pdf`. Expected: no FAIL, and main body ≤ 6.
- [ ] **Step 5:** Anonymity pass: grep the PDF text and metadata for your name, UUN and home paths. Expected: matches only on the final page.
- [ ] **Step 6:** Commit: `git commit -m "report: final"`

### Task 17: Video

**Files:**
- Create: `video/script.md`, `video/video.mp4` (not committed)

- [ ] **Step 1:** Script ≤ 8:00:
  - 0:00 question.
  - 1:00 system diagram.
  - 2:00 kernels and one HW1 result.
  - 3:30 KV cache and trimming mechanism.
  - 5:00 live demo: `run_audio_query_fast.py` V1 then V2/V3 on one dev query, with config shown.
  - 6:30 key figures.
  - 7:30 limitations.
- [ ] **Step 2:** When the questions are released (R−7 days), script the Question 1 and Question 2 answers, ≤ 1:00 each.
- [ ] **Step 3:** Record in OBS with a clean shell profile (prompt with no username or path), notifications off and `.env` never opened. Edit in Shotcut. Caption any shortened wait.
- [ ] **Step 4:** Check: duration ≤ 10:00, final 2:00 is Q1 then Q2, text readable at 1080p, metadata author empty.

### Task 18: Submission package

**Files:**
- Create: `scripts/make_submission_zip.sh`, `supplementary/README.md`
- Test: run the script, then `unzip -l` inspection

**Interfaces:**
- Produces: `supplementary_code.zip` containing the five `hw1-asr/csrc/task*` files, `rag/extensions/**` (no `__pycache__`), `supplementary/README.md` (files, dependencies, commands, own vs supplied), `results/**/*.csv|txt|md`.

- [ ] **Step 1:** Write the script with `git archive`-style explicit paths. No `.git`, `.env`, `rag/data`, `rag/models`, `.cache` or weights.
- [ ] **Step 2:** Run it, then `unzip -l supplementary_code.zip | grep -E "\.env|\.git/|rag/data|\.npy|safetensors"`. Expected: no output.
- [ ] **Step 3:** Grep the zip contents for your name, UUN and `/home/`. Expected: no matches.
- [ ] **Step 4:** Upload `report.pdf`, `video.mp4` and `supplementary_code.zip` on Learn, then download each back and open it.
- [ ] **Step 5:** Commit the script and README.

### Task 19: Individual review (after assignment)

- [ ] **Step 1:** Read the assigned report twice. Write the Summary (≤ 150 words), keeping their contribution apart from the supplied baseline.
- [ ] **Step 2:** Strengths: ≥ 3, numbered, each citing a page, figure or table (≤ 300 words).
- [ ] **Step 3:** Weaknesses: ≥ 3, numbered, each labelled error, open question or scope limit, with the affected claim (≤ 300 words).
- [ ] **Step 4:** Detailed Comments: develop 1–2 top issues into a concrete change or test and what it would establish (≤ 300 words).
- [ ] **Step 5:** Check word counts, tone, and that nothing identifies you. Submit on Learn.
