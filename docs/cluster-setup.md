# Cluster setup and run guide

Copy-paste steps for the Informatics teaching cluster (`mlp.inf.ed.ac.uk`, partition `Teaching`).
The order gets you fast feedback first: environment, then PyTorch baseline, then kernels, then the long runs.
Replace `<uun>` with your login. Paths assume the checkout is at `~/edin-mls-26-autumn`.

Rules for every step:
- The head node has internet but no GPU. Do setup, `git lfs pull` and pip installs there.
- GPU nodes may have no internet. Build and run there.
- Every timing run uses the pinned GPU: `--gres=gpu:nvidia_geforce_rtx_2080_ti:1` (builds go to `hw1-asr/build/sm_75`).
- If anything fails, paste the full output back to Claude.

---

## 1. On the Mac: push the branch

```bash
cd ~/edin-mls-26-autumn
git push -u origin coursework
```

Expected: `branch 'coursework' set up to track 'origin/coursework'`.

## 2. Head node: clone, assets, environments (once)

`mlp` only accepts connections from the Informatics VPN or a DICE machine. Otherwise go through `ssh.inf.ed.ac.uk` first.

```bash
ssh s2473583@ssh.inf.ed.ac.uk      # only if not on VPN/DICE
ssh s2473583@mlp.inf.ed.ac.uk
```

Clone and check out:

```bash
cd ~
git clone https://github.com/Ritvik2205/edin-mls-26-autumn.git
cd ~/edin-mls-26-autumn
git checkout coursework
```

Expected: `Switched to a new branch 'coursework'` (tracking `origin/coursework`).

Fetch `rag-assets.zip` (Git LFS, about 196 MB):

```bash
git lfs install
git lfs pull --include="rag-assets.zip"
ls -lh rag-assets.zip
```

Expected: about 196 MB. A file of a few hundred bytes is the LFS pointer, so the pull did not happen.
If `git: 'lfs' is not a git command`, Git LFS is not installed on that machine (check how to get it on `mlp`).

ASR environment and model download (needs about 20 GB in home, mostly download time; safe to rerun). This also downloads the GLM-ASR-Nano weights into `~/.cache/huggingface`, so GPU nodes work offline:

```bash
cd ~/edin-mls-26-autumn/hw1-asr
bash setup_env.sh
```

Expected: ends with `>>> Setup finished.`

RAG venv (bridged to ASR by `RAG_ASR_PYTHON`):

```bash
cd ~/edin-mls-26-autumn
conda activate mls-asr
export RAG_ASR_PYTHON="$(python -c 'import sys; print(sys.executable)')"
python -m venv "$HOME/mls-rag-env"
source "$HOME/mls-rag-env/bin/activate"
python -m pip install -r rag/requirements.txt
cp rag/.env.example rag/.env
```

Expected: pip finishes without errors; `rag/.env` exists (key still empty, see step 9).

Install the course data:

```bash
mkdir -p "$HOME/mls-rag-assets"
unzip rag-assets.zip -d "$HOME/mls-rag-assets"
python rag/scripts/install_assets.py --from-dir "$HOME/mls-rag-assets"
```

Expected: `Course assets verified`.

The enc_a encoder is bundled in the assets, so no further download should be needed for retrieval (check: run step 4 on a GPU node to confirm nothing tries to reach the internet).

## 3. GPU shell (every session)

```bash
srun -p Teaching --gres=gpu:nvidia_geforce_rtx_2080_ti:1 --cpus-per-task=2 --mem=16G --time=01:00:00 --pty bash
export PATH=/opt/cuda-12.8.0/bin:$PATH
conda activate mls-asr
cd ~/edin-mls-26-autumn
nvidia-smi -L
```

Expected: `nvidia-smi -L` prints `NVIDIA GeForce RTX 2080 Ti`.

For RAG scripts in the same shell, also run:

```bash
export RAG_ASR_PYTHON="$(python -c 'import sys; print(sys.executable)')"
source "$HOME/mls-rag-env/bin/activate"
```

**Do not keep the venv active for `hw1-asr/` work.** The venv's `python` (no torch) shadows the conda one,
so `./run.sh`, `make test` and the `rag/extensions` GPU tests fail with `No module named 'torch'`.
Run `deactivate` first and check `which python` points into `envs/mls-asr`. The ASR worker launched
by RAG scripts is fine either way: it uses `RAG_ASR_PYTHON`.

## 4. Validate setup (PyTorch only, no kernels needed)

```bash
cd ~/edin-mls-26-autumn/hw1-asr
./run.sh --cuda none
```

Expected: ends with `Status: PASS`. Loading takes from half a minute to a few minutes.

Save the baseline (plan Task 0):

```bash
cd ~/edin-mls-26-autumn
mkdir -p results/hw1
make -C hw1-asr && (cd hw1-asr && ./run.sh --cuda none --profile) | tee results/hw1/baseline_torch.txt
```

Expected: `Status: PASS`, about 990 ms per transcription.

Check RAG setup (needs the step-3 RAG lines):

```bash
make -C hw1-asr lib
python rag/scripts/check_setup.py
```

Expected: PASS for corpus, audio manifest, embeddings, enc_a model and ASR source path.
`[FAIL] ELM configuration: ELM_API_KEY is not configured` is expected until the key is issued.

## 5. Kernels

```bash
cd ~/edin-mls-26-autumn/hw1-asr
make
make test1
make test2
make test3
make test4
make test
./run.sh --profile
nvidia-smi -L
```

Expected: `make` without errors; each test reports ALL PASS; `./run.sh --profile` ends with `Status: PASS` and prints the per-operation timing table.
Paste the full output of any failure back to Claude.

PyTorch profile for the measurement pack (plan Task 5, same node type):

```bash
./run.sh --cuda none --profile > ../results/hw1/profile_torch.txt
```

Debugging:

```bash
ASR_DEBUG=1 ./run.sh                                            # sync after every kernel; error names the right op
./run.sh --cuda task1                                           # then task1,task2 ... to find the breaking task
compute-sanitizer build/sm_75/test_task3_gemm small             # exact line of out-of-bounds access
compute-sanitizer --tool racecheck build/sm_75/test_task3_gemm small   # shared-memory races
```

Always add `small` under the sanitizer (seconds instead of many minutes).

## 6. Extension GPU tests

From the repo root, in `mls-asr`, on a node where `make -C hw1-asr lib` was built.

PyTorch path first:

```bash
cd ~/edin-mls-26-autumn
make -C hw1-asr lib
FAST_ASR_TEST_CUDA=none python -m unittest discover -s rag/extensions -t rag -p "test_*.py" -v
```

Then your kernels (default is `all`):

```bash
python -m unittest discover -s rag/extensions -t rag -p "test_*.py" -v
```

Expected: `OK` both times, GPU tests not skipped (they skip only without CUDA).

RoPE cost comparison: `FAST_ASR_ROPE=kernel|scratch` overrides the automatic choice (default is `kernel` when available):

```bash
FAST_ASR_ROPE=scratch python rag/extensions/fast_asr/bench_asr.py --split practice --variant v2 --limit 20 \
    --out-csv results/rope/practice_v2_rope_scratch.csv --cache rag/.cache/practice_v2_rope_scratch.jsonl
python rag/extensions/fast_asr/bench_asr.py --split practice --variant v2 --limit 20 \
    --out-csv results/rope/practice_v2_rope_kernel.csv --cache rag/.cache/practice_v2_rope_kernel.jsonl
```

Kept in `results/rope/`, not `results/asr/`: `summarise.py` reads every `results/asr/{split}_*.csv` and would mix these into the V2 row.

Quick smoke run before long jobs (plan Task 11):

```bash
python rag/extensions/fast_asr/bench_asr.py --split practice --variant v2 --limit 3 \
    --out-csv results/asr/smoke_v2.csv --cache rag/.cache/smoke_v2.jsonl
```

Expected: 3 rows, non-zero timings.

## 7. Batch measurement

HW1 pack (plan Task 5), from `hw1-asr/` on the head node:

```bash
cd ~/edin-mls-26-autumn/hw1-asr
sbatch --gres=gpu:nvidia_geforce_rtx_2080_ti:1 run_asr.sbatch
squeue -u "$USER"                      # wait until the job is gone
cp asr-<jobid>.out ../results/hw1/run_asr_<jobid>.out
```

Expected: four ALL PASS, `Status: PASS`, and an `nvidia-smi -L` line.

`bench_asr.py` flags: `--split {practice,dev}` and `--variant {v1,v2,v3}` (required), `--margin S` or `--bucket S` (v3, mutually exclusive), `--cuda` (default `all`), `--out-csv` and `--cache` (required), `--warmup` (default 5), `--ids FILE`, `--repeat-subset N`, `--limit N`, `--max-new-tokens` (default 100). Rerunning the same command resumes.

Practice runs (plan Task 14 Step 2 uses `--repeat-subset 50`; naming `results/asr/{split}_{variant}[_{setting}].csv`):

```bash
python rag/extensions/fast_asr/bench_asr.py --split practice --variant v1 --repeat-subset 50 \
    --out-csv results/asr/practice_v1.csv --cache rag/.cache/practice_v1.jsonl
python rag/extensions/fast_asr/bench_asr.py --split practice --variant v2 --repeat-subset 50 \
    --out-csv results/asr/practice_v2.csv --cache rag/.cache/practice_v2.jsonl
python rag/extensions/fast_asr/bench_asr.py --split practice --variant v3 --margin 0 --repeat-subset 50 \
    --out-csv results/asr/practice_v3_m0.csv --cache rag/.cache/practice_v3_m0.jsonl
python rag/extensions/fast_asr/bench_asr.py --split practice --variant v3 --margin 2 --repeat-subset 50 \
    --out-csv results/asr/practice_v3_m2.csv --cache rag/.cache/practice_v3_m2.jsonl
python rag/extensions/fast_asr/bench_asr.py --split practice --variant v3 --bucket 10 --repeat-subset 50 \
    --out-csv results/asr/practice_v3_b10.csv --cache rag/.cache/practice_v3_b10.jsonl
```

Expected: 350 rows per CSV plus 50 repeats; V2 transcripts equal V1 in every row.
`_b10` matches the setting name `bench_asr.py` writes (`b10`); `_m<margin>` follows `supplementary/README.md`. Margins 1 and 5 are P2 (spec §8): run them the same way only if GPU time is left after P0.

Wrapper (save as e.g. `~/bench_one.sbatch`, outside the repo; the plan names `rag/extensions/fast_asr/run_all.sbatch`, which does not exist yet):

```bash
#!/bin/bash
#SBATCH -p Teaching
#SBATCH --gres=gpu:nvidia_geforce_rtx_2080_ti:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=bench-%j.out
# usage, from the repo root: sbatch ~/bench_one.sbatch <name> <bench_asr args...>
#   e.g. sbatch ~/bench_one.sbatch practice_v3_m2 --split practice --variant v3 --margin 2 --repeat-subset 50
export PATH=/opt/cuda-12.8.0/bin:$PATH
source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate mls-asr
cd "$SLURM_SUBMIT_DIR"
hostname; nvidia-smi -L
make -C hw1-asr lib
name=$1; shift
python rag/extensions/fast_asr/bench_asr.py "$@" \
    --out-csv "results/asr/${name}.csv" --cache "rag/.cache/${name}.jsonl"
```

Submit one job per variant from `~/edin-mls-26-autumn`; check with `squeue -u "$USER"` and `cat bench-<jobid>.out`.
(check: whether 350 + 50 queries fit in 1 hour per variant; the job resumes if resubmitted.)

## 8. Copy results to the Mac

Run on the Mac. Local CPU retrieval scoring from cached transcripts is allowed (forum #42); report the machine and timing boundary.

```bash
cd ~/edin-mls-26-autumn
rsync -av s2473583@mlp.inf.ed.ac.uk:edin-mls-26-autumn/results/ results/
rsync -av s2473583@mlp.inf.ed.ac.uk:edin-mls-26-autumn/rag/.cache/ rag/.cache/
```

Off VPN, add `-e "ssh -J s2473583@ssh.inf.ed.ac.uk"`. Then locally:

```bash
python rag/scripts/evaluate_rag.py --device cpu --split practice --transcript-cache rag/.cache/practice_v1.jsonl | tee results/retrieval/practice_v1.txt
```

Expected: Recall@5/10/20/100, MRR@10, NDCG@10 printed. Local needs the RAG venv and installed assets too.

## 9. ELM key

When issued, put it only in `rag/.env` on the machine that runs generation. Never in chat, git, or the report.

```text
ELM_API_KEY=<your key>
ELM_BASE_URL=https://elm.edina.ac.uk/api/v1
ELM_MODEL=utter-project/EuroLLM-22B-Instruct-2512
```

The base URL and model are already in `rag/.env.example`; leave them as supplied.

```bash
git check-ignore -v rag/.env
python rag/scripts/check_setup.py
```

Expected: `rag/.gitignore:3:.env	rag/.env`, and the ELM configuration line now PASS.
Generation and judging spend real requests: start with `--limit 5` / `--dry-run`.

## 10. Troubleshooting

| What you see | Cause | Fix |
|---|---|---|
| `No GPU found. Build on a GPU node` | On the head node | Step 3 `srun` first |
| `nvcc: command not found` | CUDA not on PATH | `export PATH=/opt/cuda-12.8.0/bin:$PATH` |
| `ModuleNotFoundError: No module named 'torch'` | conda env not active, or `mls-rag-env` venv active on top of it (prompt shows both) | `deactivate`, then `conda activate mls-asr`; `which python` must be in `envs/mls-asr` |
| `no kernel image is available for execution` | Library built on a different GPU type | `make` (or `make -C hw1-asr lib`) on the node you run on; check `nvidia-smi -L` |
| Timings differ wildly between runs | Different GPU type | Always pin `--gres=gpu:nvidia_geforce_rtx_2080_ti:1` |
| Test `FAIL` with `got 0` | Blank empty or wrong write index | Check that blank against its guide |
| `CUDA error ... an illegal memory access` | Index past array end | `compute-sanitizer build/sm_75/test_taskN_... small` |
| Test hangs | `__syncthreads()` reached by only some threads | Keep barriers outside thread-dependent `if`/`return` |
| `nan` in output | `exp` overflow, divide by zero, `-inf - (-inf)` | `-INFINITY` notes in task 2/4 guides |
| Tests pass, transcription wrong | Kernel fails on untested shapes | `./run.sh --cuda task1`, then `task1,task2`, ... |
| `CUDA out of memory` | GPU busy or too small | `nvidia-smi`; request another GPU |
| Hugging Face download/connection error on GPU node | Weights not cached; GPU nodes offline | Rerun `bash hw1-asr/setup_env.sh` on the head node; `HF_HUB_OFFLINE=1` makes failures immediate (check) |
| `git: 'lfs' is not a git command` | Git LFS missing | Install it, repeat step 2 LFS lines |
| `unzip`: not a ZIP archive | Still the LFS pointer | `git lfs install && git lfs pull --include="rag-assets.zip"` |
| ELM key missing / auth error | `rag/.env` | Step 9 |
| ASR source path missing | Wrong checkout | Full checkout, or set `RAG_ASR_ROOT` to `hw1-asr/` |
| ASR worker cannot import torch/transformers | `RAG_ASR_PYTHON` unset | Step 3 RAG lines |
