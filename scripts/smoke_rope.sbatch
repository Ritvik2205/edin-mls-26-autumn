#!/bin/bash
# Setup-guide steps 6b + 7a in one job: V2 smoke run (3 queries), then the RoPE kernel-vs-scratch
# comparison (20 queries each). From the repository root: sbatch scripts/smoke_rope.sbatch
# The RoPE CSVs go to results/rope/, not results/asr/, so summarise.py does not mix them into V2.
#SBATCH -p Teaching
#SBATCH --gres=gpu:nvidia_geforce_rtx_2080_ti:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=asr-smoke
#SBATCH --output=results/logs/smoke-%j.out

export PATH=/opt/cuda-12.8.0/bin:$PATH
source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate mls-asr
cd "$SLURM_SUBMIT_DIR"
mkdir -p results/asr results/rope results/logs rag/.cache
echo "node: $(hostname)"; nvidia-smi -L; git log --oneline -1
make -C hw1-asr lib
B=rag/extensions/fast_asr/bench_asr.py

echo "=== smoke v2"
python $B --split practice --variant v2 --limit 3 \
    --out-csv results/asr/smoke_v2.csv --cache rag/.cache/smoke_v2.jsonl || exit 1
echo "=== rope scratch"
FAST_ASR_ROPE=scratch python $B --split practice --variant v2 --limit 20 \
    --out-csv results/rope/practice_v2_rope_scratch.csv --cache rag/.cache/practice_v2_rope_scratch.jsonl
echo "=== rope kernel"
FAST_ASR_ROPE=kernel python $B --split practice --variant v2 --limit 20 \
    --out-csv results/rope/practice_v2_rope_kernel.csv --cache rag/.cache/practice_v2_rope_kernel.jsonl
echo "=== done"
