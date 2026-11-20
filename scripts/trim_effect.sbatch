#!/bin/bash
# The 5 s margin on dev, and why trimming changes transcripts (trim_effect.py: embedding change
# and token-probability margins per query). One job, one GPU. From the repository root:
#     sbatch scripts/trim_effect.sbatch
# Outputs: results/asr/dev_v3_m5.csv, rag/.cache/dev_v3_m5.jsonl, results/trim_effect/dev.{csv,txt}
#SBATCH -p Teaching
#SBATCH --gres=gpu:nvidia_geforce_rtx_2080_ti:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=asr-trimfx
#SBATCH --output=results/logs/trimfx-%j.out

export PATH=/opt/cuda-12.8.0/bin:$PATH
source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate mls-asr
cd "$SLURM_SUBMIT_DIR"
mkdir -p results/asr results/trim_effect results/logs rag/.cache
echo "node: $(hostname)"; nvidia-smi -L; git log --oneline -1
make -C hw1-asr lib || exit 1
python rag/extensions/fast_asr/trim_effect.py --split dev --limit 3 --out-csv results/trim_effect/smoke.csv || exit 1
python rag/extensions/fast_asr/bench_asr.py --split dev --variant v3 --margin 5 --repeat-subset 50 \
    --out-csv results/asr/dev_v3_m5.csv --cache rag/.cache/dev_v3_m5.jsonl | tail -2
python rag/extensions/fast_asr/trim_effect.py --split dev --out-csv results/trim_effect/dev.csv \
    | tee results/trim_effect/dev.txt
