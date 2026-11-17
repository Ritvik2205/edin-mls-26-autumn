#!/bin/bash
# One bench_asr.py run per job on the pinned GPU type. From the repository root, on the head node:
#     sbatch scripts/bench.sbatch <out_dir> <name> <bench_asr args...>
#     e.g. sbatch scripts/bench.sbatch results/asr practice_v3_m2 --split practice --variant v3 --margin 2 --repeat-subset 50
# Writes <out_dir>/<name>.csv and rag/.cache/<name>.jsonl. Resubmitting the same command resumes.
#SBATCH -p Teaching
#SBATCH --gres=gpu:nvidia_geforce_rtx_2080_ti:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=asr-bench
#SBATCH --output=results/logs/bench-%j.out

export PATH=/opt/cuda-12.8.0/bin:$PATH
source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate mls-asr
cd "$SLURM_SUBMIT_DIR"
out_dir=$1 name=$2; shift 2
mkdir -p "$out_dir" rag/.cache results/logs
echo "node: $(hostname)"; nvidia-smi -L; git log --oneline -1; echo "run: $name $*"
make -C hw1-asr lib
python rag/extensions/fast_asr/bench_asr.py "$@" --out-csv "$out_dir/$name.csv" --cache "rag/.cache/$name.jsonl"
