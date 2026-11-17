#!/bin/bash
# Setup-guide steps 5-6 as one batch job on the pinned GPU type:
#   CUDA + PyTorch profiles of hw1-asr, then the rag/extensions tests (PyTorch path, then kernels).
# From the repository root, on the head node:
#     sbatch scripts/check_kernels.sbatch
#     squeue -u "$USER"
#     cat results/logs/check-JOB_ID.out
# Each step runs even if an earlier one fails; exit codes are summarised at the end.
# STEPS="ext_torch ext_cuda" sbatch scripts/check_kernels.sbatch   runs only those steps.
#SBATCH -p Teaching
#SBATCH --gres=gpu:nvidia_geforce_rtx_2080_ti:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=asr-check
#SBATCH --output=results/logs/check-%j.out

export PATH=/opt/cuda-12.8.0/bin:$PATH
source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate mls-asr
cd "$SLURM_SUBMIT_DIR"
mkdir -p results/hw1 results/logs

echo "node: $(hostname)"; nvidia-smi -L; echo "python: $(which python)"; git log --oneline -1

declare -A rc
STEPS=${STEPS:-"profile_cuda profile_torch ext_torch ext_cuda"}
run() {  # run NAME OUTFILE CMD...
    local name=$1 out=$2; shift 2
    [[ " $STEPS " == *" $name "* ]] || return 0
    echo; echo "=== $name -> $out"
    "$@" > "$out" 2>&1
    rc[$name]=$?
    tail -5 "$out"
}

make -C hw1-asr lib
run profile_cuda  results/hw1/profile_cuda.txt  bash -c 'cd hw1-asr && ./run.sh --profile'
run profile_torch results/hw1/profile_torch.txt bash -c 'cd hw1-asr && ./run.sh --cuda none --profile'
run ext_torch results/hw1/ext_tests_torch.txt env FAST_ASR_TEST_CUDA=none \
    python -m unittest discover -s rag/extensions -t rag -p "test_*.py" -v
run ext_cuda  results/hw1/ext_tests_cuda.txt \
    python -m unittest discover -s rag/extensions -t rag -p "test_*.py" -v

echo; echo "=== summary (0 = ok)"
for k in $STEPS; do echo "$k: ${rc[$k]}"; done
