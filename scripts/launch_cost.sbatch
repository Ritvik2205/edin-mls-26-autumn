#!/bin/bash
# Is a KV-cache decode step bound by GPU work or by the host issuing its launches? Per GEMV
# kernel: host issue time and GPU wait per step, and the cost of the launch path alone.
# One job, one GPU. From the repository root, on the head node:
#     sbatch scripts/launch_cost.sbatch
# Outputs: results/hw1/launch_{off,seq,warp}.txt
#SBATCH -p Teaching
#SBATCH --gres=gpu:nvidia_geforce_rtx_2080_ti:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --job-name=asr-launch
#SBATCH --output=results/logs/launch-%j.out

export PATH=/opt/cuda-12.8.0/bin:$PATH
source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate mls-asr
cd "$SLURM_SUBMIT_DIR"
mkdir -p results/hw1 results/logs
echo "node: $(hostname)"; nvidia-smi -L; git log --oneline -1
make -C hw1-asr lib || exit 1
for mode in off seq warp; do
    ASR_GEMV=$mode python rag/extensions/fast_asr/launch_cost.py | tee "results/hw1/launch_${mode}.txt"
done
