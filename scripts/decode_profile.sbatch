#!/bin/bash
# What a KV-cache decode step is made of, per operation, for the tiled kernel, the two GEMV
# kernels and the PyTorch path. One job, one GPU. From the repository root, on the head node:
#     sbatch scripts/decode_profile.sbatch
# Outputs: results/hw1/decode_{off,seq,warp,torch}.txt
#SBATCH -p Teaching
#SBATCH --gres=gpu:nvidia_geforce_rtx_2080_ti:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=asr-decode
#SBATCH --output=results/logs/decode-%j.out

export PATH=/opt/cuda-12.8.0/bin:$PATH
source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate mls-asr
cd "$SLURM_SUBMIT_DIR"
mkdir -p results/hw1 results/logs
echo "node: $(hostname)"; nvidia-smi -L; git log --oneline -1
make -C hw1-asr lib || exit 1
for mode in off seq warp; do
    ASR_GEMV=$mode python rag/extensions/fast_asr/profile_decode.py --cuda all \
        | tee "results/hw1/decode_${mode}.txt"
done
python rag/extensions/fast_asr/profile_decode.py --cuda none | tee results/hw1/decode_torch.txt
