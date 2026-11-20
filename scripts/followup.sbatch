#!/bin/bash
# Follow-up controls, one job, one GPU, all on one node. From the repository root, on the head node:
#     sbatch scripts/followup.sbatch
# 1. Where a V2 transcription spends its time (inside kernels vs between them), on our kernels
#    and on the PyTorch path.
# 2. The dev split on the PyTorch path (ASR_CUDA=none): V1 (= the supplied reference, V0), V2 and
#    V3-m0, so the report can say whether the gains depend on our kernels.
# Outputs: results/hw1/profile_step_{cuda,torch}.txt, results/asr_torch/dev_*.csv.
#SBATCH -p Teaching
#SBATCH --gres=gpu:nvidia_geforce_rtx_2080_ti:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=asr-followup
#SBATCH --output=results/logs/followup-%j.out

export PATH=/opt/cuda-12.8.0/bin:$PATH
source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate mls-asr
cd "$SLURM_SUBMIT_DIR"
mkdir -p results/hw1 results/asr_torch rag/.cache results/logs
echo "node: $(hostname)"; nvidia-smi -L; git log --oneline -1
make -C hw1-asr lib

python rag/extensions/fast_asr/profile_step.py --cuda all  | tee results/hw1/profile_step_cuda.txt
python rag/extensions/fast_asr/profile_step.py --cuda none | tee results/hw1/profile_step_torch.txt

bench() {  # bench <name> <bench_asr args...>
    local name=$1; shift
    python rag/extensions/fast_asr/bench_asr.py --split dev --cuda none "$@" \
        --out-csv "results/asr_torch/$name.csv" --cache "rag/.cache/torch_$name.jsonl"
}
bench dev_v1    --variant v1
bench dev_v2    --variant v2
bench dev_v3_m0 --variant v3 --margin 0
