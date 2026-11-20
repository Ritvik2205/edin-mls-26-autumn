#!/bin/bash
# Answer generation and LLM judge for the queries whose transcript changed (see
# scripts/judge_changed.sh). CPU only, no ASR. Needs a node that can reach the ELM endpoint; the
# 5-query smoke test at the start fails fast if it cannot. From the repository root:
#     sbatch scripts/judge.sbatch
#SBATCH -p Teaching
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --job-name=rag-judge
#SBATCH --output=results/logs/judge-%j.out

cd "$SLURM_SUBMIT_DIR"
mkdir -p results/judge results/logs
echo "node: $(hostname)  cpus: $SLURM_CPUS_PER_TASK"; git log --oneline -1
if [ ! -s results/judge/smoke_dev_v1b/answer_judge_summary.json ]; then
    bash scripts/judge_changed.sh dev 5 || { echo "smoke test failed"; exit 1; }
fi
bash scripts/judge_changed.sh dev || exit 1
bash scripts/judge_changed.sh practice || exit 1
