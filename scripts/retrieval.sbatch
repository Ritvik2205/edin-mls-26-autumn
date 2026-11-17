#!/bin/bash
# Retrieval scoring (evaluate_rag.py, enc_a, exact dense search) of cached ASR transcripts. CPU only:
# every transcript is already in the cache, so the ASR worker never starts. From the repository root:
#     sbatch scripts/retrieval.sbatch practice v1 v2 v3_m0 v3_m2 v3_b10
# Reads rag/.cache/<split>_<name>.jsonl, writes results/retrieval/<split>_<name>.txt (read by summarise.py).
#SBATCH -p Teaching
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --job-name=rag-retrieval
#SBATCH --output=results/logs/retrieval-%j.out

source "$HOME/mls-rag-env/bin/activate"
cd "$SLURM_SUBMIT_DIR"
split=$1; shift
mkdir -p results/retrieval results/logs
echo "node: $(hostname)  cpus: $SLURM_CPUS_PER_TASK"; lscpu | grep "Model name"; git log --oneline -1
for name in "$@"; do
    cache=rag/.cache/${split}_${name}.jsonl
    n=$(wc -l < "$cache")
    echo "=== $name ($n cached transcripts)"
    python rag/scripts/evaluate_rag.py --device cpu --split "$split" --transcript-cache "$cache" \
        > "results/retrieval/${split}_${name}.txt" 2>&1
    echo "exit $?"; tail -8 "results/retrieval/${split}_${name}.txt"
done
