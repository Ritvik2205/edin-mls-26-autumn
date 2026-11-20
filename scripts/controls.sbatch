#!/bin/bash
# Controls and the GEMV extension, one job, one GPU, one node. From the repository root, on the
# head node:
#     sbatch scripts/controls.sbatch
# 1. hw1-asr tests and profiles with the hand-in defaults (ASR_GEMV unset).
# 2. M = 1 kernels: GEMM test, one profiled V2 transcription and the GEMV unit tests for
#    ASR_GEMV = off, seq, warp.
# 3. TILE at M = 1: the same two measurements with TILE = 8, 16, 32 (tiled kernel). The source is
#    copied to node-local scratch and only the copy is edited; hw1-asr/ is not touched.
# 4. Dev split, all on this node: a replicate of the five original runs, then V2 and V3-m0 with
#    each GEMV kernel.
# Outputs: results/hw1/{gemv,tile}_*.txt, results/asr_rep/dev_*.csv, results/asr_gemv/dev_*.csv.
#SBATCH -p Teaching
#SBATCH --gres=gpu:nvidia_geforce_rtx_2080_ti:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --job-name=asr-controls
#SBATCH --output=results/logs/controls-%j.out

export PATH=/opt/cuda-12.8.0/bin:$PATH
source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate mls-asr
cd "$SLURM_SUBMIT_DIR"
mkdir -p results/hw1 results/asr_rep results/asr_gemv rag/.cache results/logs
echo "node: $(hostname)"; nvidia-smi -L; git log --oneline -1
unset ASR_GEMV
make -C hw1-asr || exit 1

echo "=== 1. hand-in defaults"
make -C hw1-asr test > results/hw1/gemv_branch_make_test.txt 2>&1; echo "make test: $?"
tail -3 results/hw1/gemv_branch_make_test.txt

echo "=== 2. M = 1 kernels"
gemm=$(ls hw1-asr/build/sm_*/test_task3_gemm | head -1)
for mode in off seq warp; do
    ASR_GEMV=$mode "$gemm" > "results/hw1/gemv_${mode}_gemm.txt" 2>&1
    grep -A1 "lm_head" "results/hw1/gemv_${mode}_gemm.txt"
    ASR_GEMV=$mode python rag/extensions/fast_asr/profile_step.py --cuda all \
        | tee "results/hw1/gemv_${mode}_step.txt"
done
python -m unittest discover -s rag/extensions/fast_asr/tests -t rag -p "test_gemv.py" -v \
    > results/hw1/gemv_tests.txt 2>&1; echo "gemv tests: $?"
tail -12 results/hw1/gemv_tests.txt

echo "=== 3. TILE at M = 1"
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
for tile in 8 16 32; do
    src="$scratch/tile_$tile"
    mkdir -p "$src"
    cp -r hw1-asr/Makefile hw1-asr/csrc hw1-asr/tests "$src/"
    sed -i "s/^constexpr int TILE = [0-9]*;/constexpr int TILE = $tile;/" "$src/csrc/task3_gemm.cu"
    grep -n "^constexpr int TILE" "$src/csrc/task3_gemm.cu"
    make -C "$src" BUILD=build >/dev/null 2>&1 || { echo "TILE=$tile: build failed"; continue; }
    "$src/build/test_task3_gemm" > "results/hw1/tile_${tile}_gemm.txt" 2>&1
    grep -A1 "lm_head" "results/hw1/tile_${tile}_gemm.txt"
    ASR_LIB="$src/build/libasr_kernels.so" python rag/extensions/fast_asr/profile_step.py --cuda all \
        | tee "results/hw1/tile_${tile}_step.txt"
done

echo "=== 4. dev split on this node"
bench() {  # bench <out_dir> <name> <bench_asr args...>
    local dir=$1 name=$2; shift 2
    python rag/extensions/fast_asr/bench_asr.py --split dev "$@" \
        --out-csv "results/$dir/$name.csv" --cache "rag/.cache/${dir}_$name.jsonl" | tail -2
}
# The GEMV runs first: they are short and are the new result.
for mode in seq warp; do
    bench asr_gemv "dev_v2_$mode"    --variant v2 --gemv "$mode"
    bench asr_gemv "dev_v3_m0_$mode" --variant v3 --margin 0 --gemv "$mode"
done
bench asr_rep dev_v2     --variant v2
bench asr_rep dev_v3_m0  --variant v3 --margin 0
bench asr_rep dev_v3_m2  --variant v3 --margin 2
bench asr_rep dev_v3_b10 --variant v3 --bucket 10
bench asr_rep dev_v1     --variant v1
echo "=== done"
