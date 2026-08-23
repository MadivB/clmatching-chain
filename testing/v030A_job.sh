#!/bin/bash
#SBATCH -A dune
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 01:15:00
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -J rescue_pipe
#SBATCH -o /global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing/v030A_%j.log
set -uo pipefail
HERE=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing
PY=/global/common/software/nersc9/pytorch/2.8.0/bin/python
DATA=/global/cfs/cdirs/dunepro/people/abooth/nd-production/output/MiniProdN5/run-ndlar-flow/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift/FLOW/0000000
FILES="$DATA/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.0000001.FLOW.hdf5 \
$DATA/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.0000002.FLOW.hdf5 \
$DATA/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.0000003.FLOW.hdf5"
OUT=$HERE/v030A_out
mkdir -p "$OUT"
PIDS=()
for w in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=$((w % 4)) $PY "$HERE/overflow_rescue_pipeline.py" \
        --files $FILES --out "$OUT/groups_worker${w}.jsonl" \
        --backbone stack2 --sigma-mode poisson --max-clip-ticks 4 --police locked --merge-vertex --skip-baseline --phase3 matrix --event-stride 8 --event-offset $w > "$OUT/worker${w}.log" 2>&1 &
    PIDS+=("$!")
done
FAIL=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAIL=$((FAIL+1)); done
echo "done failures=$FAIL"
wc -l "$OUT"/groups_worker*.jsonl
