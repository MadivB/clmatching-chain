#!/bin/bash
#SBATCH -A dune
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -J rescue_b
#SBATCH -o /global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing/rescue_b_%j.log
set -uo pipefail
HERE=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing
PY=/global/common/software/nersc9/pytorch/2.8.0/bin/python
DATA=/global/cfs/cdirs/dunepro/people/abooth/nd-production/output/MiniProdN5/run-ndlar-flow/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift/FLOW/0000000
FILES="$DATA/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.0000001.FLOW.hdf5 \
$DATA/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.0000002.FLOW.hdf5 \
$DATA/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.0000003.FLOW.hdf5"
OUT=$HERE/rescue_b_out
mkdir -p "$OUT"
PIDS=()
for w in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=$((w % 4)) $PY "$HERE/overflow_rescue_b.py" \
        --files $FILES --out "$OUT/groups_worker${w}.jsonl" \
        --event-stride 8 --event-offset $w > "$OUT/worker${w}.log" 2>&1 &
    PIDS+=("$!")
done
FAIL=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAIL=$((FAIL+1)); done
echo "done failures=$FAIL"
wc -l "$OUT"/groups_worker*.jsonl
