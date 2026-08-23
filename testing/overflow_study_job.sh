#!/bin/bash
#SBATCH -A dune
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -J overflow_cut_study
#SBATCH -o /global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing/overflow_study_%j.log

set -uo pipefail

HERE=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing
PY=/global/common/software/nersc9/pytorch/2.8.0/bin/python
DATA=/global/cfs/cdirs/dunepro/people/abooth/nd-production/output/MiniProdN5/run-ndlar-flow/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift/FLOW/0000000
FILES="$DATA/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.0000001.FLOW.hdf5 \
$DATA/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.0000002.FLOW.hdf5 \
$DATA/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.0000003.FLOW.hdf5"

OUT_DIR=$HERE/overflow_study_out
mkdir -p "$OUT_DIR"

N_WORKERS=8
PIDS=()
for w in $(seq 0 $((N_WORKERS - 1))); do
    CUDA_VISIBLE_DEVICES=$((w % 4)) $PY "$HERE/overflow_cut_study.py" \
        --files $FILES \
        --out "$OUT_DIR/rows_worker${w}.jsonl" \
        --event-stride $N_WORKERS --event-offset $w \
        > "$OUT_DIR/worker${w}.log" 2>&1 &
    PIDS+=("$!")
done

FAIL=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAIL=$((FAIL + 1))
done
echo "all workers done; failures=$FAIL"
wc -l "$OUT_DIR"/rows_worker*.jsonl
exit 0
