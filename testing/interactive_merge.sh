#!/bin/bash
# 3.1+3.3 merge test: all P2-class clusters through the permutation pool
# (baseline for comparison: v040expB_out = canonical v0.4.0 + plain expansion)
set -uo pipefail
HERE=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing
PY=/global/common/software/nersc9/pytorch/2.8.0/bin/python
DATA=/global/cfs/cdirs/dunepro/people/abooth/nd-production/output/MiniProdN5/run-ndlar-flow/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift/FLOW/0000000
FILES="$DATA/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.0000001.FLOW.hdf5 \
$DATA/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.0000002.FLOW.hdf5 \
$DATA/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.0000003.FLOW.hdf5"
V4="--backbone stack2 --sigma-mode poisson --max-clip-ticks 4 --police locked --merge-vertex \
--skip-baseline --phase3 both --force-coverage --swap-fix --swap-ratio-max 0.85 \
--p2-threshold 30 --predict-min-energy 1.5"
run_cfg () {
  tag=$1; extra=$2
  OUT=$HERE/v040${tag}_out; mkdir -p "$OUT"
  echo "=== config $tag ($extra) start $(date +%H:%M:%S) ==="
  PIDS=()
  for w in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=$((w % 4)) $PY "$HERE/overflow_rescue_pipeline.py" \
      --files $FILES --out "$OUT/groups_worker${w}.jsonl" $V4 $extra \
      --event-stride 8 --event-offset $w > "$OUT/worker${w}.log" 2>&1 &
    PIDS+=("$!")
  done
  FAIL=0
  for pid in "${PIDS[@]}"; do wait "$pid" || FAIL=$((FAIL+1)); done
  echo "=== config $tag done failures=$FAIL rows=$(cat $OUT/groups_worker*.jsonl 2>/dev/null | wc -l) $(date +%H:%M:%S) ==="
}
run_cfg mrgA ""
run_cfg mrgB "--merge-p2"
echo ALL_CONFIGS_DONE
