#!/bin/bash
# External front-end (clmatching-frontend) through the FULL v0.4.1 matching chain.
set -uo pipefail
HERE=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing
PY=/global/common/software/nersc9/pytorch/2.8.0/bin/python
DATA=/global/cfs/cdirs/dunepro/people/abooth/nd-production/output/MiniProdN5/run-ndlar-flow/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift/FLOW/0000000
FILES=""
for i in 0000002 0000003 0000004 0000005 0000006 0000007 0000008 0000009 0000010; do
  FILES="$FILES $DATA/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.${i}.FLOW.hdf5"
done
V41="--backbone stack2 --merge-p2 --sigma-mode poisson --max-clip-ticks 4 --police locked \
--merge-vertex --skip-baseline --phase3 joint --force-coverage --swap-fix \
--swap-ratio-max 0.85 --p2-threshold 30 --predict-min-energy 1.5"
OUT=$HERE/v064conf_out; mkdir -p "$OUT"
PIDS=()
for w in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$((w % 4)) $PY "$HERE/overflow_rescue_pipeline.py" \
    --files $FILES --out "$OUT/groups_worker${w}.jsonl" $V41 \
    --frontend-repo /global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/clmatching-frontend \
    --event-stride 8 --event-offset $w > "$OUT/worker${w}.log" 2>&1 &
  PIDS+=("$!")
done
FAIL=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAIL=$((FAIL+1)); done
echo "=== v064conf done failures=$FAIL rows=$(cat $OUT/groups_worker*.jsonl 2>/dev/null | wc -l) ==="
echo ALL_CONFIGS_DONE
