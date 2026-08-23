#!/bin/bash
# v0.5.6 release production: file 0000001 (13 events; first 10 -> .pt),
# canonical v0.4.1 chain + release dumps with per-cluster confidence.
set -uo pipefail
HERE=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing
PY=/global/common/software/nersc9/pytorch/2.8.0/bin/python
DATA=/global/cfs/cdirs/dunepro/people/abooth/nd-production/output/MiniProdN5/run-ndlar-flow/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift/FLOW/0000000
FILES="$DATA/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.0000001.FLOW.hdf5"
V41="--backbone stack2 --merge-p2 --sigma-mode poisson --max-clip-ticks 4 --police locked \
--merge-vertex --skip-baseline --phase3 both --force-coverage --swap-fix \
--swap-ratio-max 0.85 --p2-threshold 30 --predict-min-energy 1.5"
OUT=$HERE/v056rel_out; mkdir -p "$OUT" "$OUT/dumps"
PIDS=()
for w in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$((w % 4)) $PY "$HERE/overflow_rescue_pipeline.py" \
    --files $FILES --out "$OUT/groups_worker${w}.jsonl" $V41 \
    --dump-dir "$OUT/dumps" \
    --event-stride 8 --event-offset $w > "$OUT/worker${w}.log" 2>&1 &
  PIDS+=("$!")
done
FAIL=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAIL=$((FAIL+1)); done
echo "=== v056 done failures=$FAIL rows=$(cat $OUT/groups_worker*.jsonl 2>/dev/null | wc -l) ==="
echo ALL_CONFIGS_DONE
