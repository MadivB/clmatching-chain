#!/bin/bash
set -uo pipefail
HERE=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing
PY=/global/common/software/nersc9/pytorch/2.8.0/bin/python
DATA=/global/cfs/cdirs/dunepro/people/abooth/nd-production/output/MiniProdN5/run-ndlar-flow/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift/FLOW/0000000
FILES="$DATA/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.0000001.FLOW.hdf5 $DATA/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.0000002.FLOW.hdf5 $DATA/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.0000003.FLOW.hdf5"
COMMON="--backbone stack2 --sigma-mode poisson --max-clip-ticks 4 --police locked --merge-vertex --skip-baseline --phase3 both --force-coverage --swap-fix"
OUT=$HERE/v030v3_out; DUMP=$OUT/dumps; mkdir -p "$DUMP"
echo "=== v0.3 (D50 + swap-fix) + dumps start $(date +%H:%M:%S) ==="
PIDS=()
for w in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$((w % 4)) $PY "$HERE/overflow_rescue_pipeline.py" --files $FILES \
    --out "$OUT/groups_worker${w}.jsonl" $COMMON --dump-dir "$DUMP" \
    --event-stride 8 --event-offset $w > "$OUT/worker${w}.log" 2>&1 &
  PIDS+=("$!")
done
F=0; for pid in "${PIDS[@]}"; do wait "$pid" || F=$((F+1)); done
echo "=== v0.3 done fail=$F rows=$(cat $OUT/groups_worker*.jsonl 2>/dev/null|wc -l) dumps=$(ls $DUMP/*.npz 2>/dev/null|wc -l) $(date +%H:%M:%S) ==="
echo ALL_V3_DONE
