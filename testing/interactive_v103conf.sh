#!/bin/bash
# (a) single-worker latency probe (contention-free), (b) files 2-10 confirmation — both with --skip-v2
set -uo pipefail
HERE=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing
PY=/global/common/software/nersc/pe/conda-envs/26.1.0/python-3.13/nersc-python/bin/python
DATA=/global/cfs/cdirs/dunepro/people/abooth/nd-production/output/MiniProdN5/run-ndlar-flow/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift/FLOW/0000000
F1="$DATA/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.0000001.FLOW.hdf5"
FILES=""
for i in 2 3 4 5 6 7 8 9 10; do FILES="$FILES $DATA/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.$(printf %07d $i).FLOW.hdf5"; done
SMALL=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/NewMLSection/runs/ndfull_small4x/best_model_arch.pt
V86="--backbone stack2 --merge-p2 --sigma-mode poisson --max-clip-ticks 4 --police locked \
--merge-vertex --skip-baseline --phase3 joint --pass2-trade --pass2-light-nom --force-coverage --swap-fix \
--swap-ratio-max 0.85 --p2-threshold 30 --predict-min-energy 1.5 --light-ckpt $SMALL --skip-v2 --gate-support --le-retime \
--frontend-repo /global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/clmatching-frontend"
# --- (a) latency probe: ONE worker, whole node to itself, 4 events ---
OUT=$HERE/v103LAT_out; mkdir -p "$OUT"
echo "=== latency probe start $(date +%H:%M:%S) ==="
CUDA_VISIBLE_DEVICES=0 $PY "$HERE/overflow_rescue_pipeline.py" \
  --files $F1 --out "$OUT/groups_worker0.jsonl" $V86 \
  --event-stride 4 --event-offset 0 > "$OUT/worker0.log" 2>&1
echo "=== probe done rows=$(cat $OUT/groups_worker0.jsonl | wc -l) $(date +%H:%M:%S) ==="
# --- (b) files 2-10 confirmation, standard 8-worker layout ---
OUT=$HERE/v103CONF_out; mkdir -p "$OUT"
PIDS=()
for w in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$((w % 4)) $PY "$HERE/overflow_rescue_pipeline.py" \
    --files $FILES --out "$OUT/groups_worker${w}.jsonl" $V86 \
    --event-stride 8 --event-offset $w > "$OUT/worker${w}.log" 2>&1 &
  PIDS+=("$!")
done
FAIL=0
for pid in "${PIDS[@]}"; do wait $pid || FAIL=$((FAIL+1)); done
echo "=== conf done failures=$FAIL rows=$(cat $OUT/groups_worker*.jsonl 2>/dev/null | wc -l) ==="
echo ALL_CONFIGS_DONE
