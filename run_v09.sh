#!/bin/bash
# v0.9 canonical charge-light matching = v0.8.9 chain + stage-6 continuous
# t0s (--subtick, batched interaction-level re-prediction + 9-pt quadratic)
# with per-group timing uncertainty output (hit_t0_sigma, ticks).
#
# Usage:   bash run_v09.sh <FLOW.hdf5> <out_dir> [n_workers] [n_gpus]
# Typical: 8 workers on a 4-GPU node (2 events resident per GPU).
set -uo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/global/common/software/nersc9/pytorch/2.8.0/bin/python
FILES="$1"; OUT="$2"; NW="${3:-8}"; NG="${4:-4}"
SMALL=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/NewMLSection/runs/ndfull_small4x/best_model_arch.pt
FRONTEND=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/clmatching-frontend

V09="--backbone stack2 --merge-p2 --sigma-mode poisson --max-clip-ticks 4 \
--police locked --merge-vertex --skip-baseline --phase3 joint --pass2-trade \
--pass2-light-nom --force-coverage --swap-fix --swap-ratio-max 0.85 \
--p2-threshold 30 --predict-min-energy 1.5 --skip-v2 --gate-support \
--le-retime --edge-penalty --subtick \
--light-ckpt $SMALL --frontend-repo $FRONTEND"

mkdir -p "$OUT"
PIDS=()
for w in $(seq 0 $((NW - 1))); do
  CUDA_VISIBLE_DEVICES=$((w % NG)) $PY "$HERE/testing/overflow_rescue_pipeline.py" \
    --files $FILES --out "$OUT/groups_worker${w}.jsonl" $V09 \
    --event-stride $NW --event-offset $w > "$OUT/worker${w}.log" 2>&1 &
  PIDS+=("$!")
done
FAIL=0
for p in "${PIDS[@]}"; do wait "$p" || FAIL=$((FAIL + 1)); done
ROWS=$(cat "$OUT"/groups_worker*.jsonl 2>/dev/null | wc -l)
echo "=== done failures=$FAIL rows=$ROWS ==="
echo "ALL_CONFIGS_DONE"
