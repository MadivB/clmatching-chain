#!/bin/bash
# v1.2 canonical charge-light matching = v1.1 chain (run_v11.sh) + stage P per-hit precision timing:
#   --precision --precision-calib testing/v120_calib_lowi.json --precision-rounds 3
#   (per-family support channels, delay-aware per-channel fits on the pile-up residual, weighted-median
#    family apex time, apex-clock per-hit deposition times, in-situ k_idx / per-type slew calibrated on the
#    low-intensity sample; delivers hit_t0_sigma_p per hit).  Assignment identical to v1.1.
# v1.1 = v1.0 chain (run_v1.sh) +
#   --pool-mode light            light-coupled component batching + exact combination search (stage 3.1)
#   --p2-threshold 20            20-30 MeV single-TPC clusters join the component search (pool = light-judgeable clusters)
#   --redecide --redecide-emin 20  stage 3.98: final-state component re-decision, light-trustworthy band
#   --predict-min-energy 0.5 --predict-min-energy-shower 1.5   templates for 0.5-1.5 MeV clusters (non-shower TPCs)
# v1.1 flag set = arm K1 (5-file cert: NS +0.26 / all-TPC +0.17); v1.2 certification (files 1-5) pending
# (see V110_DESIGN.md).  Confidence output unchanged (hit_t0_conf = pcu_vote10).
#
# Usage:   bash run_v12.sh <FLOW.hdf5 ...> <out_dir> [n_workers] [n_gpus]
set -uo pipefail
export PYTHONUSERBASE=/global/homes/y/yuxuan/.local/perlmutter/pytorch2.8.0; unset PYTHONPATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/global/common/software/nersc9/pytorch/2.8.0/bin/python
FILES="$1"; OUT="$2"; NW="${3:-8}"; NG="${4:-4}"
SMALL=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/NewMLSection/runs/ndfull_small4x/best_model_arch.pt
FRONTEND=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/clmatching-frontend

V12="--backbone stack2 --merge-p2 --sigma-mode poisson --max-clip-ticks 4 \
--police locked --merge-vertex --skip-baseline --phase3 joint --pass2-trade \
--pass2-light-nom --force-coverage --swap-fix --swap-ratio-max 0.85 \
--p2-threshold 20 --predict-min-energy 0.5 --predict-min-energy-shower 1.5 \
--skip-v2 --gate-support --force-simple --isolate-cm 25 --fast-scan --edge-penalty \
--pool-mode light --redecide --redecide-emin 20 \
--precision --precision-rounds 3 --precision-calib $HERE/testing/v120_calib_lowi.json \
--subtick --conf-cluster --light-ckpt $SMALL --frontend-repo $FRONTEND"

mkdir -p "$OUT"
PIDS=()
for w in $(seq 0 $((NW - 1))); do
  CUDA_VISIBLE_DEVICES=$((w % NG)) $PY "$HERE/testing/overflow_rescue_pipeline.py" \
    --files $FILES --out "$OUT/groups_worker${w}.jsonl" $V12 \
    --event-stride $NW --event-offset $w > "$OUT/worker${w}.log" 2>&1 &
  PIDS+=("$!")
done
FAIL=0
for p in "${PIDS[@]}"; do wait "$p" || FAIL=$((FAIL + 1)); done
ROWS=$(cat "$OUT"/groups_worker*.jsonl 2>/dev/null | wc -l)
echo "=== done failures=$FAIL rows=$ROWS ==="
echo "ALL_CONFIGS_DONE"
