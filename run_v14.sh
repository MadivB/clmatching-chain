#!/bin/bash
# v1.4 (defined 2026-10-01, user decision) = v1.3 chain (run_v13.sh) + stage 4.9 light-vote force rule:
#   --force-light-cm 20   noise hits beyond 3 cm from any assigned hit take the cluster-density vote over the TPC's
#                         family times; beyond 20 cm the vote is multiplied by the positive light residual on the 6
#                         nearest channels (testing/force_light_vote.py).  Offline on the 63 cert spills: noise class
#                         58.6 -> 65.2 %, all-TPC +0.095 pp, 0.84 s/spill.  Budget for new late stages: < 3 s/spill.
#   --resid-fill          stage 4.8 residual-flash filling (testing/resid_fill.py): unexplained light peaks (summed
#                         residual, prominence 10 sigma, residual >= 0.5 x model at the peak = not a tail bump, not within
#                         5 ticks of a family) are filled, brightest first, by the templated cluster < 100 MeV of the TPC
#                         whose predicted pattern on its 90 % support channels matches the residual best (cos >= 0.9 and
#                         > cos at its current slot + 0.1, residual/template amplitude in [0.5, 3], >= 20 cm from any
#                         co-timed host).  Offline dev (files 1-3) +0.026 pp, holdout (4-5) +0.021 pp, 46/16 spills, 0.4 s/spill.
#   5-file certification (arm V14A vs V2A5, 63 spills): NS 96.77 -> 96.89 (+0.114, 63/0), all-TPC 95.76 -> 95.86 (+0.097, 62/1),
#   LE 60.89 -> 65.55; new stages 1.0 s/spill.  Tag CLMatching_v1.4.
# v1.3 canonical = v1.2.1 chain (run_v12.sh) + stage 3.99 exact-block re-solve of clusters >= 20 MeV
#   (--v200-min-e 20, testing/v200, V200_DESIGN.md).  5-file cert vs v1.2: all-TPC +0.22 pp (42/20), NS +0.08,
#   >= 100 MeV wrong 1.91 -> 1.68 %, 20-100 MeV 8.62 -> 8.28 %, < 20 MeV unchanged, +5-12 s/ev.
#   Also carries v1.2.2 (light saturation ceiling = int16 rail minus the dynamic baseline).
# v1.2.2: light saturation ceiling = int16 rail (2**15-1) minus the per-channel dynamic baseline,
#   computed per event in the first stage and used by every clip / veto (was the constant 60780).
# v1.2 canonical charge-light matching = v1.1 chain (run_v11.sh) + stage P per-hit precision timing:
#   --precision --precision-calib testing/v120_calib_lowi.json --precision-rounds 3 --precision-pieces
#   (v1.2.1: displaced pieces >= 5 MeV with their own channels get a free time; per-piece delay / distance /
#    beta / sigma and a per-family apex-time table are written to the event rows for time-of-flight pairing)
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
export V200_DOM=3

HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/global/common/software/nersc9/pytorch/2.8.0/bin/python
FILES="$1"; OUT="$2"; NW="${3:-8}"; NG="${4:-4}"
SMALL=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/NewMLSection/runs/ndfull_small4x/best_model_arch.pt
FRONTEND=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/clmatching-frontend

V14="--backbone stack2 --merge-p2 --sigma-mode poisson --max-clip-ticks 4 \
--police locked --merge-vertex --skip-baseline --phase3 joint --pass2-trade \
--pass2-light-nom --force-coverage --swap-fix --swap-ratio-max 0.85 \
--p2-threshold 20 --predict-min-energy 0.5 --predict-min-energy-shower 1.5 \
--skip-v2 --gate-support --force-simple --isolate-cm 25 --fast-scan --edge-penalty \
--pool-mode light --redecide --redecide-emin 20 \
--precision --precision-rounds 3 --precision-calib $HERE/testing/v120_calib_lowi.json --precision-pieces \
--v200-min-e 20 --force-light-cm 20 --resid-fill \
--subtick --conf-cluster --light-ckpt $SMALL --frontend-repo $FRONTEND"

mkdir -p "$OUT"
PIDS=()
for w in $(seq 0 $((NW - 1))); do
  CUDA_VISIBLE_DEVICES=$((w % NG)) $PY "$HERE/testing/overflow_rescue_pipeline.py" \
    --files $FILES --out "$OUT/groups_worker${w}.jsonl" $V14 \
    --event-stride $NW --event-offset $w > "$OUT/worker${w}.log" 2>&1 &
  PIDS+=("$!")
done
FAIL=0
for p in "${PIDS[@]}"; do wait "$p" || FAIL=$((FAIL + 1)); done
ROWS=$(cat "$OUT"/groups_worker*.jsonl 2>/dev/null | wc -l)
echo "=== done failures=$FAIL rows=$ROWS ==="
echo "ALL_CONFIGS_DONE"
