#!/bin/bash
# v1.2 first in-chain run (via salloc -q interactive ... srun -n 1 -c 32 --gpus 4 bash this_script):
#   P1   v1.1 (K1 flags) + --precision (calib v120_calib_dev13.json, 3 rounds) on files 1-3
#   LOWI v1.1 chain WITH dumps on 3 low-intensity files (the user's calibration source) for v120_calibrate.py
set -uo pipefail
export PYTHONUSERBASE=/global/homes/y/yuxuan/.local/perlmutter/pytorch2.8.0; unset PYTHONPATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
HERE=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing
R=$HERE/run_v11arm.sh
K1="--p2-threshold 20 --pool-mode light --predict-min-energy 0.5 --predict-min-energy-shower 1.5 --redecide --redecide-emin 20"
bash $R P1 $K1 --precision --precision-rounds 3
# ---- low-intensity dumps (v1.1 flags, no precision) ----
PY=/global/common/software/nersc9/pytorch/2.8.0/bin/python
LD=/pscratch/sd/d/dunepro/yuxuan/output/MiniProdN5/run-ndlar-flow/MiniProdN5p1_NDComplex_FHC.flow.full.lowintensity.sanddrift/FLOW/0001000
LFILES="$LD/MiniProdN5p1_NDComplex_FHC.flow.full.lowintensity.sanddrift.0001000.FLOW.hdf5 $LD/MiniProdN5p1_NDComplex_FHC.flow.full.lowintensity.sanddrift.0001001.FLOW.hdf5 $LD/MiniProdN5p1_NDComplex_FHC.flow.full.lowintensity.sanddrift.0001002.FLOW.hdf5"
SMALL=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/NewMLSection/runs/ndfull_small4x/best_model_arch.pt
FRONTEND=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/clmatching-frontend
V1="--backbone stack2 --merge-p2 --sigma-mode poisson --max-clip-ticks 4 --police locked --merge-vertex --skip-baseline --phase3 joint --pass2-trade --pass2-light-nom --force-coverage --swap-fix --swap-ratio-max 0.85 --skip-v2 --gate-support --force-simple --isolate-cm 25 --fast-scan --edge-penalty --subtick --conf-cluster --light-ckpt $SMALL --frontend-repo $FRONTEND"
OUT=/pscratch/sd/y/yuxuan/v11arms/LOWI; mkdir -p "$OUT/full_dumps"; cd "$HERE"
echo "=== ARM LOWI start $(date) ==="
PIDS=()
for w in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$((w % 4)) $PY "$HERE/overflow_rescue_pipeline.py" --files $LFILES --out "$OUT/groups_worker${w}.jsonl" $V1 $K1 \
    --dump-dir "$OUT/full_dumps" --event-stride 8 --event-offset $w > "$OUT/worker${w}.log" 2>&1 &
  PIDS+=("$!")
done
FAIL=0; for pid in "${PIDS[@]}"; do wait $pid || FAIL=$((FAIL+1)); done
echo "=== ARM LOWI done $(date) failures=$FAIL rows=$(cat $OUT/groups_worker*.jsonl 2>/dev/null | wc -l) ==="
echo ALL_ARMS_DONE
