#!/bin/bash
# v1.2.2 saturation-ceiling regression: v1.2.1 chain (P2 flags) on file 0000001 only (13 events),
# dumps on, output /pscratch/sd/y/yuxuan/v11arms/SAT1.  Compare against the v1.2.1 arm P2 with
#   python testing/v11_ab.py /pscratch/sd/y/yuxuan/v11arms/P2 /pscratch/sd/y/yuxuan/v11arms/SAT1
# (paired per (file, event); only file-1 events pair).  Expected: identical assignments up to the
# 13-count shift of the veto threshold (60700 -> rail - 80).
# Launch: salloc -A dune_g -q interactive -C gpu -N 1 --gpus 4 -c 32 -t 01:00:00 \
#         srun -n 1 -c 32 --gpus 4 bash testing/sbatch_v122sat.sh
set -uo pipefail
export PYTHONUSERBASE=/global/homes/y/yuxuan/.local/perlmutter/pytorch2.8.0; unset PYTHONPATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE
HERE=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing
PY=/global/common/software/nersc9/pytorch/2.8.0/bin/python
DATA=/global/cfs/cdirs/dunepro/people/abooth/nd-production/output/MiniProdN5/run-ndlar-flow/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift/FLOW/0000000
FILES="$DATA/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.0000001.FLOW.hdf5"
SMALL=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/NewMLSection/runs/ndfull_small4x/best_model_arch.pt
FRONTEND=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/clmatching-frontend
V1="--backbone stack2 --merge-p2 --sigma-mode poisson --max-clip-ticks 4 \
--police locked --merge-vertex --skip-baseline --phase3 joint --pass2-trade \
--pass2-light-nom --force-coverage --swap-fix --swap-ratio-max 0.85 \
--predict-min-energy 1.5 --skip-v2 --gate-support \
--force-simple --isolate-cm 25 --fast-scan --edge-penalty --subtick --conf-cluster \
--light-ckpt $SMALL --frontend-repo $FRONTEND"
K1="--p2-threshold 20 --pool-mode light --predict-min-energy 0.5 --predict-min-energy-shower 1.5 --redecide --redecide-emin 20"
P2="--precision --precision-rounds 3 --precision-calib $HERE/v120_calib_lowi.json --precision-pieces"
NAME=${1:-SAT1}
OUT=/pscratch/sd/y/yuxuan/v11arms/$NAME; mkdir -p "$OUT/full_dumps"
cd "$HERE"
echo "=== ARM $NAME start $(date) ==="
PIDS=()
for w in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$((w % 4)) $PY "$HERE/overflow_rescue_pipeline.py" \
    --files $FILES --out "$OUT/groups_worker${w}.jsonl" $V1 $K1 $P2 \
    --dump-dir "$OUT/full_dumps" --event-stride 8 --event-offset $w > "$OUT/worker${w}.log" 2>&1 &
  PIDS+=("$!")
done
FAIL=0
for pid in "${PIDS[@]}"; do wait $pid || FAIL=$((FAIL+1)); done
echo "=== ARM $NAME done $(date) failures=$FAIL rows=$(cat $OUT/groups_worker*.jsonl 2>/dev/null | wc -l) ==="
echo ALL_CONFIGS_DONE
