#!/bin/bash
# Generic v1.1 arm runner (files 0000001-3, 8 workers x 4 GPUs, dumps on):
#   bash run_v11arm.sh <ARM_NAME> <extra flags...>
# Output: /pscratch/sd/y/yuxuan/v11arms/<ARM_NAME>
set -uo pipefail
export PYTHONUSERBASE=/global/homes/y/yuxuan/.local/perlmutter/pytorch2.8.0; unset PYTHONPATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
HERE=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing
PY=/global/common/software/nersc9/pytorch/2.8.0/bin/python
DATA=/global/cfs/cdirs/dunepro/people/abooth/nd-production/output/MiniProdN5/run-ndlar-flow/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift/FLOW/0000000
FILES="$DATA/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.0000001.FLOW.hdf5 $DATA/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.0000002.FLOW.hdf5 $DATA/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.0000003.FLOW.hdf5"
SMALL=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/NewMLSection/runs/ndfull_small4x/best_model_arch.pt
FRONTEND=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/clmatching-frontend
V1="--backbone stack2 --merge-p2 --sigma-mode poisson --max-clip-ticks 4 \
--police locked --merge-vertex --skip-baseline --phase3 joint --pass2-trade \
--pass2-light-nom --force-coverage --swap-fix --swap-ratio-max 0.85 \
--predict-min-energy 1.5 --skip-v2 --gate-support \
--force-simple --isolate-cm 25 --fast-scan --edge-penalty --subtick --conf-cluster \
--light-ckpt $SMALL --frontend-repo $FRONTEND"
NAME=$1; shift
OUT=/pscratch/sd/y/yuxuan/v11arms/$NAME; mkdir -p "$OUT/full_dumps"
cd "$HERE"
echo "=== ARM $NAME start $(date) flags: $* ==="
PIDS=()
for w in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$((w % 4)) $PY "$HERE/overflow_rescue_pipeline.py" \
    --files $FILES --out "$OUT/groups_worker${w}.jsonl" $V1 "$@" \
    --dump-dir "$OUT/full_dumps" --event-stride 8 --event-offset $w > "$OUT/worker${w}.log" 2>&1 &
  PIDS+=("$!")
done
FAIL=0
for pid in "${PIDS[@]}"; do wait $pid || FAIL=$((FAIL+1)); done
echo "=== ARM $NAME done $(date) failures=$FAIL rows=$(cat $OUT/groups_worker*.jsonl 2>/dev/null | wc -l) ==="
