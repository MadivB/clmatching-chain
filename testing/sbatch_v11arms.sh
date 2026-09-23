#!/bin/bash
#SBATCH -A dune_g
#SBATCH -q interactive
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH --gpus 4
#SBATCH -c 32
#SBATCH -t 03:30:00
#SBATCH -J v11arms
#SBATCH -o /pscratch/sd/y/yuxuan/v11arms/sbatch.log
# v1.1 development arms on files 0000001-3 (39 events), one node, arms back-to-back:
#   A  v1.0 canonical rerun (same-node timing reference; accuracy ref = v138FS_out)
#   B  --pool-mode light                       (component batching + combination search)
#   C  --pool-mode light --pool-gain 0.25      (+ amplitude profiling)
#   D  --pool-mode light --pool-gain 0.25 --p2-threshold 10   (+ more clusters into the pool)
#   E  --pool-gain 0.25 (energy batching)      (gain alone)
set -uo pipefail
export PYTHONUSERBASE=/global/homes/y/yuxuan/.local/perlmutter/pytorch2.8.0; unset PYTHONPATH   # = what `module load pytorch/2.8.0` sets (plotly lives there)
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
ROOT=/pscratch/sd/y/yuxuan/v11arms
cd "$HERE"

run_arm () {
  local NAME=$1; shift
  local OUT=$ROOT/$NAME; mkdir -p "$OUT/full_dumps"
  echo "=== ARM $NAME start $(date) flags: $* ==="
  local PIDS=()
  for w in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=$((w % 4)) $PY "$HERE/overflow_rescue_pipeline.py" \
      --files $FILES --out "$OUT/groups_worker${w}.jsonl" $V1 "$@" \
      --dump-dir "$OUT/full_dumps" --event-stride 8 --event-offset $w > "$OUT/worker${w}.log" 2>&1 &
    PIDS+=("$!")
  done
  local FAIL=0
  for pid in "${PIDS[@]}"; do wait $pid || FAIL=$((FAIL+1)); done
  echo "=== ARM $NAME done $(date) failures=$FAIL rows=$(cat $OUT/groups_worker*.jsonl 2>/dev/null | wc -l) ==="
}

# smoke: arm C on ONE event (worker 0 of stride 39 -> event index 0 of file 1), fail fast
SM=$ROOT/smoke; mkdir -p "$SM/full_dumps"
echo "=== SMOKE start $(date) ==="
CUDA_VISIBLE_DEVICES=0 $PY "$HERE/overflow_rescue_pipeline.py" --files $DATA/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.0000001.FLOW.hdf5 \
  --out "$SM/groups_worker0.jsonl" $V1 --p2-threshold 30 --pool-mode light --pool-gain 0.25 \
  --dump-dir "$SM/full_dumps" --event-stride 13 --event-offset 1 > "$SM/worker0.log" 2>&1
RC=$?
echo "=== SMOKE rc=$RC rows=$(cat $SM/groups_worker0.jsonl 2>/dev/null | wc -l) $(date) ==="
if [ "$RC" -ne 0 ] || [ "$(cat $SM/groups_worker0.jsonl 2>/dev/null | wc -l)" -lt 1 ]; then
  echo "SMOKE FAILED - aborting arms"; tail -40 "$SM/worker0.log"; exit 1
fi

run_arm B --p2-threshold 30 --pool-mode light
run_arm C --p2-threshold 30 --pool-mode light --pool-gain 0.25
run_arm D --p2-threshold 10 --pool-mode light --pool-gain 0.25
run_arm E --p2-threshold 30 --pool-gain 0.25
run_arm A --p2-threshold 30
echo ALL_ARMS_DONE
