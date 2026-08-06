#!/bin/bash
# v0.6 debug session: A = instrumented baseline (must reproduce 96.02),
# B = + shower retype (>=3 tracks), then walkthrough smoke on the failure subject.
set -uo pipefail
HERE=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing
PY=/global/common/software/nersc/pe/conda-envs/26.1.0/python-3.13/nersc-python/bin/python
DATA=/global/cfs/cdirs/dunepro/people/abooth/nd-production/output/MiniProdN5/run-ndlar-flow/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift/FLOW/0000000
FILES="$DATA/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.0000001.FLOW.hdf5"
V41="--backbone stack2 --merge-p2 --sigma-mode poisson --max-clip-ticks 4 --police locked \
--merge-vertex --skip-baseline --phase3 both --force-coverage --swap-fix \
--swap-ratio-max 0.85 --p2-threshold 30 --predict-min-energy 1.5 \
--frontend-repo /global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/clmatching-frontend"
run_cfg () {
  tag=$1; extra=$2
  OUT=$HERE/v061${tag}_out; mkdir -p "$OUT"
  echo "=== config $tag ($extra) start $(date +%H:%M:%S) ==="
  PIDS=()
  for w in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=$((w % 4)) $PY "$HERE/overflow_rescue_pipeline.py" \
      --files $FILES --out "$OUT/groups_worker${w}.jsonl" $V41 $extra \
      --event-stride 8 --event-offset $w > "$OUT/worker${w}.log" 2>&1 &
    PIDS+=("$!")
  done
  FAIL=0
  for pid in "${PIDS[@]}"; do wait "$pid" || FAIL=$((FAIL+1)); done
  echo "=== config $tag done failures=$FAIL rows=$(cat $OUT/groups_worker*.jsonl 2>/dev/null | wc -l) $(date +%H:%M:%S) ==="
}
run_cfg A ""
run_cfg B "--frontend-shower-min-tracks 3"
echo "=== walkthrough smoke (failure subject ev12 TPC38) ==="
module load pytorch/2.8.0 2>/dev/null || true
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HDF5_USE_FILE_LOCKING=FALSE
export WT_FRONTEND_REPO=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/clmatching-frontend
export WT_SUBJECT=0000001,12,38
cd $HERE && python -u smoke_walkthrough.py 2>&1 | tail -30
echo ALL_CONFIGS_DONE
