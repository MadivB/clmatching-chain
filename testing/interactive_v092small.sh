#!/bin/bash
# small amplitude perceiver A/B: v0.8.5 (canonical v0.8.1 + small) and
# v0.6.2 (v0.6 flags + small), file 0000001
set -uo pipefail
HERE=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing
PY=/global/common/software/nersc9/pytorch/2.8.0/bin/python
DATA=/global/cfs/cdirs/dunepro/people/abooth/nd-production/output/MiniProdN5/run-ndlar-flow/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift/FLOW/0000000
FILES="$DATA/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.0000001.FLOW.hdf5"
SMALL=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/NewMLSection/runs/ndfull_small4x/best_model_arch.pt
FRONT="--frontend-repo /global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/clmatching-frontend"
V81="--backbone stack2 --merge-p2 --sigma-mode poisson --max-clip-ticks 4 --police locked \
--merge-vertex --skip-baseline --phase3 joint --pass2-trade --pass2-light-nom --force-coverage --swap-fix \
--swap-ratio-max 0.85 --p2-threshold 30 --predict-min-energy 1.5 $FRONT"
V60="--backbone stack2 --merge-p2 --sigma-mode poisson --max-clip-ticks 4 --police locked \
--merge-vertex --skip-baseline --phase3 both --force-coverage --swap-fix \
--swap-ratio-max 0.85 --p2-threshold 30 --predict-min-energy 1.5 $FRONT"
run_cfg () {
  tag=$1; flags=$2
  OUT=$HERE/v092${tag}_out; mkdir -p "$OUT"
  echo "=== config $tag start $(date +%H:%M:%S) ==="
  PIDS=()
  for w in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=$((w % 4)) $PY "$HERE/overflow_rescue_pipeline.py" \
      --files $FILES --out "$OUT/groups_worker${w}.jsonl" $flags \
      --light-ckpt $SMALL --event-stride 8 --event-offset $w > "$OUT/worker${w}.log" 2>&1 &
    PIDS+=("$!")
  done
  FAIL=0
  for pid in "${PIDS[@]}"; do wait $pid || FAIL=$((FAIL+1)); done
  echo "=== config $tag done failures=$FAIL rows=$(cat $OUT/groups_worker*.jsonl 2>/dev/null | wc -l) $(date +%H:%M:%S) ==="
}
run_cfg V85 "$V81"
run_cfg V62 "$V60"
echo ALL_CONFIGS_DONE
