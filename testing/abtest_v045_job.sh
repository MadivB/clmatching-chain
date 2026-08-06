#!/bin/bash
#SBATCH -A dune
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -J v045ab
#SBATCH -o /global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing/v045ab_%j.log
# v0.4.5 experiment A/B: 4 arms in parallel (one GPU each), 5 events each (stride 8).
#   BASE = canonical v0.4.1 | A = --track-absorb-scale 2.0 | B = --early-expand | AB = both
set -uo pipefail
HERE=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing
PY=/global/common/software/nersc/pe/conda-envs/26.1.0/python-3.13/nersc-python/bin/python
DATA=/global/cfs/cdirs/dunepro/people/abooth/nd-production/output/MiniProdN5/run-ndlar-flow/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift/FLOW/0000000
FILES="$DATA/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.0000001.FLOW.hdf5 \
$DATA/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.0000002.FLOW.hdf5 \
$DATA/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.0000003.FLOW.hdf5"
V4="--backbone stack2 --sigma-mode poisson --max-clip-ticks 4 --police locked --merge-vertex \
--skip-baseline --phase3 both --force-coverage --swap-fix --swap-ratio-max 0.85 \
--p2-threshold 30 --predict-min-energy 1.5 --merge-p2"
STR="--event-stride 8 --event-offset 0"
declare -A ARMS=(
  [BASE]=""
  [A]="--track-absorb-scale 2.0"
  [B]="--early-expand"
  [AB]="--track-absorb-scale 2.0 --early-expand"
)
echo "v0.4.5 A/B start $(date +%H:%M:%S)"
PIDS=(); g=0
for arm in BASE A B AB; do
  OUT=$HERE/v045test_${arm}_out; mkdir -p "$OUT"
  CUDA_VISIBLE_DEVICES=$g $PY "$HERE/overflow_rescue_pipeline.py" --files $FILES \
    --out "$OUT/g.jsonl" $V4 ${ARMS[$arm]} $STR > "$OUT/log" 2>&1 &
  PIDS+=("$!"); g=$((g+1))
done
F=0; for pid in "${PIDS[@]}"; do wait "$pid" || F=$((F+1)); done
for arm in BASE A B AB; do
  echo "$arm rows=$(wc -l <$HERE/v045test_${arm}_out/g.jsonl 2>/dev/null)"
done
echo "v0.4.5 A/B done fail=$F $(date +%H:%M:%S)"
