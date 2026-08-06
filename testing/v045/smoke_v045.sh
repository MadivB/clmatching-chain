#!/bin/bash
#SBATCH -A dune
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -J v045smoke
#SBATCH -o /global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing/v045/smoke_%j.log
set -uo pipefail
HERE=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing
PY=/global/common/software/nersc/pe/conda-envs/26.1.0/python-3.13/nersc-python/bin/python
D=/global/cfs/cdirs/dunepro/people/abooth/nd-production/output/MiniProdN5/run-ndlar-flow/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift/FLOW/0000000
FILES="$D/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.0000003.FLOW.hdf5"
COMMON="--backbone stack2 --sigma-mode poisson --max-clip-ticks 4 --police locked --merge-vertex --skip-baseline --phase3 both --force-coverage --swap-fix --swap-ratio-max 0.85 --p2-threshold 30 --predict-min-energy 1.5"
OUT=$HERE/v045/smoke_out; mkdir -p "$OUT"
echo "=== C: stage2-expand + 2x track absorb ==="
$PY "$HERE/v045/v045_pipeline.py" --files $FILES --out "$OUT/C.jsonl" $COMMON \
  --track-absorb-scale 2.0 --stage2-expand --event-stride 20 --event-offset 6 2>&1 | tail -25
