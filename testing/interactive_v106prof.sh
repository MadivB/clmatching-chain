#!/bin/bash
set -uo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
HERE=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing
PY=/global/common/software/nersc9/pytorch/2.8.0/bin/python
DATA=/global/cfs/cdirs/dunepro/people/abooth/nd-production/output/MiniProdN5/run-ndlar-flow/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift/FLOW/0000000
FILES="$DATA/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.0000001.FLOW.hdf5"
SMALL=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/NewMLSection/runs/ndfull_small4x/best_model_arch.pt
OUT=$HERE/v106PROF_out; mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=0 $PY -m cProfile -o "$OUT/ev0.prof" "$HERE/overflow_rescue_pipeline.py" \
  --files $FILES --out "$OUT/groups.jsonl" \
  --backbone stack2 --merge-p2 --sigma-mode poisson --max-clip-ticks 4 --police locked \
  --merge-vertex --skip-baseline --phase3 joint --pass2-trade --pass2-light-nom --force-coverage --swap-fix \
  --swap-ratio-max 0.85 --p2-threshold 30 --predict-min-energy 1.5 --light-ckpt $SMALL --skip-v2 \
  --frontend-repo /global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/clmatching-frontend \
  --gate-support --le-retime --event-stride 13 --event-offset 4 > "$OUT/worker.log" 2>&1
echo PROF_DONE
