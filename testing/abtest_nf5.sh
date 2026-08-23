#!/bin/bash
set -uo pipefail
HERE=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing
PY=/global/common/software/nersc9/pytorch/2.8.0/bin/python
DATA=/global/cfs/cdirs/dunepro/people/abooth/nd-production/output/MiniProdN5/run-ndlar-flow/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift/FLOW/0000000
FILES="$DATA/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.0000001.FLOW.hdf5 $DATA/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.0000002.FLOW.hdf5 $DATA/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.0000003.FLOW.hdf5"
COMMON="--backbone stack2 --sigma-mode poisson --max-clip-ticks 4 --police locked --merge-vertex --skip-baseline --phase3 both --force-coverage --swap-fix"
STR="--event-stride 8 --event-offset 0"
export CLM_IMAGE_BATCH=2 PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
mkdir -p "$HERE/abtest_nf5_out"
echo "v3-gated RESC start $(date +%H:%M:%S)"
$PY "$HERE/overflow_rescue_pipeline.py" --files $FILES --out "$HERE/abtest_nf5_out/g.jsonl" $COMMON $STR --new-flash-rescue > "$HERE/abtest_nf5_out/log" 2>&1
echo "RESC5 done $(date +%H:%M:%S) rows=$(wc -l <$HERE/abtest_nf5_out/g.jsonl 2>/dev/null)"
echo NF5DONE
