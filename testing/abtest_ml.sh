#!/bin/bash
set -uo pipefail
HERE=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing
PY=/global/common/software/nersc/pe/conda-envs/26.1.0/python-3.13/nersc-python/bin/python
DATA=/global/cfs/cdirs/dunepro/people/abooth/nd-production/output/MiniProdN5/run-ndlar-flow/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift/FLOW/0000000
FILES="$DATA/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.0000001.FLOW.hdf5 $DATA/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.0000002.FLOW.hdf5 $DATA/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.0000003.FLOW.hdf5"
COMMON="--backbone stack2 --sigma-mode model --z-cut 8 --max-clip-ticks 4 --police locked --merge-vertex --skip-baseline --phase3 both --force-coverage --swap-fix"
STR="--event-stride 8 --event-offset 0"
mkdir -p "$HERE/abtest_mlA_out" "$HERE/abtest_mlB_out"
echo "ML A/B start $(date +%H:%M:%S) (A=model no-tiebreak, B=model + ML-tiebreak; stride 8)"
CUDA_VISIBLE_DEVICES=0 $PY "$HERE/overflow_rescue_pipeline.py" --files $FILES \
  --out "$HERE/abtest_mlA_out/g.jsonl" $COMMON $STR > "$HERE/abtest_mlA_out/log" 2>&1 &
PA=$!
CLM_UNIT_TIEBREAK=1 CUDA_VISIBLE_DEVICES=0 $PY "$HERE/overflow_rescue_pipeline.py" --files $FILES \
  --out "$HERE/abtest_mlB_out/g.jsonl" $COMMON $STR > "$HERE/abtest_mlB_out/log" 2>&1 &
PB=$!
wait $PA; RA=$?; wait $PB; RB=$?
echo "ML A/B done $(date +%H:%M:%S) A_rc=$RA B_rc=$RB A_rows=$(wc -l <$HERE/abtest_mlA_out/g.jsonl 2>/dev/null) B_rows=$(wc -l <$HERE/abtest_mlB_out/g.jsonl 2>/dev/null)"
echo MLABDONE
