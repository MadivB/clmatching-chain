#!/bin/bash
set -uo pipefail
HERE=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing
PY=/global/common/software/nersc9/pytorch/2.8.0/bin/python
DATA=/global/cfs/cdirs/dunepro/people/abooth/nd-production/output/MiniProdN5/run-ndlar-flow/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift/FLOW/0000000
FILES="$DATA/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.0000001.FLOW.hdf5 $DATA/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.0000002.FLOW.hdf5 $DATA/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.0000003.FLOW.hdf5"
COMMON="--backbone stack2 --sigma-mode poisson --max-clip-ticks 4 --police locked --merge-vertex --skip-baseline --phase3 both --force-coverage --swap-fix"
STR="--event-stride 5 --event-offset 0"
NG=$(nvidia-smi -L 2>/dev/null | wc -l)
mkdir -p "$HERE/abtest_base_out" "$HERE/abtest_unit_out"
echo "A/B start $(date +%H:%M:%S) NGPU=$NG (baseline vs unit-tiebreak, stride 5)"
CUDA_VISIBLE_DEVICES=0 $PY "$HERE/overflow_rescue_pipeline.py" --files $FILES \
  --out "$HERE/abtest_base_out/g.jsonl" $COMMON $STR > "$HERE/abtest_base_out/log" 2>&1 &
P1=$!
GPU2=$([ "$NG" -ge 2 ] && echo 1 || echo 0)
CLM_UNIT_TIEBREAK=1 CUDA_VISIBLE_DEVICES=$GPU2 $PY "$HERE/overflow_rescue_pipeline.py" --files $FILES \
  --out "$HERE/abtest_unit_out/g.jsonl" $COMMON $STR > "$HERE/abtest_unit_out/log" 2>&1 &
P2=$!
wait $P1; R1=$?; wait $P2; R2=$?
echo "A/B done $(date +%H:%M:%S) base_rc=$R1 unit_rc=$R2 base_rows=$(wc -l <$HERE/abtest_base_out/g.jsonl 2>/dev/null) unit_rows=$(wc -l <$HERE/abtest_unit_out/g.jsonl 2>/dev/null)"
echo ABDONE
