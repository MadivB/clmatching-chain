#!/bin/bash
set -uo pipefail
HERE=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing
PY=/global/common/software/nersc9/pytorch/2.8.0/bin/python
DATA=/global/cfs/cdirs/dunepro/people/abooth/nd-production/output/MiniProdN5/run-ndlar-flow/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift/FLOW/0000000
FILES="$DATA/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.0000001.FLOW.hdf5 $DATA/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.0000002.FLOW.hdf5 $DATA/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.0000003.FLOW.hdf5"
COMMON="--backbone stack2 --sigma-mode poisson --max-clip-ticks 4 --police locked --merge-vertex --skip-baseline --phase3 both --force-coverage --swap-fix"
STR="--event-stride 8 --event-offset 0"
export PYTORCH_ALLOC_CONF=expandable_segments:True
echo "waiting for GPU (need >14GB free)... $(date +%H:%M:%S)"
for i in $(seq 1 180); do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [ "${FREE:-0}" -gt 14000 ]; then echo "GPU free=${FREE}MiB at $(date +%H:%M:%S) -> running"; break; fi
  sleep 60
done
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
if [ "${FREE:-0}" -le 14000 ]; then echo "GPU still busy (${FREE}MiB free) after wait -> ABORT"; echo NFWAITDONE; exit 0; fi
mkdir -p "$HERE/abtest_nfBASE_out" "$HERE/abtest_nfRESC_out"
CUDA_VISIBLE_DEVICES=0 $PY "$HERE/overflow_rescue_pipeline.py" --files $FILES --out "$HERE/abtest_nfBASE_out/g.jsonl" $COMMON $STR > "$HERE/abtest_nfBASE_out/log" 2>&1
echo "BASE done $(date +%H:%M:%S) rows=$(wc -l <$HERE/abtest_nfBASE_out/g.jsonl 2>/dev/null)"
CUDA_VISIBLE_DEVICES=0 $PY "$HERE/overflow_rescue_pipeline.py" --files $FILES --out "$HERE/abtest_nfRESC_out/g.jsonl" $COMMON $STR --new-flash-rescue > "$HERE/abtest_nfRESC_out/log" 2>&1
echo "RESC done $(date +%H:%M:%S) rows=$(wc -l <$HERE/abtest_nfRESC_out/g.jsonl 2>/dev/null)"
echo NFWAITDONE
