#!/bin/bash
# GPU: piece-first ND Michel stage, 13 spills with and without the Bragg requirement (apply mode), then hold for GO_cmd
set -uo pipefail
export PYTHONUSERBASE=/global/homes/y/yuxuan/.local/perlmutter/pytorch2.8.0; unset PYTHONPATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=4
PY=/global/common/software/nersc9/pytorch/2.8.0/bin/python
HERE=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing; O=$HERE/michel_nd_out
$PY $HERE/v14_michel_nd_fast.py /pscratch/sd/y/yuxuan/v11arms/V14A --events 13 --apply --out $O/fast_13_bragg.json > $O/fast_13_bragg.log 2>&1
$PY $HERE/v14_michel_nd_fast.py /pscratch/sd/y/yuxuan/v11arms/V14A --events 13 --apply --no-bragg --out $O/fast_13_nobragg.json > $O/fast_13_nobragg.log 2>&1
echo FAST13_DONE >> $O/fast_13_nobragg.log
for i in $(seq 1 540); do
  if [ -f $O/GO_cmd ]; then mv $O/GO_cmd $O/GO_cmd.running; bash $O/GO_cmd.running > $O/GO_cmd.out 2>&1; echo GO_DONE >> $O/GO_cmd.out; mv $O/GO_cmd.running $O/GO_cmd.done.$(date +%s); fi
  sleep 10
done
