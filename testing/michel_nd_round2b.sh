#!/bin/bash
# round 2b: label 13 spills (family-time candidates, unexplained >= 5000 ADC and >= 20 %), then hold the GPU and
# execute whatever bash snippet appears in michel_nd_out/GO_cmd (gate-scan decision), up to 2 h
set -uo pipefail
export PYTHONUSERBASE=/global/homes/y/yuxuan/.local/perlmutter/pytorch2.8.0; unset PYTHONPATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=4
PY=/global/common/software/nersc9/pytorch/2.8.0/bin/python
HERE=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing
O=$HERE/michel_nd_out
$PY $HERE/v14_michel_nd.py /pscratch/sd/y/yuxuan/v11arms/V14A --events 13 --unexpl-frac 0.1 --out $O/label2_13.json > $O/label2_13.log 2>&1
echo LABEL2_DONE >> $O/label2_13.log
for i in $(seq 1 720); do
  if [ -f $O/GO_cmd ]; then
    mv $O/GO_cmd $O/GO_cmd.running
    bash $O/GO_cmd.running > $O/GO_cmd.out 2>&1
    echo GO_DONE >> $O/GO_cmd.out
    mv $O/GO_cmd.running $O/GO_cmd.done.$(date +%s)
  fi
  sleep 10
done
