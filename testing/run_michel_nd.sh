#!/bin/bash
# GPU: label the ND stage-M hypotheses on the first N certification dumps (no moves), then an --apply pass
set -uo pipefail
export PYTHONUSERBASE=/global/homes/y/yuxuan/.local/perlmutter/pytorch2.8.0; unset PYTHONPATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=4
PY=/global/common/software/nersc9/pytorch/2.8.0/bin/python
HERE=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing
N=${N:-13}
$PY $HERE/v14_michel_nd.py /pscratch/sd/y/yuxuan/v11arms/V14A --events $N --out $HERE/michel_nd_out/label_${N}.json > $HERE/michel_nd_out/label_${N}.log 2>&1
$PY $HERE/v14_michel_nd.py /pscratch/sd/y/yuxuan/v11arms/V14A --events $N --apply --out $HERE/michel_nd_out/apply_${N}.json > $HERE/michel_nd_out/apply_${N}.log 2>&1
echo MICHEL_ND_DONE >> $HERE/michel_nd_out/apply_${N}.log
