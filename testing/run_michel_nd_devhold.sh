#!/bin/bash
# GPU: ND stage-M apply pass on dev (files 1-3) and holdout (files 4-5) with gate overrides from $GATES
set -uo pipefail
export PYTHONUSERBASE=/global/homes/y/yuxuan/.local/perlmutter/pytorch2.8.0; unset PYTHONPATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=4
PY=/global/common/software/nersc9/pytorch/2.8.0/bin/python
HERE=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing
TAG=${TAG:-g1}
$PY $HERE/v14_michel_nd.py /pscratch/sd/y/yuxuan/v11arms/V14A --files 0000001,0000002,0000003 --apply ${GATES:-} --out $HERE/michel_nd_out/dev_${TAG}.json > $HERE/michel_nd_out/dev_${TAG}.log 2>&1
$PY $HERE/v14_michel_nd.py /pscratch/sd/y/yuxuan/v11arms/V14A --files 0000004,0000005 --apply ${GATES:-} --out $HERE/michel_nd_out/hold_${TAG}.json > $HERE/michel_nd_out/hold_${TAG}.log 2>&1
echo MICHEL_ND_DEVHOLD_DONE >> $HERE/michel_nd_out/hold_${TAG}.log
