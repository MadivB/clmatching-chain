#!/bin/bash
set -uo pipefail
export PYTHONUSERBASE=/global/homes/y/yuxuan/.local/perlmutter/pytorch2.8.0; unset PYTHONPATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=4
PY=/global/common/software/nersc9/pytorch/2.8.0/bin/python
HERE=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing/v200
$PY $HERE/v200_varcache.py /pscratch/sd/y/yuxuan/v11arms/P2 --out /pscratch/sd/y/yuxuan/v11arms/P2/var_cache
echo VARCACHE_DONE
