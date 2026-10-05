#!/bin/bash
# GPU: stage 4.86 (unexplained flashes at track ends) on 13 dev spills: flash-only, and after the piece-first kernel (v1.5 config)
set -uo pipefail
export PYTHONUSERBASE=/global/homes/y/yuxuan/.local/perlmutter/pytorch2.8.0; unset PYTHONPATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=4
PY=/global/common/software/nersc9/pytorch/2.8.0/bin/python
HERE=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing; O=$HERE/michel_nd_out
K="host_extent_min_cm=20 bragg_end_cm=20 piece_max_mev=60 tip_face_min_cm=5 band_dedx_max=5 max_frac=1.0 remainder_min_mev=10 mf_min=0.75 apc_max=-0.25 whole_mf_min=0.8 whole_apc_max=-0.5"
$PY $HERE/v14_michel_nd_eval.py /pscratch/sd/y/yuxuan/v11arms/V14A --events 13 --batch 16 --flash-only --out $O/flash_13_only.json > $O/flash_13_only.log 2>&1
$PY $HERE/v14_michel_nd_eval.py /pscratch/sd/y/yuxuan/v11arms/V14A --events 13 --batch 16 --set $K --flash --out $O/flash_13_after.json > $O/flash_13_after.log 2>&1
echo FLASHDEV_DONE >> $O/flash_13_after.log
