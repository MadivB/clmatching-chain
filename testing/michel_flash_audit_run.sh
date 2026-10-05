#!/bin/bash
set -uo pipefail
export PYTHONUSERBASE=/global/homes/y/yuxuan/.local/perlmutter/pytorch2.8.0; unset PYTHONPATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=4
PY=/global/common/software/nersc9/pytorch/2.8.0/bin/python
HERE=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing; O=$HERE/michel_nd_out
K="host_extent_min_cm=20 bragg_end_cm=20 piece_max_mev=60 tip_face_min_cm=5 band_dedx_max=5 max_frac=1.0 remainder_min_mev=10 mf_min=0.75 apc_max=-0.25 whole_mf_min=0.8 whole_apc_max=-0.5"
# per-Michel recall chain of the flash stage run after the v1.5 piece-first kernel, 20 spills
$PY $HERE/v15_michel_audit.py /pscratch/sd/y/yuxuan/v11arms/V14A --events 20 --set $K --flash --out $O/flash3_audit_20.json > $O/flash3_audit_20.log 2>&1
echo FAUDIT3_DONE >> $O/flash3_audit_20.log
