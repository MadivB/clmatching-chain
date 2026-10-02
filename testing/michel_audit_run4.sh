#!/bin/bash
set -uo pipefail
export PYTHONUSERBASE=/global/homes/y/yuxuan/.local/perlmutter/pytorch2.8.0; unset PYTHONPATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=4
PY=/global/common/software/nersc9/pytorch/2.8.0/bin/python
HERE=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing; O=$HERE/michel_nd_out
# audit v4: piece vs whole records; no dE/dx cap, face >= 5
$PY $HERE/v15_michel_audit.py /pscratch/sd/y/yuxuan/v11arms/V14A --events 20 --set host_extent_min_cm=20 bragg_end_cm=20 piece_max_mev=60 tip_face_min_cm=5 --out $O/michel_audit4_20.json > $O/michel_audit4_20.log 2>&1
echo AUDIT4_DONE >> $O/michel_audit4_20.log
