#!/bin/bash
set -uo pipefail
export PYTHONUSERBASE=/global/homes/y/yuxuan/.local/perlmutter/pytorch2.8.0; unset PYTHONPATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=4
PY=/global/common/software/nersc9/pytorch/2.8.0/bin/python
HERE=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing; O=$HERE/michel_nd_out
# audit v3: candidate status instrumentation; certified gates + face >= 5, NO dE/dx cap (cap evaluated offline from host_dedx)
$PY $HERE/v15_michel_audit.py /pscratch/sd/y/yuxuan/v11arms/V14A --events 20 --set host_extent_min_cm=20 bragg_end_cm=20 piece_max_mev=60 tip_face_min_cm=5 --out $O/michel_audit3_20.json > $O/michel_audit3_20.log 2>&1
echo AUDIT3_DONE >> $O/michel_audit3_20.log
