#!/bin/bash
set -uo pipefail
export PYTHONUSERBASE=/global/homes/y/yuxuan/.local/perlmutter/pytorch2.8.0; unset PYTHONPATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=4
PY=/global/common/software/nersc9/pytorch/2.8.0/bin/python
HERE=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing; O=$HERE/michel_nd_out
# audit v5 on all 63 spills: every scored hypothesis saved with truth + geometry (offline gate scans); loose geometry
# (extent 10, no dE/dx cap, face 5, 3 minima) so that tighter gates can be emulated offline
$PY $HERE/v15_michel_audit.py /pscratch/sd/y/yuxuan/v11arms/V14A --files 0000001,0000002,0000003,0000004,0000005 --set host_extent_min_cm=10 bragg_end_cm=20 piece_max_mev=60 tip_face_min_cm=5 n_minima=3 --out $O/michel_audit5_63.json > $O/michel_audit5_63.log 2>&1
echo AUDIT5_DONE >> $O/michel_audit5_63.log
