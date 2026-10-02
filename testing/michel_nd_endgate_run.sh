#!/bin/bash
# GPU: kernel on 13 spills with the ending gates (host dE/dx <= 5 MeV/cm, tips >= 5 cm from an outer face) vs without
set -uo pipefail
export PYTHONUSERBASE=/global/homes/y/yuxuan/.local/perlmutter/pytorch2.8.0; unset PYTHONPATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=4
PY=/global/common/software/nersc9/pytorch/2.8.0/bin/python
HERE=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing; O=$HERE/michel_nd_out
G="host_extent_min_cm=20 bragg_end_cm=20 piece_max_mev=60"
$PY $HERE/v14_michel_nd_eval.py /pscratch/sd/y/yuxuan/v11arms/V14A --events 13 --set $G host_dedx_max=5 tip_face_min_cm=5 --out $O/kern_13_endgate.json > $O/kern_13_endgate.log 2>&1
$PY $HERE/v14_michel_nd_eval.py /pscratch/sd/y/yuxuan/v11arms/V14A --events 13 --set $G host_dedx_max=4 tip_face_min_cm=5 --out $O/kern_13_endgate4.json > $O/kern_13_endgate4.log 2>&1
$PY $HERE/v14_michel_nd_eval.py /pscratch/sd/y/yuxuan/v11arms/V14A --files 0000001,0000002,0000003,0000004,0000005 --set $G host_dedx_max=5 tip_face_min_cm=5 --out $O/kern_63_endgate.json > $O/kern_63_endgate.log 2>&1
echo ENDGATE_DONE >> $O/kern_63_endgate.log
