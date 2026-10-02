#!/bin/bash
# GPU: kernel on 63 spills, un-zeroed predictions, batch 16, face >= 5, band dE/dx <= 5: host extent gate 20 -> 10 -> 0
set -uo pipefail
export PYTHONUSERBASE=/global/homes/y/yuxuan/.local/perlmutter/pytorch2.8.0; unset PYTHONPATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=4
PY=/global/common/software/nersc9/pytorch/2.8.0/bin/python
HERE=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing; O=$HERE/michel_nd_out
F=0000001,0000002,0000003,0000004,0000005
$PY $HERE/v14_michel_nd_eval.py /pscratch/sd/y/yuxuan/v11arms/V14A --files $F --batch 16 --set host_extent_min_cm=10 bragg_end_cm=20 piece_max_mev=60 tip_face_min_cm=5 band_dedx_max=5 --out $O/kern_63_ext10_b5.json > $O/kern_63_ext10_b5.log 2>&1
$PY $HERE/v14_michel_nd_eval.py /pscratch/sd/y/yuxuan/v11arms/V14A --files $F --batch 16 --set host_extent_min_cm=0 bragg_end_cm=20 piece_max_mev=60 tip_face_min_cm=5 band_dedx_max=5 --out $O/kern_63_ext0_b5.json > $O/kern_63_ext0_b5.log 2>&1
echo EXT_DONE >> $O/kern_63_ext0_b5.log
