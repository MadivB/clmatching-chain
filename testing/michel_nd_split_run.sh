#!/bin/bash
# GPU: kernel on 63 spills: loose light gates for Bragg-attached pieces (mf 0.75, apc -0.25), certified gates for whole clusters
set -uo pipefail
export PYTHONUSERBASE=/global/homes/y/yuxuan/.local/perlmutter/pytorch2.8.0; unset PYTHONPATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=4
PY=/global/common/software/nersc9/pytorch/2.8.0/bin/python
HERE=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing; O=$HERE/michel_nd_out
F=0000001,0000002,0000003,0000004,0000005
B="host_extent_min_cm=20 bragg_end_cm=20 piece_max_mev=60 tip_face_min_cm=5 band_dedx_max=5 max_frac=1.0 remainder_min_mev=10"
$PY $HERE/v14_michel_nd_eval.py /pscratch/sd/y/yuxuan/v11arms/V14A --files $F --batch 16 --set $B mf_min=0.75 apc_max=-0.25 whole_mf_min=0.8 whole_apc_max=-0.5 --out $O/kern_63_split.json > $O/kern_63_split.log 2>&1
echo SPLIT_DONE >> $O/kern_63_split.log
