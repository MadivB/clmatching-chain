#!/bin/bash
# GPU: kernel on 63 spills (un-zeroed predictions, batch 16): remainder rule / Bragg-end 40 / looser light gates
set -uo pipefail
export PYTHONUSERBASE=/global/homes/y/yuxuan/.local/perlmutter/pytorch2.8.0; unset PYTHONPATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=4
PY=/global/common/software/nersc9/pytorch/2.8.0/bin/python
HERE=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing; O=$HERE/michel_nd_out
F=0000001,0000002,0000003,0000004,0000005
B="host_extent_min_cm=20 piece_max_mev=60 tip_face_min_cm=5 band_dedx_max=5"
$PY $HERE/v14_michel_nd_eval.py /pscratch/sd/y/yuxuan/v11arms/V14A --files $F --batch 16 --set $B bragg_end_cm=20 max_frac=1.0 remainder_min_mev=10 --out $O/kern_63_r4_rem.json > $O/kern_63_r4_rem.log 2>&1
$PY $HERE/v14_michel_nd_eval.py /pscratch/sd/y/yuxuan/v11arms/V14A --files $F --batch 16 --set $B bragg_end_cm=40 max_frac=1.0 remainder_min_mev=10 --out $O/kern_63_r4_rem_be40.json > $O/kern_63_r4_rem_be40.log 2>&1
$PY $HERE/v14_michel_nd_eval.py /pscratch/sd/y/yuxuan/v11arms/V14A --files $F --batch 16 --set $B bragg_end_cm=20 max_frac=1.0 remainder_min_mev=10 mf_min=0.75 apc_max=-0.25 --out $O/kern_63_r4_rem_loose.json > $O/kern_63_r4_rem_loose.log 2>&1
$PY $HERE/v14_michel_nd_eval.py /pscratch/sd/y/yuxuan/v11arms/V14A --files $F --batch 16 --set $B bragg_end_cm=40 max_frac=1.0 remainder_min_mev=10 mf_min=0.75 apc_max=-0.25 --out $O/kern_63_r4_all.json > $O/kern_63_r4_all.log 2>&1
echo R4_DONE >> $O/kern_63_r4_all.log
