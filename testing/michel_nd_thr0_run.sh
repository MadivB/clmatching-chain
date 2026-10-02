#!/bin/bash
# GPU: kernel on 63 spills, piece predictions no longer zeroed below 100 ADC (min_prediction_threshold=0), batch 16;
# dE/dx handles: cluster E/extent cap 8 vs local band (10 cm behind the Bragg stop) cap 5 / 4
set -uo pipefail
export PYTHONUSERBASE=/global/homes/y/yuxuan/.local/perlmutter/pytorch2.8.0; unset PYTHONPATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=4
PY=/global/common/software/nersc9/pytorch/2.8.0/bin/python
HERE=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing; O=$HERE/michel_nd_out
G="host_extent_min_cm=20 bragg_end_cm=20 piece_max_mev=60 tip_face_min_cm=5"
F=0000001,0000002,0000003,0000004,0000005
$PY $HERE/v14_michel_nd_eval.py /pscratch/sd/y/yuxuan/v11arms/V14A --files $F --batch 16 --set $G host_dedx_max=5 --out $O/kern_63_thr0_d5.json > $O/kern_63_thr0_d5.log 2>&1
$PY $HERE/v14_michel_nd_eval.py /pscratch/sd/y/yuxuan/v11arms/V14A --files $F --batch 16 --set $G host_dedx_max=8 --out $O/kern_63_thr0_d8.json > $O/kern_63_thr0_d8.log 2>&1
$PY $HERE/v14_michel_nd_eval.py /pscratch/sd/y/yuxuan/v11arms/V14A --files $F --batch 16 --set $G band_dedx_max=5 --out $O/kern_63_thr0_b5.json > $O/kern_63_thr0_b5.log 2>&1
$PY $HERE/v14_michel_nd_eval.py /pscratch/sd/y/yuxuan/v11arms/V14A --files $F --batch 16 --set $G band_dedx_max=4 --out $O/kern_63_thr0_b4.json > $O/kern_63_thr0_b4.log 2>&1
echo THR0_DONE >> $O/kern_63_thr0_b4.log
