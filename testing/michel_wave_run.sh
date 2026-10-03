#!/bin/bash
set -uo pipefail
export PYTHONUSERBASE=/global/homes/y/yuxuan/.local/perlmutter/pytorch2.8.0; unset PYTHONPATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=4 MPLBACKEND=Agg
PY=/global/common/software/nersc9/pytorch/2.8.0/bin/python
HERE=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing; O=$HERE/michel_nd_out
$PY $HERE/v15_michel_waveforms.py /pscratch/sd/y/yuxuan/v11arms/V14A --audit $O/michel_audit5_63.json --files 0000001,0000002,0000003,0000004,0000005 --examples 10 > $O/michel_waveforms_63.log 2>&1
echo WAVE_DONE >> $O/michel_waveforms_63.log
