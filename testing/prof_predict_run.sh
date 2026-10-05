#!/bin/bash
export PYTHONUSERBASE=/global/homes/y/yuxuan/.local/perlmutter/pytorch2.8.0; unset PYTHONPATH; export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=4
/global/common/software/nersc9/pytorch/2.8.0/bin/python /global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing/prof_predict.py > /global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing/michel_nd_out/prof_predict.log 2>&1
echo PROF_DONE >> /global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing/michel_nd_out/prof_predict.log
