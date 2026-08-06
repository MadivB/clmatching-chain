#!/bin/bash
module load pytorch/2.8.0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HDF5_USE_FILE_LOCKING=FALSE
cd /global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing
python -u smoke_walkthrough.py
