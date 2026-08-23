#!/bin/bash
HERE=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing
PY=/global/common/software/nersc9/pytorch/2.8.0/bin/python
export WT_FRONTEND_REPO=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/clmatching-frontend
export WT_SUBJECT=0000001,1,16
$PY -u $HERE/smoke_walkthrough.py
