#!/bin/bash
# v2.0_test arms on one debug CPU node (128 cores): B0 (light only), D3 (own-dominated decidability),
# F1 (D3 + systematics weights f=0.10), then in-situ prior calibration on F1 and the pass-2 arm F1P.
set -uo pipefail
export PYTHONUSERBASE=/global/homes/y/yuxuan/.local/perlmutter/pytorch2.8.0; unset PYTHONPATH
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 HDF5_USE_FILE_LOCKING=FALSE
HERE=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing/v200
PY=/global/common/software/nersc9/pytorch/2.8.0/bin/python
DUMPS=/pscratch/sd/y/yuxuan/v11arms/P2
OUT=$HERE/v200_out; mkdir -p $OUT
run() { name=$1; shift; echo "=== $name start $(date) ==="; "$@" $PY $HERE/v200_run.py $DUMPS --workers 32 --out $OUT/$name.json > $OUT/$name.log 2>&1; grep "===" $OUT/$name.log | tail -1; echo "=== $name done $(date) ==="; }
run B0 env V200_DOM=0
run D3 env V200_DOM=3
run F1 env V200_DOM=3 V200_FSYS=0.10
run S1 env V200_DOM=3 V200_SUP=1
pass2() { name=$1; src=$2; shift 2; $PY $HERE/v200_calib.py $OUT/$src.json --out $OUT/prior_$src.json > $OUT/prior_$src.log 2>&1; echo "=== $name start $(date) ==="; "$@" $PY $HERE/v200_run.py $DUMPS --workers 32 --beta $OUT/prior_$src.json --out $OUT/$name.json > $OUT/$name.log 2>&1; grep "===" $OUT/$name.log | tail -1; echo "=== $name done $(date) ==="; }
pass2 D3P D3 env V200_DOM=3
pass2 S1P S1 env V200_DOM=3 V200_SUP=1
echo ALL_ARMS_DONE
