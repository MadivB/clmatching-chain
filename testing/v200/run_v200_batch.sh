#!/bin/bash
# v2.0_test batch on a shared CPU node:  bash run_v200_batch.sh <ARM> <dump_dir> [extra v200_run args...]
# env vars for the solver (V200_DOM=...) are passed through.  Output: v200_out/<ARM>.{log,json}
set -uo pipefail
export PYTHONUSERBASE=/global/homes/y/yuxuan/.local/perlmutter/pytorch2.8.0; unset PYTHONPATH
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 HDF5_USE_FILE_LOCKING=FALSE
HERE=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing/v200
PY=/global/common/software/nersc9/pytorch/2.8.0/bin/python
ARM=$1; DUMPS=$2; shift 2
mkdir -p "$HERE/v200_out"
echo "=== $ARM start $(date) env V200_DOM=${V200_DOM:-0} args: $* ==="
$PY "$HERE/v200_run.py" "$DUMPS" --workers 12 --out "$HERE/v200_out/$ARM.json" "$@"
echo "=== $ARM done $(date) ==="
