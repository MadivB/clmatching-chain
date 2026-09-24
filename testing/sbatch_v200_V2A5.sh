#!/bin/bash
# v2.0 test 5-file certification: v1.2.1 chain + stage 3.99 (--v200-min-e 20), files 1-5, dumps on.
# Reference: /pscratch/sd/y/yuxuan/v11arms/CERT_P (v1.2 cert; pieces do not change the assignment).
set -uo pipefail
HERE=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing
K1="--p2-threshold 20 --pool-mode light --predict-min-energy 0.5 --predict-min-energy-shower 1.5 --redecide --redecide-emin 20"
P2="--precision --precision-rounds 3 --precision-calib $HERE/v120_calib_lowi.json --precision-pieces"
export HDF5_USE_FILE_LOCKING=FALSE V200_DOM=3
bash $HERE/run_v11cert.sh ${1:-V2A5} $K1 $P2 --v200-min-e 20
echo ALL_CONFIGS_DONE
