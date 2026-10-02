#!/bin/bash
# v1.4 5-file certification: v1.3 chain + --force-light-cm 20 (stage 4.9 light-vote force) + --resid-fill (stage 4.8).
# Reference: /pscratch/sd/y/yuxuan/v11arms/V2A5 (v1.3 cert, same 63 spills).
set -uo pipefail
HERE=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing
K1="--p2-threshold 20 --pool-mode light --predict-min-energy 0.5 --predict-min-energy-shower 1.5 --redecide --redecide-emin 20"
P2="--precision --precision-rounds 3 --precision-calib $HERE/v120_calib_lowi.json --precision-pieces"
export HDF5_USE_FILE_LOCKING=FALSE V200_DOM=3
bash $HERE/run_v11cert.sh ${1:-V14A} $K1 $P2 --v200-min-e 20 --force-light-cm 20 --resid-fill \
  --rf-small-max 100 --rf-cos-min 0.9 --rf-dcos 0.1 --rf-ratio-hi 3 --rf-tail 0.5 --rf-host-min 20 --rf-mode sup
echo ALL_CONFIGS_DONE
