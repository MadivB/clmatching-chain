#!/bin/bash
# v2.0 test in-chain arm V2A: v1.2.1 chain (K1 + precision pieces) + stage 3.99 exact-block LNS re-solve
# of clusters >= 20 MeV (cold re-init, small clusters frozen).  Files 1-3, dumps on.
# Launch: salloc -A dune_g -q interactive -C gpu -N 1 --gpus 4 -c 32 -t 01:00:00 srun -n 1 -c 32 --gpus 4 bash testing/sbatch_v200_V2A.sh
set -uo pipefail
HERE=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing
K1="--p2-threshold 20 --pool-mode light --predict-min-energy 0.5 --predict-min-energy-shower 1.5 --redecide --redecide-emin 20"
P2="--precision --precision-rounds 3 --precision-calib $HERE/v120_calib_lowi.json --precision-pieces"
export HDF5_USE_FILE_LOCKING=FALSE V200_DOM=3
bash $HERE/run_v11arm.sh ${1:-V2A} $K1 $P2 --v200-min-e 20
echo ALL_CONFIGS_DONE
