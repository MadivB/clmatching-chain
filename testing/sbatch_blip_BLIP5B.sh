#!/bin/bash
# Stage 4.7 blip re-homing + operator 3 re-attach + 4.9 noise-hit vote, 5-file certification: v1.3 + --blip-rehome --blip-reattach --noise-vote.
# Reference: /pscratch/sd/y/yuxuan/v11arms/V2A5 (v1.3 cert, same files, deterministic chain).
# Launch: salloc -A dune_g -q interactive -C gpu -N 1 --gpus 4 -c 32 -t 01:30:00 srun -n 1 -c 32 --gpus 4 \
#         bash testing/gpu_wait_run.sh <GO_FILE> testing/sbatch_blip_BLIP5.sh
set -uo pipefail
HERE=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing
K1="--p2-threshold 20 --pool-mode light --predict-min-energy 0.5 --predict-min-energy-shower 1.5 --redecide --redecide-emin 20"
P2="--precision --precision-rounds 3 --precision-calib $HERE/v120_calib_lowi.json --precision-pieces"
export HDF5_USE_FILE_LOCKING=FALSE V200_DOM=3
bash $HERE/run_v11cert.sh ${1:-BLIP5B} $K1 $P2 --v200-min-e 20 --blip-rehome --blip-reattach --noise-vote
echo ALL_CONFIGS_DONE
