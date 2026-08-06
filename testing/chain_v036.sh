#!/bin/bash
HERE=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing
until grep -q "ALL_CONFIGS_DONE" $HERE/interactive_v035r2.log 2>/dev/null; do sleep 60; done
cd $HERE
srun -A dune -C gpu -q interactive -t 02:00:00 -N 1 --gpus-per-node=4 -J v036 bash interactive_v036.sh > interactive_v036.log 2>&1
