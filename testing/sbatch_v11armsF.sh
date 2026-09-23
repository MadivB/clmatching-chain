#!/bin/bash
# v1.1 follow-up arms (run via: salloc -q interactive ... srun -n 1 -c 32 --gpus 4 bash this_script)
# Measured so far: B (light components) = wash, 42% faster pool; C (gain at commit) negative;
# contact prior at the >=5 MeV level is physically unjustified (99% of disagreeing 3-cm
# contacts are genuine cross-interaction contacts) -> not run.
#   O   light-order: v1.0 ORDERING search on light-component batches (batching rule alone)
#   F2  light + --redecide                   (final-state component re-decision by light)
#   B2  light (clean rerun WITH the per-chunk audit: greedy vs joint, loss margin, truth)
#   D2  light + --p2-threshold 10            (10-30 MeV single-TPC clusters into the component search)
#   D3  light + --p2-threshold 5             (5-30 MeV single-TPC clusters into the component search)
#   G   light + --predict-min-energy 0.5     (templates for 0.5-1.5 MeV clusters)
set -uo pipefail
R=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing/run_v11arm.sh
bash $R O  --p2-threshold 30 --pool-mode light-order
bash $R F2 --p2-threshold 30 --pool-mode light --redecide
bash $R B2 --p2-threshold 30 --pool-mode light
bash $R D2 --p2-threshold 10 --pool-mode light
bash $R D3 --p2-threshold 5 --pool-mode light
bash $R G  --p2-threshold 30 --pool-mode light --predict-min-energy 0.5
echo ALL_ARMS_DONE
