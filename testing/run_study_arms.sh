#!/bin/bash
set -uo pipefail
T=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing
bash $T/sbatch_v142st05.sh
echo "=== ARM 0.5 DONE ==="
bash $T/sbatch_v143st20.sh
echo "=== ARM 2.0 DONE ==="
echo BOTH_ARMS_DONE
