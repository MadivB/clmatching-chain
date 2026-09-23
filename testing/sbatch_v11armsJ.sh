#!/bin/bash
# v1.1 third batch (run via: salloc -q interactive ... srun -n 1 -c 32 --gpus 4 bash this_script)
# D2/D3: lower pool floors gain in the light-trustworthy bands (crowded components) and lose in
# 5-20 MeV (light-weak clusters do worse in the light-only pool than in the tail solver).
# F2: re-decision helps >= 20 MeV, hurts 5-20. => both floors at the light boundary (20 MeV),
# plus the hybrid: small members PARTICIPATE in the joint decision but are not COMMITTED by it.
#   F20     light + --redecide --redecide-emin 20                             (floor 30)
#   D20     light + --p2-threshold 20                                         (pool = light-judgeable)
#   H20     light + --p2-threshold 5 + --pool-commit-min 20                   (participate down to 5, commit >= 20)
#   F20D20  light + --p2-threshold 20 + --redecide --redecide-emin 20
#   H20R    light + --p2-threshold 5 + --pool-commit-min 20 + --redecide --redecide-emin 20
#   F20D    light + --p2-threshold 10 + --redecide --redecide-emin 20         (D2 + re-decision)
set -uo pipefail
R=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing/run_v11arm.sh
bash $R F20    --p2-threshold 30 --pool-mode light --redecide --redecide-emin 20
bash $R D20    --p2-threshold 20 --pool-mode light
bash $R H20    --p2-threshold 5  --pool-mode light --pool-commit-min 20
bash $R F20D20 --p2-threshold 20 --pool-mode light --redecide --redecide-emin 20
bash $R H20R   --p2-threshold 5  --pool-mode light --pool-commit-min 20 --redecide --redecide-emin 20
bash $R F20D   --p2-threshold 10 --pool-mode light --redecide --redecide-emin 20
echo ALL_ARMS_DONE
