#!/bin/bash
# v1.1 fourth batch (run via: salloc -q interactive ... srun -n 1 -c 32 --gpus 4 bash this_script)
# G (predict floor 0.5 MeV) = +0.15 pp NS (36/3) but shower TPCs gain nothing and the matrix
# stage doubles in cost -> the 0.5 floor in non-shower TPCs only, then stacked with the pool
# floor and the 20 MeV re-decision.
#   K0  light + predict 0.5 (non-shower) / 1.5 (shower)
#   K1  K0 + --p2-threshold 20 + --redecide --redecide-emin 20
#   K2  K0 + --p2-threshold 10 + --redecide --redecide-emin 20
set -uo pipefail
R=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing/run_v11arm.sh
P="--predict-min-energy 0.5 --predict-min-energy-shower 1.5"
bash $R K0 --p2-threshold 30 --pool-mode light $P
bash $R K1 --p2-threshold 20 --pool-mode light $P --redecide --redecide-emin 20
bash $R K2 --p2-threshold 10 --pool-mode light $P --redecide --redecide-emin 20
echo ALL_ARMS_DONE
