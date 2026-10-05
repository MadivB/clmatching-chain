#!/bin/bash
# launch the S2 certification when S1 has finished (interactive QOS: the other slot is the user's own job)
O=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing/michel_nd_out
until grep -q ALL_CONFIGS_DONE $O/salloc_cert_V15S1.log 2>/dev/null; do sleep 60; done
sleep 45
for try in 1 2 3 4 5 6 7 8 9 10 11 12; do
  salloc -A dune_g -q interactive -C gpu -t 03:00:00 -N 1 --gpus 4 srun -n 1 bash /global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing/sbatch_speed_V15S2.sh > $O/salloc_cert_V15S2.log 2>&1
  grep -q QOSMax $O/salloc_cert_V15S2.log || break
  sleep 300
done
