#!/bin/bash
O=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing/michel_nd_out
until grep -q ALL_CONFIGS_DONE $O/salloc_cert_V15S2d.log 2>/dev/null; do sleep 60; done
sleep 45
for try in $(seq 1 12); do
  salloc -A dune_g -q interactive -C gpu -t 03:00:00 -N 1 --gpus 4 srun -n 1 bash /global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing/sbatch_speed_V15S3.sh > $O/salloc_cert_V15S3.log 2>&1
  grep -q QOSMax $O/salloc_cert_V15S3.log || break
  sleep 300
done
