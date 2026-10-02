#!/bin/bash
# wait for the thr0 interactive job to end (QOS submit limit), then launch the extent run
while squeue -h -j 59220179 2>/dev/null | grep -q .; do sleep 60; done
nohup salloc -A dune_g -q interactive -C gpu -t 01:00:00 -N 1 --gpus 1 srun -n 1 bash /global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing/michel_nd_ext_run.sh > /global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing/michel_nd_out/salloc_ext.log 2>&1
