#!/bin/bash
# wait for the round-4 interactive job to end (QOS submit limit), then launch the V14M2 certification (8 workers x 4 GPUs)
while squeue -h -j 59222846 2>/dev/null | grep -q .; do sleep 60; done
nohup salloc -A dune_g -q interactive -C gpu -t 03:30:00 -N 1 --gpus 4 srun -n 1 bash /global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing/sbatch_v14m_V14M2.sh > /global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing/michel_nd_out/salloc_cert_V14M2.log 2>&1
