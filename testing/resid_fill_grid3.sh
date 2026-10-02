#!/bin/bash
export PYTHONUSERBASE=/global/homes/y/yuxuan/.local/perlmutter/pytorch2.8.0; unset PYTHONPATH; export HDF5_USE_FILE_LOCKING=FALSE
PY=/global/common/software/nersc9/pytorch/2.8.0/bin/python; D=/pscratch/sd/y/yuxuan/v11arms/V2A5; N=13
run() { nice -n 5 $PY resid_fill_offline.py $D --events $N --procs 8 "$@" 2>&1 | grep -v "^    0000" ; }
B="--host-min 20 --ratio-hi 3 --mode sup"
run --tag sup_c90_tail05 $B --tail 0.5
run --tag sup_c90_tail10 $B --tail 1.0
run --tag sup_c90_tail05_smax100 $B --tail 0.5 --small-max 100
run --tag sup_c90_smax100 $B --small-max 100
run --tag sup_c95_smax100 $B --small-max 100 --cos-min 0.95
run --tag sup_c90_dcos005 $B --dcos 0.05
run --tag sup_c90_r5 --host-min 20 --ratio-hi 5 --mode sup
run --tag sup_c90_host10 --host-min 10 --ratio-hi 3 --mode sup
run --tag sup_c90_host30 --host-min 30 --ratio-hi 3 --mode sup
run --tag sup_c90_minadc1500 $B --min-adc 1500
echo GRID_DONE
