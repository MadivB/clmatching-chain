#!/bin/bash
export PYTHONUSERBASE=/global/homes/y/yuxuan/.local/perlmutter/pytorch2.8.0; unset PYTHONPATH; export HDF5_USE_FILE_LOCKING=FALSE
PY=/global/common/software/nersc9/pytorch/2.8.0/bin/python; D=/pscratch/sd/y/yuxuan/v11arms/V2A5
run() { nice -n 5 $PY resid_fill_offline.py $D --procs 8 "$@" 2>&1 ; }
for cfg in "A_sup_c90_smax100|--host-min 20 --ratio-hi 3 --mode sup --small-max 100" \
           "B_sup_c90_tail05_smax100|--host-min 20 --ratio-hi 3 --mode sup --small-max 100 --tail 0.5" \
           "C_sup_c95_smax100|--host-min 20 --ratio-hi 3 --mode sup --small-max 100 --cos-min 0.95" \
           "D_sup_c90_smax20|--host-min 20 --ratio-hi 3 --mode sup --small-max 20"; do
  name=${cfg%%|*}; args=${cfg#*|}
  run --tag "DEV_$name" --files 0000001,0000002,0000003 $args
  run --tag "HOLD_$name" --files 0000004,0000005 $args
done
echo GRID_DONE
