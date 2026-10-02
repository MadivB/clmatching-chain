#!/bin/bash
export PYTHONUSERBASE=/global/homes/y/yuxuan/.local/perlmutter/pytorch2.8.0; unset PYTHONPATH; export HDF5_USE_FILE_LOCKING=FALSE
PY=/global/common/software/nersc9/pytorch/2.8.0/bin/python; D=/pscratch/sd/y/yuxuan/v11arms/V2A5; N=13
run() { nice -n 5 $PY resid_fill_offline.py $D --events $N --procs 8 "$@" 2>&1 | grep -v "^    0000" ; }
run --tag base
run --tag gates --glare-min 0.17 --host-min 20
run --tag gates_r3 --glare-min 0.17 --host-min 20 --ratio-hi 3
run --tag gates_r3_c95 --glare-min 0.17 --host-min 20 --ratio-hi 3 --cos-min 0.95
run --tag gates_r3_near16 --glare-min 0.17 --host-min 20 --ratio-hi 3 --K 16
run --tag gates_r3_near30 --glare-min 0.17 --host-min 20 --ratio-hi 3 --K 30
run --tag gates_r3_top8 --glare-min 0.17 --host-min 20 --ratio-hi 3 --mode top
run --tag gates_r3_sup --glare-min 0.17 --host-min 20 --ratio-hi 3 --mode sup
run --tag gates_r3_small2 --glare-min 0.17 --host-min 20 --ratio-hi 3 --small-min 2
run --tag gates_r3_kprom20 --glare-min 0.17 --host-min 20 --ratio-hi 3 --kprom 20
run --tag gates_r3_tail05 --glare-min 0.17 --host-min 20 --ratio-hi 3 --tail 0.5
run --tag gates_r3_dcos02 --glare-min 0.17 --host-min 20 --ratio-hi 3 --dcos 0.2
echo GRID_DONE
