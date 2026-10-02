#!/bin/bash
export PYTHONUSERBASE=/global/homes/y/yuxuan/.local/perlmutter/pytorch2.8.0; unset PYTHONPATH; export HDF5_USE_FILE_LOCKING=FALSE
PY=/global/common/software/nersc9/pytorch/2.8.0/bin/python; D=/pscratch/sd/y/yuxuan/v11arms/V2A5; N=13
run() { nice -n 5 $PY resid_fill_offline.py $D --events $N --procs 8 "$@" 2>&1 | grep -v "^    0000" ; }
run --tag noglare_host20 --host-min 20 --ratio-hi 3
run --tag noglare_host20_sup --host-min 20 --ratio-hi 3 --mode sup
run --tag noglare_host20_sup_c95 --host-min 20 --ratio-hi 3 --mode sup --cos-min 0.95
run --tag noglare_host20_near30_c95 --host-min 20 --ratio-hi 3 --K 30 --cos-min 0.95
run --tag noglare_host20_sup_c95_r2_d02 --host-min 20 --ratio-hi 2 --mode sup --cos-min 0.95 --dcos 0.2
run --tag noglare_host20_sup_c95_small2 --host-min 20 --ratio-hi 3 --mode sup --cos-min 0.95 --small-min 2
run --tag noglare_nohost_sup_c95_small2 --ratio-hi 3 --mode sup --cos-min 0.95 --small-min 2
run --tag glare_host20_sup_c95_small2_kprom20 --glare-min 0.17 --host-min 20 --ratio-hi 3 --mode sup --cos-min 0.95 --small-min 2 --kprom 20
run --tag glare005_host20_sup_small2 --glare-min 0.05 --host-min 20 --ratio-hi 3 --mode sup --small-min 2
echo GRID_DONE
