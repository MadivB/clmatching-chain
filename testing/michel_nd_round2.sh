#!/bin/bash
# round 2 (family-time candidates): label 13 spills at unexpl 0.2 and 0.1, then hold the GPU for the dev/holdout pass
set -uo pipefail
export PYTHONUSERBASE=/global/homes/y/yuxuan/.local/perlmutter/pytorch2.8.0; unset PYTHONPATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=4
PY=/global/common/software/nersc9/pytorch/2.8.0/bin/python
HERE=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing
O=$HERE/michel_nd_out
$PY $HERE/v14_michel_nd.py /pscratch/sd/y/yuxuan/v11arms/V14A --events 13 --unexpl-frac 0.2 --out $O/label2_13_u20.json > $O/label2_13_u20.log 2>&1
$PY $HERE/v14_michel_nd.py /pscratch/sd/y/yuxuan/v11arms/V14A --events 13 --unexpl-frac 0.1 --out $O/label2_13_u10.json > $O/label2_13_u10.log 2>&1
echo LABEL2_DONE >> $O/label2_13_u10.log
# hold for the gate decision (GO file holds the extra flags), up to 45 min
for i in $(seq 1 270); do
  if [ -f $O/GO_devhold ]; then
    GATES=$(cat $O/GO_devhold)
    for TAG in A B; do
      G=$(echo "$GATES" | sed -n "${TAG/A/1}p" 2>/dev/null); [ "$TAG" = "B" ] && G=$(echo "$GATES" | sed -n 2p)
      [ -z "$G" ] && continue
      $PY $HERE/v14_michel_nd.py /pscratch/sd/y/yuxuan/v11arms/V14A --files 0000001,0000002,0000003 --apply $G --out $O/dev2_$TAG.json > $O/dev2_$TAG.log 2>&1
      $PY $HERE/v14_michel_nd.py /pscratch/sd/y/yuxuan/v11arms/V14A --files 0000004,0000005 --apply $G --out $O/hold2_$TAG.json > $O/hold2_$TAG.log 2>&1
    done
    echo DEVHOLD2_DONE >> $O/hold2_A.log; exit 0
  fi
  sleep 10
done
echo "no GO file; releasing the GPU" >> $O/label2_13_u10.log
