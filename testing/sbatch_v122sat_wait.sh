#!/bin/bash
# Hold a GPU allocation until the code edits are done, then run the v1.2.2 regression.
# The go-signal is the file /pscratch/sd/y/yuxuan/v11arms/SAT1_GO (touch it); gives up after 50 min.
HERE=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing
GO=/pscratch/sd/y/yuxuan/v11arms/SAT1_GO
echo "=== allocation up $(date) on $(hostname); waiting for $GO ==="
for i in $(seq 1 300); do
  if [ -f "$GO" ]; then
    echo "=== go at $(date) ==="
    bash "$HERE/sbatch_v122sat.sh" SAT1
    exit $?
  fi
  sleep 10
done
echo "=== gave up waiting $(date) ==="
exit 1
