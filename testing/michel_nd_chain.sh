#!/bin/bash
# after the 13-event apply pass releases its GPU: dev + holdout apply passes with two gate sets in one allocation
HERE=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing
until grep -q "MICHEL_ND_DONE" $HERE/michel_nd_out/apply_13.log 2>/dev/null; do sleep 30; done
sleep 20
cat > $HERE/michel_nd_out/_devhold_inner.sh <<'INNER'
#!/bin/bash
HERE=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing
TAG=v2 GATES="" bash $HERE/run_michel_nd_devhold.sh
TAG=s1 GATES="--mf-min 0.9 --mf-min-whole 0.9 --apc-max -0.3" bash $HERE/run_michel_nd_devhold.sh
echo ALL_DEVHOLD_DONE >> $HERE/michel_nd_out/hold_s1.log
INNER
chmod +x $HERE/michel_nd_out/_devhold_inner.sh
salloc -A dune_g -q interactive -C gpu -t 03:00:00 -N 1 --gpus 1 srun -n 1 bash $HERE/michel_nd_out/_devhold_inner.sh > $HERE/michel_nd_out/salloc_devhold.log 2>&1
