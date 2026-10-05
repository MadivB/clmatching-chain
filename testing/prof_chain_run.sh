#!/bin/bash
# GPU (exclusive): (1) bf16/cudnn profile of the perceiver; (2) cProfile of the full chain on 2 events of file 1 with the V14M3 flags
set -uo pipefail
export PYTHONUSERBASE=/global/homes/y/yuxuan/.local/perlmutter/pytorch2.8.0; unset PYTHONPATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 V200_DOM=3
PY=/global/common/software/nersc9/pytorch/2.8.0/bin/python
HERE=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing; O=$HERE/michel_nd_out
$PY $HERE/prof_bf16.py > $O/prof_bf16.log 2>&1
DATA=/global/cfs/cdirs/dunepro/people/abooth/nd-production/output/MiniProdN5/run-ndlar-flow/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift/FLOW/0000000
SMALL=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/NewMLSection/runs/ndfull_small4x/best_model_arch.pt
FRONTEND=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/clmatching-frontend
V1="--backbone stack2 --merge-p2 --sigma-mode poisson --max-clip-ticks 4 --police locked --merge-vertex --skip-baseline --phase3 joint --pass2-trade --pass2-light-nom --force-coverage --swap-fix --swap-ratio-max 0.85 --predict-min-energy 1.5 --skip-v2 --gate-support --force-simple --isolate-cm 25 --fast-scan --edge-penalty --subtick --conf-cluster --light-ckpt $SMALL --frontend-repo $FRONTEND"
K1="--p2-threshold 20 --pool-mode light --predict-min-energy 0.5 --predict-min-energy-shower 1.5 --redecide --redecide-emin 20"
P2="--precision --precision-rounds 3 --precision-calib $HERE/v120_calib_lowi.json --precision-pieces"
V14="--v200-min-e 20 --force-light-cm 20 --resid-fill --rf-small-max 100 --rf-cos-min 0.9 --rf-dcos 0.1 --rf-ratio-hi 3 --rf-tail 0.5 --rf-host-min 20 --rf-mode sup"
MND="--michel-nd --mnd-host-extent-cm 20 --mnd-bragg-end-cm 20 --mnd-piece-max-mev 60 --mnd-delay-max 300 --mnd-batch 16 --mnd-tip-face-cm 5 --mnd-set band_dedx_max=5 max_frac=1.0 remainder_min_mev=10 mf_min=0.75 apc_max=-0.25 whole_mf_min=0.8 whole_apc_max=-0.5 --michel-flash"
mkdir -p /pscratch/sd/y/yuxuan/v11arms/PROF
cd $HERE
$PY -m cProfile -o $O/prof_chain.pstats $HERE/overflow_rescue_pipeline.py --files $DATA/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.0000001.FLOW.hdf5 --out /pscratch/sd/y/yuxuan/v11arms/PROF/groups.jsonl $V1 $K1 $P2 $V14 $MND --event-stride 8 --event-offset 1 > $O/prof_chain.log 2>&1
$PY - <<'PYEOF' > $O/prof_chain_top.log 2>&1
import pstats
p = pstats.Stats("/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing/michel_nd_out/prof_chain.pstats")
print("=== by cumulative time"); p.sort_stats("cumulative").print_stats(60)
print("=== by internal time"); p.sort_stats("tottime").print_stats(50)
PYEOF
echo PROFCHAIN_DONE >> $O/prof_chain_top.log
