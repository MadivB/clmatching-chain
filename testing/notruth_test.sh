#!/bin/bash
# GPU: one spill through the full v1.5 chain with the MC truth ignored (production / data mode check)
export PYTHONUSERBASE=/global/homes/y/yuxuan/.local/perlmutter/pytorch2.8.0; unset PYTHONPATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 V200_DOM=3 CLM_IGNORE_TRUTH=1
PY=/global/common/software/nersc9/pytorch/2.8.0/bin/python
HERE=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly
D=/global/cfs/cdirs/dunepro/people/abooth/nd-production/output/MiniProdN5/run-ndlar-flow/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift/FLOW/0000000
SMALL=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/NewMLSection/runs/ndfull_small4x/best_model_arch.pt
FRONTEND=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/clmatching-frontend
V15="--backbone stack2 --merge-p2 --sigma-mode poisson --max-clip-ticks 4 --police locked --merge-vertex --skip-baseline --phase3 joint --pass2-trade --pass2-light-nom --force-coverage --swap-fix --swap-ratio-max 0.85 --p2-threshold 20 --predict-min-energy 0.5 --predict-min-energy-shower 1.5 --skip-v2 --gate-support --force-simple --isolate-cm 25 --fast-scan --edge-penalty --pool-mode light --redecide --redecide-emin 20 --precision --precision-rounds 3 --precision-calib $HERE/testing/v120_calib_lowi.json --precision-pieces --v200-min-e 20 --force-light-cm 20 --resid-fill --michel-nd --mnd-host-extent-cm 20 --mnd-bragg-end-cm 20 --mnd-piece-max-mev 60 --mnd-delay-max 300 --mnd-batch 16 --mnd-tip-face-cm 5 --mnd-set band_dedx_max=5 max_frac=1.0 remainder_min_mev=10 mf_min=0.75 apc_max=-0.25 whole_mf_min=0.8 whole_apc_max=-0.5 --michel-flash --subtick --conf-cluster --light-ckpt $SMALL --frontend-repo $FRONTEND"
mkdir -p /pscratch/sd/y/yuxuan/v11arms/NOTRUTH
cd $HERE/testing
$PY $HERE/testing/overflow_rescue_pipeline.py --files $D/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.0000001.FLOW.hdf5 --out /pscratch/sd/y/yuxuan/v11arms/NOTRUTH/groups.jsonl $V15 --dump-dir /pscratch/sd/y/yuxuan/v11arms/NOTRUTH/full_dumps --event-stride 13 --event-offset 2 > /pscratch/sd/y/yuxuan/v11arms/NOTRUTH/worker.log 2>&1
echo NOTRUTH_DONE >> /pscratch/sd/y/yuxuan/v11arms/NOTRUTH/worker.log
