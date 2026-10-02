#!/bin/bash
# in-chain smoke of v1.4 + --michel-nd (stage 4.85) on 2 events of file 1 (worker 0 of 8)
export PYTHONUSERBASE=/global/homes/y/yuxuan/.local/perlmutter/pytorch2.8.0; unset PYTHONPATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HDF5_USE_FILE_LOCKING=FALSE V200_DOM=3
HERE=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly
PY=/global/common/software/nersc9/pytorch/2.8.0/bin/python
F=/global/cfs/cdirs/dunepro/people/abooth/nd-production/output/MiniProdN5/run-ndlar-flow/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift/FLOW/0000000/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.0000001.FLOW.hdf5
OUT=/pscratch/sd/y/yuxuan/v11arms/MNDsmoke
SMALL=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/NewMLSection/runs/ndfull_small4x/best_model_arch.pt
FRONTEND=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/clmatching-frontend
V13="--backbone stack2 --merge-p2 --sigma-mode poisson --max-clip-ticks 4 --police locked --merge-vertex --skip-baseline --phase3 joint --pass2-trade --pass2-light-nom --force-coverage --swap-fix --swap-ratio-max 0.85 --p2-threshold 20 --predict-min-energy 0.5 --predict-min-energy-shower 1.5 --skip-v2 --gate-support --force-simple --isolate-cm 25 --fast-scan --edge-penalty --pool-mode light --redecide --redecide-emin 20 --precision --precision-rounds 3 --precision-calib $HERE/testing/v120_calib_lowi.json --precision-pieces --v200-min-e 20 --subtick --conf-cluster --light-ckpt $SMALL --frontend-repo $FRONTEND"
cd $HERE/testing
CUDA_VISIBLE_DEVICES=0 $PY $HERE/testing/overflow_rescue_pipeline.py --files $F --out $OUT/groups_worker0.jsonl $V13 --force-light-cm 20 --resid-fill --michel-nd --event-stride 8 --event-offset 0 > $OUT/worker0.log 2>&1
echo SMOKE_DONE >> $OUT/worker0.log
