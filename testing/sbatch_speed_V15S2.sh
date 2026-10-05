#!/bin/bash
# SPEED S2 = S1 + CPU fixes (flash-table unchanged-list cache, CPU cache of GPU image maps in the backbone shift helper, truth slab read, frontend module memoisation, vectorised pass2 pair features) (Michel perceiver batch stays 16: 32 collides with the other worker on the shared GPU); S1 = V14M3 + exact FFT shift scan in the matrix stage (CLM_FFT_SCAN=1) + perceiver image batch 32 (CLM_IMAGE_BATCH); v1.4 + stage 4.85 Michel + stage 4.86 unexplained flashes at track ends (michel_flash.py, mf 0.8 / apc -0.5) (piece-first, Bragg-attached end pieces + small whole clusters, host gates + ending gates: local band dE/dx <= 5 MeV/cm behind the Bragg stop, tip >= 5 cm from an outer face, un-zeroed piece predictions, host-remainder rule, piece light gates mf 0.75 / apc -0.25 (wholes keep 0.8 / -0.5)) 5-file certification.
# Reference: /pscratch/sd/y/yuxuan/v11arms/V14A (v1.4 cert, same 63 spills).
set -uo pipefail
HERE=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing
K1="--p2-threshold 20 --pool-mode light --predict-min-energy 0.5 --predict-min-energy-shower 1.5 --redecide --redecide-emin 20"
P2="--precision --precision-rounds 3 --precision-calib $HERE/v120_calib_lowi.json --precision-pieces"
V14="--v200-min-e 20 --force-light-cm 20 --resid-fill --rf-small-max 100 --rf-cos-min 0.9 --rf-dcos 0.1 --rf-ratio-hi 3 --rf-tail 0.5 --rf-host-min 20 --rf-mode sup"
MND="--michel-nd --mnd-host-extent-cm 20 --mnd-bragg-end-cm 20 --mnd-piece-max-mev 60 --mnd-delay-max 300 --mnd-batch 16 --mnd-tip-face-cm 5 --mnd-set band_dedx_max=5 max_frac=1.0 remainder_min_mev=10 mf_min=0.75 apc_max=-0.25 whole_mf_min=0.8 whole_apc_max=-0.5 --michel-flash"
export HDF5_USE_FILE_LOCKING=FALSE V200_DOM=3 CLM_FFT_SCAN=1 CLM_IMAGE_BATCH=32
bash $HERE/run_v11cert.sh ${1:-V15S2} $K1 $P2 $V14 $MND
echo ALL_CONFIGS_DONE
