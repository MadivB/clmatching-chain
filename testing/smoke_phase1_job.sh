#!/bin/bash
#SBATCH -A dune
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:28:00
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -J smoke_p1nb
#SBATCH -o /global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing/smoke_phase1_%j.log
set -uo pipefail
PY=/global/common/software/nersc9/pytorch/2.8.0/bin/python
$PY -u /global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing/smoke_phase1_nb.py
