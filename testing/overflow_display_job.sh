#!/bin/bash
#SBATCH -A dune
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:20:00
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -J overflow_display_dump
#SBATCH -o /global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing/display_dump_%j.log
PY=/global/common/software/nersc9/pytorch/2.8.0/bin/python
$PY /global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing/overflow_display_dump.py
