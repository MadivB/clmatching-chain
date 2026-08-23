#!/bin/bash
cd /global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing
export LE_MODEL=hgb
PY=/global/common/software/nersc9/pytorch/2.8.0/bin/python
$PY -u le_pre_spine.py fit && $PY -u le_pre_spine.py emulate
echo LE_HGB_DONE
