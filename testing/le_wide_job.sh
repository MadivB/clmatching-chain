#!/bin/bash
cd /global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing
export LE_MODEL=hgb LE_CANDR=100000 LE_CANDK=40
PY=/global/common/software/nersc/pe/conda-envs/26.1.0/python-3.13/nersc-python/bin/python
$PY -u le_pre_spine.py fit && $PY -u le_pre_spine.py emulate
echo LE_WIDE_DONE
