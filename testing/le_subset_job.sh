#!/bin/bash
cd /global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing
export LE_MODEL=hgb LE_USE_CACHE=1
PY=/global/common/software/nersc9/pytorch/2.8.0/bin/python
for SUB in "" \
  "d_min,log_fe,is_inc" \
  "d_min,log_fe,is_inc,same_tpc,dt_ambig" \
  "d_min,log_fe,is_inc,cos_fam" \
  "d_min,log_fe,is_inc,cos_blob" \
  "d_min,log_fe,is_inc,cos_fam,cos_blob" \
  "d_min,log_fe,is_inc,dt_ambig,cos_fam"; do
  export LE_SUBSET="$SUB"
  echo "===== SUBSET: ${SUB:-ALL-11} ====="
  $PY -u le_pre_spine.py fit && $PY -u le_pre_spine.py emulate
done
echo LE_SUBSET_DONE
