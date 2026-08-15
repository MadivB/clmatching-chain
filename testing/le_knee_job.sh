#!/bin/bash
cd /global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing
export LE_MODEL=hgb LE_USE_CACHE=1
PY=/global/common/software/nersc/pe/conda-envs/26.1.0/python-3.13/nersc-python/bin/python
for SUB in \
  "d_min,d_cent,log_fe,fam_n,dt_ambig,same_tpc,d_min_rank,is_inc" \
  "d_min,d_cent,log_fe,fam_n,dt_ambig,d_min_rank,is_inc" \
  "d_min,d_cent,log_fe,fam_n,dt_ambig,is_inc" \
  "d_min,log_fe,fam_n,dt_ambig,d_min_rank,is_inc" \
  "d_min,d_cent,log_fe,fam_n,is_inc"; do
  export LE_SUBSET="$SUB"
  echo "===== SUBSET: $SUB ====="
  $PY -u le_pre_spine.py fit && $PY -u le_pre_spine.py emulate
done
echo LE_KNEE_DONE
