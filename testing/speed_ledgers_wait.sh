#!/bin/bash
# build the S1 and S2 ledgers (vs V14A and vs V14M3) when each certification finishes
cd /global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing
export PYTHONUSERBASE=/global/homes/y/yuxuan/.local/perlmutter/pytorch2.8.0; unset PYTHONPATH; export HDF5_USE_FILE_LOCKING=FALSE
PY=/global/common/software/nersc9/pytorch/2.8.0/bin/python
for ARM in V15S1 V15S2; do
  until grep -q ALL_CONFIGS_DONE michel_nd_out/salloc_cert_$ARM.log 2>/dev/null; do sleep 60; done
  { echo "=== $ARM ledger vs V14A"; nice -n 5 $PY v11_ab.py /pscratch/sd/y/yuxuan/v11arms/V14A /pscratch/sd/y/yuxuan/v11arms/$ARM --dumps | head -16
    echo "=== $ARM ledger vs V14M3 (same algorithm, speed changes only)"; nice -n 5 $PY v11_ab.py /pscratch/sd/y/yuxuan/v11arms/V14M3 /pscratch/sd/y/yuxuan/v11arms/$ARM --dumps | head -14
    echo "=== stage timing"; $PY stage_time_report.py /pscratch/sd/y/yuxuan/v11arms/$ARM | head -30
    $PY - <<PYEOF
import json, glob, numpy as np
rows = [json.loads(l) for f in sorted(glob.glob("/pscratch/sd/y/yuxuan/v11arms/$ARM/groups_worker*.jsonl")) for l in open(f)]
t = np.array([r["t_event_s"] for r in rows]); print(f"t_event median {np.median(t):.1f} s  p90 {np.percentile(t, 90):.1f}  mean {t.mean():.1f}  (n {t.size})")
for k in ("michel_nd", "michel_flash"):
    m = [r["phase3"].get(k, {}) for r in rows]; print(f"{k}: time {np.mean([x.get('time_s', 0) for x in m]):.1f} s/ev, moves {sum(x.get('n_moves', 0) for x in m)}, fixed {sum(x.get('fixed_e', 0) for x in m)/1e3:.2f} broke {sum(x.get('broke_e', 0) for x in m)/1e3:.2f} GeV")
PYEOF
  } > speed_cert_$ARM.log 2>&1
  echo LEDGER_DONE >> speed_cert_$ARM.log
done
