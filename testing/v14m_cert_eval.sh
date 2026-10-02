#!/bin/bash
# wait for the V14M certification, then evaluate (detached from the harness)
export PYTHONUSERBASE=/global/homes/y/yuxuan/.local/perlmutter/pytorch2.8.0; unset PYTHONPATH; export HDF5_USE_FILE_LOCKING=FALSE
PY=/global/common/software/nersc9/pytorch/2.8.0/bin/python
cd /global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing
until grep -q "ALL_CONFIGS_DONE\|CERT V14M done" /pscratch/sd/y/yuxuan/v11arms/sallocV14M.log 2>/dev/null; do sleep 120; done
{
echo "=== cert log tail"; tail -3 /pscratch/sd/y/yuxuan/v11arms/sallocV14M.log
echo "=== ledger vs V14A"; nice -n 5 $PY v11_ab.py /pscratch/sd/y/yuxuan/v11arms/V14A /pscratch/sd/y/yuxuan/v11arms/V14M --dumps
echo "=== stage timing"; $PY stage_time_report.py /pscratch/sd/y/yuxuan/v11arms/V14M | head -30
echo "=== new-stage rows"; $PY - <<'PYEOF'
import json, glob, numpy as np
rows = [json.loads(l) for f in sorted(glob.glob("/pscratch/sd/y/yuxuan/v11arms/V14M/groups_worker*.jsonl")) for l in open(f)]
rf = [r["phase3"].get("michel_nd", {}) for r in rows]; fo = [r["phase3"].get("force", {}) for r in rows]
print("events", len(rows))
print("michel_nd: pieces/ev %.0f moves/ev %.1f fixed %.2f GeV broke %.2f GeV time %.1f s/ev" % (np.mean([x.get("n_cand",0) for x in rf]), np.mean([x.get("n_moves",0) for x in rf]), sum(x.get("fixed_e",0) for x in rf)/1e3, sum(x.get("broke_e",0) for x in rf)/1e3, np.mean([x.get("time_s",0) for x in rf])))
w = np.array([x.get("force_new_efrac",0)*r["truth_energy"] for x, r in zip(fo, rows)])
print("force: n_light/ev %.0f forced-hit acc E-wt %.3f time %.2f s/ev" % (np.mean([x.get("n_light",0) for x in fo]), np.average([x.get("force_acc_new",0) for x in fo], weights=np.maximum(w,1e-9)), np.mean([x.get("time_s",0) for x in fo])))
print("t_event median %.1f s" % np.median([r["t_event_s"] for r in rows]))
PYEOF
} > v14_cert_V14M.log 2>&1
echo CERT_EVAL_DONE >> v14_cert_V14M.log
