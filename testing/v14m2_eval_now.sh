#!/bin/bash
export PYTHONUSERBASE=/global/homes/y/yuxuan/.local/perlmutter/pytorch2.8.0; unset PYTHONPATH; export HDF5_USE_FILE_LOCKING=FALSE
PY=/global/common/software/nersc9/pytorch/2.8.0/bin/python
cd /global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing
{
echo "=== ledger vs V14A"; nice -n 5 $PY v11_ab.py /pscratch/sd/y/yuxuan/v11arms/V14A /pscratch/sd/y/yuxuan/v11arms/V14M2 --dumps
echo "=== stage timing"; $PY stage_time_report.py /pscratch/sd/y/yuxuan/v11arms/V14M2 | head -30
echo "=== michel rows"; $PY - <<'PYEOF'
import json, glob, numpy as np
rows = [json.loads(l) for f in sorted(glob.glob("/pscratch/sd/y/yuxuan/v11arms/V14M2/groups_worker*.jsonl")) for l in open(f)]
m = [r["phase3"].get("michel_nd", {}) for r in rows]
print("events", len(rows))
print("michel_nd: pieces/ev %.0f wholes/ev %.0f predictions/ev %.0f moves/ev %.2f fixed %.2f GeV broke %.2f GeV time %.1f s/ev (p90 %.1f)" % (np.mean([x.get("n_cand",0) for x in m]), np.mean([x.get("n_whole",0) for x in m]), np.mean([x.get("n_pred",0) for x in m]), np.mean([x.get("n_moves",0) for x in m]), sum(x.get("fixed_e",0) for x in m)/1e3, sum(x.get("broke_e",0) for x in m)/1e3, np.mean([x.get("time_s",0) for x in m]), np.percentile([x.get("time_s",0) for x in m], 90)))
tt = {k: round(float(np.mean([x["tim"][k] for x in m if "tim" in x])), 2) for k in ("dense", "geom", "pred1", "scan", "pred2", "refine")}
print("time by part:", tt)
mv = [q for r in rows for q in r["phase3"].get("michel_nd_moves", [])]
d = np.array([q["now"] - q["was"] for q in mv]); wh = np.array([q["whole"] for q in mv])
print(f"moves {len(mv)} (whole {wh.sum()}): right {np.mean(d>0.5):.2f} wrong {np.mean(d<-0.5):.2f}; pieces fixed {d[~wh][d[~wh]>0].sum():.0f} broke {-d[~wh][d[~wh]<0].sum():.0f}; whole fixed {d[wh][d[wh]>0].sum():.0f} broke {-d[wh][d[wh]<0].sum():.0f} MeV")
print("t_event median %.1f s" % np.median([r["t_event_s"] for r in rows]))
PYEOF
} > v14m_cert_V14M2.log 2>&1
echo CERT_EVAL_DONE >> v14m_cert_V14M2.log
