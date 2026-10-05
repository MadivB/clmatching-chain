"""Per-hit precision-timing resolution of two arms from their dumps (ts_final sub-tick vs truth_t0, assigned hits with
truth), plus the precision rows (sig_med_ns) from the groups files.  python speed_precision_compare.py <armA> <armB>"""
import glob, json, os, sys
import numpy as np
NS = 16.0
def per_hit(arm):
    d = []
    for f in sorted(glob.glob(f"/pscratch/sd/y/yuxuan/v11arms/{arm}/full_dumps/*.npz")):
        z = np.load(f, allow_pickle=True); ts = z["ts_final"].astype(float); tru = z["truth_t0"].astype(float); e = np.clip(z["e"], 0, None)
        m = np.isfinite(ts) & np.isfinite(tru) & (ts >= 0); dt = (ts[m] - tru[m]) * NS
        ok = np.abs(dt) <= 10 * NS                      # correctly assigned hits only (within 10 ticks)
        d.append(np.column_stack([dt[ok], e[m][ok]]))
    d = np.concatenate(d); dt, w = d[:, 0], d[:, 1]
    med = np.median(dt); mad = 1.4826 * np.median(np.abs(dt - med)); q = np.percentile(np.abs(dt - med), [50, 68, 90])
    return dict(n=int(dt.size), median_ns=float(med), robust_sigma_ns=float(mad), abs_p50=float(q[0]), abs_p68=float(q[1]), abs_p90=float(q[2]), std_ns=float(dt.std()))
def rows(arm):
    R = [json.loads(l) for f in sorted(glob.glob(f"/pscratch/sd/y/yuxuan/v11arms/{arm}/groups_worker*.jsonl")) for l in open(f)]
    pr = [r["phase3"].get("precision", {}) for r in R]
    return dict(n_ev=len(R), sig_med_ns=float(np.median([p.get("sig_med_ns", np.nan) for p in pr])), n_fit=float(np.mean([p.get("n_fit", 0) for p in pr])), time_s=float(np.mean([p.get("time_s", 0) for p in pr])), t_event=float(np.median([r["t_event_s"] for r in R])), truth_s=float(np.median([r.get("timing", {}).get("truth_s", np.nan) for r in R])))
for arm in sys.argv[1:]:
    print(arm, "per-hit (assigned & within 10 ticks):", {k: round(v, 2) for k, v in per_hit(arm).items()}, "| rows:", {k: round(v, 2) for k, v in rows(arm).items()})
