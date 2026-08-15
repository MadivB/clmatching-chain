"""Variance-as-TIE-BREAKER test (user idea, mask variant).

Question: in near-tie placements (small |dz| = assigned vs best-alternative
quality nearly equal), does EXCLUDING the cells the variance model flags as
least reliable (top-q predicted sigma^2) improve wrong-vs-right separation?
Mask use is scale-calibration-free — it only needs sigma^2 to RANK cell
reliability correctly.

Data: v091STD dumps (real sigma^2 field). For each assigned cluster:
dz computed from confidence machinery with (a) all support cells,
(b) support cells minus the top-q sigma^2 cells (q = 10/25/50%).
Report wrong-vs-right AUC overall and in the near-tie band, per E band.
"""
import glob, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from confidence_cluster import cluster_scores
from sptl_v081 import auc_midrank

def run(q_mask):
    rows = []
    for fp in sorted(glob.glob("v091STD_out/dumps/*.npz")):
        d = np.load(fp, allow_pickle=True)
        actual = d["full_light"].astype(np.float64)
        base = d["base_image"].astype(np.float64)
        var = d["full_light_std"].astype(np.float64)         # sigma^2
        if q_mask is not None:
            thr = np.quantile(var, 1.0 - q_mask)
            actual = actual.copy()
            actual[var >= thr] = 60780.0 * 0.99              # -> "saturated" => cell excluded
        labs = d["labels"]; e = np.clip(d["e"].astype(float), 0, None)
        tsf = d["ts_final"].astype(float); tru = d["truth_t0"].astype(float)
        conf = cluster_scores(labs, d["tpc"], d["y"].astype(float), e, tsf,
                              actual, base)
        for l, c in conf.items():
            m = (labs == l)
            mt = m & np.isfinite(tru) & (tru >= 0)
            ma = m & np.isfinite(tsf) & (tsf >= 0)
            if not mt.any() or not ma.any():
                continue
            t_r = float(np.median(tru[mt])); t_a = float(np.median(tsf[ma]))
            rows.append((c["z_own"] - c["z_min"], abs(t_a - t_r) > 10.0, c["E"]))
        d.close()
    dz = np.array([r[0] for r in rows]); y = np.array([r[1] for r in rows], float)
    E = np.array([r[2] for r in rows])
    out = []
    for lo, hi in ((2, 20), (20, 100), (100, 1e6)):
        m = (E >= lo) & (E < hi) & np.isfinite(dz)
        tie = m & (np.abs(dz) < np.nanmedian(np.abs(dz[m])))   # near-tie half
        out.append((lo, hi, auc_midrank(y[m], dz[m]) if m.sum() > 200 else np.nan,
                    auc_midrank(y[tie], dz[tie]) if tie.sum() > 200 else np.nan))
    return out

for q in (None, 0.10, 0.25, 0.50):
    tag = "no mask " if q is None else f"mask {int(q*100):2d}%"
    res = run(q)
    line = "  ".join(f"E{lo}-{hi if hi<1e5 else '+'}: all {a:.3f} tie {t:.3f}"
                     for lo, hi, a, t in res)
    print(f"{tag}: {line}", flush=True)
print("TIEBREAK_DONE")
