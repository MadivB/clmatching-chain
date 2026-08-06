"""Compare overflow-statistic variants for the pause-on-placement check.

Per fit channel the study dumped [own, base_before, actual] at the placed
pulse-peak tick.  Each variant maps those arrays to one scalar per placement;
we compare how well each separates truth-wrong from truth-correct placements.
"""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.abspath(os.path.dirname(__file__))
OUT_DIR = os.path.join(HERE, "overflow_study_out")
COHERENCE_MIN = 0.7
SIGMA_FLOOR = 100.0

rows = []
for fp in sorted(glob.glob(os.path.join(OUT_DIR, "rows_worker*.jsonl"))):
    with open(fp) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
n_events = len({(r["file"], r["event"]) for r in rows})
usable = [r for r in rows
          if r["wrong"] is not None and r["coherence"] >= COHERENCE_MIN
          and r["type"] == "track" and r.get("fit_channels_oba")]
print(f"{len(rows)} rows, {n_events} events; usable track placements: {len(usable)}")


def stats_for(r):
    oba = np.asarray(r["fit_channels_oba"], dtype=np.float64)
    o, b, a = oba[:, 0], oba[:, 1], oba[:, 2]
    p = o + b
    sig_pois = np.sqrt(np.maximum(a, SIGMA_FLOOR))
    sig_mod = np.sqrt(np.maximum(a, SIGMA_FLOOR) + (0.25 * p) ** 2)
    ov = p - a
    ov_own = np.minimum(ov, o)               # excess attributable to THIS cluster
    active = o >= 0.2 * max(float(o.max()), 1e-9)

    out = {
        "raw_max": float(np.max(ov / sig_pois)),
        "moderr_max": float(np.max(ov / sig_mod)),
        "own_max": float(np.max(ov_own / sig_pois)),
        "own_moderr_max": float(np.max(ov_own / sig_mod)),
        "frac_max": float(np.max(ov / np.maximum(p, 200.0))),
    }
    if active.any():
        out["own_moderr_med_active"] = float(np.median((ov_own / sig_mod)[active]))
        out["frac_med_active"] = float(np.median((ov / np.maximum(p, 200.0))[active]))
    else:
        out["own_moderr_med_active"] = 0.0
        out["frac_med_active"] = 0.0
    return out


VARIANTS = ["raw_max", "moderr_max", "own_max", "own_moderr_max", "frac_max",
            "own_moderr_med_active", "frac_med_active"]
vals = {v: {True: [], False: []} for v in VARIANTS}
for r in usable:
    s = stats_for(r)
    for v in VARIANTS:
        vals[v][bool(r["wrong"])].append(s[v])

print(f"\n{'variant':>24} | {'TPR@FPR=2%':>10} | {'TPR@FPR=5%':>10} | {'TPR@FPR=10%':>11} | AUC-ish")
best = None
for v in VARIANTS:
    w = np.asarray(vals[v][True]); c = np.asarray(vals[v][False])
    cuts = np.unique(np.concatenate([w, c]))
    fpr = np.array([np.mean(c > x) for x in cuts])
    tpr = np.array([np.mean(w > x) for x in cuts])
    order = np.argsort(fpr)
    auc = float(np.trapezoid(tpr[order], fpr[order]))
    def tpr_at(f):
        ok = fpr <= f
        return float(np.max(tpr[ok])) if ok.any() else 0.0
    print(f"{v:>24} | {tpr_at(0.02)*100:>9.1f}% | {tpr_at(0.05)*100:>9.1f}% | {tpr_at(0.10)*100:>10.1f}% | {auc:.3f}")
    if best is None or auc > best[1]:
        best = (v, auc)
print(f"\nbest variant by AUC: {best[0]} ({best[1]:.3f})")

# save per-placement variant values for figure-making
np.savez_compressed(
    os.path.join(HERE, "overflow_variant_values.npz"),
    **{f"{v}_wrong": np.asarray(vals[v][True]) for v in VARIANTS},
    **{f"{v}_correct": np.asarray(vals[v][False]) for v in VARIANTS},
    n_events=n_events,
)
print("saved overflow_variant_values.npz")
