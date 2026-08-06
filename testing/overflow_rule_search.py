"""Find a simple pause-trigger reaching >= 60% catch of polluted probes.

Features per placement (from the full 120-ch [own, base, actual] maps at the
probe tick), an event-split ML upper bound, then a greedy search over simple
OR-rules of 2-3 thresholded conditions.
"""
import glob
import itertools
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.abspath(os.path.dirname(__file__))
BASE = os.path.abspath(os.path.join(HERE, "..", "CLMatching_AlphaRelease"))
OUT_DIR = os.path.join(HERE, "overflow_study_out")
HITS_DIR = os.path.join(OUT_DIR, "hits")
PEAK = 105
THRESH_MEV = 5.0
W_MIN = 0.02

tmpl = np.load(os.path.join(BASE, "assets", "avg_pulse.npy")).astype(np.float64)
tmpl = np.maximum(tmpl / tmpl[PEAK], 0.0)

rows = []
for fp in sorted(glob.glob(os.path.join(OUT_DIR, "rows_worker*.jsonl"))):
    with open(fp) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
by_event = {}
for r in rows:
    by_event.setdefault((r["file"], r["event"]), []).append(r)

feats, labels, groups = [], [], []
for (fname, ev_id), ev_rows in by_event.items():
    ftag = fname.split(".")[-3]
    hf = os.path.join(HITS_DIR, f"{ftag}_ev{ev_id}.npz")
    if not os.path.exists(hf):
        continue
    d = np.load(hf)
    h_lab = d["labels"].astype(np.int64)
    h_tpc = d["tpc"].astype(np.int64)
    h_e = np.clip(d["e"].astype(np.float64), 0, None)
    h_reco = d["reco_t0"].astype(np.float64)
    h_truth = d["truth_t0"].astype(np.float64)
    mis = ((h_lab >= 0) & np.isfinite(h_reco) & (h_reco >= 0)
           & np.isfinite(h_truth) & (np.abs(h_reco - h_truth) > 10.0))

    for r in sorted(ev_rows, key=lambda q: q["label"]):
        if r["type"] != "track" or r["wrong"] is None or not r.get("tpc_channels"):
            continue
        pol = 0.0
        for c in r["tpcs"]:
            m = mis & (h_tpc == int(c)) & (h_lab <= int(r["label"]))
            if not m.any():
                continue
            idx = np.round(PEAK + float(r["t0"]) - h_reco[m]).astype(int)
            w = np.where((idx >= 0) & (idx < tmpl.size), tmpl[np.clip(idx, 0, tmpl.size - 1)], 0.0)
            w[w < W_MIN] = 0.0
            pol = max(pol, float(np.sum(h_e[m] * w)))

        f = {"adj3": -np.inf, "n5": 0, "n3": 0, "base_max": -np.inf,
             "frac_own": -np.inf, "own_frac3": 0.0, "sup_max": r["max_overflow_sigma"],
             "all_max": r["max_overflow_allch_sigma"] or -np.inf, "excess_e": 0.0,
             "under_max": -np.inf}
        for td in r["tpc_channels"]:
            o = np.asarray(td["own"], float)
            b = np.asarray(td["base"], float)
            a = np.asarray(td["actual"], float)
            sig = np.sqrt(np.maximum(a, 100.0))
            ov = (o + b - a) / sig
            fin = np.ones(120, dtype=bool)
            if td["veto"]:
                fin[np.asarray(td["veto"], int)] = False
            sup = np.asarray(td["support"], int)
            # 3-adjacent (min of trio at best support channel neighborhood)
            for c in sup:
                lo, hi = (0, 59) if c < 60 else (60, 119)
                if c - 1 >= lo and c + 1 <= hi and fin[c - 1] and fin[c] and fin[c + 1]:
                    f["adj3"] = max(f["adj3"], float(np.min(ov[[c - 1, c, c + 1]])))
            f["n5"] = max(f["n5"], int(np.sum(ov[fin] > 5.0)))
            f["n3"] = max(f["n3"], int(np.sum(ov[fin] > 3.0)))
            bov = (b - a) / sig
            f["base_max"] = max(f["base_max"], float(np.max(bov[fin])) if fin.any() else -np.inf)
            act = fin & (o >= max(0.2 * float(o.max()), 50.0))
            if act.any():
                f["frac_own"] = max(f["frac_own"], float(np.max((o + b - a)[act] / np.maximum((o + b)[act], 200.0))))
                f["own_frac3"] = max(f["own_frac3"], float(np.mean(ov[act] > 3.0)))
            f["excess_e"] = max(f["excess_e"], float(np.sum(np.clip((o + b - a)[fin], 0, None))))
            f["under_max"] = max(f["under_max"], float(np.max(((a - o - b) / sig)[fin])) if fin.any() else -np.inf)
        feats.append(f)
        labels.append(pol > THRESH_MEV)
        groups.append(ftag)

y = np.asarray(labels)
grp = np.asarray(groups)
names = ["sup_max", "all_max", "adj3", "n5", "n3", "base_max", "frac_own", "own_frac3", "excess_e", "under_max"]
X = np.array([[f[n] if np.isfinite(f[n]) else -100.0 for n in names] for f in feats])
print(f"{y.size} placements, polluted {int(y.sum())} ({y.mean():.1%}); features: {names}")

# ---- ML upper bound (event-file split CV) ----
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import LeaveOneGroupOut

probs = np.zeros(y.size)
logo = LeaveOneGroupOut()
for tr, te in logo.split(X, y, grp):
    clf = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=0)
    clf.fit(X[tr], y[tr])
    probs[te] = clf.predict_proba(X[te])[:, 1]
cuts = np.unique(probs)
tpr = np.array([np.mean(probs[y] > c) for c in cuts])
fpr = np.array([np.mean(probs[~y] > c) for c in cuts])
ok = tpr >= 0.60
print(f"\nML upper bound (GBDT, leave-one-file-out): at TPR>=60%, min FPR = "
      f"{fpr[ok].min()*100:.1f}%" if ok.any() else "ML cannot reach 60%")
for t in (0.60, 0.65, 0.70):
    okt = tpr >= t
    if okt.any():
        print(f"  TPR>={t:.0%}: FPR={fpr[okt].min()*100:.1f}%")

# feature importances (fit on all for reporting)
clf = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=0).fit(X, y)
imp = sorted(zip(names, clf.feature_importances_), key=lambda kv: -kv[1])
print("  importances:", ", ".join(f"{n}={v:.2f}" for n, v in imp))

# ---- greedy simple OR-rule search ----
cond_space = {
    "sup_max": [3, 5, 8, 10, 15, 20],
    "all_max": [3, 5, 8, 10, 15, 20],
    "adj3": [2, 3, 5, 8, 10, 14],
    "n5": [5, 10, 15, 20, 25, 30],
    "n3": [10, 15, 20, 25, 30, 40],
    "base_max": [3, 5, 10, 20],
    "frac_own": [0.3, 0.5, 0.7, 0.9],
    "own_frac3": [0.3, 0.5, 0.7, 0.9],
    "excess_e": [20000, 50000, 100000, 200000],
    "under_max": [10, 20, 50, 100],
}
conds = []
for n, ths in cond_space.items():
    j = names.index(n)
    for t in ths:
        conds.append((n, t, X[:, j] > t))

results = []
for k in (1, 2, 3):
    best = None
    for combo in itertools.combinations(range(len(conds)), k):
        fire = np.zeros(y.size, dtype=bool)
        for ci in combo:
            fire |= conds[ci][2]
        t = float(np.mean(fire[y]))
        if t < 0.60:
            continue
        fp = float(np.mean(fire[~y]))
        if best is None or fp < best[0]:
            best = (fp, t, combo)
    if best:
        desc = " OR ".join(f"{conds[ci][0]}>{conds[ci][1]}" for ci in best[2])
        results.append((k, desc, best[1], best[0]))
        print(f"\nbest {k}-arm rule >=60%: {desc}\n  catch {best[1]*100:.1f}% | false {best[0]*100:.1f}%")

# holdout check of the best rule: pick thresholds on 2 files, evaluate on the third
if results:
    k, desc, _, _ = results[-1]
    print("\nper-file holdout of the best rule:")
    for hold in np.unique(grp):
        tr = grp != hold
        best = None
        for combo in itertools.combinations(range(len(conds)), k):
            fire = np.zeros(y.size, dtype=bool)
            for ci in combo:
                fire |= conds[ci][2]
            t_tr = np.mean(fire[tr & y]) if (tr & y).any() else 0
            if t_tr < 0.60:
                continue
            fp_tr = np.mean(fire[tr & ~y])
            if best is None or fp_tr < best[0]:
                best = (fp_tr, combo)
        if best:
            fire = np.zeros(y.size, dtype=bool)
            for ci in best[1]:
                fire |= conds[ci][2]
            te = ~tr
            print(f"  train w/o {hold}: rule = "
                  + " OR ".join(f"{conds[ci][0]}>{conds[ci][1]}" for ci in best[1])
                  + f" | test catch {np.mean(fire[te & y])*100:.1f}% false {np.mean(fire[te & ~y])*100:.1f}%")

# ---- figure: rule points on the composite ROC ----
fig, ax = plt.subplots(figsize=(9, 6.5))
order = np.argsort(fpr)
ax.plot(fpr[order] * 100, tpr[order] * 100, color="gray", lw=2, label="ML upper bound (GBDT, file-CV)")
j = names.index("all_max")
c_all = np.unique(X[:, j])
ax.plot([np.mean(X[~y, j] > c) * 100 for c in c_all], [np.mean(X[y, j] > c) * 100 for c in c_all],
        color="lightsteelblue", lw=1.5, ls="--", label="single statistic: all-ch max")
for k, desc, t, fp in results:
    ax.scatter(fp * 100, t * 100, s=90, zorder=5, label=f"{k}-arm rule: {desc}")
ax.axhline(60, color="k", ls=":", lw=1)
ax.set_xlabel("false-pause rate on clean placements [%]")
ax.set_ylabel("polluted placements caught [%]")
ax.set_title("Simple OR-rules reaching >= 60% catch (hit-level +-10-tick truth)")
ax.grid(alpha=0.3)
ax.legend(fontsize=8, loc="lower right")
plt.tight_layout()
fig.savefig(os.path.join(HERE, "overflow_rule60.png"), dpi=140)
print("\nwrote overflow_rule60.png")
