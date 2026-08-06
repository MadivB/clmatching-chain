"""Round 2: reach >=60% catch with the lowest false-pause, still simple.

Adds two composite features:
  mismatch = min(overflow_max, underflow_max)  -- light PATTERN disagreement
             (some channels over, others under, at the same tick)
  cov      = sum(pred)/sum(actual) at the probe tick (placement coverage;
             makes underflow meaningful only once the slot should be explained)
Search: greedy OR of up to 3 arms, each arm = single threshold or AND-pair.
Also an interpretable depth-3 decision tree. Everything validated per-file.
"""
import glob
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
            w[w < 0.02] = 0.0
            pol = max(pol, float(np.sum(h_e[m] * w)))

        f = {"adj3": -100.0, "n5": 0, "base_max": -100.0, "all_max": -100.0,
             "under_max": -100.0, "mismatch": -100.0, "cov": 0.0, "frac_own": -100.0}
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
            for c in sup:
                lo, hi = (0, 59) if c < 60 else (60, 119)
                if c - 1 >= lo and c + 1 <= hi and fin[c - 1] and fin[c] and fin[c + 1]:
                    f["adj3"] = max(f["adj3"], float(np.min(ov[[c - 1, c, c + 1]])))
            omax = float(np.max(ov[fin]))
            umax = float(np.max((-ov)[fin]))
            f["all_max"] = max(f["all_max"], omax)
            f["under_max"] = max(f["under_max"], umax)
            f["mismatch"] = max(f["mismatch"], min(omax, umax))
            f["n5"] = max(f["n5"], int(np.sum(ov[fin] > 5.0)))
            f["base_max"] = max(f["base_max"], float(np.max(((b - a) / sig)[fin])))
            f["cov"] = max(f["cov"], float(np.sum((o + b)[fin]) / max(np.sum(a[fin]), 1.0)))
            act = fin & (o >= max(0.2 * float(o.max()), 50.0))
            if act.any():
                f["frac_own"] = max(f["frac_own"], float(np.max((o + b - a)[act] / np.maximum((o + b)[act], 200.0))))
        feats.append(f)
        labels.append(pol > 5.0)
        groups.append(ftag)

y = np.asarray(labels)
grp = np.asarray(groups)
names = ["adj3", "n5", "base_max", "all_max", "under_max", "mismatch", "cov", "frac_own"]
X = np.array([[f[n] for n in names] for f in feats])
print(f"{y.size} placements, polluted {int(y.sum())} ({y.mean():.1%})")

# ---- condition pool: singles + physically sensible AND pairs ----
TH = {
    "adj3": [3, 5, 8, 12], "n5": [10, 20, 30], "base_max": [3, 5, 10, 20],
    "all_max": [5, 10, 15, 20, 30], "under_max": [10, 20, 50, 100],
    "mismatch": [3, 5, 8, 12, 20], "cov": [0.3, 0.5, 0.8], "frac_own": [0.5, 0.7, 0.9],
}
singles = []
for n, ths in TH.items():
    j = names.index(n)
    for t in ths:
        singles.append((f"{n}>{t}", X[:, j] > t))
AND_PAIRS = [("all_max", "under_max"), ("all_max", "cov"), ("under_max", "cov"),
             ("adj3", "under_max"), ("mismatch", "cov"), ("all_max", "base_max"),
             ("frac_own", "under_max"), ("adj3", "n5")]
arms = list(singles)
for na, nb in AND_PAIRS:
    ja, jb = names.index(na), names.index(nb)
    for ta in TH[na]:
        for tb in TH[nb]:
            arms.append((f"({na}>{ta} & {nb}>{tb})", (X[:, ja] > ta) & (X[:, jb] > tb)))
print(f"arm pool: {len(arms)}")


def greedy(sel_mask, target=0.60):
    """greedy OR-of-arms on the selected subset; returns list of arm indices."""
    chosen, fire = [], np.zeros(sel_mask.sum(), dtype=bool)
    ys = y[sel_mask]
    A = [a[1][sel_mask] for a in arms]
    while np.mean(fire[ys]) < target and len(chosen) < 3:
        best = None
        for i, av in enumerate(A):
            if i in chosen:
                continue
            nf = fire | av
            t, fp = np.mean(nf[ys]), np.mean(nf[~ys])
            gain_t = t - np.mean(fire[ys])
            gain_f = fp - np.mean(fire[~ys])
            if gain_t <= 0:
                continue
            score = gain_t / (gain_f + 0.01)
            # prefer arms that get us to target with low added FPR
            if best is None or score > best[0]:
                best = (score, i, nf)
        if best is None:
            break
        chosen.append(best[1])
        fire = best[2]
    return chosen, fire


# full-data rule (for display) + per-file holdout (for honesty)
chosen, fire = greedy(np.ones(y.size, dtype=bool))
desc = " OR ".join(arms[i][0] for i in chosen)
print(f"\ngreedy rule (all data): {desc}")
print(f"  catch {np.mean(fire[y])*100:.1f}% | false-pause {np.mean(fire[~y])*100:.1f}%")

print("\nper-file holdout (rule refit on the other two files):")
hold_t, hold_f = [], []
for hold in np.unique(grp):
    tr = grp != hold
    ch, _ = greedy(tr)
    fire_all = np.zeros(y.size, dtype=bool)
    for i in ch:
        fire_all |= arms[i][1]
    te = ~tr
    t, fp = np.mean(fire_all[te & y]), np.mean(fire_all[te & ~y])
    hold_t.append(t); hold_f.append(fp)
    print(f"  test {hold}: {' OR '.join(arms[i][0] for i in ch)}")
    print(f"    catch {t*100:.1f}% | false {fp*100:.1f}%")
print(f"holdout mean: catch {np.mean(hold_t)*100:.1f}% | false {np.mean(hold_f)*100:.1f}%")

# ---- interpretable depth-3 tree ----
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import LeaveOneGroupOut

probs = np.zeros(y.size)
for tr, te in LeaveOneGroupOut().split(X, y, grp):
    t3 = DecisionTreeClassifier(max_depth=3, min_samples_leaf=60, class_weight="balanced", random_state=0)
    t3.fit(X[tr], y[tr])
    probs[te] = t3.predict_proba(X[te])[:, 1]
cuts = np.unique(probs)
tpr = np.array([np.mean(probs[y] >= c) for c in cuts])
fpr = np.array([np.mean(probs[~y] >= c) for c in cuts])
ok = tpr >= 0.60
if ok.any():
    i = int(np.argmin(fpr[ok]))
    print(f"\ndepth-3 tree (file-CV): TPR={tpr[ok][i]*100:.1f}% at FPR={fpr[ok][i]*100:.1f}%")
t3 = DecisionTreeClassifier(max_depth=3, min_samples_leaf=60, class_weight="balanced", random_state=0).fit(X, y)
print("\ntree rules (fit on all data, for reading):")
print(export_text(t3, feature_names=names, max_depth=3))
