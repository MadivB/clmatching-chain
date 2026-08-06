"""Joint optimization of the gated single-channel overflow rule.

Statistic S(A, X) = max over channels with actual >= A and (pred-actual) >= X
of (pred-actual)/sqrt(max(actual,100)); trigger = S > cut.
Scans (scope, A, X, cut); reports Pareto frontier, best-at-budget table,
per-file holdout of the chosen optimum, and a re-tune of the 3-arm 60% rule
with the gated arm.
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
PEAK = 105
tmpl = np.load(os.path.join(BASE, "assets", "avg_pulse.npy")).astype(np.float64)
tmpl = np.maximum(tmpl / tmpl[PEAK], 0.0)

rows = []
for fp in sorted(glob.glob(os.path.join(HERE, "overflow_study_out", "rows_worker*.jsonl"))):
    with open(fp) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
by_event = {}
for r in rows:
    by_event.setdefault((r["file"], r["event"]), []).append(r)

A_GATES = [0, 250, 500, 1000, 2000, 4000]
X_GATES = [0, 250, 500, 1000, 2000, 4000]
CUTS = [5, 7.5, 10, 12.5, 15, 20, 25, 30, 40, 50, 75]

# per placement: channel arrays needed for any gate combo
recs = []
for (fname, ev_id), ev_rows in by_event.items():
    ftag = fname.split(".")[-3]
    hf = os.path.join(HERE, "overflow_study_out", "hits", f"{ftag}_ev{ev_id}.npz")
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
        ovs, acts, exs, sups, adj = [], [], [], [], -100.0
        cov = 0.0
        for td in r["tpc_channels"]:
            o = np.asarray(td["own"], float)
            b = np.asarray(td["base"], float)
            a = np.asarray(td["actual"], float)
            ex = o + b - a
            ov = ex / np.sqrt(np.maximum(a, 100.0))
            fin = np.ones(120, bool)
            if td["veto"]:
                fin[np.asarray(td["veto"], int)] = False
            supm = np.zeros(120, bool)
            supm[np.asarray(td["support"], int)] = True
            ovs.append(ov[fin]); acts.append(a[fin]); exs.append(ex[fin]); sups.append(supm[fin])
            for c in td["support"]:
                lo, hi = (0, 59) if c < 60 else (60, 119)
                if c - 1 >= lo and c + 1 <= hi and fin[c - 1] and fin[c] and fin[c + 1]:
                    adj = max(adj, float(np.min(ov[[c - 1, c, c + 1]])))
            cov = max(cov, float(np.sum((o + b)[fin]) / max(np.sum(a[fin]), 1.0)))
        recs.append({
            "bad": pol > 5.0, "ftag": ftag, "cov": cov, "adj3": adj,
            "ov": np.concatenate(ovs), "act": np.concatenate(acts),
            "ex": np.concatenate(exs), "sup": np.concatenate(sups),
        })

bad = np.array([x["bad"] for x in recs])
grp = np.array([x["ftag"] for x in recs])
print(f"{len(recs)} placements, polluted {int(bad.sum())} ({bad.mean():.1%})")


def stat(rec, A, X, scope):
    sel = (rec["act"] >= A) & (rec["ex"] >= X)
    if scope == "sup":
        sel &= rec["sup"]
    return float(np.max(rec["ov"][sel])) if sel.any() else -100.0


# precompute S for every gate combo (all placements)
S = {}
for scope in ("all", "sup"):
    for A in A_GATES:
        for X in X_GATES:
            S[(scope, A, X)] = np.array([stat(rec, A, X, scope) for rec in recs])

# ---- best catch at fixed false-pause budgets ----
BUDGETS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25]
print("\n=== best gated rule per false-pause budget (scanning scope x A x X x cut) ===")
print(f"{'budget':>7} | {'catch':>6} | {'config':<44}")
best_by_budget = {}
for bud in BUDGETS:
    best = None
    for key, v in S.items():
        for c in CUTS:
            f = float(np.mean(v[~bad] > c))
            if f > bud:
                continue
            t = float(np.mean(v[bad] > c))
            if best is None or t > best[0]:
                best = (t, f, key, c)
    best_by_budget[bud] = best
    t, f, (scope, A, X), c = best
    print(f"{bud*100:>6.0f}% | {t*100:>5.1f}% | scope={scope} actual>={A} excess>={X} cut>{c}  (fpr {f*100:.1f}%)")

# ---- Pareto frontier figure ----
fig, ax = plt.subplots(figsize=(9.5, 6.5))
pts = []
for key, v in S.items():
    for c in CUTS:
        pts.append((float(np.mean(v[~bad] > c)), float(np.mean(v[bad] > c)), key, c))
pts.sort()
pareto = []
best_t = -1
for f, t, key, c in pts:
    if t > best_t:
        pareto.append((f, t, key, c))
        best_t = t
ax.scatter([p[0] * 100 for p in pts], [p[1] * 100 for p in pts], s=8, color="lightgray", label="all scanned configs")
ax.plot([p[0] * 100 for p in pareto], [p[1] * 100 for p in pareto], "o-", color="crimson", ms=4,
        lw=1.5, label="Pareto frontier (gated single-channel)")
v0 = S[("all", 0, 0)]
cs = np.unique(v0)
ax.plot([np.mean(v0[~bad] > c) * 100 for c in cs], [np.mean(v0[bad] > c) * 100 for c in cs],
        color="steelblue", ls="--", lw=1.5, label="ungated all-channel max")
for lbl, A, X, c, mk in (("user: actual>=1000, 20s", 1000, 0, 20, "s"),
                         ("excess>=1000, 20s", 0, 1000, 20, "^")):
    v = S[("all", A, X)]
    ax.scatter(np.mean(v[~bad] > c) * 100, np.mean(v[bad] > c) * 100, marker=mk, s=90, zorder=5, label=lbl)
ax.set_xlabel("false-pause on clean placements [%]")
ax.set_ylabel("polluted placements caught [%]")
ax.set_xlim(-1, 45)
ax.set_title("Gate scan: actual-gate x excess-gate x sigma cut (hit-level truth)")
ax.grid(alpha=0.3)
ax.legend(fontsize=8.5)
plt.tight_layout()
fig.savefig(os.path.join(HERE, "overflow_gate_scan.png"), dpi=140)
print("\nwrote overflow_gate_scan.png")

# print Pareto configs in the useful 5-25% band
print("\nPareto configs with false-pause in [3%, 27%]:")
for f, t, (scope, A, X), c in pareto:
    if 0.03 <= f <= 0.27:
        print(f"  catch {t*100:5.1f}% @ {f*100:5.1f}% | scope={scope} actual>={A:>4} excess>={X:>4} cut>{c}")

# ---- holdout of the overall best-at-20% config ----
t, f, key, c = best_by_budget[0.20]
scope, A, X = key
print(f"\nper-file holdout of budget-20% optimum (scope={scope}, A={A}, X={X}, cut={c}):")
for hold in np.unique(grp):
    te = grp == hold
    v = S[key]
    print(f"  test {hold}: catch {np.mean(v[te & bad] > c)*100:.1f}% | false {np.mean(v[te & ~bad] > c)*100:.2f}%")

# ---- re-tune the 3-arm 60% rule with the gated arm ----
cov = np.array([x["cov"] for x in recs])
adj3 = np.array([x["adj3"] for x in recs])
print("\n=== 3-arm rule re-tune: cov<cc OR gated-arm>cut OR adj3>cj (target >=60%) ===")
best = None
for key in (("all", 0, 0), ("all", 1000, 0), ("all", 0, 1000), ("all", 1000, 1000),
            ("all", 2000, 2000), ("all", 500, 500)):
    v = S[key]
    for cc in np.arange(0.30, 0.75, 0.05):
        for c in CUTS:
            for cj in (8, 10, 12, 15, 20, 1e9):
                fire = (cov < cc) | (v > c) | (adj3 > cj)
                t = float(np.mean(fire[bad]))
                if t < 0.60:
                    continue
                f = float(np.mean(fire[~bad]))
                if best is None or f < best[0]:
                    best = (f, t, key, cc, c, cj)
f, t, (scope, A, X), cc, c, cj = best
print(f"BEST: cov<{cc:.2f} OR [actual>={A} & excess>={X} & ov>{c}] OR adj3>{cj:g}")
print(f"  catch {t*100:.1f}% | false-pause {f*100:.1f}%  (previous best rule: 60.1% @ 24.0%)")
for hold in np.unique(grp):
    te = grp == hold
    v = S[(scope, A, X)]
    fire = (cov < cc) | (v > c) | (adj3 > cj)
    print(f"  holdout {hold}: catch {np.mean(fire[te & bad])*100:.1f}% | false {np.mean(fire[te & ~bad])*100:.2f}%")
