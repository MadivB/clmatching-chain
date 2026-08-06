"""Definitive quality-metric figure from quality_final.json (no cube reload)."""
import json, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.abspath(os.path.dirname(__file__))
R = json.load(open(os.path.join(HERE, "quality_final.json")))
wf = np.array([r["wf"] for r in R]); m = np.array([r["chi2s"] for r in R]); E = np.array([r["E"] for r in R])
bad = wf > 0.20; good = wf < 0.05; marg = (~bad) & (~good)

def auc(sc, y):
    pos = sc[y == 1]; neg = sc[y == 0]
    o = np.argsort(np.concatenate([pos, neg])); rk = np.empty_like(o, float); rk[o] = np.arange(1, len(o) + 1)
    return (rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

fig, ax = plt.subplots(2, 2, figsize=(13, 9))

# (a) distributions
b = np.linspace(0, 0.6, 45)
ax[0, 0].hist(m[good], b, density=True, alpha=.55, color="tab:green", label=f"GOOD <5%  (n={good.sum()})")
ax[0, 0].hist(m[marg], b, density=True, alpha=.45, color="tab:orange", label=f"MARGINAL 5-20%  (n={marg.sum()})")
ax[0, 0].hist(m[bad], b, density=True, alpha=.55, color="tab:red", label=f"BAD >20%  (n={bad.sum()})")
ax[0, 0].axvline(np.median(m[good]), color="tab:green", ls="--"); ax[0, 0].axvline(np.median(m[bad]), color="tab:red", ls="--")
ax[0, 0].set_xlabel("scaled  chi2 / channel  (truth-free)"); ax[0, 0].set_ylabel("density")
ax[0, 0].set_title("(a) Metric distribution by truth wrong-energy fraction"); ax[0, 0].legend(fontsize=8)

# (b) ROC BAD vs GOOD
sel = bad | good; y = bad[sel].astype(int); s = m[sel]
ts = np.unique(s); tpr = []; fpr = []
for t in ts:
    fl = s >= t; tpr.append((fl & (y == 1)).sum() / max((y == 1).sum(), 1)); fpr.append((fl & (y == 0)).sum() / max((y == 0).sum(), 1))
ax[0, 1].plot(fpr, tpr, color="tab:blue"); ax[0, 1].plot([0, 1], [0, 1], "k:", lw=.8)
ax[0, 1].set_xlabel("false-positive rate (GOOD flagged)"); ax[0, 1].set_ylabel("true-positive rate (BAD caught)")
ax[0, 1].set_title(f"(b) ROC  BAD>20% vs GOOD<5%   AUC = {auc(s, y):.3f}")

# (c) continuous scatter metric vs wrong-fraction
ax[1, 0].scatter(m, 100 * wf, s=8, alpha=.25, color="tab:purple", edgecolors="none")
# binned median
qs = np.linspace(0, 1, 11); edges = np.quantile(m, qs); cx = []; cy = []
for i in range(10):
    sel2 = (m >= edges[i]) & (m < edges[i + 1]) if i < 9 else (m >= edges[i]) & (m <= edges[i + 1])
    if sel2.sum(): cx.append(np.median(m[sel2])); cy.append(100 * np.median(wf[sel2]))
ax[1, 0].plot(cx, cy, "o-", color="black", lw=2, ms=5, label="binned median")
rx = np.argsort(np.argsort(m)); ry = np.argsort(np.argsort(wf)); rho = np.corrcoef(rx, ry)[0, 1]
ax[1, 0].axhline(20, color="tab:red", ls=":", lw=.8); ax[1, 0].axhline(5, color="tab:green", ls=":", lw=.8)
ax[1, 0].set_xlabel("scaled chi2 / channel (truth-free)"); ax[1, 0].set_ylabel("truth wrong-energy %")
ax[1, 0].set_xlim(0, 0.6); ax[1, 0].set_ylim(-2, 100)
ax[1, 0].set_title(f"(c) Continuous:  metric vs wrong-frac   Spearman rho = {rho:+.3f}"); ax[1, 0].legend(fontsize=8)

# (d) efficiency-estimator curves vs cut
cuts = np.linspace(0.04, 0.35, 40); rec = []; con = []; pur = []; efl = []
for c in cuts:
    fl = m >= c
    rec.append((fl & bad).sum() / max(bad.sum(), 1)); con.append((fl & good).sum() / max(good.sum(), 1))
    pur.append((fl & (wf > 0.05)).sum() / max(fl.sum(), 1)); efl.append(E[fl].sum() / E.sum())
ax[1, 1].plot(cuts, rec, color="tab:red", label="BAD recall (>20% caught)")
ax[1, 1].plot(cuts, con, color="tab:green", label="GOOD contamination")
ax[1, 1].plot(cuts, pur, color="tab:blue", label="flagged purity (>5% wrong)")
ax[1, 1].plot(cuts, efl, color="tab:gray", ls="--", label="energy fraction flagged")
ax[1, 1].axvline(0.15, color="k", ls=":", lw=.8)
ax[1, 1].set_xlabel("cut on scaled chi2 / channel"); ax[1, 1].set_ylabel("fraction")
ax[1, 1].set_title("(d) On-data triage: pick a cut on the trade-off"); ax[1, 1].legend(fontsize=8); ax[1, 1].set_ylim(0, 1.02)

fig.suptitle("Truth-free per-TPC quality metric  (scaled predicted-vs-actual light chi2/channel, v0.3, 2320 TPCs)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = os.path.join(HERE, "deck_figs", "quality_final.png")
fig.savefig(out, dpi=115); print("wrote", out)
