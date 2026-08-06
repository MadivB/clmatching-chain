"""Definitive figure: chi2_flash + displaced-light (conservation) + their combo,
vs MC-truth mis-classified energy.  Shows the specificity win and the complementarity.
"""
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

HERE = os.path.abspath(os.path.dirname(__file__))
d = json.load(open(os.path.join(HERE, "eval_final_out.json")))
chi2 = np.array(d["chi2"]); disp = np.array(d["disp"]); cmax = np.array(d["combo_max"])
cmean = np.array(d["combo_mean"]); WF = np.array(d["wf"]); EE = np.array(d["E"]); WE = WF * EE
bad = WF > 0.20; good = WF < 0.05; sel = bad | good; y = bad[sel].astype(int)


def auc(sc, yy):
    pos = sc[yy == 1]; neg = sc[yy == 0]
    o = np.argsort(np.concatenate([pos, neg])); rk = np.empty_like(o, float); rk[o] = np.arange(1, len(o) + 1)
    return (rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
def spr(a, b): return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])
def roc(s, yy):
    P = (yy == 1).sum(); N = (yy == 0).sum(); ts = np.unique(s); tp = []; fp = []
    for t in ts:
        f = s >= t; tp.append((f & (yy == 1)).sum() / P); fp.append((f & (yy == 0)).sum() / N)
    return np.array(fp), np.array(tp)
def fpr_curve(s, yy):
    bpos = s[yy == 1]; gneg = s[yy == 0]; recs = np.linspace(0.5, 1.0, 40); out = []
    for rr in recs:
        thr = np.quantile(bpos, 1 - rr); out.append((gneg >= thr).mean())
    return recs, np.array(out)

fig, ax = plt.subplots(2, 2, figsize=(14, 10))

# (a) combo(mean) vs mis-classified fraction
xm = cmean; XX = (1, 100)
hb = ax[0, 0].hexbin(xm, 100 * WF, gridsize=40, cmap="viridis", norm=LogNorm(vmin=1),
                     extent=(0, 1, 0, 100), mincnt=1)
order = np.argsort(xm); cx, md = [], []
for i in range(14):
    lo, hi = i / 14, (i + 1) / 14; s = (xm >= lo) & (xm < hi)
    if s.sum() >= 5: cx.append(xm[s].mean()); md.append(100 * np.median(WF[s]))
ax[0, 0].plot(cx, md, "ko-", lw=2, ms=4, label="binned median")
ax[0, 0].axhline(20, color="tab:red", ls=":", lw=1); ax[0, 0].axhline(5, color="w", ls=":", lw=1)
ax[0, 0].set_xlim(0, 1); ax[0, 0].set_ylim(0, 100); ax[0, 0].legend(loc="upper left", fontsize=8)
ax[0, 0].set_xlabel("combined metric  mean(z_chi2, z_displaced)"); ax[0, 0].set_ylabel("mis-classified energy [% of TPC]")
ax[0, 0].set_title(f"(a) combined metric vs mis-classified fraction   Spearman={spr(xm, WF):+.3f}")
fig.colorbar(hb, ax=ax[0, 0], pad=.01).set_label("TPCs/cell")

# (b) ROC
for s, lab, col in [(cmax, "combo max(z)", "black"), (cmean, "combo mean(z)", "tab:purple"),
                    (disp, "displaced-light (k=3sigma)", "tab:blue"),
                    (chi2, "chi2_flash", "tab:orange")]:
    f, t = roc(s[sel], y); ax[0, 1].plot(f, t, color=col, lw=1.8, label=f"{lab}  (AUC {auc(s[sel], y):.3f})")
ax[0, 1].plot([0, 1], [0, 1], "k:", lw=.8); ax[0, 1].set_xlabel("false-positive rate (GOOD flagged)")
ax[0, 1].set_ylabel("true-positive rate (BAD caught)"); ax[0, 1].set_title("(b) ROC: BAD>20% vs GOOD<5%")
ax[0, 1].legend(fontsize=8, loc="lower right")

# (c) specificity: FPR at matched recall (lower = better/purer)
for s, lab, col in [(chi2, "chi2_flash", "tab:orange"), (disp, "displaced-light", "tab:blue"),
                    (cmax, "combo max(z)", "black")]:
    rr, fp = fpr_curve(s[sel], y); ax[1, 0].plot(100 * rr, 100 * fp, color=col, lw=2, label=lab)
ax[1, 0].set_xlabel("BAD recall [%]"); ax[1, 0].set_ylabel("GOOD false-positive rate [%]")
ax[1, 0].set_title("(c) Purity: displaced-light fires on almost no GOOD at high purity")
ax[1, 0].legend(fontsize=9, loc="upper left"); ax[1, 0].set_ylim(0, 55)

# (d) complementarity: z_chi2 vs z_displaced, BAD vs GOOD
zc = np.argsort(np.argsort(chi2)).astype(float) / len(chi2)
zd = np.argsort(np.argsort(disp)).astype(float) / len(disp)
ax[1, 1].scatter(zc[good], zd[good], s=5, c="0.75", alpha=.4, label="GOOD (<5%)", edgecolors="none")
marg = (~bad) & (~good)
ax[1, 1].scatter(zc[marg], zd[marg], s=6, c="tab:orange", alpha=.35, label="marginal (5-20%)", edgecolors="none")
ax[1, 1].scatter(zc[bad], zd[bad], s=45, c="tab:red", edgecolors="k", lw=.5, label="BAD (>20%)", zorder=5)
ax[1, 1].plot([0, 1], [0, 1], "k:", lw=.6)
ax[1, 1].set_xlabel("z(chi2_flash)  [rank]"); ax[1, 1].set_ylabel("z(displaced-light)  [rank]")
ax[1, 1].set_title("(d) Complementarity: BAD sit high on EITHER axis -> max(z) catches both")
ax[1, 1].legend(fontsize=8, loc="lower right")

fig.suptitle("Charge-light quality metric: chi2 at the t0 list + displaced-light (conservation)  vs  MC-truth mis-classified energy  (v0.3, 2320 TPCs)", fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(HERE, "deck_figs", "combined_metric_vs_misclass.png")
fig.savefig(out, dpi=118); print("wrote", out)
print(f"combo max AUC={auc(cmax[sel], y):.3f}  mean AUC={auc(cmean[sel], y):.3f}  rhoFrac(mean)={spr(cmean, WF):+.3f}")
