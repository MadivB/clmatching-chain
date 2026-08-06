"""2D density: truth-free quality metric vs MC-truth mis-classified energy, per TPC.

Left  : metric vs mis-classified energy FRACTION (%)
Right : metric vs ABSOLUTE mis-classified energy (MeV, symlog)
Both y-axes use MC truth (E * wrong-fraction).  Hexbin log-count density (a plain
scatter is misleading — most of the 2320 TPCs pile up at ~0 wrong energy), with an
equal-count binned-median trend + 25-75% IQR band and the Spearman rank correlation.
"""
import json, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

HERE = os.path.abspath(os.path.dirname(__file__))
R = json.load(open(os.path.join(HERE, "quality_final.json")))
x = np.array([r["chi2s"] for r in R])            # truth-free metric (scaled chi2/channel)
wf = np.array([r["wf"] for r in R])              # truth wrong-energy fraction
E = np.array([r["E"] for r in R])                # TPC energy (MeV)
we = E * wf                                       # absolute mis-classified energy (MeV)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def trend(ax, xv, yv, nbin=12):
    """equal-count bins on x: median + 25-75% band."""
    order = np.argsort(xv); xs = xv[order]; ys = yv[order]
    edges = np.interp(np.linspace(0, len(xs), nbin + 1), np.arange(len(xs) + 1),
                      np.concatenate([xs[:1], xs]))
    cx, med, lo, hi = [], [], [], []
    for i in range(nbin):
        sel = (xv >= edges[i]) & (xv <= edges[i + 1]) if i == nbin - 1 else (xv >= edges[i]) & (xv < edges[i + 1])
        if sel.sum() < 5:
            continue
        cx.append(np.median(xv[sel])); med.append(np.median(yv[sel]))
        lo.append(np.percentile(yv[sel], 25)); hi.append(np.percentile(yv[sel], 75))
    cx, med, lo, hi = map(np.array, (cx, med, lo, hi))
    ax.fill_between(cx, lo, hi, color="k", alpha=0.12, zorder=4, label="25-75% band")
    ax.plot(cx, med, "o-", color="k", lw=2.2, ms=5, zorder=5, label="binned median")


fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6.2))

# ---- LEFT: fraction ----
hb = axL.hexbin(x, 100 * wf, gridsize=42, cmap="viridis", norm=LogNorm(vmin=1),
                extent=(0, 0.6, 0, 100), mincnt=1)
trend(axL, x, 100 * wf)
axL.axhline(20, color="tab:red", ls=":", lw=1.1); axL.axhline(5, color="w", ls=":", lw=1.1)
axL.text(0.585, 21, "BAD >20%", color="tab:red", ha="right", va="bottom", fontsize=9)
axL.text(0.585, 6, "GOOD <5%", color="0.85", ha="right", va="bottom", fontsize=9)
axL.set_xlim(0, 0.6); axL.set_ylim(0, 100)
axL.set_xlabel("truth-free quality metric   (scaled  chi2 / channel)")
axL.set_ylabel("MC-truth mis-classified energy  [% of TPC]")
axL.set_title(f"(a)  metric vs mis-classified FRACTION\nSpearman rho = {spearman(x, wf):+.3f}   (n={len(R)} TPCs)")
axL.legend(loc="upper left", fontsize=8, framealpha=.9)
cb1 = fig.colorbar(hb, ax=axL, pad=.01); cb1.set_label("TPCs per cell")

# ---- RIGHT: absolute MeV (symlog handles the wrong-energy=0 pile) ----
hb2 = axR.hexbin(x, np.maximum(we, 0.5), gridsize=42, cmap="magma", norm=LogNorm(vmin=1),
                 yscale="log", extent=(0, 0.6, np.log10(0.5), np.log10(1500)), mincnt=1)
# trend on log10 then transform back
def trend_log(ax, xv, yv, nbin=12):
    yl = np.log10(np.maximum(yv, 0.5))
    order = np.argsort(xv)
    edges = np.interp(np.linspace(0, len(xv), nbin + 1), np.arange(len(xv) + 1),
                      np.concatenate([xv[order][:1], xv[order]]))
    cx, med, lo, hi = [], [], [], []
    for i in range(nbin):
        sel = (xv >= edges[i]) & (xv <= edges[i + 1]) if i == nbin - 1 else (xv >= edges[i]) & (xv < edges[i + 1])
        if sel.sum() < 5:
            continue
        cx.append(np.median(xv[sel])); med.append(10 ** np.median(yl[sel]))
        lo.append(10 ** np.percentile(yl[sel], 25)); hi.append(10 ** np.percentile(yl[sel], 75))
    cx, med, lo, hi = map(np.array, (cx, med, lo, hi))
    ax.fill_between(cx, lo, hi, color="w", alpha=0.18, zorder=4, label="25-75% band")
    ax.plot(cx, med, "o-", color="w", lw=2.2, ms=5, zorder=5, label="binned median")
trend_log(axR, x, we)
axR.set_xlim(0, 0.6); axR.set_ylim(0.5, 1500)
axR.set_xlabel("truth-free quality metric   (scaled  chi2 / channel)")
axR.set_ylabel("MC-truth mis-classified energy  [MeV]")
axR.set_title(f"(b)  metric vs ABSOLUTE mis-classified energy\nSpearman rho = {spearman(x, we):+.3f}   (n={len(R)} TPCs)")
axR.legend(loc="upper left", fontsize=8, framealpha=.9)
cb2 = fig.colorbar(hb2, ax=axR, pad=.01); cb2.set_label("TPCs per cell")

fig.suptitle("Per-TPC quality metric vs mis-classified energy  (v0.3, truth-free x  vs  MC-truth y)", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = os.path.join(HERE, "deck_figs", "metric_vs_misclass.png")
fig.savefig(out, dpi=120); print("wrote", out)
print(f"n={len(R)}  Spearman(metric,frac)={spearman(x,wf):+.3f}  Spearman(metric,absMeV)={spearman(x,we):+.3f}")
print(f"total mis-classified E = {we.sum():.0f} MeV over {len(R)} TPCs; median wrong-frac={np.median(wf)*100:.2f}%")
