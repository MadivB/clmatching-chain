"""Definitive figure for the winning metric: chi2_flash (significance-weighted chi2
evaluated only at the TPC's t0 list) vs MC-truth mis-classified energy, plus a ROC
comparison against the whole-TPC chi2 and the best dot-product / structural metrics.
"""
import os, pickle, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

HERE = os.path.abspath(os.path.dirname(__file__))
GAIN = 200.0
cache = pickle.load(open(os.path.join(HERE, "lightcache.pkl"), "rb"))

snr, cell = [], []
for r in cache:
    A_all = r["Fa"].astype(np.float64); P_all = r["Fp"].astype(np.float64) * r["alpha"]
    if len(A_all) == 0:
        snr.append(0.0); cell.append(0.0); continue
    ampA = A_all.sum(1); red = []; fchi = []; fdof = []
    for A, P in zip(A_all, P_all):
        v = GAIN * np.maximum(A, 100.0); c = float(((P - A) ** 2 / v).sum())
        fchi.append(c); fdof.append(len(A)); red.append(c / len(A))
    red = np.array(red); w = np.sqrt(np.maximum(ampA, 0) / GAIN)
    snr.append(float((red * w).sum() / w.sum()) if w.sum() > 0 else 0.0)
    cell.append(sum(fchi) / sum(fdof))
snr = np.array(snr); cell = np.array(cell)
WF = np.array([r["wf"] for r in cache]); EE = np.array([r["E"] for r in cache]); WE = WF * EE
bad = WF > 0.20; good = WF < 0.05; sel = bad | good; y = bad[sel].astype(int)

Q = json.load(open(os.path.join(HERE, "quality_final.json"))); chi2s = np.array([q["chi2s"] for q in Q])
EM = json.load(open(os.path.join(HERE, "eval_metrics_out.json")))["metrics"]
cos_snr = np.array(EM["cos_snr"]); gcos = np.array(EM["global_cos"])


def spr(a, b): return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])
def roc(score, yy):
    ts = np.unique(score); tpr = []; fpr = []
    P = (yy == 1).sum(); N = (yy == 0).sum()
    for t in ts:
        fl = score >= t; tpr.append((fl & (yy == 1)).sum() / P); fpr.append((fl & (yy == 0)).sum() / N)
    return np.array(fpr), np.array(tpr)
def auc(sc, yy):
    pos = sc[yy == 1]; neg = sc[yy == 0]
    o = np.argsort(np.concatenate([pos, neg])); rk = np.empty_like(o, float); rk[o] = np.arange(1, len(o) + 1)
    return (rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def trend(ax, xv, yv, nbin=12, log=False):
    order = np.argsort(xv)
    edges = np.interp(np.linspace(0, len(xv), nbin + 1), np.arange(len(xv) + 1),
                      np.concatenate([xv[order][:1], xv[order]]))
    Y = np.log10(np.maximum(yv, 0.5)) if log else yv
    cx, md, lo, hi = [], [], [], []
    for i in range(nbin):
        s = (xv >= edges[i]) & (xv <= edges[i + 1]) if i == nbin - 1 else (xv >= edges[i]) & (xv < edges[i + 1])
        if s.sum() < 5: continue
        cx.append(np.median(xv[s])); f = (lambda z: 10 ** z) if log else (lambda z: z)
        md.append(f(np.median(Y[s]))); lo.append(f(np.percentile(Y[s], 25))); hi.append(f(np.percentile(Y[s], 75)))
    col = "w" if log else "k"
    ax.fill_between(cx, lo, hi, color=col, alpha=.15, zorder=4)
    ax.plot(cx, md, "o-", color=col, lw=2.2, ms=5, zorder=5, label="binned median")


fig, ax = plt.subplots(2, 2, figsize=(14, 10))
XMAX = np.percentile(snr, 99.5)

# (a) vs fraction
hb = ax[0, 0].hexbin(snr, 100 * WF, gridsize=42, cmap="viridis", norm=LogNorm(vmin=1),
                     extent=(0, XMAX, 0, 100), mincnt=1)
trend(ax[0, 0], snr, 100 * WF)
ax[0, 0].axhline(20, color="tab:red", ls=":", lw=1); ax[0, 0].axhline(5, color="w", ls=":", lw=1)
ax[0, 0].set_xlim(0, XMAX); ax[0, 0].set_ylim(0, 100); ax[0, 0].legend(loc="upper left", fontsize=8)
ax[0, 0].set_xlabel("chi2_flash  (SNR-weighted chi2 at the t0 list)"); ax[0, 0].set_ylabel("mis-classified energy [% of TPC]")
ax[0, 0].set_title(f"(a) vs mis-classified FRACTION   Spearman={spr(snr, WF):+.3f}")
fig.colorbar(hb, ax=ax[0, 0], pad=.01).set_label("TPCs/cell")

# (b) vs absolute MeV
hb2 = ax[0, 1].hexbin(snr, np.maximum(WE, 0.5), gridsize=42, cmap="magma", norm=LogNorm(vmin=1),
                      yscale="log", extent=(0, XMAX, np.log10(0.5), np.log10(1500)), mincnt=1)
trend(ax[0, 1], snr, WE, log=True)
ax[0, 1].set_xlim(0, XMAX); ax[0, 1].set_ylim(0.5, 1500)
ax[0, 1].set_xlabel("chi2_flash  (SNR-weighted chi2 at the t0 list)"); ax[0, 1].set_ylabel("mis-classified energy [MeV]")
ax[0, 1].set_title(f"(b) vs ABSOLUTE mis-classified energy   Spearman={spr(snr, WE):+.3f}")
fig.colorbar(hb2, ax=ax[0, 1], pad=.01).set_label("TPCs/cell")

# (c) ROC comparison
for score, lab, col in [(snr, "chi2_flash @ t0 list (SNR-wt)", "tab:blue"),
                        (chi2s, "chi2 whole-TPC", "tab:orange"),
                        (cos_snr, "per-flash cosine (SNR-wt)", "tab:green"),
                        (gcos, "global cosine (big-peak)", "tab:red")]:
    f, t = roc(score[sel], y); ax[1, 0].plot(f, t, color=col, lw=1.8, label=f"{lab}  (AUC {auc(score[sel], y):.3f})")
ax[1, 0].plot([0, 1], [0, 1], "k:", lw=.8)
ax[1, 0].set_xlabel("false-positive rate (GOOD flagged)"); ax[1, 0].set_ylabel("true-positive rate (BAD caught)")
ax[1, 0].set_title("(c) ROC: BAD>20% vs GOOD<5%"); ax[1, 0].legend(fontsize=8, loc="lower right")

# (d) binned-median trend: chi2_flash vs whole-TPC chi2 (rank-x for fair overlay)
def rankpct(x): return 100 * np.argsort(np.argsort(x)) / len(x)
for score, lab, col in [(snr, "chi2_flash @ t0 list", "tab:blue"), (chi2s, "chi2 whole-TPC", "tab:orange")]:
    rx = rankpct(score); order = np.argsort(rx)
    cx, md = [], []
    for i in range(20):
        s = (rx >= i * 5) & (rx < (i + 1) * 5)
        if s.sum() >= 5: cx.append(rx[s].mean()); md.append(100 * np.median(WF[s]))
    ax[1, 1].plot(cx, md, "o-", color=col, lw=2, ms=4, label=lab)
ax[1, 1].axhline(20, color="tab:red", ls=":", lw=1)
ax[1, 1].set_xlabel("metric percentile"); ax[1, 1].set_ylabel("median mis-classified [% of TPC]")
ax[1, 1].set_title("(d) higher metric -> more mis-classified energy"); ax[1, 1].legend(fontsize=8, loc="upper left")

fig.suptitle("Winning quality metric: chi2 evaluated at the t0 list (SNR-weighted)  vs  MC-truth mis-classified energy  (v0.3, 2320 TPCs)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(HERE, "deck_figs", "chi2flash_vs_misclass.png")
fig.savefig(out, dpi=118); print("wrote", out)
print(f"chi2_flash(snr): AUC={auc(snr[sel], y):.3f} rhoFrac={spr(snr, WF):+.3f} rhoMeV={spr(snr, WE):+.3f}")
