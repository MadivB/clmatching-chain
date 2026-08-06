"""Capstone: every metric tried across v1 / vFable / vGPT, graded on the two
confound-free targets.  Shows the winner (v2) and the amount-vs-structural lesson."""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.abspath(os.path.dirname(__file__))
# (label, approach, AUC, rhoFrac, is_structural_detector)
rows = [
    ("chi2_flash (at t0 list)",        "v1",     0.903, 0.532, False),
    ("displaced-light (amount)",       "v1",     0.919, 0.551, False),
    ("Anscombe chi2",                  "vFable", 0.883, 0.585, False),
    ("multiscale maxk",                "vFable", 0.886, 0.605, False),
    ("Z_mis covariance",               "vFable", 0.882, 0.490, True),
    ("Q_noise (whitened GoF)",         "vGPT",   0.883, 0.581, False),
    ("Q_dipole (Hungarian)",           "vGPT",   0.851, 0.413, True),
    ("Q_shift (lag-corr)",             "vGPT",   0.855, 0.393, True),
    ("Q_failure (z^2 sum, Sec10)",     "vGPT",   0.828, 0.158, False),
    ("v1 combo (chi2+disp)",           "combo",  0.936, 0.630, False),
    ("confidence v2 (+maxk)",          "combo",  0.935, 0.654, False),
]
COL = {"v1": "tab:orange", "vFable": "tab:blue", "vGPT": "tab:purple", "combo": "black"}
labels = [r[0] for r in rows]
yy = np.arange(len(rows))[::-1]

fig, ax = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
for j, (metric, xlab, xlim) in enumerate([(2, "AUC  (BAD>20% vs GOOD<5%)", (0.80, 0.95)),
                                          (3, "Spearman vs mis-classified fraction", (0.10, 0.70))]):
    for i, r in enumerate(rows):
        c = COL[r[1]]
        hatch = "//" if r[4] else None
        ax[j].barh(yy[i], r[metric], color=c, alpha=0.5 if r[1] != "combo" else 0.9, hatch=hatch,
                   edgecolor="k" if r[1] == "combo" else "none", lw=1.2)
        ax[j].text(r[metric] + (0.002 if j == 0 else 0.004), yy[i], f"{r[metric]:.3f}", va="center", fontsize=8)
    ax[j].set_xlim(*xlim); ax[j].set_xlabel(xlab)
    ax[j].axhline(1.5, color="0.8", lw=.6)
ax[0].set_yticks(yy); ax[0].set_yticklabels(labels, fontsize=9)
# legend
from matplotlib.patches import Patch
leg = [Patch(color=COL[k], alpha=.6, label=k) for k in ["v1", "vFable", "vGPT", "combo"]]
leg.append(Patch(facecolor="0.7", hatch="//", label="structural detector\n(loses the AMOUNT)"))
ax[1].legend(handles=leg, fontsize=8, loc="upper right")
fig.suptitle("Charge-light confidence metric: v1 vs vFable vs vGPT  (v0.3, 2320 TPCs, MC-truth targets)\n"
             "Winner = v2 (complexity-norm chi2_flash + displaced + maxk).  Structural swap-detectors (hatched) lose to amount-based displaced.",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
out = os.path.join(HERE, "deck_figs", "metric_comparison_summary.png")
fig.savefig(out, dpi=120); print("wrote", out)
