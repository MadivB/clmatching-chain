"""#3 for BACKBONE hits only: accuracy vs TPC energy (backbone energy), pink = proportion
of wrong backbone energy per bin.  Backbone = hits with finite ts_p1 (phase-1 placement)."""
import os, pickle, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.abspath(os.path.dirname(__file__)); OUT = os.path.join(HERE, "deck_figs")
M = pickle.load(open(os.path.join(HERE, "master_v04.pkl"), "rb")); HIT = M["HIT"]
sel = HIT["has_truth"].astype(bool) & HIT["backbone"].astype(bool)      # backbone-only
f = HIT["file"][sel]; ev = HIT["event"][sel].astype(np.int64); tp = HIT["tpc"][sel].astype(np.int64)
e = HIT["e"][sel].astype(float); cor = HIT["correct"][sel].astype(bool)
fu, fi = np.unique(f, return_inverse=True)
key = fi.astype(np.int64) * 100000 + ev * 100 + tp
uk, inv = np.unique(key, return_inverse=True)
Et = np.zeros(len(uk)); wE = np.zeros(len(uk))
np.add.at(Et, inv, e); np.add.at(wE, inv, np.where(cor, 0.0, e))
acc_tpc = 100 * (1 - wE / np.maximum(Et, 1e-9))
EMIN = 30.0; keep = Et >= EMIN; Et, wE, acc_tpc = Et[keep], wE[keep], acc_tpc[keep]
print(f"backbone TPCs (E>={EMIN:.0f}): N={len(Et)}  E range [{Et.min():.0f}, {Et.max():.0f}] MeV  "
      f"overall backbone energy-eff={100*(1-wE.sum()/Et.sum()):.2f}%")

edges = np.logspace(np.log10(EMIN), np.log10(Et.max() * 1.02), 22); cen = np.sqrt(edges[:-1] * edges[1:])
black, pink = [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (Et >= lo) & (Et < hi); tot = Et[m].sum()
    black.append(np.mean(acc_tpc[m]) if m.any() else np.nan)
    pink.append(100 * wE[m].sum() / tot if tot > 0 else np.nan)

fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.4))
ax2 = axL.twinx()
ax2.bar(cen, pink, width=np.diff(edges), color="pink", alpha=0.7, align="center", edgecolor="none", zorder=1)
ax2.set_ylabel("proportion of wrong energy in bin [%]", color="palevioletred")
ax2.tick_params(axis="y", labelcolor="palevioletred"); ax2.set_ylim(0, max(np.nanmax(pink) * 1.15, 1))
axL.step(cen, black, where="mid", color="black", lw=2, zorder=3); axL.axhline(100, color="green", ls=":", lw=.8)
axL.set_xscale("log"); axL.set_xlabel("TPC backbone hit energy [MeV]"); axL.set_ylabel("assignment accuracy [%]")
axL.set_ylim(min(np.nanmin(black) - 2, 90), 100.5); axL.set_zorder(ax2.get_zorder() + 1); axL.patch.set_visible(False)
axL.set_title("accuracy vs TPC energy  (BACKBONE hits)")

be = np.logspace(np.log10(EMIN), np.log10(Et.max() * 1.02), 26)
axR.hist(Et, bins=be, color="#8fb3d9", alpha=.85, label="all backbone TPCs")
axR.hist(Et[wE >= 10], bins=be, color="#c0392b", alpha=.8, label="backbone wrong >= 10 MeV")
axR.set_xscale("log"); axR.set_yscale("log"); axR.set_xlabel("TPC backbone hit energy [MeV]")
axR.set_ylabel("N TPCs"); axR.set_title("where the backbone mistakes live"); axR.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "v04_accuracy_vs_energy_proportion_backbone.png"), dpi=140); plt.close(fig)
print("saved deck_figs/v04_accuracy_vs_energy_proportion_backbone.png")
