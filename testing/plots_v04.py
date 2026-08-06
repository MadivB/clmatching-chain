"""v0.4 deliverables #1 (eff vs cluster energy), #2 (confidence vs mis-class, square
bins), #3 (accuracy vs TPC energy + proportion-wrong), #5 (efficiency table), #6 (stage
timing).  All from master_v04.pkl."""
import os, json, pickle, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib.colors import LogNorm

HERE = os.path.abspath(os.path.dirname(__file__)); OUT = os.path.join(HERE, "deck_figs")
M = pickle.load(open(os.path.join(HERE, "master_v04.pkl"), "rb"))
TPC, CL, HIT, EV = M["TPC"], M["CL"], M["HIT"], M["EV"]
NEV = len(EV["event"]); NCL = len(CL["energy"])

# ================= #1 efficiency vs cluster energy =================
Ecl = CL["energy"].astype(float); Ccl = CL["correct"].astype(float)
def eff_bar(bins, labels, colors, fname, title):
    effs, ns = [], []
    for lo, hi in bins:
        m = (Ecl >= lo) & (Ecl < hi); ns.append(int(m.sum())); effs.append(100 * Ccl[m].mean() if m.any() else 0)
    fig, ax = plt.subplots(figsize=(13, 5.6)); x = np.arange(len(labels))
    ax.bar(x, effs, color=colors, edgecolor="k", width=0.82)
    for i, (e, n) in enumerate(zip(effs, ns)): ax.text(i, e + 1, f"{e:.0f}%\nN={n}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=0, fontsize=8.5)
    ax.set_xlabel("cluster energy [MeV]"); ax.set_ylabel("matching efficiency [%]"); ax.set_ylim(0, 108)
    ax.grid(axis="y", alpha=.3, ls=":"); ax.set_title(title)
    ax.text(0.01, 0.04, f"ND-LAr v0.4, {NEV} events, {NCL} clusters with truth", transform=ax.transAxes,
            fontsize=9, va="bottom", bbox=dict(boxstyle="round", facecolor="white", alpha=.85))
    fig.tight_layout(); fig.savefig(os.path.join(OUT, fname), dpi=140, bbox_inches="tight"); plt.close(fig)
    print(fname, "eff:", [f"{l}:{e:.0f}%(N{n})" for l, e, n in zip(labels, effs, ns)])

# A: exact reference axis
binsA = [(i, i + 1) for i in range(10)] + [(10, 50), (50, np.inf)]
labA = [f"{i}-{i+1}" for i in range(10)] + ["10-50", ">=50"]
colA = ["#E45756"] + ["#4C78A8"] * 9 + ["#59A14F", "#59A14F"]
eff_bar(binsA, labA, colA, "v04_eff_vs_energy_1MeV.png",
        "ND-LAr v0.4 charge-light matching efficiency vs cluster energy  (per-cluster, |reco-truth|<=160 ns)")
# B: wide range
edgesB = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100, 1000, np.inf]
binsB = list(zip(edgesB[:-1], edgesB[1:]))
labB = [f"{int(a)}-{int(b)}" if np.isfinite(b) else f">={int(a)}" for a, b in binsB]
colB = ["#E45756"] + ["#4C78A8"] * 9 + ["#59A14F", "#59A14F", "#B07AA1"]
eff_bar(binsB, labB, colB, "v04_eff_vs_energy_wide.png",
        "ND-LAr v0.4 matching efficiency vs cluster energy  (wide range, per-cluster, |reco-truth|<=160 ns)")

# ================= #2 confidence vs mis-classified energy (SQUARE bins) =================
conf = TPC["conf"]; wfp = TPC["wf"] * 100
fig, ax = plt.subplots(figsize=(7.8, 6))
h = ax.hist2d(conf, wfp, bins=[np.linspace(0, 1, 45), np.linspace(0, 100, 45)], norm=LogNorm(), cmap="viridis")
# binned median trend
cx, md = [], []
for i in range(14):
    lo, hi = i / 14, (i + 1) / 14; s = (conf >= lo) & (conf < hi)
    if s.sum() >= 5: cx.append(conf[s].mean()); md.append(np.median(wfp[s]))
ax.plot(cx, md, "-", color="white", lw=2.4); ax.plot(cx, md, "o", color="black", ms=4)
ax.set_xlabel("confidence  (v2)"); ax.set_ylabel("mis-classified energy [% of TPC]")
ax.set_title("confidence vs mis-classified energy")
fig.colorbar(h[3], ax=ax, label="TPCs / bin"); fig.tight_layout()
fig.savefig(os.path.join(OUT, "v04_confidence_vs_misclass_square.png"), dpi=140); plt.close(fig)
print("v04_confidence_vs_misclass_square.png done")

# ================= #3 accuracy vs TPC energy + proportion-of-wrong =================
Et = TPC["E"]; wE = TPC["wrongE"]
edges = np.logspace(np.log10(max(Et.min(), 20)), np.log10(Et.max() * 1.02), 22)
cen = np.sqrt(edges[:-1] * edges[1:])
acc = []; propw = []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (Et >= lo) & (Et < hi); tot = Et[m].sum()
    acc.append(100 * (1 - wE[m].sum() / tot) if tot > 0 else np.nan)
    propw.append(100 * wE[m].sum() / tot if tot > 0 else np.nan)
fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.4))
axR2 = axL.twinx()
axR2.bar(cen, propw, width=np.diff(edges), color="pink", alpha=0.75, align="center", edgecolor="none")
axR2.set_ylabel("proportion of wrong energy in bin [%]", color="palevioletred")
axR2.tick_params(axis="y", labelcolor="palevioletred")
axL.step(cen, acc, where="mid", color="black", lw=2)
axL.axhline(100, color="green", ls=":", lw=.8)
axL.set_xscale("log"); axL.set_xlabel("TPC total hit energy [MeV]"); axL.set_ylabel("assignment accuracy [%]")
axL.set_ylim(84, 100.5); axL.set_zorder(axR2.get_zorder() + 1); axL.patch.set_visible(False)
axL.set_title("accuracy vs TPC total energy")
# right: where mistake-TPCs live
allE = Et; badE = Et[wE >= 10]
be = np.logspace(np.log10(max(Et.min(), 20)), np.log10(Et.max() * 1.02), 24)
axR.hist(allE, bins=be, color="#8fb3d9", alpha=.85, label="all TPCs")
axR.hist(badE, bins=be, color="#c0392b", alpha=.8, label="TPCs with wrong >= 10 MeV")
axR.set_xscale("log"); axR.set_yscale("log"); axR.set_xlabel("TPC total hit energy [MeV]")
axR.set_ylabel("N TPCs"); axR.set_title("where the mistake-TPCs live"); axR.legend(fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "v04_accuracy_vs_energy_proportion.png"), dpi=140); plt.close(fig)
print("v04_accuracy_vs_energy_proportion.png done")

# ================= #6 stage timing =================
stages = ["first", "backbone", "rescue", "staged", "rest"]
labs = ["first stage\n(ML predict\n+ cluster)", "backbone\nmatch", "rescue\nadopt", "staged\n(phase3+matrix)", "force-cov\n+ dump"]
means = [np.nanmean(EV[s]) for s in stages]; stds = [np.nanstd(EV[s]) for s in stages]
fig, ax = plt.subplots(figsize=(9.5, 5.5))
b = ax.bar(range(len(stages)), means, yerr=stds, color=["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2"],
           edgecolor="k", capsize=4)
for i, m in enumerate(means): ax.text(i, m + max(means) * 0.01, f"{m:.1f}s", ha="center", va="bottom", fontsize=9)
ax.set_xticks(range(len(stages))); ax.set_xticklabels(labs, fontsize=8.5)
ax.set_ylabel("wall time per event [s]")
ax.set_title(f"v0.4 computation time per stage  (mean +- std over {NEV} events; total {np.nanmean(EV['t_event']):.0f}s/event)")
ax.grid(axis="y", alpha=.3, ls=":")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "v04_stage_timing.png"), dpi=140); plt.close(fig)
print("v04_stage_timing.png done; stage means:", dict(zip(stages, [round(m, 1) for m in means])))

# ================= #5 efficiency table =================
ht = HIT["has_truth"].astype(bool); bb = HIT["backbone"].astype(bool); cor = HIT["correct"].astype(bool); he = HIT["e"]
rows = []
for nm, mk in [("all hits", np.ones(len(ht), bool)), ("backbone", bb), ("non-backbone", ~bb)]:
    m = mk & ht; ew = he[m]
    rows.append(dict(category=nm, n_hits=int(m.sum()), hit_eff_pct=round(100 * cor[m].mean(), 3),
                     energy_eff_pct=round(100 * ew[cor[m]].sum() / ew.sum(), 3),
                     total_energy_MeV=round(ew.sum(), 1), wrong_energy_MeV=round(ew[~cor[m]].sum(), 1),
                     energy_frac_of_all=round(ew.sum() / he[ht].sum(), 4)))
tbl = dict(events=NEV, cut_ticks=10, rows=rows)
json.dump(tbl, open(os.path.join(HERE, "v04_efficiency_table.json"), "w"), indent=2)
with open(os.path.join(HERE, "v04_efficiency_table.txt"), "w") as f:
    f.write(f"ND-LAr v0.4 charge-light matching efficiency  ({NEV} events, |reco-truth|<=10 ticks)\n")
    f.write(f"{'category':>14} {'N hits':>10} {'hit eff %':>10} {'energy eff %':>13} {'energy frac':>12}\n")
    for r in rows:
        f.write(f"{r['category']:>14} {r['n_hits']:>10} {r['hit_eff_pct']:>10.2f} {r['energy_eff_pct']:>13.2f} {r['energy_frac_of_all']:>12.3f}\n")
print(open(os.path.join(HERE, "v04_efficiency_table.txt")).read())
print("DONE")
