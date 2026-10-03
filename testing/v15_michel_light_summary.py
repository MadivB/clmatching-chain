"""Population figure for the delayed Michels' light (from v15_michel_waveforms.py output): how bright the Michel's own
light is on its best channel versus what is actually there at that time, split by covered / uncovered and isolated or not."""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "michel_nd_out", "waveforms")
R = json.load(open(os.path.join(D, "michel_light.json")))
P = json.load(open(os.path.join(D, "michel_photons.json"))) if os.path.exists(os.path.join(D, "michel_photons.json")) else {}
for r in R:
    p = P.get(f"{r['ev']}:{r['michel']}"); r["produced"] = p["produced"] if p else np.nan; r["dE"] = p["dE"] if p else np.nan
unc = [r for r in R if r["uncovered"]]; cov = [r for r in R if not r["uncovered"]]
fig, axs = plt.subplots(1, 3, figsize=(17, 5.2))
ax = axs[0]
for sel, col, lab in ((cov, "0.6", "covered by a probe piece"), (unc, "red", "never a candidate piece")):
    pm = np.array([r["pred_max"] for r in sel]); act = np.array([max(r["actual_top"][0], 1) for r in sel]); iso = np.array([r["n_family_15"] == 0 for r in sel])
    ax.scatter(act[~iso], pm[~iso], s=10, c=col, alpha=0.6, label=f"{lab}, family within 15 ticks (n {np.sum(~iso)})")
    ax.scatter(act[iso], pm[iso], s=22, c=col, marker="^", alpha=0.9, edgecolors="k", linewidths=0.3, label=f"{lab}, no family within 15 ticks (n {iso.sum()})")
lim = [20, 1e5]; ax.plot(lim, lim, "k:", lw=1, label="Michel = everything else there"); ax.plot(lim, [0.1 * v for v in lim], "k--", lw=0.7, label="Michel = 10 % of it")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_xlabel("actual amplitude at the Michel time on the Michel's brightest channel [ADC]"); ax.set_ylabel("Michel's own light there (perceiver) [ADC]")
ax.legend(fontsize=7, loc="upper left"); ax.set_title("delayed Michels (> 12 ticks, >= 5 MeV), 63 spills", fontsize=10)
ax = axs[1]
bins = np.logspace(1.5, 4.5, 31)
for sel, col, lab in ((unc, "red", "never a candidate piece"), (cov, "0.5", "covered by a probe piece")):
    ax.hist([r["pred_max"] for r in sel], bins=bins, histtype="step", color=col, lw=1.6, label=f"{lab} (n {len(sel)}, median {np.median([r['pred_max'] for r in sel]):.0f} ADC)")
ax.hist([r["pred_max"] for r in unc if r["n_family_15"] == 0], bins=bins, color="red", alpha=0.25, label=f"never a candidate AND isolated (n {sum(r['n_family_15']==0 for r in unc)})")
ax.axvline(100, color="k", ls=":", lw=1); ax.text(105, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] > 0 else 1, "100 ADC\n(the old prediction floor)", fontsize=7)
ax.set_xscale("log"); ax.set_xlabel("Michel's own light, brightest channel [ADC]"); ax.set_ylabel("Michels"); ax.legend(fontsize=7); ax.set_title("how bright is the Michel's light?", fontsize=10)
ax = axs[2]
for sel, col, lab in ((unc, "red", "never a candidate piece"), (cov, "0.5", "covered by a probe piece")):
    ratio = np.sort([r["pred_max"] / max(r["actual_top"][0], 1) for r in sel]); ax.plot(ratio, np.arange(1, ratio.size + 1) / ratio.size, color=col, lw=1.6, label=lab)
ax.axvline(1, color="k", ls=":", lw=1); ax.set_xscale("log"); ax.set_xlim(1e-3, 30); ax.set_xlabel("Michel light / actual amplitude at that channel and time"); ax.set_ylabel("cumulative fraction of Michels"); ax.legend(fontsize=8); ax.set_title("how buried is it?", fontsize=10)
fn = os.path.join(D, "michel_light_population.png"); fig.tight_layout(); fig.savefig(fn, dpi=120); print("figure", fn)
# text summary
def q(v, ps=(10, 25, 50, 75, 90)):
    v = np.asarray(v, float); v = v[np.isfinite(v)]; return np.percentile(v, ps).round(0)
for lab, sel in (("never a candidate piece", unc), ("covered by a probe piece", cov)):
    pm = np.array([r["pred_max"] for r in sel]); act = np.array([max(r["actual_top"][0], 1) for r in sel]); iso = np.array([r["n_family_15"] == 0 for r in sel]); E = np.array([r["E"] for r in sel])
    print(f"{lab}: n {len(sel)}, E {E.sum()/1e3:.2f} GeV; Michel brightest-channel light p10/25/50/75/90 {q(pm)} ADC; summed {q([r['pred_sum'] for r in sel])} ADC; produced photons median {np.nanmedian([r['produced'] for r in sel]):.0f}")
    print(f"   Michel >= 1000 ADC on its best channel: {np.mean(pm >= 1000):.2f}; >= 10 % of what is there: {np.mean(pm >= 0.1 * act):.2f}; >= 50 %: {np.mean(pm >= 0.5 * act):.2f}; >= 100 %: {np.mean(pm >= act):.2f}")
    print(f"   isolated (no placed family within 15 ticks): {iso.sum()} ({E[iso].sum()/1e3:.2f} GeV); of these Michel >= 50 % of the actual: {np.mean(pm[iso] >= 0.5 * act[iso]) if iso.any() else 0:.2f}; residual (actual - model) on the best channel vs Michel prediction, median ratio {np.median([r['resid_top'][0] / max(r['pred_max'], 1) for r in sel if r['n_family_15'] == 0]) if iso.any() else float('nan'):.2f}")
