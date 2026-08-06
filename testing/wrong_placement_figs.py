"""Figures for wrong phase-2 placements: red wrong hits + the loss landscape.

Per wrong cluster, one PNG with 4 panels:
  (a) 3D hits of the TPC -- this cluster's wrongly-placed hits in RED, everything
      else grey; a lime marker set shows where the cluster's TRUE slot-mates are.
  (b) DECISION-TIME loss L(t0): the matcher's own loss on this cluster's support
      channels against the base image AS IT WAS when the cluster was placed
      (reconstructed by replaying the recorded assignment order).  The discrete
      t0-candidate grid is drawn as ticks -- an ITERATIVE (greedy_discrete)
      cluster may ONLY be placed on that grid, so a truth far from any candidate
      is unwinnable by construction.
  (c) HINDSIGHT loss: same cluster against the FINAL base minus its own light.
      If truth becomes the minimum here but not in (b), the mistake was caused by
      missing context (ordering) rather than by the light itself.
  (d) zoom of both curves around the chosen and true t0.

Usage: python wrong_placement_figs.py [TOP_N]
"""
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
DUMPS = os.path.join(HERE, "p12wrong_out", "dumps")
OUT = os.path.join(HERE, "wrong_placement_figs")
os.makedirs(OUT, exist_ok=True)
CUT = 10.0
TOP_N = int(sys.argv[1]) if len(sys.argv) > 1 else 10

rows = []
for fp in sorted(glob.glob(os.path.join(HERE, "p12wrong_out", "rows_worker*.jsonl"))):
    for line in open(fp):
        if line.strip():
            rows.append(json.loads(line))
p2 = [r for r in rows if r["phase"] == "phase2"]
p2.sort(key=lambda r: -r["energy"])
print(f"{len(rows)} wrong placements ({len(p2)} phase-2); rendering up to {TOP_N}")

made = 0
for r in p2:
    if made >= TOP_N:
        break
    tag = f"{r['file']}_ev{r['event']}"
    dp = os.path.join(DUMPS, f"{tag}.npz")
    if not os.path.exists(dp):
        continue
    d = np.load(dp)
    cid = int(r["label"])
    kdec, khind, kgrid = f"L{cid}_curve_dec", f"L{cid}_curve_hind", f"L{cid}_grid"
    if kdec not in d.files:
        continue
    dec = np.asarray(d[kdec], float)
    hind = np.asarray(d[khind], float) if khind in d.files else None
    grid = np.asarray(d[kgrid], int) if kgrid in d.files else np.array([], int)
    t0, tr = float(r["t0"]), float(r["truth"])
    tpc0 = int(r["tpcs"][0]) if r["tpcs"] else -1

    x, y, z = d["x"], d["y"], d["z"]
    tpcs_h, labs = d["tpc"], d["labels"]
    ts2, tt = np.asarray(d["ts2"], float), np.asarray(d["truth_t0"], float)
    mine = labs == cid
    inTPC = np.isin(tpcs_h, np.asarray(r["tpcs"], int))
    wrong_hits = mine & np.isfinite(tt) & (np.abs(ts2 - tt) > CUT)
    # hits that truly belong at this cluster's PLACED t0 (who really owns that slot)
    slot_owner = inTPC & np.isfinite(tt) & (np.abs(tt - t0) <= CUT) & ~mine

    fig = plt.figure(figsize=(17, 9.5))
    kind = "ITERATIVE (greedy/band)" if r["iterative"] else r["stage"]
    fig.suptitle(
        f"{r['file']} ev{r['event']} | cluster {cid} | {kind} | TPCs {r['tpcs']} | "
        f"{r['energy']:.0f} MeV | placed t0={t0:.0f} vs TRUTH {tr:.0f}  (off by {abs(t0-tr):.0f} ticks)",
        fontsize=13)

    # ---- (a) 3D hits ----
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    o = inTPC & ~mine
    ax.scatter(z[o], x[o], y[o], s=1.5, c="lightgray", alpha=0.35, label="other hits (TPC)")
    if slot_owner.any():
        ax.scatter(z[slot_owner], x[slot_owner], y[slot_owner], s=4, c="limegreen", alpha=0.8,
                   label=f"true owner of t0={t0:.0f}")
    ax.scatter(z[wrong_hits], x[wrong_hits], y[wrong_hits], s=6, c="red", alpha=0.95,
               label=f"THIS cluster, placed WRONG ({wrong_hits.sum()} hits)")
    ax.set_xlabel("z"); ax.set_ylabel("x"); ax.set_zlabel("y")
    ax.set_title("(a) where the newly-placed wrong hits are", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")

    # ---- (b) decision-time loss ----
    ax2 = fig.add_subplot(2, 2, 2)
    ts = np.arange(dec.size)
    ax2.plot(ts, dec, color="tab:blue", lw=1.2, label="loss at DECISION time")
    ax2.axvline(t0, color="red", lw=1.6, label=f"chosen t0={t0:.0f}")
    ax2.axvline(tr, color="green", ls=":", lw=1.8, label=f"TRUTH t0={tr:.0f}")
    am = int(np.argmin(dec))
    ax2.plot([am], [dec[am]], "v", color="tab:blue", ms=9,
             label=f"free argmin={am}")
    if grid.size:
        yl = ax2.get_ylim()
        ax2.vlines(grid, yl[0], yl[0] + 0.06 * (yl[1] - yl[0]), color="darkorange", lw=1.0,
                   label=f"allowed t0 candidates ({grid.size})")
        near = grid[np.argmin(np.abs(grid - tr))] if grid.size else None
        if near is not None:
            ax2.annotate(f"nearest candidate to truth: {near}\n(|Δ|={abs(near-tr):.0f} ticks)",
                         xy=(near, yl[0] + 0.10 * (yl[1] - yl[0])), fontsize=8,
                         color="darkorange", ha="center")
    ax2.set_xlabel("t0 [ticks]"); ax2.set_ylabel("loss  mean((model-actual)^2 / var)")
    ax2.set_title("(b) the landscape the algorithm actually faced", fontsize=10)
    ax2.legend(fontsize=8); ax2.grid(alpha=0.25)

    # ---- (c) hindsight loss ----
    ax3 = fig.add_subplot(2, 2, 3)
    if hind is not None:
        ax3.plot(ts, hind, color="tab:purple", lw=1.2, label="loss with FINAL base (hindsight)")
        hm = int(np.argmin(hind))
        ax3.plot([hm], [hind[hm]], "v", color="tab:purple", ms=9, label=f"argmin={hm}")
        verdict = ("truth WINS in hindsight -> mistake caused by ORDERING/missing context"
                   if abs(hm - tr) <= CUT else
                   "truth still does NOT win -> intrinsic (light/prediction) failure")
        ax3.set_title(f"(c) hindsight: {verdict}", fontsize=9.5,
                      color="tab:green" if abs(hm - tr) <= CUT else "crimson")
    else:
        ax3.text(0.5, 0.5, "no hindsight curve", ha="center")
        ax3.set_title("(c) hindsight", fontsize=10)
    ax3.axvline(t0, color="red", lw=1.6); ax3.axvline(tr, color="green", ls=":", lw=1.8)
    ax3.set_xlabel("t0 [ticks]"); ax3.set_ylabel("loss"); ax3.legend(fontsize=8); ax3.grid(alpha=0.25)

    # ---- (d) zoom ----
    ax4 = fig.add_subplot(2, 2, 4)
    lo = int(max(0, min(t0, tr) - 60)); hi = int(min(dec.size - 1, max(t0, tr) + 60))
    ax4.plot(ts[lo:hi], dec[lo:hi], color="tab:blue", lw=1.4, label="decision-time")
    if hind is not None:
        ax4.plot(ts[lo:hi], hind[lo:hi], color="tab:purple", lw=1.2, ls="--", label="hindsight")
    ax4.axvline(t0, color="red", lw=1.6, label=f"chosen ({dec[int(round(t0))]:.3g})")
    ax4.axvline(tr, color="green", ls=":", lw=1.8, label=f"truth ({dec[int(round(tr))]:.3g})")
    if grid.size:
        g2 = grid[(grid >= lo) & (grid <= hi)]
        if g2.size:
            yl = ax4.get_ylim()
            ax4.vlines(g2, yl[0], yl[0] + 0.08 * (yl[1] - yl[0]), color="darkorange", lw=1.2,
                       label="candidates")
    ax4.set_xlabel("t0 [ticks]"); ax4.set_ylabel("loss")
    ax4.set_title("(d) zoom: is truth even reachable / preferred?", fontsize=10)
    ax4.legend(fontsize=8); ax4.grid(alpha=0.25)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fn = f"wp{made}_{r['file']}_ev{r['event']}_c{cid}.png"
    fig.savefig(os.path.join(OUT, fn), dpi=125)
    plt.close(fig)
    dl = dec[int(round(t0))] - dec[int(round(tr))]
    print(f"wp{made}: ev{r['event']} c{cid} {r['energy']:.0f} MeV {kind} | "
          f"t0={t0:.0f} truth={tr:.0f} | loss(chosen)-loss(truth)={dl:+.4g} | {fn}", flush=True)
    made += 1
print("figures ->", OUT)
