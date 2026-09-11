"""Stage-E refinement study (2x2-style plot for ND v1.0).

Four arms from the merge-0.5 study dumps:
  1. integer placement only        (ts_before)
  2. 0.25-tick discrete argmin     (ts_grid)
  3. analytic quadratic = v1.0     (ts_after)
  4. per-CLUSTER quadratic fit     (ts_perclu)
Plus the family-tolerance comparison (merge 0.5 vs 2.0) + multiplicity.

Selection: correctly-assigned core hits (|integer - truth| <= 2 ticks),
counts (per hit), residuals in ticks; mu/sigma reported in ns (x16).

Usage: python subtick_study_plot.py /pscratch/.../v142ST05_out /pscratch/.../v143ST20_out
"""
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUT, GRID = "#1b2230", "#5d6879", "#d8dee7"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                     "axes.edgecolor": MUT, "axes.labelcolor": INK,
                     "savefig.dpi": 270, "savefig.bbox": "tight",
                     "figure.facecolor": "white"})


def load_arms(outdir, keys):
    arrs = {k: [] for k in keys}
    for sf in sorted(glob.glob(f"{outdir}/dumps/*.npz")):
        sd = np.load(sf)
        if "ts_grid" in sd.files and sd["ts_grid"].size == 0:
            continue                      # not a study dump
        tru = sd["truth"].astype(np.float64)
        tb = sd["ts_before"].astype(np.float64)
        core = (np.isfinite(tru) & np.isfinite(tb) & (tb >= 0)
                & (np.abs(tb - tru) <= 2.0))
        for k in keys:
            v = sd[k].astype(np.float64)
            arrs[k].append((v - tru)[core])
    return {k: np.concatenate(v) for k, v in arrs.items() if v}


def mu_sig(r):
    m = np.abs(r) <= 2.0
    return 16.0 * float(r[m].mean()), 16.0 * float(r[m].std())


def main(d05, d20=None):
    a = load_arms(d05, ["ts_before", "ts_grid", "ts_after", "ts_perclu"])
    b = load_arms(d20, ["ts_after"]) if d20 and os.path.isdir(f"{d20}/dumps") else {}

    arms = [("Alpha Release: integer fit only", a["ts_before"], "#9aa3ad"),
            ("0.25-tick discrete argmin", a["ts_grid"], "#9fc5e8"),
            ("v1.0 release: quadratic minimum", a["ts_after"], "#0e8fa0"),
            ("per-cluster quadratic fit", a["ts_perclu"], "#b5307a")]
    fig, ax = plt.subplots(figsize=(10.5, 6.6))
    bins = np.arange(-2, 2 + 0.0625, 0.125)
    for name, r, col in arms:
        mu, sg = mu_sig(r)
        ax.hist(r[np.abs(r) <= 2], bins=bins, histtype="step", lw=2.2,
                color=col,
                label=f"{name:31s} \u03bc = {mu:+5.2f} ns   \u03c3 = {sg:5.2f} ns")
    ax.axvline(0, color=INK, lw=0.8, ls=":")
    ax.set_xlabel("reco t0 \u2212 truth t0   (ticks, 1 tick = 16 ns)", color=INK)
    ax.set_ylabel("hits", color=INK)
    ax.set_title("v1 upgrade: discrete vs. continuous minimum",
                 fontsize=14, fontweight="bold", color=INK)
    ax.legend(loc="upper left", prop={"family": "DejaVu Sans Mono", "size": 9.5})
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig("subtick_out/v1_stageE_arms.png", dpi=140)
    print("FIGURE: /global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/"
          "CLMatching_v0.1_backboneOnly/testing/subtick_out/v1_stageE_arms.png")
    for name, r, _ in arms:
        mu, sg = mu_sig(r)
        print(f"  {name:36s} mu {mu:+6.2f} ns  sigma {sg:5.2f} ns")
    if "ts_after" in b:
        mu, sg = mu_sig(b["ts_after"])
        print(f"  {'quadratic, 2.0-tick families':36s} mu {mu:+6.2f} ns  sigma {sg:5.2f} ns")

    # multiplicity comparison
    for tag, d in (("0.5", d05), ("2.0", d20)):
        mult = []
        for sf in sorted(glob.glob(f"{d}/dumps/*.npz")):
            sd = np.load(sf)
            if "nlab_grp" not in sd.files or sd["nlab_grp"].size == 0:
                continue
            gid, nl = sd["group_id"], sd["nlab_grp"]
            for g in np.unique(gid[gid >= 0]):
                m = gid == g
                if nl[m].max() > 0:
                    mult.append(int(nl[m].max()))
        mult = np.array(mult)
        if mult.size:
            print(f"family multiplicity (merge {tag}): {mult.size} families, "
                  f"median {np.median(mult):.0f}, mean {mult.mean():.1f}, "
                  f"p90 {np.percentile(mult, 90):.0f}, max {mult.max()}, "
                  f"singletons {(mult == 1).mean()*100:.0f}%")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
