"""v0.10.0 final validation plot: IN-CHAIN cluster confidence (hit_conf93
from the baked model) vs MC-truth wrong fraction — the plot style the user
asked for (one axis = confidence, other = how often it is actually wrong).

Usage: python conf93_plot.py v127V93_out
"""
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subtick_conf_study2 import auc_rank

C_OK, C_BAD, C_REF = "#2471a3", "#c0392b", "#999999"


def load(outdir):
    rows = []
    for ff in sorted(glob.glob(f"{outdir}/full_dumps/*.npz")):
        fd = np.load(ff, allow_pickle=True)
        if "hit_conf93" not in fd.files:
            continue
        hc = fd["hit_conf93"].astype(np.float64)
        labs = fd["labels"]
        e = np.clip(fd["e"], 0, None).astype(np.float64)
        ts = fd["ts_final"].astype(np.float64)
        tru = fd["truth_t0"].astype(np.float64)
        for L in np.unique(labs[labs >= 0]):
            m = labs == L
            fin = m & np.isfinite(tru) & np.isfinite(ts) & (ts >= 0)
            etot = float(e[fin].sum())
            cvals = hc[m][np.isfinite(hc[m])]
            if etot <= 0 or cvals.size == 0:
                continue
            ew = float(e[fin & (np.abs(ts - tru) > 10.0)].sum())
            rows.append((float(cvals[0]), etot, ew / etot))
    p = np.array([r[0] for r in rows])
    eg = np.array([r[1] for r in rows])
    wf = np.array([r[2] for r in rows])
    return p, eg, (wf <= 0.5).astype(int)


def main():
    outdir = sys.argv[1]
    p, eg, y = load(outdir)
    wrong = y == 0
    print(f"{len(p)} clusters with in-chain conf, {wrong.sum()} wrong "
          f"({100*eg[wrong].sum()/eg.sum():.2f}% E)")
    print(f"in-chain AUC {auc_rank(y, p):.3f} (E-wt {auc_rank(y, p, eg):.3f})")
    print("reliability (in-chain conf, E-weighted):")
    for lo, hi in [(0, .2), (.2, .5), (.5, .8), (.8, .95), (.95, 1.001)]:
        m = (p >= lo) & (p < hi)
        if m.sum():
            fc = eg[m & (y == 1)].sum() / eg[m].sum()
            print(f"  [{lo:.2f},{hi:.2f}): n={m.sum():5d}  E-frac correct {fc*100:5.1f}%")

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    rng = np.random.default_rng(7)
    ax.scatter(p, 1 - y + rng.uniform(-0.02, 0.02, len(y)),
               s=6 + 2 * np.sqrt(eg), alpha=0.12, color=C_OK, lw=0)
    bins = np.linspace(0, 1.0001, 11)
    bx, by, be = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() < 3:
            continue
        w = eg[m]
        q = float(np.average(1 - y[m], weights=w))
        bx.append(0.5 * (lo + hi))
        by.append(q)
        be.append(np.sqrt(max(q * (1 - q), 1e-4) / m.sum()))
    ax.errorbar(bx, by, yerr=be, color=C_BAD, lw=2, marker="o", ms=5,
                capsize=3, label="E-weighted wrong-frac", zorder=5)
    xx = np.linspace(0, 1, 20)
    ax.plot(xx, 1 - xx, "--", color=C_REF, lw=1.2, label="perfect (1 - conf)")
    ax.set_xlabel("v0.10.0 cluster confidence (in-chain, calibrated)")
    ax.set_ylabel("wrong fraction (MC truth)")
    ax.set_title("v0.10.0: three-family calibrated confidence vs truth\n"
                 f"file 0000001, {len(p)} clusters, in-chain output")
    ax.legend(fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.06, 1.06)
    fig.tight_layout()
    out = "subtick_out/v0100_conf_vs_truth.png"
    fig.savefig(out, dpi=140)
    print("FIGURE:", os.path.abspath(out))


if __name__ == "__main__":
    main()
