"""Before/after validation of the sub-tick refine (stage 6).

Reads per-hit dumps (ts_before, ts_after, truth, e), reports sigma/bias on
correct core hits (|D| <= 5 ticks on the BEFORE arm, 2x2 methodology: same
hit population in both arms), checks eff(+-160 ns) invariance, and draws the
before/after residual histograms.

Usage: python subtick_validate.py v109ST_out/dumps out_prefix
"""
import glob
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TICK_NS = 16.0


def main():
    dumps, prefix = sys.argv[1], sys.argv[2]
    rb, ra, tt, ee = [], [], [], []
    for f in sorted(glob.glob(f"{dumps}/*.npz")):
        d = np.load(f)
        m = (np.isfinite(d["truth"]) & np.isfinite(d["ts_before"])
             & (d["ts_before"] >= 0) & (d["e"] > 0))
        rb.append(d["ts_before"][m] - d["truth"][m])
        ra.append(d["ts_after"][m] - d["truth"][m])
        ee.append(d["e"][m])
    rb, ra, e = np.concatenate(rb), np.concatenate(ra), np.concatenate(ee)
    print(f"{rb.size} assigned truth-valid hits, {e.sum()/1000:.1f} GeV")

    for cut, name in [(10.0, "eff(+-160ns)")]:
        for lab, r in [("before", rb), ("after", ra)]:
            eff = e[np.abs(r) <= cut].sum() / e.sum()
            print(f"{name} {lab}: {eff*100:.3f}%")

    core = np.abs(rb) <= 5.0          # fixed population: correct core on BEFORE arm
    stats = {}
    for lab, r in [("before", rb), ("after", ra)]:
        mu = np.average(r[core], weights=e[core]) * TICK_NS
        sig = np.sqrt(np.average((r[core]*TICK_NS - mu) ** 2, weights=e[core]))
        stats[lab] = (mu, sig)
        print(f"core sigma {lab}: {sig:.2f} ns   bias {mu:+.2f} ns")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    bins = np.linspace(-80, 80, 161)
    for ax, rng in zip(axes, [(-80, 80), (-30, 30)]):
        b = np.linspace(*rng, 121)
        for lab, r, c in [("before (integer ticks)", rb, "#888888"),
                          ("after (sub-tick refine)", ra, "#c0392b")]:
            mu, sig = stats[lab.split(" ")[0]]
            ax.hist(r[core]*TICK_NS, bins=b, weights=e[core],
                    histtype="step", lw=1.8, color=c,
                    label=f"{lab}: sigma={sig:.2f} ns, bias={mu:+.2f} ns")
        ax.set_xlabel("reco - truth  [ns]")
        ax.set_ylabel("energy [MeV] / bin")
        ax.legend(fontsize=8.5)
        ax.set_title(f"correct core hits (|D|<=5 ticks), E-weighted, range {rng}")
    fig.suptitle("ND charge-light matching t0 residuals - stage-6 sub-tick refinement")
    fig.tight_layout()
    out = f"{prefix}_residuals.png"
    fig.savefig(out, dpi=140)
    print("FIGURE:", out)

    # zoomed linear + log overlay on the full +-160ns window
    fig2, ax = plt.subplots(figsize=(7.2, 5))
    b = np.linspace(-160, 160, 201)
    for lab, r, c in [("before", rb, "#888888"), ("after", ra, "#c0392b")]:
        ax.hist(r*TICK_NS, bins=b, weights=e, histtype="step", lw=1.6,
                color=c, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("reco - truth  [ns]")
    ax.set_ylabel("energy [MeV] / bin")
    ax.set_title("all assigned hits, E-weighted (log)")
    ax.legend()
    fig2.tight_layout()
    out2 = f"{prefix}_residuals_log.png"
    fig2.savefig(out2, dpi=140)
    print("FIGURE:", out2)


if __name__ == "__main__":
    main()
