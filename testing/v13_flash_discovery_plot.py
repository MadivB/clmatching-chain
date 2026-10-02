"""Figure for the flash-discovery comparison (reads flash_discovery/flash_discovery_data.npz).
A: accidental-corrected truth-flash discovery efficiency vs energy deposited in the TPC: chain vs summed-waveform flash
   finder at a loose threshold and at the threshold whose fake rate matches the chain's.
B: the same for 2-20 MeV flashes vs how buried they are under a >= 3x brighter flash in the module.
C: fakes per TPC-event (module level, accidental-corrected) vs efficiency for 2-20 MeV flashes: every finder as a point.
python v13_flash_discovery_plot.py"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "flash_discovery")


def corr(f, a):
    return (f - a) / np.maximum(1 - a, 1e-9)


def main():
    z = np.load(os.path.join(OUT, "flash_discovery_data.npz"), allow_pickle=True)
    T_n, T_f, T_a, C_tot, C_acc = z["T_n"], z["T_f"], z["T_a"], z["C_tot"], z["C_acc"]
    names = list(z["names"]); E_LAB = list(z["E_LAB"]); B_LAB = list(z["B_LAB"]); n_tpc = int(z["n_tpc"]); n_ev = int(z["n_ev"])
    nF = len(names); ic = names.index("chain")
    # per finder: module-level purity and fakes per TPC-event
    n = np.maximum(C_tot[:, 0], 1); pm = C_tot[:, 2] / n; am = C_acc[:, 2] / n; pm_c = corr(pm, am)
    fakes = C_tot[:, 0] / n_tpc * (1 - pm_c)
    # matched-fake-rate summed finder = the S finder with fakes closest to the chain's
    iS = [i for i, nm in enumerate(names) if nm.startswith("S")]
    i_match = min(iS, key=lambda i: abs(fakes[i] - fakes[ic]))
    i_loose = iS[0]
    eff_E = lambda i: np.array([corr(T_f[a, :, i].sum() / max(T_n[a].sum(), 1), T_a[a, :, i].sum() / max(T_n[a].sum(), 1)) for a in range(len(E_LAB))])
    eff_B = lambda i, lo, hi: np.array([corr(T_f[lo:hi, b, i].sum() / max(T_n[lo:hi, b].sum(), 1), T_a[lo:hi, b, i].sum() / max(T_n[lo:hi, b].sum(), 1)) for b in range(len(B_LAB))])
    fig, ax = plt.subplots(1, 3, figsize=(21, 6))
    x = np.arange(len(E_LAB))
    for i, c, lab in ((i_loose, "0.6", f"flash finder {names[i_loose]} (loose: {fakes[i_loose]:.1f} fakes / TPC-event)"),
                      (i_match, "C1", f"flash finder {names[i_match]} (matched fake rate: {fakes[i_match]:.2f} / TPC-event)"),
                      (ic, "C3", f"chain t0s ({fakes[ic]:.2f} fakes / TPC-event)")):
        ax[0].plot(x, eff_E(i), "o-", color=c, lw=2, ms=7, label=lab)
    ax[0].set_xticks(x); ax[0].set_xticklabels([f"{l}\n(n={int(T_n[a].sum())})" for a, l in enumerate(E_LAB)])
    ax[0].set_xlabel("energy deposited in the TPC at the true flash time [MeV]"); ax[0].set_ylabel("discovery efficiency (accidental matches removed)")
    ax[0].set_title(f"A. which true flashes get a t0  ({n_ev} spills, {n_tpc} TPC-events)"); ax[0].set_ylim(0, 1.02); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=9, loc="lower right")
    xb = np.arange(len(B_LAB)); w = 0.26
    for k, (i, c, lab) in enumerate(((i_loose, "0.6", names[i_loose]), (i_match, "C1", names[i_match]), (ic, "C3", "chain"))):
        ax[1].bar(xb + (k - 1) * w, eff_B(i, 1, 3), w, color=c, label=lab)
    ax[1].set_xticks(xb); ax[1].set_xticklabels([f"{l}\n(n={int(T_n[1:3, b].sum())})" for b, l in enumerate(B_LAB)], fontsize=9)
    ax[1].set_xlabel("gap to the nearest >= 3x brighter true flash in the module [ticks of 16 ns]"); ax[1].set_ylabel("discovery efficiency, 2-20 MeV flashes")
    ax[1].set_title("B. buried flashes: 2-20 MeV deposits near a brighter flash"); ax[1].set_ylim(0, 1.02); ax[1].grid(alpha=0.3, axis="y"); ax[1].legend(fontsize=9)
    e_small = [eff_E(i)[1:3].mean() for i in range(nF)]
    for i in range(nF):
        c = "C3" if i == ic else ("C1" if names[i].startswith("S") else "C2")
        ax[2].plot(fakes[i], e_small[i], "o", color=c, ms=9); ax[2].annotate(names[i], (fakes[i], e_small[i]), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax[2].set_xscale("log"); ax[2].set_xlabel("fake t0s per TPC-event (no true flash in the module within 5 ticks)"); ax[2].set_ylabel("efficiency for 2-20 MeV flashes (mean of the 2-5 and 5-20 MeV bins)")
    ax[2].set_title("C. operating points: summed-waveform (S), 2-PE coincidence (C), chain"); ax[2].grid(alpha=0.3, which="both"); ax[2].set_ylim(0, 1.02)
    fig.suptitle("Flash discovery: conventional light-only flash finders vs the charge-light chain's t0s, ND simulation (v1.3 certification spills)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(OUT, "flash_discovery_v13.png"); fig.savefig(out, dpi=140); plt.close(fig); print(out)


if __name__ == "__main__":
    main()
