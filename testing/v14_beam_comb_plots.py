"""Figures for the beam-comb study (reads beam_comb_figs/beam_comb_data.npz written by
v14_beam_comb.py and the truth interaction times of the five flow files).
python v14_beam_comb_plots.py"""
import glob
import os

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "beam_comb_figs")
D = "/global/cfs/cdirs/dunepro/people/abooth/nd-production/output/MiniProdN5/run-ndlar-flow/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift/FLOW/0000000/"
SPILL = 1.2e9


def wrap(x, P):
    return ((x + P / 2) % P) - P / 2


def truth_arrays():
    tv_all, cont_all = [], []
    box = (-346, 346, -215, 82, 418, 913)
    for i in (1, 2, 3, 4, 5):
        f = h5py.File(D + f"MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.{i:07d}.FLOW.hdf5", "r")
        it = f["mc_truth/interactions/data"]
        tv = np.asarray(it["t_vert"], float) * 1000.0
        tin = tv - np.round(tv / SPILL) * SPILL
        xv, yv, zv = (np.asarray(it[k], float) for k in ("x_vert", "y_vert", "z_vert"))
        ins = (xv > box[0]) & (xv < box[1]) & (yv > box[2]) & (yv < box[3]) & (zv > box[4]) & (zv < box[5])
        tv_all.append(tin); cont_all.append(ins)
    return np.concatenate(tv_all), np.concatenate(cont_all)


def main():
    z = np.load(os.path.join(OUT, "beam_comb_data.npz"), allow_pickle=True)
    I, Fm, P, HB, EV, cal = z["I"], z["Fm"], float(z["P"]), float(z["half_bunch"]), z["ev"], z["cal"]
    cont, nfam, Esum, sigmin, dpr, dpt, d_rt, flight = I.T
    m = cont == 1
    tin, ins = truth_arrays()
    tin = tin[(tin > -100) & (tin < 12000)]; ins = ins[: tin.size] if ins.size != tin.size else ins
    paths = []

    # 1. truth: periodogram and spill profile
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    periods = np.arange(17.0, 22.0, 0.005)
    pw = np.array([np.abs(np.exp(2j * np.pi * tin / p).mean()) ** 2 * tin.size for p in periods])
    ax[0].plot(periods, pw, lw=0.8); ax[0].set_xlabel("trial period [ns]"); ax[0].set_ylabel("Rayleigh power")
    ax[0].set_title(f"ND truth interaction times, {tin.size} interactions (5 files): bunch period {periods[np.argmax(pw)]:.2f} ns")
    ax[1].hist(tin / 1000.0, bins=240, range=(0, 12), histtype="step", color="k")
    ax[1].set_xlabel("time in spill [us]"); ax[1].set_ylabel("interactions / 50 ns"); ax[1].set_title("spill profile: batch gaps every ~1.6 us")
    fig.tight_layout(); p1 = os.path.join(OUT, "fig1_truth_period_and_spill.png"); fig.savefig(p1, dpi=130); plt.close(fig); paths.append(p1)

    # 2. truth comb fold (all interactions vs contained)
    fig, ax = plt.subplots(figsize=(7, 4))
    ph = np.mod(tin, P)
    ax.hist(ph, bins=76, range=(0, P), histtype="step", color="k", label="all interactions")
    ax.hist(ph[ins[: ph.size]] if ins.size >= ph.size else ph, bins=76, range=(0, P), histtype="step", color="C3", label="contained in ND-LAr")
    ax.set_xlabel(f"vertex time mod {P:.2f} ns  [ns]"); ax.set_ylabel("interactions / 0.25 ns"); ax.set_title("truth bunch structure (flat-top ~8.6 ns bunches)")
    ax.legend(); fig.tight_layout(); p2 = os.path.join(OUT, "fig2_truth_comb.png"); fig.savefig(p2, dpi=130); plt.close(fig); paths.append(p2)

    # 3. reconstructed comb: contained interactions, truth vs reco phase relative to the event's comb phase
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(-P / 2, P / 2, 39)
    ax.hist(dpt[m], bins=bins, histtype="stepfilled", alpha=0.35, color="C0", label=f"truth vertex time (n={int(m.sum())})")
    ax.hist(dpr[m], bins=bins, histtype="step", lw=2, color="C3", label="reconstructed start time (earliest family apex)")
    ax.axvspan(-HB, HB, color="0.9", zorder=0); ax.axvline(-HB, color="0.5", ls="--"); ax.axvline(HB, color="0.5", ls="--")
    ax.set_xlabel("phase relative to the spill's bunch phase [ns]"); ax.set_ylabel("contained interactions / 0.5 ns")
    ax.set_title(f"beam comb from reconstruction: in-window truth {np.mean(np.abs(dpt[m])<=HB):.2f}, reco {np.mean(np.abs(dpr[m])<=HB):.2f} (random {2*HB/P:.2f}); bunch ID {np.mean(np.abs(dpr[m]-dpt[m])<P/2):.3f}", fontsize=9)
    ax.legend(fontsize=9); fig.tight_layout(); p3 = os.path.join(OUT, "fig3_reco_comb_contained.png"); fig.savefig(p3, dpi=130); plt.close(fig); paths.append(p3)

    # 4. resolution: reco start - vertex time, contained, by energy
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].hist(d_rt[m], bins=80, range=(-20, 20), histtype="step", color="k", lw=1.5)
    rob = 1.4826 * np.median(np.abs(d_rt[m]))
    ax[0].set_xlabel("reco start - true vertex time [ns] (global constant removed)"); ax[0].set_ylabel("contained interactions / 0.5 ns")
    ax[0].set_title(f"contained interactions: robust {rob:.2f} ns, core std {np.std(d_rt[m][np.abs(d_rt[m])<20]):.2f} ns")
    eb = [(5, 20), (20, 100), (100, 500), (500, 1e9)]; lab = ["5-20", "20-100", "100-500", ">500 MeV"]
    res = [1.4826 * np.median(np.abs(d_rt[m & (Esum >= lo) & (Esum < hi)])) if (m & (Esum >= lo) & (Esum < hi)).any() else np.nan for lo, hi in eb]
    pur = [np.mean(np.abs(dpr[mm] - dpt[mm]) < P / 2) if (mm := m & (Esum >= lo) & (Esum < hi)).any() else np.nan for lo, hi in eb]
    x = np.arange(4)
    ax[1].bar(x - 0.2, res, 0.4, color="C0", label="resolution [ns]")
    ax2 = ax[1].twinx(); ax2.bar(x + 0.2, [100 * v for v in pur], 0.4, color="C3", label="bunch ID purity [%]"); ax2.set_ylim(80, 100)
    ax[1].set_xticks(x); ax[1].set_xticklabels(lab); ax[1].set_ylabel("robust resolution [ns]"); ax2.set_ylabel("bunch identification purity [%]")
    ax[1].set_ylim(0, 1.45 * np.nanmax(res)); ax2.set_ylim(80, 103)
    ax[1].set_title("by interaction energy"); ax[1].legend(loc="upper left", fontsize=9); ax2.legend(loc="upper right", fontsize=9)
    fig.tight_layout(); p4 = os.path.join(OUT, "fig4_resolution_by_energy.png"); fig.savefig(p4, dpi=130); plt.close(fig); paths.append(p4)

    # 5. external (rock) interactions: flight time and reco phase
    mr = ~m
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].hist(flight[mr], bins=100, range=(0, 200), histtype="step", color="k", lw=1.5)
    ax[0].set_xlabel("true flight time: detector entry - vertex time [ns]"); ax[0].set_ylabel("external interactions / 2 ns"); ax[0].set_title(f"rock / external interactions (n={int(mr.sum())})")
    ax[1].hist(dpr[mr], bins=bins, histtype="step", lw=2, color="C3", label="reco start phase (external)")
    ax[1].hist(dpr[m], bins=bins, histtype="step", lw=1.5, color="C0", label="reco start phase (contained)")
    ax[1].axvspan(-HB, HB, color="0.9", zorder=0); ax[1].set_xlabel("phase relative to the bunch phase [ns]"); ax[1].set_ylabel("interactions / 0.5 ns")
    ax[1].set_title("external interactions sit off the comb by their flight time (mod 19.09 ns)", fontsize=9); ax[1].legend(fontsize=9)
    fig.tight_layout(); p5 = os.path.join(OUT, "fig5_external_flight_time.png"); fig.savefig(p5, dpi=130); plt.close(fig); paths.append(p5)

    # 6. spill phase from reconstruction alone vs truth, per event; global constant per event
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    dph = EV[:, 5][np.isfinite(EV[:, 5])]
    ax[0].hist(dph, bins=40, range=(-10, 10), histtype="step", color="k", lw=1.5)
    ax[0].set_xlabel("spill bunch phase: reconstruction alone - truth [ns]"); ax[0].set_ylabel("spills")
    ax[0].set_title(f"beam RF phase per spill from light alone: robust {1.4826*np.median(np.abs(dph)):.2f} ns ({dph.size} spills)")
    ax[1].plot(cal, "k.", ms=4); ax[1].axhline(np.median(cal), color="C3", ls="--")
    ax[1].set_xlabel("event"); ax[1].set_ylabel("global reco time constant [ns]"); ax[1].set_title(f"global constant {np.median(cal):+.2f} ns, spread {1.4826*np.median(np.abs(cal-np.median(cal))):.2f} ns")
    fig.tight_layout(); p6 = os.path.join(OUT, "fig6_spill_phase_and_constant.png"); fig.savefig(p6, dpi=130); plt.close(fig); paths.append(p6)

    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
