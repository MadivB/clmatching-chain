"""The 2x2-style beam-comb figure for ND simulation (cf. 2x2QLMatching/StartTimeFit/beam_comb_a10.png).

Everything on the reconstruction side is truth-free:
  * family times come from the v1.3 precision stage (apex time = the family's reference start time);
  * families are grouped into interactions by reconstructed time alone (single linkage, gap < 10 ns);
  * an interaction's start time = the earliest family of the group among those timed to sigma < 1 ns
    (a badly timed family is not allowed to define the start; fallback: earliest of all);
  * "not entering" = no family of the group has hits within 10 cm of the upstream face, the two x side
    faces or the top face of the event's hit box (the faces beam-induced rock particles enter through;
    the downstream face and the bottom are exits for forward-going products);
  * "E > 100 MeV" = summed family energy of the group (rock neutrons/photons make small deep blips);
  * "elite" = not entering, E > 100 MeV, start-family sigma < 0.5 ns.
Truth is used only for (i) the per-spill time frame (the RWM analogue in data: the event's tick frame
relative to the spill start), (ii) ONE global time constant for the detector (-5.6 ns), (iii) the
overlaid reference curves (true vertex times of contained interactions) and (iv) panel C, the residual
to the true vertex time, which data cannot show.
Panel A: comb search = Rayleigh power vs fold period of the reconstructed interaction start times (linear; the truth peak is off scale).
Panel B: folded phase at the bunch spacing (genuine 4-period fold), densities.
Panel C: reconstructed start - true vertex time, truth-contained members of each population.
python v14_beam_comb_nd_figure.py  -> beam_comb_figs/beam_comb_nd_v13.png"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PI = "--pi" in sys.argv     # two-curve version: all reconstructed interactions + truth (beam_comb_nd_v13_pi.png)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "beam_comb_figs")
GAP_NS = 10.0        # time-linkage gap between families of different interactions
FACE_CM = 10.0       # a family "touches" a face when it has hits within this distance of it
HALF_BUNCH = 4.3     # half of the ~8.6 ns flat-top bunch
SIG_START = 1.0      # families timed worse than this cannot define the start
SIG_ELITE = 0.5
E_MIN = 100.0


def rayleigh(t, periods):
    return np.array([np.abs(np.exp(2j * np.pi * t / p).mean()) ** 2 * t.size for p in periods])


def build_interactions(fam, ev_box):
    """fam cols: 0 t_reco_abs(per-event cal removed) 1 t_true_abs 2 E 3 sig 4 apex margin 5 n_hits 6 ev 7 cal
    8 matched 9 cont 10-12 apex 13-15 hit min 16-18 hit max 19-21 truth-earliest hit 22 v_entry 23 v_time."""
    ev = fam[:, 6].astype(int); cal = fam[:, 7]; sig = fam[:, 3]; E = fam[:, 2]
    cal_ev = np.array([cal[ev == e][0] for e in np.unique(ev)]); cal_global = float(np.median(cal_ev))
    t = fam[:, 0] + cal - cal_global
    B = ev_box[ev]; lo = fam[:, 13:16]; hi = fam[:, 16:19]
    enters = (lo[:, 2] < B[:, 4] + FACE_CM) | (lo[:, 0] < B[:, 0] + FACE_CM) | (hi[:, 0] > B[:, 1] - FACE_CM) | (hi[:, 1] > B[:, 3] - FACE_CM)
    rows = []
    for e in np.unique(ev):
        m = np.flatnonzero(ev == e); o = m[np.argsort(t[m])]; tt = t[o]
        for grp in np.split(o, np.flatnonzero(np.diff(tt) > GAP_NS) + 1):
            good = sig[grp] < SIG_START
            i0 = grp[good][np.argmin(t[grp][good])] if good.any() else grp[np.argmin(t[grp])]
            rows.append((t[i0], sig[i0], E[grp].sum(), float(enters[grp].any()), len(grp), fam[i0, 1], fam[i0, 9], fam[i0, 23]))
    R = np.array(rows)
    return R, cal_global, 1.4826 * np.median(np.abs(cal_ev - cal_global))


def main():
    z = np.load(os.path.join(OUT, "beam_comb_data.npz"), allow_pickle=True)
    fam = z["fam_abs"]; IA = z["inter_abs"]; ev_box = z["ev_box"]
    R, cal_global, cal_spread = build_interactions(fam, ev_box)
    t_all, sig0, E_g, enters, nfam, t_true0, cont0, v_time0 = R.T
    n_ev = ev_box.shape[0]
    truth_c = IA[IA[:, 3] == 1, 1]                 # true vertex times of contained interactions (reference)
    periods = np.arange(17.0, 22.0, 0.002)
    pw_t = rayleigh(truth_c, periods); Pt = float(periods[np.argmax(pw_t)])
    ne = enters == 0
    pop = [
        ("all reconstructed interactions", np.ones(len(R), bool), "C0", True),
        (f"not entering (no family within {FACE_CM:.0f} cm of upstream/side/top faces)", ne, "C2", False),
        (f"not entering & E > {E_MIN:.0f} MeV", ne & (E_g > E_MIN), "C1", False),
        (f"elite: not entering & E > {E_MIN:.0f} MeV & start sigma < {SIG_ELITE} ns", ne & (E_g > E_MIN) & (sig0 < SIG_ELITE), "C3", False),
    ]
    if PI:
        pop = pop[:1]
    fig = plt.figure(figsize=(24, 7.2))
    fig.suptitle(f"Beam bunch structure from ND-LAr light timing alone (v1.3 precision stage) — ND simulation, {n_ev} spills, "
                 f"spill-start (RWM-analogue) frame, one global time constant ({cal_global:+.2f} ns, spill-to-spill spread {cal_spread:.2f} ns)",
                 fontsize=13.5, fontweight="bold")
    axA = fig.add_subplot(1, 3, 1); axB = fig.add_subplot(1, 3, 2); axC = fig.add_subplot(1, 3, 3)
    stats = []
    for name, m, c, filled in pop:
        t = t_all[m]
        pw = rayleigh(t, periods); Pk = float(periods[np.argmax(pw)])
        axA.plot(periods, pw, lw=0.9, color=c, label=f"{name.split(' (')[0]} (N={t.size}): P = {pw.max():.0f}")
        stats.append((name, m, c, filled, Pk, float(pw.max())))
    axA.plot(periods, pw_t, lw=0.8, color="0.45", ls="--", label=f"truth, contained vertices (N={truth_c.size}): P = {pw_t.max():.0f}")
    ymax = 1.3 * max(pk for *_, pk in stats)
    axA.set_ylim(0, ymax); axA.set_xlim(17, 22)
    axA.annotate(f"truth peak {pw_t.max():.0f} (off scale)\nbunch spacing {Pt:.3f} ns", xy=(Pt, ymax), xytext=(Pt + 0.6, 0.6 * ymax), fontsize=9, color="k",
                 arrowprops=dict(arrowstyle="->", color="k", lw=0.7))
    axA.set_xlabel("fold period T [ns]"); axA.set_ylabel("Rayleigh power (noise ~ 1)")
    axA.set_title("A. comb search on reconstructed interaction start times")
    axA.legend(fontsize=9, loc="upper right", framealpha=0.95); axA.grid(alpha=0.25)
    # ---- B: genuine 4-period fold at the truth spacing; one common phase shift (truth bunch centre at T/2)
    P4 = 4 * Pt
    ph0 = (np.angle(np.exp(2j * np.pi * truth_c / Pt).mean()) / (2 * np.pi) * Pt) % Pt
    shift = Pt / 2 - ph0
    bins = np.linspace(0, P4, int(round(P4 / 1.0)) + 1)
    for name, m, c, filled, Pk, pk in stats:
        t = t_all[m]
        axB.hist(np.mod(t + shift, P4), bins=bins, density=True, histtype="stepfilled" if filled else "step",
                 alpha=0.35 if filled else 1.0, lw=1.8, color=c, label=f"{name.split(' (')[0]} (N={t.size})")
    axB.hist(np.mod(truth_c + shift, P4), bins=np.linspace(0, P4, int(round(P4 / 0.5)) + 1), density=True, histtype="step", lw=1.1, ls="--", color="k",
             label=f"truth: true vertex times, contained (N={truth_c.size})")
    for k in range(1, 4):
        axB.axvline(k * Pt, color="0.5", ls=":")
    for k in range(4):
        axB.axvspan(k * Pt + Pt / 2 - HALF_BUNCH, k * Pt + Pt / 2 + HALF_BUNCH, color="0.93", zorder=0)
    axB.axhline(1.0 / P4, color="k", ls="--", lw=0.8)
    axB.set_xlabel(f"spill-start-referenced start time mod {P4:.3f} ns  (genuine 4-period fold)"); axB.set_ylabel("density")
    axB.set_title(f"B. folded phase at the bunch spacing {Pt:.3f} ns  (grey: +-{HALF_BUNCH} ns bunch flat-top)")
    axB.legend(fontsize=10 if PI else 8, loc="upper right"); axB.grid(alpha=0.3); axB.set_xlim(0, P4)
    # ---- C: simulation only: residual to the true vertex time for the truth-contained members of each population
    lines = []
    for name, m, c, filled, Pk, pk in stats:
        mm = m & np.isfinite(t_true0) & (cont0 == 1)
        d = t_all[mm] - v_time0[mm]
        if d.size < 10:
            continue
        rob = 1.4826 * np.median(np.abs(d - np.median(d))); sd = float(np.std(d[np.abs(d) < 15]))
        axC.hist(d, bins=np.linspace(-15, 15, 121), density=True, histtype="stepfilled" if filled else "step", alpha=0.35 if filled else 1.0, lw=1.8, color=c,
                 label=f"{name.split(' (')[0]}, truth-contained (N={d.size}): \u03c3 = {sd:.2f} ns, |d| < {HALF_BUNCH} ns: {np.mean(np.abs(d) < HALF_BUNCH):.2f}")
        lines.append((name, d.size, rob, float(np.mean(np.abs(d) < HALF_BUNCH)), float(np.median(d)), sd))
    axC.axvspan(-HALF_BUNCH, HALF_BUNCH, color="0.93", zorder=0)
    axC.set_xlabel("reconstructed start - true vertex time [ns]"); axC.set_ylabel("density / 0.25 ns")
    axC.set_title("C. simulation only: residual to the true vertex time  (\u03c3 over the plotted range)")
    axC.legend(fontsize=10 if PI else 8, loc="upper left"); axC.grid(alpha=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(OUT, "beam_comb_nd_v13_pi.png" if PI else "beam_comb_nd_v13.png"); fig.savefig(out, dpi=140); plt.close(fig)
    print(out)
    # ---- numbers for the report
    print(f"spills {n_ev}; families {len(fam)}; reconstructed interactions (time-linked) {len(R)}; global constant {cal_global:+.2f} ns (spill spread {cal_spread:.2f} ns)")
    print(f"truth period {Pt:.3f} ns, truth power {pw_t.max():.0f} (N={truth_c.size}), |R|={np.sqrt(pw_t.max()/truth_c.size):.2f}")
    for name, m, c, filled, Pk, pk in stats:
        t = t_all[m]; ph = ((t - ph0 + Pt / 2) % Pt) - Pt / 2
        known = np.isfinite(cont0[m])
        print(f"  {name}: N={t.size}, peak {Pk:.3f} ns, power {pk:.0f}, |R|={np.sqrt(pk/t.size):.2f}; in-window {np.mean(np.abs(ph) <= HALF_BUNCH):.3f} (random {2*HALF_BUNCH/Pt:.2f}); "
              f"truth-contained purity {np.mean(cont0[m][known]==1):.2f}")
    for name, n, rob, fw, med, sd in lines:
        print(f"  residual {name.split(' (')[0]}: N={n} sigma {sd:.2f} ns (|d|<15) robust {rob:.2f} median {med:+.2f} ns in-bunch {fw:.2f}")


if __name__ == "__main__":
    main()
