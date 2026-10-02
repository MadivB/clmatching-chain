"""Beam-comb figure with every interaction PROJECTED to a common beam-time reference (user's proposal:
do not throw entering interactions away; assume c and project every start point to the beam origin).

Physics: everything that comes out of the beam travels at ~c along the beam axis n (the neutrino to its
vertex, a rock muon on to the detector), so for any start point r observed at time t the time the parent
left the target is   t_beam = t - (r . n) / c   (plane wave: the target is ~570 m upstream, the transverse
curvature term is < 0.05 ns).  Projection removes the flight-time spread of entering interactions
without a containment cut, and every interaction becomes a comb sample.

SIMULATION CAVEAT (measured, see VERSION_HISTORY 2026-09-24): the ND MC assigns the bunch time AT the
vertex for every interaction (rock vertices at z = -20 m get the same bunch phase as contained ones) and
only the products' flight from the vertex is simulated.  In the MC as delivered the projection therefore
smears the comb (truth itself: |R| 0.78 -> 0.14 for contained vertices) and cannot repair rock muons,
because the delay they carry is set by an unknown rock-vertex position.  To evaluate the proposal we
EMULATE the physical convention: every true interaction's time is shifted by + (r_v . n)/c, the target-to-
vertex flight the MC left out (a per-interaction constant taken from truth, applied identically to the
reconstructed time of that interaction).  On top of that the reconstruction-side projection is truth-free:
families time-linked into interactions (gap 10 ns), start = earliest family timed to sigma < 1 ns, start
point = the upstream-most member hit of that family along n (beam-direction prior), t_beam = t - (r . n)/c.
Reconstructed interactions whose start family is not associated with a true vertex cannot receive the
emulated shift and are left out (counted).  n = mean true neutrino direction (data: survey).

Panel A: Rayleigh comb search on projected start times.  Panel B: 4-period fold.
Panel C: projected reco - projected truth, truth-matched interactions, contained and rock stacked.
python v14_beam_comb_projected.py  -> beam_comb_figs/beam_comb_nd_v13_projected.png"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "beam_comb_figs")
C = 29.9792458       # cm / ns
GAP_NS = 10.0
SIG_START = 1.0
HALF_BUNCH = 4.3
Z_FRONT = 418.0      # cm: the ND-LAr front (upstream) face; the common reference plane is the plane through (0, 0, Z_FRONT) normal to n


def rayleigh(t, periods):
    return np.array([np.abs(np.exp(2j * np.pi * t / p).mean()) ** 2 * t.size for p in periods])


def build(fam):
    """per reconstructed interaction: t_start, r_up(3), sig, E, n_fam, t_true_v, cont, v_xyz(3)"""
    ev = fam[:, 6].astype(int); cal = fam[:, 7]; sig = fam[:, 3]; E = fam[:, 2]
    cal_ev = np.array([cal[ev == e][0] for e in np.unique(ev)]); cal_g = float(np.median(cal_ev))
    t = fam[:, 0] + cal - cal_g
    rows = []
    for e in np.unique(ev):
        m = np.flatnonzero(ev == e); o = m[np.argsort(t[m])]; tt = t[o]
        for grp in np.split(o, np.flatnonzero(np.diff(tt) > GAP_NS) + 1):
            good = sig[grp] < SIG_START
            i0 = grp[good][np.argmin(t[grp][good])] if good.any() else grp[np.argmin(t[grp])]
            rows.append((t[i0], *fam[i0, 24:27], sig[i0], E[grp].sum(), len(grp), fam[i0, 23], fam[i0, 9], *fam[i0, 27:30]))
    return np.array(rows), cal_g


def main():
    z = np.load(os.path.join(OUT, "beam_comb_data.npz"), allow_pickle=True)
    fam = z["fam_abs"]; TA = z["truth_all"]; n = np.asarray(z["nhat"], float); n_ev = z["ev_box"].shape[0]
    R, cal_g = build(fam)
    t_start, r_up, cont0, t_true_v, v_xyz = R[:, 0], R[:, 1:4], R[:, 8], R[:, 7], R[:, 9:12]
    known = np.isfinite(v_xyz[:, 0])
    r_front = np.array([0.0, 0.0, Z_FRONT]); s_front = float(r_front @ n)   # reference: time at which the parent crossed the ND front plane
    # --- MC as delivered: projection applied directly (diagnostic only)
    tb_raw = t_start - (r_up @ n - s_front) / C
    # --- physical convention: + target flight (r_v.n)/c per true interaction, then the truth-free projection
    t_phys = t_start + v_xyz @ n / C
    t_beam = t_phys - (r_up @ n - s_front) / C
    tv_all, v_all, cont_all = TA[:, 1], TA[:, 2:5], TA[:, 5]
    tv_phys = tv_all + v_all @ n / C                    # what a physical MC would have simulated
    tb_truth = tv_phys - (v_all @ n - s_front) / C      # projected truth = the bunch times at the front plane
    periods = np.arange(17.0, 22.0, 0.002)
    pw_t = rayleigh(tb_truth, periods); Pt = float(periods[np.argmax(pw_t)])
    pw_r = rayleigh(t_beam[known], periods); Pr = float(periods[np.argmax(pw_r)])
    pw_phys_u = rayleigh(t_phys[known], periods)        # physical convention, NOT projected
    pw_raw = rayleigh(tb_raw, periods)                  # MC as delivered, projected
    pw_sim_u = rayleigh(t_start[known], periods)        # MC as delivered, not projected (same interactions)
    Nk = int(known.sum())
    fig = plt.figure(figsize=(24, 7.2))
    fig.suptitle(f"Beam bunch structure from ND-LAr light timing with every interaction projected along the beam to the ND front plane:  t - ((r - r_front).n)/c  "
                 f"— ND simulation, {n_ev} spills, physical time convention emulated", fontsize=13, fontweight="bold")
    axA = fig.add_subplot(1, 3, 1); axB = fig.add_subplot(1, 3, 2); axC = fig.add_subplot(1, 3, 3)
    axA.plot(periods, pw_phys_u, lw=0.9, color="0.6", label=f"same interactions, not projected (N={Nk}): P = {pw_phys_u.max():.0f}")
    axA.plot(periods, pw_r, lw=0.9, color="C0", label=f"all reconstructed interactions, projected (N={Nk}): P = {pw_r.max():.0f}")
    axA.plot(periods, pw_t, lw=0.8, color="0.45", ls="--", label=f"truth, all interactions incl. rock (N={tb_truth.size}): P = {pw_t.max():.0f}")
    ymax = 1.3 * float(pw_r.max())
    axA.set_ylim(0, ymax); axA.set_xlim(17, 22)
    axA.annotate(f"truth peak {pw_t.max():.0f} (off scale)\nbunch spacing {Pt:.3f} ns", xy=(Pt, ymax), xytext=(Pt + 0.6, 0.6 * ymax), fontsize=9, color="k",
                 arrowprops=dict(arrowstyle="->", color="k", lw=0.7))
    axA.set_xlabel("fold period T [ns]"); axA.set_ylabel("Rayleigh power (noise ~ 1)")
    axA.set_title("A. comb search on start times projected to the ND front plane (no containment cut)")
    axA.legend(fontsize=9, loc="upper right", framealpha=0.95); axA.grid(alpha=0.25)
    # ---- B: 4-period fold, one common shift (truth bunch centre at T/2)
    P4 = 4 * Pt
    ph0 = (np.angle(np.exp(2j * np.pi * tb_truth / Pt).mean()) / (2 * np.pi) * Pt) % Pt
    shift = Pt / 2 - ph0
    axB.hist(np.mod(t_beam[known] + shift, P4), bins=np.linspace(0, P4, int(round(P4 / 1.0)) + 1), density=True, histtype="stepfilled", alpha=0.35, color="C0",
             label=f"all reconstructed interactions, projected (N={Nk})")
    axB.hist(np.mod(tb_truth + shift, P4), bins=np.linspace(0, P4, int(round(P4 / 0.5)) + 1), density=True, histtype="step", lw=1.1, ls="--", color="k",
             label=f"truth: all true interactions incl. rock, projected (N={tb_truth.size})")
    for k in range(1, 4):
        axB.axvline(k * Pt, color="0.5", ls=":")
    for k in range(4):
        axB.axvspan(k * Pt + Pt / 2 - HALF_BUNCH, k * Pt + Pt / 2 + HALF_BUNCH, color="0.93", zorder=0)
    axB.axhline(1.0 / P4, color="k", ls="--", lw=0.8)
    axB.set_xlabel(f"time at the ND front plane  t - ((r - r_front).n)/c  mod {P4:.3f} ns  (genuine 4-period fold)"); axB.set_ylabel("density")
    axB.set_title(f"B. folded phase at the bunch spacing {Pt:.3f} ns  (grey: +-{HALF_BUNCH} ns bunch flat-top)")
    axB.legend(fontsize=9.5, loc="upper right"); axB.grid(alpha=0.3); axB.set_xlim(0, P4)
    # ---- C: projected reco - projected truth (= bunch time of the matched vertex), contained and rock stacked
    d = t_beam - (t_true_v + s_front / C)
    dc, de = d[known & (cont0 == 1)], d[known & (cont0 == 0)]
    rob = lambda x: 1.4826 * np.median(np.abs(x - np.median(x))); sd = lambda x: float(np.std(x[np.abs(x) < 15]))
    axC.hist([dc, de], bins=np.linspace(-15, 15, 121), stacked=True, color=["C0", "C1"], alpha=0.6,
             label=[f"contained (N={dc.size}): \u03c3 = {sd(dc):.2f} ns, |d| < {HALF_BUNCH} ns: {np.mean(np.abs(dc) < HALF_BUNCH):.2f}",
                    f"rock / external (N={de.size}): \u03c3 = {sd(de):.2f} ns, |d| < {HALF_BUNCH} ns: {np.mean(np.abs(de) < HALF_BUNCH):.2f}"])
    axC.axvspan(-HALF_BUNCH, HALF_BUNCH, color="0.93", zorder=0)
    axC.set_xlabel("projected reconstructed start - true time of the parent at the ND front plane [ns]"); axC.set_ylabel("interactions / 0.25 ns")
    axC.set_title("C. simulation only: residual after projection (stacked; \u03c3 over the plotted range)")
    axC.legend(fontsize=9.5, loc="upper left"); axC.grid(alpha=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(OUT, "beam_comb_nd_v13_projected.png"); fig.savefig(out, dpi=140); plt.close(fig)
    print(out)
    # ---- numbers
    ph = lambda x: ((x - ph0 + Pt / 2) % Pt) - Pt / 2
    Rv = lambda pw, N: np.sqrt(pw.max() / N)
    print(f"beam axis n = {np.round(n, 4)}; spills {n_ev}; reconstructed interactions {len(R)}, with a true vertex for the emulated shift {Nk} (dropped {len(R)-Nk})")
    print(f"truth: bunch times of all {tb_truth.size} interactions P={pw_t.max():.0f} |R|={Rv(pw_t, tb_truth.size):.2f} in-window {np.mean(np.abs(ph(tb_truth))<=HALF_BUNCH):.3f}")
    print(f"MC as delivered: not projected P={pw_sim_u.max():.0f} |R|={Rv(pw_sim_u, Nk):.2f};  projected P={pw_raw.max():.0f} |R|={Rv(pw_raw, len(R)):.2f} (projection smears the MC's vertex-time convention)")
    print(f"physical convention: not projected P={pw_phys_u.max():.0f} |R|={Rv(pw_phys_u, Nk):.2f};  PROJECTED P={pw_r.max():.0f} at {Pr:.3f} ns |R|={Rv(pw_r, Nk):.2f} in-window {np.mean(np.abs(ph(t_beam[known]))<=HALF_BUNCH):.3f} (random {2*HALF_BUNCH/Pt:.2f})")
    print(f"residual projected reco - bunch time: contained sigma {sd(dc):.2f} (|d|<15) robust {rob(dc):.2f} ns median {np.median(dc):+.2f} in-bunch {np.mean(np.abs(dc)<HALF_BUNCH):.2f}; rock sigma {sd(de):.2f} robust {rob(de):.2f} median {np.median(de):+.2f} in-bunch {np.mean(np.abs(de)<HALF_BUNCH):.2f}; all {np.mean(np.abs(d[known])<HALF_BUNCH):.2f}")
    du = t_start - t_true_v
    print(f"reference, MC convention without projection: contained robust {rob(du[known & (cont0==1)]):.2f} ns; rock robust {rob(du[known & (cont0==0)]):.2f} median {np.median(du[known & (cont0==0)]):+.1f} ns (flight delay)")


if __name__ == "__main__":
    main()
