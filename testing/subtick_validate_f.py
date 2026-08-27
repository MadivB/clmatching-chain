"""Stage E+F validation: three-arm residual histograms + the stage-F
mechanism check (corr of per-hit correction vs true within-group deviation
- the decisive plot per the 2x2 handoff; 2x2 got corr 0.914, slope 0.985).

Usage: python subtick_validate_f.py v110SF_out/dumps subtick_out/v110
"""
import glob
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T = 16.0


def wstat(r, w):
    mu = np.average(r, weights=w) * T
    return mu, float(np.sqrt(np.average(np.square(r * T - mu), weights=w)))


def main():
    dumps, prefix = sys.argv[1], sys.argv[2]
    RB, RE, RF, E = [], [], [], []
    corr_x, corr_y, corr_w = [], [], []
    ev_off = 0
    for f in sorted(glob.glob(f"{dumps}/*.npz")):
        d = np.load(f)
        tsb = d["ts_before"].astype(np.float64)
        tse = d["ts_groupE"].astype(np.float64)
        tsf = d["ts_after"].astype(np.float64)
        tr = d["truth"].astype(np.float64)
        e = d["e"].astype(np.float64)
        gid = d["group_id"].astype(np.int64)
        m = np.isfinite(tr) & np.isfinite(tsb) & (tsb >= 0) & (e > 0)
        core = m & (np.abs(tsb - tr) <= 5.0)
        RB.append((tsb - tr)[core]); RE.append((tse - tr)[core])
        RF.append((tsf - tr)[core]); E.append(e[core])
        # mechanism: per-hit correction vs true deviation from group truth mean
        for g in np.unique(gid[core & (gid >= 0)]):
            gm = core & (gid == g)
            if gm.sum() < 2:
                continue
            tmu = np.average(tr[gm], weights=e[gm])
            corr_x.append((tr[gm] - tmu))
            corr_y.append((tsf[gm] - tse[gm]))
            corr_w.append(e[gm])
        ev_off += 1
    RB, RE, RF, E = map(np.concatenate, (RB, RE, RF, E))
    cx, cy, cw = map(np.concatenate, (corr_x, corr_y, corr_w))
    print(f"{RB.size} core hits, {E.sum()/1000:.1f} GeV; mechanism hits: {cx.size}")

    stats = {}
    for lab, r in [("before", RB), ("stage E", RE), ("stage F", RF)]:
        mu, sg = wstat(r, E)
        stats[lab] = (mu, sg)
        print(f"{lab:8s}: sigma {sg:.2f} ns  bias {mu:+.2f} ns")

    # eff invariance on all assigned hits is checked in-run; here core only.
    c_plain = float(np.corrcoef(cx, cy)[0, 1])
    wm_x = np.average(cx, weights=cw); wm_y = np.average(cy, weights=cw)
    cov = np.average((cx - wm_x) * (cy - wm_y), weights=cw)
    vx = np.average(np.square(cx - wm_x), weights=cw)
    vy = np.average(np.square(cy - wm_y), weights=cw)
    c_w = cov / np.sqrt(vx * vy)
    slope = cov / vx
    print(f"mechanism: corr {c_plain:.3f} (E-wt {c_w:.3f})  slope {slope:.3f}")
    print(f"correction std {np.sqrt(np.average(np.square(cy-wm_y),weights=cw))*T:.2f} ns "
          f"vs true within-group spread {np.sqrt(vx)*T:.2f} ns")

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0))
    b = np.linspace(-40, 40, 121)
    for lab, r, c in [("before (integer ticks)", RB, "#666666"),
                      ("stage E (group sub-tick)", RE, "#2471a3"),
                      ("stage F (per-hit)", RF, "#c0392b")]:
        mu, sg = stats[lab.split(" (")[0]]
        axes[0].hist(r * T, bins=b, weights=E, histtype="step", lw=1.9, color=c,
                     label=f"{lab}\n  sigma = {sg:.2f} ns, bias = {mu:+.2f} ns")
    axes[0].set_xlabel("reco t0 - truth  [ns]")
    axes[0].set_ylabel("energy [MeV] / bin")
    axes[0].set_title("per-hit residuals, correct core (|D|<=5 ticks), E-weighted")
    axes[0].legend(fontsize=8.5)

    hb = axes[1].hexbin(cx * T, cy * T, C=cw, reduce_C_function=np.sum,
                        gridsize=60, extent=(-25, 25, -25, 25), cmap="viridis",
                        mincnt=1)
    lim = np.array([-25, 25])
    axes[1].plot(lim, lim, "w--", lw=1.0, alpha=0.8)
    axes[1].plot(lim, slope * (lim / T - wm_x) * T + wm_y * T, "r-", lw=1.2,
                 label=f"fit slope {slope:.3f}")
    axes[1].set_xlabel("true per-hit deviation from group time  [ns]")
    axes[1].set_ylabel("stage-F per-hit correction  [ns]")
    axes[1].set_title(f"mechanism check: corr = {c_w:.3f} (2x2: 0.914)")
    axes[1].legend(fontsize=9)
    fig.colorbar(hb, ax=axes[1], label="energy [MeV]")
    fig.suptitle("ND stage E+F sub-tick / per-hit t0 refinement - file 0000001 (13 ev)")
    fig.tight_layout()
    out = f"{prefix}_stageF.png"
    fig.savefig(out, dpi=140)
    print("FIGURE:", out)


if __name__ == "__main__":
    main()
