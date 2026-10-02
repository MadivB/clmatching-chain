"""Physics-facing evaluation of the v1.3 output on certification dumps: interaction-level recovery,
pile-up dependence, cluster type/energy, confidence and sigma_p reading keys, timing precision vs energy,
two-flash separation, saturation and position dependence.   python v13_physics_eval.py <arm_dir>"""
import glob
import json
import sys

import numpy as np

CUT = 10.0; NS = 16.0; WIN = 885.0


def wstd(v, w):
    mu = np.average(v, weights=w); return float(np.sqrt(np.average((v - mu) ** 2, weights=w)))


def wrob(v, w):
    o = np.argsort(v); cw = np.cumsum(w[o]); med = v[o][np.searchsorted(cw, 0.5 * cw[-1])]
    d = np.abs(v - med); o2 = np.argsort(d); cw2 = np.cumsum(w[o2]); return float(1.4826 * d[o2][np.searchsorted(cw2, 0.5 * cw2[-1])])


def main(arm):
    files = sorted(glob.glob(f"{arm}/full_dumps/*.npz"))
    inter = []            # per truth interaction-time group per event: E, n_tpc, E_found_frac, found(>=50%), n_clusters>=20MeV
    gap_bins = [0, 5, 10, 30, 100, 300, 1e9]; gap_acc = np.zeros((len(gap_bins) - 1, 2))
    ntpc_bins = [1, 2, 4, 8, 16, 1000]; ntpc_acc = np.zeros((len(ntpc_bins) - 1, 2))
    pair_bins = [0, 3, 5, 10, 20, 50, 1e9]; pair_both = np.zeros((len(pair_bins) - 1, 2))
    conf_bins = [-0.001, 0.2, 0.4, 0.6, 0.8, 0.95, 1.001]; conf_acc = np.zeros((len(conf_bins) - 1, 2)); conf_nan = np.zeros(2)
    sig_bins = [0, 0.3, 0.6, 1.0, 2.0, 4.0, 1e9]; sig_res = [[] for _ in sig_bins[:-1]]; sig_w = [[] for _ in sig_bins[:-1]]; sig_nan = np.zeros(2)
    e_bins = [0, 1.5, 5, 20, 100, 500, 1e9]; e_res = [[] for _ in e_bins[:-1]]; e_w = [[] for _ in e_bins[:-1]]
    type_acc = {}
    sat_bins = [0, 1, 5, 15, 40, 121]; sat_acc = np.zeros((len(sat_bins) - 1, 2))
    x_bins = np.linspace(-350, 350, 8); x_acc = np.zeros((7, 2)); y_bins = np.linspace(-215, 82, 7); y_acc = np.zeros((6, 2))
    win_e = np.zeros(3)   # energy with truth in window / before 0 / after 885
    ev_stats = []
    for ff in files:
        z = np.load(ff, allow_pickle=True)
        e = np.clip(z["e"], 0, None).astype(float); ts = z["ts_final"].astype(float); tru = z["truth_t0"].astype(float)
        labs = z["labels"].astype(int); tpc = z["tpc"].astype(int); x = z["x"].astype(float); y = z["y"].astype(float)
        conf = z["hit_conf93"].astype(float); sp = z["hit_sigma_p"].astype(float); veto = z["veto_mask"]
        ok = np.isfinite(tru); asg = np.isfinite(ts) & (ts >= 0)
        good = ok & asg & (np.abs(ts - tru) <= CUT)
        inwin = ok & (tru >= 0) & (tru <= WIN)
        win_e += [e[inwin].sum(), e[ok & (tru < 0)].sum(), e[ok & (tru > WIN)].sum()]
        okw = ok & inwin
        # ---- cluster table
        cl = {}
        for L in np.unique(labs[labs >= 0]):
            m = labs == L
            cl[int(L)] = (e[m].sum(), int(np.bincount(tpc[m]).argmax()), float(np.median(tru[m & ok])) if (m & ok).any() else np.nan)
        # ---- truth interaction-time groups per event (1-tick clustering of truth t0)
        tt = np.round(tru[okw]); ee = e[okw]; gg = good[okw]; tp = tpc[okw]
        order = np.argsort(tt); tt_s = tt[order]
        cuts = np.flatnonzero(np.diff(tt_s) > 1.5) + 1
        grp = np.zeros(tt.size, int); grp[order] = np.repeat(np.arange(len(cuts) + 1), np.diff(np.r_[0, cuts, tt.size]))
        n_grp = grp.max() + 1
        gE = np.bincount(grp, weights=ee, minlength=n_grp); gG = np.bincount(grp, weights=ee * gg, minlength=n_grp)
        gT = np.bincount(grp, weights=ee * tt, minlength=n_grp) / np.maximum(gE, 1e-9)
        for g in range(n_grp):
            mg = grp == g
            inter.append((gE[g], np.unique(tp[mg]).size, gG[g] / max(gE[g], 1e-9)))
        # ---- pile-up: per (tpc, group) energy, gap to the nearest other group in the same TPC
        key = tp * 100000 + grp
        uk, inv = np.unique(key, return_inverse=True)
        kE = np.bincount(inv, weights=ee); kG = np.bincount(inv, weights=ee * gg)
        k_tpc = uk // 100000; k_grp = uk % 100000; k_t = gT[k_grp]
        for t_ in np.unique(k_tpc):
            sel = np.flatnonzero(k_tpc == t_)
            times = k_t[sel]; n_here = sel.size
            for i, si in enumerate(sel):
                others = np.delete(times, i)
                gap = np.min(np.abs(others - times[i])) if others.size else 1e9
                b = min(np.searchsorted(gap_bins, gap, side="right") - 1, len(gap_bins) - 2); gap_acc[b] += [kE[si], kG[si]]
                b2 = min(np.searchsorted(ntpc_bins, n_here, side="right") - 1, len(ntpc_bins) - 2); ntpc_acc[b2] += [kE[si], kG[si]]
            # two-flash separation: pairs of groups >= 20 MeV each in this TPC
            big = sel[kE[sel] >= 20]
            for a in range(big.size):
                for b_ in range(a + 1, big.size):
                    dt = abs(k_t[big[a]] - k_t[big[b_]]); both = (kG[big[a]] / kE[big[a]] >= 0.5) and (kG[big[b_]] / kE[big[b_]] >= 0.5)
                    pb = min(np.searchsorted(pair_bins, dt, side="right") - 1, len(pair_bins) - 2); pair_both[pb] += [1, float(both)]
        # ---- confidence and sigma_p reading keys (hit-level, E-weighted)
        m = okw & asg
        cn = np.isnan(conf) & m; conf_nan += [e[cn].sum(), e[cn & good].sum()]
        for i in range(len(conf_bins) - 1):
            mb = m & (conf > conf_bins[i]) & (conf <= conf_bins[i + 1]); conf_acc[i] += [e[mb].sum(), e[mb & good].sum()]
        res = (ts - tru) * NS
        sn = np.isnan(sp) & good; sig_nan += [e[sn].sum(), 1]
        for i in range(len(sig_bins) - 1):
            mb = good & (sp >= sig_bins[i]) & (sp < sig_bins[i + 1])
            if mb.any():
                sig_res[i].append(res[mb]); sig_w[i].append(e[mb])
        cE = np.array([cl[int(L)][0] if L >= 0 else 0.0 for L in labs])
        for i in range(len(e_bins) - 1):
            mb = good & (labs >= 0) & (cE >= e_bins[i]) & (cE < e_bins[i + 1])
            if mb.any():
                e_res[i].append(res[mb]); e_w[i].append(e[mb])
        # ---- cluster type
        for r in json.loads(str(z["lab_rows"])):
            t_ = r.get("type") or "?"; Et = r["e_ok"] + r["e_wrong"]
            eb = 0 if Et < 20 else 1 if Et < 100 else 2
            type_acc.setdefault(t_, np.zeros((3, 2)))[eb] += [Et, r["e_ok"]]
        # ---- saturation (vetoed channels in the hit's light TPC = charge tpc ^ 1) and position
        nveto = veto.sum(1)
        light_tpc = tpc ^ 1
        nv = nveto[np.clip(light_tpc, 0, nveto.size - 1)]
        for i in range(len(sat_bins) - 1):
            mb = okw & (nv >= sat_bins[i]) & (nv < sat_bins[i + 1]); sat_acc[i] += [e[mb].sum(), e[mb & good].sum()]
        for i in range(7):
            mb = okw & (x >= x_bins[i]) & (x < x_bins[i + 1]); x_acc[i] += [e[mb].sum(), e[mb & good].sum()]
        for i in range(6):
            mb = okw & (y >= y_bins[i]) & (y < y_bins[i + 1]); y_acc[i] += [e[mb].sum(), e[mb & good].sum()]
        ev_stats.append((n_grp, e[okw].sum(), e[good & okw].sum() / max(e[okw].sum(), 1e-9)))
    I = np.array(inter)
    print(f"events {len(files)}; truth interaction-time groups per event median {np.median([s[0] for s in ev_stats]):.0f}")
    print(f"energy with truth time inside the 885-tick window {100*win_e[0]/win_e.sum():.2f} %, before the window {100*win_e[1]/win_e.sum():.2f} %, after {100*win_e[2]/win_e.sum():.2f} %")
    print("\nINTERACTION-LEVEL (truth interaction-time groups; found = >= 50 % of its energy within 10 ticks):")
    for lo, hi in ((1, 5), (5, 20), (20, 100), (100, 500), (500, 2000), (2000, 1e9)):
        m = (I[:, 0] >= lo) & (I[:, 0] < hi)
        if m.any():
            print(f"  E {lo:5.0f}-{hi:7.0f} MeV: n {m.sum():5d}  found {np.mean(I[m,2] >= 0.5):.3f}  >=90% {np.mean(I[m,2] >= 0.9):.3f}  median E-fraction correct {np.median(I[m,2]):.3f}  mean {np.mean(I[m,2]):.3f}  TPCs/interaction median {np.median(I[m,1]):.0f}")
    print("\nPILE-UP: accuracy of a (TPC, interaction) deposit vs time gap to the nearest other interaction in the same TPC [ticks]:")
    for i in range(len(gap_bins) - 1):
        if gap_acc[i, 0] > 0:
            print(f"  gap {gap_bins[i]:5.0f}-{gap_bins[i+1]:7.0f}: E {gap_acc[i,0]/1e3:7.1f} GeV  correct {gap_acc[i,1]/gap_acc[i,0]:.3f}")
    print("  vs number of interactions with deposits in the TPC:")
    for i in range(len(ntpc_bins) - 1):
        if ntpc_acc[i, 0] > 0:
            print(f"  n {ntpc_bins[i]:3d}-{ntpc_bins[i+1]:4d}: E {ntpc_acc[i,0]/1e3:7.1f} GeV  correct {ntpc_acc[i,1]/ntpc_acc[i,0]:.3f}")
    print("\nTWO-FLASH SEPARATION (pairs of >= 20 MeV deposits in one TPC; both >= 50 % correct):")
    for i in range(len(pair_bins) - 1):
        if pair_both[i, 0] > 0:
            print(f"  dt {pair_bins[i]:3.0f}-{pair_bins[i+1]:5.0f} ticks: pairs {int(pair_both[i,0]):5d}  both correct {pair_both[i,1]/pair_both[i,0]:.3f}")
    print("\nCONFIDENCE reading key (hit_t0_conf, E-weighted purity of assigned hits):")
    for i in range(len(conf_bins) - 1):
        if conf_acc[i, 0] > 0:
            print(f"  conf {max(conf_bins[i],0):.2f}-{min(conf_bins[i+1],1):.2f}: E share {conf_acc[i,0]/(conf_acc[:,0].sum()+conf_nan[0]):.3f}  purity {conf_acc[i,1]/conf_acc[i,0]:.3f}")
    print(f"  conf NaN (light-blind): E share {conf_nan[0]/(conf_acc[:,0].sum()+conf_nan[0]):.3f}  purity {conf_nan[1]/max(conf_nan[0],1e-9):.3f}")
    print("\nSIGMA_P reading key (correctly assigned hits, residual to truth in ns):")
    for i in range(len(sig_bins) - 1):
        if sig_res[i]:
            v = np.concatenate(sig_res[i]); w = np.concatenate(sig_w[i])
            print(f"  sigma_p {sig_bins[i]:.1f}-{min(sig_bins[i+1],99):.1f} ns: E share {w.sum()/sum(np.concatenate(ww).sum() for ww in sig_w if ww):.3f}  std {wstd(v,w):.2f}  robust {wrob(v,w):.2f}  |res|<5 ns {np.average(np.abs(v)<5, weights=w):.2f}")
    print("\nTIMING PRECISION vs cluster energy (correct hits, ns):")
    for i in range(len(e_bins) - 1):
        if e_res[i]:
            v = np.concatenate(e_res[i]); w = np.concatenate(e_w[i])
            print(f"  cluster E {e_bins[i]:5.1f}-{min(e_bins[i+1],1e5):7.1f} MeV: std {wstd(v,w):.2f}  robust {wrob(v,w):.2f}  median {np.average(v, weights=w):+.2f}")
    print("\nCLUSTER TYPE (frontend type; energy-weighted correct fraction by cluster energy <20 / 20-100 / >=100 MeV):")
    for t_, A in sorted(type_acc.items(), key=lambda kv: -kv[1][:, 0].sum()):
        print(f"  {t_:8s} E {A[:,0].sum()/1e3:7.1f} GeV: " + "  ".join(f"{A[i,1]/A[i,0]:.3f} ({A[i,0]/1e3:.1f} GeV)" if A[i,0] > 0 else "-" for i in range(3)))
    print("\nSATURATION: accuracy vs number of vetoed (saturated) channels in the hit's light TPC:")
    for i in range(len(sat_bins) - 1):
        if sat_acc[i, 0] > 0:
            print(f"  vetoed channels {sat_bins[i]:3d}-{sat_bins[i+1]:3d}: E {sat_acc[i,0]/1e3:7.1f} GeV  correct {sat_acc[i,1]/sat_acc[i,0]:.3f}")
    print("\nPOSITION: accuracy vs x (drift/side) and y (vertical):")
    print("  x: " + "  ".join(f"[{x_bins[i]:.0f},{x_bins[i+1]:.0f}) {x_acc[i,1]/x_acc[i,0]:.3f}" for i in range(7) if x_acc[i,0] > 0))
    print("  y: " + "  ".join(f"[{y_bins[i]:.0f},{y_bins[i+1]:.0f}) {y_acc[i,1]/y_acc[i,0]:.3f}" for i in range(6) if y_acc[i,0] > 0))
    ES = np.array(ev_stats)
    print("\nPER-SPILL: accuracy vs activity (interaction groups per spill):")
    for lo, hi in ((0, 60), (60, 90), (90, 120), (120, 1e9)):
        m = (ES[:, 0] >= lo) & (ES[:, 0] < hi)
        if m.any():
            print(f"  {lo:3.0f}-{hi:4.0f} groups: spills {m.sum():2d}  accuracy mean {np.mean(ES[m,2]):.4f} min {np.min(ES[m,2]):.4f}")


if __name__ == "__main__":
    main(sys.argv[1])
