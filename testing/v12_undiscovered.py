"""Anatomy of 'undiscovered' wrong energy per cluster class on chain dumps (v1.2.1).
undiscovered := wrong hit whose truth t0 has no reco family (rounded ts_final of any assigned hit
in the same TPC) within CUT ticks.  Pieces = (TPC, label, truth-time bucket) groups of such hits.
python v12_undiscovered.py <dump_dir>"""
import glob, os, sys
import numpy as np
CUT = 10.0; WIN_MAX = 885.0; PEAK = 105; LIGHT_SIG = 600.0
CLS = ["LE", "1.5-20", "20-100", ">=100"]


def main(d):
    keys = ["wrong", "und", "outwin", "in_unasg", "in_elsewhere", "in_local", "piece", "whole", "light", "nolight",
            "dt10_30", "dt30_100", "dt100p", "delay_pos", "delay_neg"]
    acc = {k: np.zeros(4) for k in keys}
    delays = {2: [], 3: []}; pieces = []
    for ff in sorted(glob.glob(f"{d}/full_dumps/*.npz")):
        z = np.load(ff, allow_pickle=True)
        e = np.clip(z["e"], 0, None).astype(float); tru = z["truth_t0"].astype(float); ts = z["ts_final"].astype(float)
        labs = z["labels"].astype(int); tpc = z["tpc"].astype(int)
        ok = np.isfinite(tru) & (tru >= 0); asg = np.isfinite(ts) & (ts >= 0)
        wrong = ok & (~asg | (np.abs(ts - tru) > CUT))
        imaged = set(int(l) for l in np.unique(z["amp_lab"]))
        labE = {int(L): e[labs == L].sum() for L in np.unique(labs[labs >= 0])}
        cE = np.array([labE.get(int(L), 0.0) if L >= 0 else 0.0 for L in labs])
        isimg = np.array([int(L) in imaged for L in labs])
        cls = np.zeros(labs.size, int)
        cls[isimg & (cE >= 1.5) & (cE < 20)] = 1; cls[isimg & (cE >= 20) & (cE < 100)] = 2; cls[isimg & (cE >= 100)] = 3
        fam_all = np.unique(np.round(ts[asg]))
        full = z["full_light"]; base = z["base_image"]; veto = z["veto_mask"]
        tag = os.path.basename(ff).replace(".npz", "")
        for k in range(4):
            acc["wrong"][k] += e[wrong & (cls == k)].sum()
        for c in np.unique(tpc[ok]):
            m = tpc == c
            fam = np.unique(np.round(ts[m & asg]))
            idx = np.flatnonzero(m & wrong)
            if idx.size == 0:
                continue
            dmin = np.abs(tru[idx][:, None] - fam[None, :]).min(axis=1) if fam.size else np.full(idx.size, np.inf)
            und = idx[dmin > CUT]
            if und.size == 0:
                continue
            key = labs[und] * 1000 + np.floor(tru[und] / 20).astype(int)
            for kk in np.unique(key):
                p = und[key == kk]
                k = int(cls[p[0]]); Ep = e[p].sum(); L = int(labs[p[0]])
                acc["und"][k] += Ep
                t_med = float(np.median(tru[p]))
                if t_med > WIN_MAX:
                    acc["outwin"][k] += Ep; kind = "outwin"
                    pieces.append((k, Ep, tag, int(c), L, t_med, np.nan, kind, np.nan, np.nan, np.nan, np.nan)); continue
                a = asg[p]
                if not a.any():
                    acc["in_unasg"][k] += Ep; kind = "unassigned"
                else:
                    kind = "assigned"
                elsewhere = bool(fam_all.size and np.abs(fam_all - t_med).min() <= CUT)
                acc["in_elsewhere" if elsewhere else "in_local"][k] += Ep
                mc = m & (labs == L)
                Ecl = e[mc].sum(); Ecl_w = e[mc & wrong].sum(); fw = Ecl_w / max(Ecl, 1e-9)
                acc["piece" if fw < 0.5 else "whole"][k] += Ep
                corr = mc & ok & ~wrong
                delay = t_med - float(np.median(tru[corr])) if corr.any() else np.nan
                if fw < 0.5 and np.isfinite(delay):
                    acc["delay_pos" if delay > 0 else "delay_neg"][k] += Ep
                    if k >= 2:
                        delays[k].append((delay, Ep))
                ts_med = float(np.median(ts[p][a])) if a.any() else np.nan
                if a.any():
                    dt = abs(ts_med - t_med)
                    acc["dt10_30" if dt < 30 else ("dt30_100" if dt < 100 else "dt100p")][k] += Ep
                s0 = int(round(t_med)) + PEAK
                R = np.nan
                if s0 + 8 < 1000 and s0 - 40 >= 0:
                    res = full[c].astype(float) - base[c].astype(float)
                    res = res - np.median(res[:, s0 - 40:s0 - 10], axis=1, keepdims=True)
                    chan_ok = ~veto[c]
                    R = float(res[chan_ok, s0 - 3:s0 + 8].max()) if chan_ok.any() else np.nan
                    acc["light" if R > LIGHT_SIG else "nolight"][k] += Ep
                pieces.append((k, Ep, tag, int(c), L, t_med, ts_med, kind, Ecl, fw, delay, R))
    print(f"{'class':>7} {'wrong':>8} {'und':>8} {'und%':>5} | {'outwin':>7} {'unasg':>6} {'elsewh':>7} {'local':>7} | {'piece':>7} {'whole':>7} | {'light':>7} {'nolite':>7} | {'dt<30':>6} {'30-100':>6} {'>100':>6} | {'dly+':>6} {'dly-':>6}")
    for k in range(4):
        w, u = acc["wrong"][k], acc["und"][k]
        print(f"{CLS[k]:>7} {w:8.0f} {u:8.0f} {100*u/max(w,1e-9):5.1f} | {acc['outwin'][k]:7.0f} {acc['in_unasg'][k]:6.0f} {acc['in_elsewhere'][k]:7.0f} {acc['in_local'][k]:7.0f} | {acc['piece'][k]:7.0f} {acc['whole'][k]:7.0f} | {acc['light'][k]:7.0f} {acc['nolight'][k]:7.0f} | {acc['dt10_30'][k]:6.0f} {acc['dt30_100'][k]:6.0f} {acc['dt100p'][k]:6.0f} | {acc['delay_pos'][k]:6.0f} {acc['delay_neg'][k]:6.0f}")
    print("(MeV, 39 events; outwin = truth t0 beyond the 885-tick search window; elsewh = a reco family at that time exists in another TPC of the event; piece = less than half of the host cluster is wrong; light = residual peak above " + str(LIGHT_SIG) + " at truth time)")
    for k in (3, 2):
        dl = np.array(delays[k])
        if dl.size:
            w = dl[:, 1]; dd = dl[:, 0]
            print(f"\nclass {CLS[k]} sub-piece delay vs host (ticks, E-weighted): "
                  + " ".join(f"{lo}..{hi}:{w[(dd>=lo)&(dd<hi)].sum():.0f}" for lo, hi in [(-1e9,-100),(-100,-10),(-10,10),(10,30),(30,100),(100,300),(300,1e9)]))

    for k in (3, 2):
        P = [p for p in pieces if p[0] == k and p[7] == "assigned" and p[9] < 0.5 and np.isfinite(p[10]) and p[10] > 10]
        Etot = sum(p[1] for p in P)
        Emich = sum(p[1] for p in P if p[1] <= 55)
        Elit = sum(p[1] for p in P if np.isfinite(p[11]) and p[11] > LIGHT_SIG)
        n_end = len(P)
        print(f"\nclass {CLS[k]} delayed sub-pieces (delay>10 ticks): n={n_end} E={Etot:.0f} MeV; piece E<=55 MeV: {100*Emich/max(Etot,1e-9):.0f}%; own flash visible (residual>{LIGHT_SIG:.0f}): {100*Elit/max(Etot,1e-9):.0f}%")
        Eall = sum(p[1] for p in pieces if p[0] == k)
        Ewhole_edge = sum(p[1] for p in pieces if p[0] == k and p[7] == "unassigned")
        print(f"   class total undiscovered {Eall:.0f}; whole-cluster unassigned (edge t0~0) {Ewhole_edge:.0f}")
    for k in (3, 2):
        P = [p for p in pieces if p[0] == k]
        P.sort(key=lambda r: -r[1])
        print(f"\nTop undiscovered pieces, class {CLS[k]}:")
        print(f"{'E_p':>6} {'event':>14} {'tpc':>3} {'lab':>5} {'truth':>7} {'reco':>7} {'kind':>10} {'E_cl':>7} {'fw':>4} {'delay':>7} {'lightR':>7}")
        for r in P[:14]:
            print(f"{r[1]:6.0f} {r[2]:>14} {r[3]:3d} {r[4]:5d} {r[5]:7.1f} {r[6]:7.1f} {r[7]:>10} {r[8]:7.0f} {r[9]:4.2f} {r[10]:7.1f} {r[11]:7.0f}")


if __name__ == "__main__":
    main(sys.argv[1])
