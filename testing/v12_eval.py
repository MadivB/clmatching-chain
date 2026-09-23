"""Per-hit timing resolution of a chain run from its dumps:  python v12_eval.py <arm_dir> [<ref_dir>]
Core hits = correctly assigned (|reco - truth| <= 5 ticks in the REFERENCE or, without one, in the arm).
Reports E-weighted std and robust sigma of (ts_final - truth) per hit and per family, plus the
assignment efficiency (must be unchanged by a timing stage) and the delivered per-hit sigma."""
import glob, os, sys
import numpy as np
NS, CUT = 16.0, 10.0


def rob(v, w):
    o = np.argsort(v); cw = np.cumsum(w[o]); med = v[o][np.searchsorted(cw, 0.5 * cw[-1])]
    d = np.abs(v - med); o2 = np.argsort(d); cw2 = np.cumsum(w[o2]); return 1.4826 * d[o2][np.searchsorted(cw2, 0.5 * cw2[-1])]


def wsd(v, w):
    mu = np.average(v, weights=w); return np.sqrt(np.average((v - mu) ** 2, weights=w))


def main(arm, ref=None):
    files = {os.path.basename(f): f for f in glob.glob(f"{arm}/full_dumps/*.npz")}
    rfiles = {os.path.basename(f): f for f in glob.glob(f"{ref}/full_dumps/*.npz")} if ref else {}
    R, W, RR, F, FW, FR = [], [], [], [], [], []
    eff_num = eff_den = 0.0; n_sig = n_hits = 0; sig_all = []
    for k, f in sorted(files.items()):
        z = np.load(f, allow_pickle=True)
        e = np.clip(z["e"], 0, None).astype(float); tru = z["truth_t0"].astype(float); ts = z["ts_final"].astype(float)
        tpc = z["tpc"].astype(int); ok_t = np.isfinite(tru) & (tru >= 0); asg = np.isfinite(ts) & (ts >= 0)
        eff_num += e[ok_t & asg & (np.abs(ts - tru) <= CUT)].sum(); eff_den += e[ok_t].sum()
        tsr = None
        if k in rfiles:
            zr = np.load(rfiles[k], allow_pickle=True); tsr = zr["ts_final"].astype(float)
            if tsr.size != ts.size: tsr = None
        base = tsr if tsr is not None else ts
        core = ok_t & asg & np.isfinite(base) & (np.abs(base - tru) <= 5)
        R.append((ts[core] - tru[core]) * NS); W.append(e[core])
        if tsr is not None: RR.append((tsr[core] - tru[core]) * NS)
        if "hit_sigma_p" in z.files:
            sp = z["hit_sigma_p"].astype(float); n_sig += int(np.isfinite(sp).sum()); n_hits += sp.size; sig_all.append(sp[np.isfinite(sp)] * NS)
        # families = same base (reference) time per TPC
        for c in np.unique(tpc[core]):
            m = core & (tpc == c)
            for tv in np.unique(np.round(base[m], 3)):
                mm = m & (np.round(base, 3) == tv)
                if e[mm].sum() >= 5:
                    F.append(np.average((ts[mm] - tru[mm]) * NS, weights=e[mm])); FW.append(e[mm].sum())
                    if tsr is not None: FR.append(np.average((tsr[mm] - tru[mm]) * NS, weights=e[mm]))
    R, W = np.concatenate(R), np.concatenate(W); F, FW = np.array(F), np.array(FW)
    print(f"{arm}: {len(files)} events | assignment eff {100*eff_num/eff_den:.2f}% | per-hit (core, E-wt) std {wsd(R,W):.2f} robust {rob(R,W):.2f} ns | per-family std {wsd(F,FW):.2f} robust {rob(F,FW):.2f} ns"
          + (f" | delivered sigma_p on {100*n_sig/max(n_hits,1):.1f}% of hits, median {np.median(np.concatenate(sig_all)):.2f} ns" if sig_all else ""))
    if RR:
        RR = np.concatenate(RR); FR = np.array(FR)
        print(f"  reference {ref}: per-hit std {wsd(RR,W):.2f} robust {rob(RR,W):.2f} ns | per-family std {wsd(FR,FW):.2f} robust {rob(FR,FW):.2f} ns")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
