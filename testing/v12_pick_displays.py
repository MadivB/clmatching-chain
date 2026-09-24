"""Rank (event, TPC) by wrong energy per mistake class on chain dumps, to pick display examples.
Classes: big (>=100 MeV imaged cluster wrong), mid (20-100), small imaged (1.5-20), LE (noise + <1.5 MeV),
undiscovered (wrong hits whose truth time has no reco family within 10 ticks in the TPC).
python v12_pick_displays.py <dump_dir>"""
import glob, os, sys
import numpy as np
CUT = 10.0


def main(d):
    cand = []
    for ff in sorted(glob.glob(f"{d}/full_dumps/*.npz")):
        z = np.load(ff, allow_pickle=True)
        e = np.clip(z["e"], 0, None).astype(float); tru = z["truth_t0"].astype(float); ts = z["ts_final"].astype(float)
        labs = z["labels"].astype(int); tpc = z["tpc"].astype(int)
        ok = np.isfinite(tru) & (tru >= 0); asg = np.isfinite(ts) & (ts >= 0)
        wrong = ok & (~asg | (np.abs(ts - tru) > CUT))
        imaged = set(int(l) for l in np.unique(z["amp_lab"]))
        labE = {int(L): e[labs == L].sum() for L in np.unique(labs[labs >= 0])}
        cE = np.array([labE.get(int(L), 0.0) if L >= 0 else 0.0 for L in labs])
        cls = np.full(labs.size, 0)   # 0 LE (noise or no template), 1 small imaged, 2 mid, 3 big
        isimg = np.array([int(L) in imaged for L in labs])
        cls[isimg & (cE >= 1.5) & (cE < 20)] = 1; cls[isimg & (cE >= 20) & (cE < 100)] = 2; cls[isimg & (cE >= 100)] = 3
        tag = os.path.basename(ff).replace(".npz", "")
        for c in np.unique(tpc[ok]):
            m = tpc == c
            E_t = e[m & ok].sum(); E_w = e[m & wrong].sum()
            if E_t < 200:
                continue
            fam = np.unique(np.round(ts[m & asg]))
            und = 0.0
            if fam.size:
                idx = np.flatnonzero(m & wrong)
                dmin = np.abs(tru[idx][:, None] - fam[None, :]).min(axis=1)
                und = e[idx][dmin > CUT].sum()
            byc = [e[m & wrong & (cls == k)].sum() for k in range(4)]
            cand.append((tag, int(c), E_t, E_w, byc[0], byc[1], byc[2], byc[3], und, fam.size))
    C = np.array([(r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9]) for r in cand])
    keys = [(r[0], r[1]) for r in cand]
    def top(col, name, n=4):
        o = np.argsort(C[:, col])[::-1][:n]
        print(f"\n{name}:")
        for i in o:
            t, c = keys[i]
            print(f"  {t} tpc {c:2d}: TPC E {C[i,0]:7.0f} MeV, wrong {C[i,1]:6.0f} ({100*C[i,1]/C[i,0]:4.1f}%) | LE {C[i,2]:5.0f} small {C[i,3]:5.0f} mid {C[i,4]:5.0f} big {C[i,5]:6.0f} | undiscovered {C[i,6]:5.0f} | families {int(C[i,7])}")
    top(5, "BIG cluster wrong (>= 100 MeV): twin swaps / piggyback")
    top(4, "MID (20-100 MeV) wrong")
    top(3, "SMALL imaged (1.5-20 MeV) wrong")
    top(2, "LOW-ENERGY (noise + < 1.5 MeV, no template) wrong: the neutron-like population")
    top(6, "UNDISCOVERED flashes (wrong energy with no reco family at its truth time)")


if __name__ == "__main__":
    main(sys.argv[1])
