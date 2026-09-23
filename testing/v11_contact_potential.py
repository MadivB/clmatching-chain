"""Reach of a 3-cm contact prior on an arm's own errors: for imaged clusters
>= 5 MeV, do they have a contact cluster (3 cm) placed at their TRUTH time,
at their OWN (reco) time, or at some OTHER time?  python v11_contact_potential.py <arm_dir>"""
import glob, sys
import numpy as np
from scipy.spatial import cKDTree
CUT, R, SAME_T = 10.0, 3.0, 2.0


def main(d):
    T = {}   # (wrong?, class) -> E
    def add(k, E): T[k] = T.get(k, 0.0) + E
    for ff in sorted(glob.glob(f"{d}/full_dumps/*.npz")):
        z = np.load(ff, allow_pickle=True)
        labs = z["labels"].astype(int); e = np.clip(z["e"], 0, None).astype(np.float64)
        tru = z["truth_t0"].astype(np.float64); ts = z["ts_final"].astype(np.float64)
        xyz = np.stack([z["x"], z["y"], z["z"]], 1).astype(np.float64)
        tpc = z["tpc"].astype(int)
        ok_t = np.isfinite(tru) & (tru >= 0)
        imaged = set(int(l) for l in np.unique(z["amp_lab"]))
        tpc_e = np.bincount(tpc, weights=e, minlength=int(tpc.max()) + 1)
        tree = cKDTree(xyz)
        lab_t = {}
        for L in np.unique(labs[labs >= 0]):
            v = ts[labs == L]; v = v[np.isfinite(v) & (v >= 0)]
            lab_t[int(L)] = float(np.median(v)) if v.size else np.nan
        for L in np.unique(labs[labs >= 0]):
            L = int(L)
            if L not in imaged: continue
            m = labs == L; E = e[m].sum()
            if E < 5 or not np.isfinite(lab_t[L]): continue
            if tpc_e[np.unique(tpc[m])].max() > 1400: continue    # shower TPC: excluded from re-decision
            mt = m & ok_t
            if not mt.any(): continue
            truth = float(np.median(tru[mt]))
            wrong = e[mt & (np.abs(ts - tru) > CUT)].sum() > 0.5 * e[mt].sum()
            h = np.flatnonzero(m)
            if h.size > 3000: h = h[np.linspace(0, h.size - 1, 3000).astype(int)]
            nb = tree.query_ball_point(xyz[h], R)
            idx = np.unique(np.concatenate([np.asarray(v, int) for v in nb]))
            nl = [int(v) for v in np.unique(labs[idx]) if v >= 0 and int(v) != L]
            ct = np.array([lab_t[l] for l in nl if np.isfinite(lab_t.get(l, np.nan))])
            if ct.size == 0:
                add((wrong, "no contact"), E); continue
            at_truth = np.any(np.abs(ct - truth) <= SAME_T)
            at_own = np.any(np.abs(ct - lab_t[L]) <= SAME_T)
            if wrong:
                add((True, "contact AT TRUTH time (prior can fix)" if at_truth else "contact only elsewhere"), E)
            else:
                if at_own and not np.any(np.abs(ct - lab_t[L]) > SAME_T):
                    add((False, "contact agrees (safe)"), E)
                elif at_own:
                    add((False, "contact agrees + others disagree (mixed)"), E)
                else:
                    add((False, "contact only DISAGREES (prior risk)"), E)
    Ew = sum(v for (w, k), v in T.items() if w); Er = sum(v for (w, k), v in T.items() if not w)
    print(f"{d}: imaged >=5 MeV clusters in non-shower TPCs: wrong E {Ew/1e3:.2f} GeV, right E {Er/1e3:.1f} GeV")
    for (w, k), v in sorted(T.items(), key=lambda kv: (not kv[0][0], -kv[1])):
        print(f"  {'WRONG' if w else 'right'}: {k:45s} {v/1e3:8.2f} GeV ({100*v/(Ew if w else Er):5.1f}%)")


if __name__ == "__main__":
    main(sys.argv[1])
