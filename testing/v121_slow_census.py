"""Slow-fragment population for time-of-flight: imaged clusters >= 5 MeV whose truth time is
> 10 ticks LATER than the truth time of the nearest bigger cluster within 100 cm (the parent).
How does the chain time them?  python v121_slow_census.py <dump_dir> [max_events]"""
import glob, sys
import numpy as np
from scipy.spatial import cKDTree
NS, C = 16.0, 29.9792458


def main(d, max_events):
    rows = []
    for ff in sorted(glob.glob(f"{d}/full_dumps/*.npz"))[:max_events]:
        z = np.load(ff, allow_pickle=True)
        e = np.clip(z["e"], 0, None).astype(float); tru = z["truth_t0"].astype(float); ts = z["ts_final"].astype(float)
        labs = z["labels"].astype(int); xyz = np.stack([z["x"], z["y"], z["z"]], 1).astype(float)
        imaged = set(int(l) for l in np.unique(z["amp_lab"]))
        ok = np.isfinite(tru) & (tru >= 0) & (labs >= 0)
        info = {}
        for L in np.unique(labs[ok]):
            m = ok & (labs == L); E = e[m].sum()
            if E < 5 or int(L) not in imaged:
                continue
            info[int(L)] = dict(E=E, cen=np.average(xyz[m], axis=0, weights=e[m]), tru=float(np.median(tru[m])),
                                tru_rob=1.4826 * float(np.median(np.abs(tru[m] - np.median(tru[m])))),
                                ts=float(np.median(ts[m])) if np.isfinite(ts[m]).any() else np.nan, m=m)
        Ls = list(info); cens = np.array([info[L]["cen"] for L in Ls]); Es = np.array([info[L]["E"] for L in Ls])
        tree = cKDTree(cens)
        for i, L in enumerate(Ls):
            if info[L]["tru_rob"] > 1.0:          # impure cluster (mixed truth times): skip
                continue
            nb = tree.query_ball_point(cens[i], 100.0)
            parents = [j for j in nb if j != i and Es[j] >= 2 * Es[i]]
            if not parents:
                continue
            j = min(parents, key=lambda k: np.linalg.norm(cens[k] - cens[i]))
            P = info[Ls[j]]
            dt = (info[L]["tru"] - P["tru"]) * NS
            if dt <= 10 * NS:
                continue
            dist = float(np.linalg.norm(cens[i] - P["cen"]))
            at_own = np.isfinite(info[L]["ts"]) and abs(info[L]["ts"] - info[L]["tru"]) <= 10
            at_parent = np.isfinite(info[L]["ts"]) and abs(info[L]["ts"] - P["ts"]) <= 10 if np.isfinite(P["ts"]) else False
            rows.append((info[L]["E"], dt, dist, dist / (C * dt), at_own, at_parent))
    R = np.array(rows, float)
    w = R[:, 0]
    print(f"slow fragments (>= 5 MeV, truth > 160 ns after the nearest bigger neighbour within 1 m): n={len(R)}, E {w.sum()/1e3:.2f} GeV")
    print(f"  delay p10/p50/p90: {np.percentile(R[:,1],10):.0f} / {np.percentile(R[:,1],50):.0f} / {np.percentile(R[:,1],90):.0f} ns; distance median {np.median(R[:,2]):.0f} cm; beta median {np.median(R[:,3]):.3f}")
    print(f"  timed at their OWN flash (|reco - truth| <= 10 ticks): {np.average(R[:,4], weights=w):.2f} of E ({np.mean(R[:,4]):.2f} of count); parked at the parent's time: {np.average(R[:,5], weights=w):.2f} of E")
    for lo, hi in ((5, 20), (20, 100), (100, 1e9)):
        s = (R[:, 0] >= lo) & (R[:, 0] < hi)
        if s.sum() > 3:
            print(f"    E [{lo},{hi}): n={s.sum():4d} E {R[s,0].sum()/1e3:.2f} GeV | own flash {np.average(R[s,4], weights=R[s,0]):.2f} | at parent {np.average(R[s,5], weights=R[s,0]):.2f} | delay median {np.median(R[s,1]):.0f} ns")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 39)
