"""Phase 0 for per-fragment timing: on v1.2 dumps, split every family (same final
integer t0 across TPCs) into a CORE (clusters chained within 3 cm of the biggest
cluster) and DISPLACED fragments (the rest).  For each fragment: energy, distance
from the family apex (truth-earliest core position, measurement only), the TRUE
delay of its deposition relative to the apex time, the implied velocity
beta = d / (c dt), and the current v1.2 timing error (which assumes v = c).
python v121_fragments.py <dump_dir> [max_events]"""
import glob, sys
import numpy as np
from scipy.spatial import cKDTree
C, NS = 29.9792458, 16.0


def main(d, max_events):
    F = []          # fragment rows: E, d_cm, dt_true_ns, beta, err_v12_ns, err_v11like (t0+0), core_E, n_frag_in_family
    core_stats = []  # core within-family truth spread after apex clock
    E_core = E_frag = 0.0
    for ff in sorted(glob.glob(f"{d}/full_dumps/*.npz"))[:max_events]:
        z = np.load(ff, allow_pickle=True)
        e = np.clip(z["e"], 0, None).astype(float); tru = z["truth_t0"].astype(float); ts = z["ts_final"].astype(float)
        labs = z["labels"].astype(int); xyz = np.stack([z["x"], z["y"], z["z"]], 1).astype(float)
        ok = np.isfinite(ts) & (ts >= 0) & np.isfinite(tru) & (tru >= 0) & (labs >= 0)
        # families = integer chain tick (v1.2 per-hit times vary within a family; the chain tick is the rounded family time
        # of the biggest member; approximate by rounding the cluster-median time)
        lab_t = {}; lab_hits = {}
        for L in np.unique(labs[ok]):
            m = ok & (labs == L)
            if e[m].sum() < 1.5:
                continue
            if abs(np.median(ts[m]) - np.median(tru[m])) > 10.0:      # only correctly assigned members
                continue
            lab_t[int(L)] = float(np.round(np.median(ts[m])))
            lab_hits[int(L)] = np.flatnonzero(m)
        fams = {}
        for L, t in lab_t.items():
            fams.setdefault(t, []).append(L)
        for t, members in fams.items():
            if len(members) < 2:
                continue
            Em = {L: e[lab_hits[L]].sum() for L in members}
            big = max(members, key=lambda L: Em[L])
            # chain within 3 cm from the biggest cluster
            core = {big}; frontier = [big]
            trees = {L: cKDTree(xyz[lab_hits[L]]) for L in members}
            while frontier:
                cur = frontier.pop()
                for L in members:
                    if L in core:
                        continue
                    dd, _ = trees[cur].query(xyz[lab_hits[L]], k=1, distance_upper_bound=3.0)
                    if np.isfinite(dd).any():
                        core.add(L); frontier.append(L)
            frag = [L for L in members if L not in core]
            ch = np.concatenate([lab_hits[L] for L in core])
            # apex = truth-earliest 2 % of the core (measurement only)
            # robust apex reference: the core hit whose (truth time - d/c) is the median -> apex time = that median
            o = np.argsort(tru[ch]); n2 = max(3, int(0.02 * o.size))
            apex = np.average(xyz[ch][o[:n2]], axis=0, weights=e[ch][o[:n2]])
            t_apex = float(np.median(tru[ch] * NS - np.linalg.norm(xyz[ch] - apex, axis=1) / C))
            Ecore = e[ch].sum(); E_core += Ecore
            dcore = np.linalg.norm(xyz[ch] - apex, axis=1) / C
            core_stats.append((np.sqrt(np.average((tru[ch] * NS - t_apex - dcore) ** 2, weights=e[ch])), Ecore))
            for L in frag:
                h = lab_hits[L]; Ef = e[h].sum(); E_frag += Ef
                cen = np.average(xyz[h], axis=0, weights=e[h])
                dist = float(np.linalg.norm(cen - apex))
                dt_true = float(np.average(tru[h], weights=e[h])) * NS - t_apex          # ns after the apex time
                beta = dist / (C * dt_true) if dt_true > 0.5 else np.nan
                err_v12 = float(np.average((ts[h] - tru[h]) * NS, weights=e[h]))            # current per-hit error on the fragment
                F.append((Ef, dist, dt_true, beta, err_v12, Ecore, len(frag)))
    F = np.array(F); cs = np.array(core_stats)
    print(f"families with >= 2 clusters: fragments {len(F)}, fragment E {E_frag/1e3:.2f} GeV vs core E {E_core/1e3:.1f} GeV ({100*E_frag/(E_frag+E_core):.1f} % of family energy is displaced)")
    print(f"core hits: truth spread after the apex clock (E-wt mean of per-family rms) {np.average(cs[:,0], weights=cs[:,1]):.2f} ns  -> v = c holds for the core")
    w = F[:, 0]
    print(f"fragment true delay dt (ns): p10/p50/p90 {np.percentile(F[:,2],10):.1f} / {np.percentile(F[:,2],50):.1f} / {np.percentile(F[:,2],90):.1f}; E-wt frac with dt > 10 ns {np.average(F[:,2]>10, weights=w):.2f}, > 30 ns {np.average(F[:,2]>30, weights=w):.2f}")
    for lo, hi in ((1.5, 5), (5, 20), (20, 100), (100, 1e9)):
        s = (F[:, 0] >= lo) & (F[:, 0] < hi)
        if s.sum() > 5:
            b = F[s, 3]; b = b[np.isfinite(b)]
            print(f"  frag E [{lo},{hi}): n={s.sum():5d} E {F[s,0].sum()/1e3:5.2f} GeV | dist median {np.median(F[s,1]):.0f} cm | dt median {np.median(F[s,2]):.1f} ns, >10 ns {np.mean(F[s,2]>10):.2f} | beta median {np.median(b):.2f}, <0.5 {np.mean(b<0.5):.2f} | v1.2 err on fragment: median {np.median(F[s,4]):+.1f} ns, rms {np.sqrt(np.mean(F[s,4]**2)):.1f}")
    slow = F[(F[:, 3] < 0.5) & np.isfinite(F[:, 3])]
    print(f"slow fragments (beta < 0.5): {len(slow)}, E {slow[:,0].sum()/1e3:.2f} GeV, dt median {np.median(slow[:,2]):.0f} ns, dist median {np.median(slow[:,1]):.0f} cm")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 13)
