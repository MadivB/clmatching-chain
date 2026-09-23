"""Purity census of the 2x2 topology-corrector 'home' rules on ND (v1.1 dumps, truth-scored):
  R1 endpoint-at-shower: a track whose END lies within 3 cm of a shower-like cluster (and is not
     embedded in it) -> home = the shower's time
  R2 embedded fragment: a small cluster with >= 50 % of its hits within 5 cm of a shower's hits
  R3 isolated fragment: a small cluster with no same-time cluster within 15 cm but a bigger
     cluster of another time within 15 cm -> home = the neighbour's time
ND has no front-end shower typing: shower proxy = E >= 50 MeV, >= 100 hits, PCA linearity < 0.85.
For each rule: cases, energy, purity (truth of cluster == truth of home), and among cases where the
v1.1 placement disagrees with the home: fixable energy (home right) vs breakable (placement right).
python v12_home_rules.py <dump_dir> [max_events]"""
import glob, sys
import numpy as np
from scipy.spatial import cKDTree
CUT = 10.0


def shape(P, w):
    c = np.average(P, axis=0, weights=w)
    X = (P - c) * np.sqrt(w)[:, None]
    s = np.linalg.svd(X, full_matrices=False)
    lam = s[1] ** 2; u = s[2][0]
    lin = lam[0] / max(lam.sum(), 1e-12)
    proj = (P - c) @ u
    o = np.argsort(proj); n2 = max(2, int(0.03 * o.size))
    e1 = np.average(P[o[:n2]], axis=0, weights=w[o[:n2]]); e2 = np.average(P[o[-n2:]], axis=0, weights=w[o[-n2:]])
    return lin, e1, e2, float(np.ptp(proj))


def main(d, max_events):
    T = {r: {"n": 0, "E": 0.0, "E_pure": 0.0, "E_dis": 0.0, "E_fix": 0.0, "E_break": 0.0, "n_fix": 0, "n_break": 0} for r in ("R1", "R2", "R3")}
    for ff in sorted(glob.glob(f"{d}/full_dumps/*.npz"))[:max_events]:
        z = np.load(ff, allow_pickle=True)
        e = np.clip(z["e"], 0, None).astype(np.float64); tru = z["truth_t0"].astype(np.float64); ts = z["ts_final"].astype(np.float64)
        labs = z["labels"].astype(int); xyz = np.stack([z["x"], z["y"], z["z"]], 1).astype(np.float64)
        imaged = set(int(l) for l in np.unique(z["amp_lab"]))
        ok_t = np.isfinite(tru) & (tru >= 0)
        info = {}
        for L in np.unique(labs[labs >= 0]):
            m = labs == L; E = e[m].sum()
            if E < 1.5 or m.sum() < 5:
                continue
            v = ts[m]; v = v[np.isfinite(v) & (v >= 0)]
            mt = m & ok_t
            if not v.size or not mt.any():
                continue
            lin, e1, e2, length = shape(xyz[m], np.maximum(e[m], 1e-6))
            info[int(L)] = dict(E=E, n=int(m.sum()), t=float(np.median(v)), tru=float(np.median(tru[mt])), lin=lin, e1=e1, e2=e2, len=length, hits=np.flatnonzero(m), imaged=int(L) in imaged)
        showers = [L for L, i in info.items() if i["E"] >= 50 and i["n"] >= 100 and i["lin"] < 0.85]
        sh_tree = {L: cKDTree(xyz[info[L]["hits"]]) for L in showers}
        all_tree = cKDTree(xyz)
        def account(rule, L, home_L):
            i, h = info[L], info[home_L]
            T[rule]["n"] += 1; T[rule]["E"] += i["E"]
            pure = abs(i["tru"] - h["tru"]) <= CUT
            if pure: T[rule]["E_pure"] += i["E"]
            if abs(i["t"] - h["t"]) > CUT:                      # placement disagrees with the home
                T[rule]["E_dis"] += i["E"]
                self_right = abs(i["t"] - i["tru"]) <= CUT
                if pure and not self_right: T[rule]["E_fix"] += i["E"]; T[rule]["n_fix"] += 1
                elif self_right and not pure: T[rule]["E_break"] += i["E"]; T[rule]["n_break"] += 1
        for L, i in info.items():
            if not i["imaged"]:
                continue
            # R1: tracks (linear, >= 15 cm, >= 10 MeV) with an endpoint at a shower, not embedded
            if i["lin"] >= 0.95 and i["len"] >= 15 and i["E"] >= 10:
                for S in showers:
                    if S == L: continue
                    dmin = min(sh_tree[S].query(i["e1"])[0], sh_tree[S].query(i["e2"])[0])
                    if dmin <= 3.0:
                        frac_in = np.mean(sh_tree[S].query(xyz[i["hits"]], distance_upper_bound=5.0)[0] < np.inf)
                        if frac_in <= 0.3:
                            account("R1", L, S); break
            # R2 / R3: small clusters
            if i["E"] < 30:
                homed = False
                for S in showers:
                    if S == L: continue
                    frac_in = np.mean(sh_tree[S].query(xyz[i["hits"]], distance_upper_bound=5.0)[0] < np.inf)
                    if frac_in >= 0.5:
                        account("R2", L, S); homed = True; break
                if homed:
                    continue
                nb = all_tree.query_ball_point(xyz[i["hits"]], 15.0)
                nl = set(int(v) for v in np.unique(labs[np.unique(np.concatenate([np.asarray(q, int) for q in nb]))]) if v >= 0 and int(v) != L and int(v) in info)
                same = [K for K in nl if abs(info[K]["t"] - i["t"]) <= CUT]
                if same:
                    continue
                bigger = [K for K in nl if info[K]["E"] >= max(30.0, 2 * i["E"]) and abs(info[K]["t"] - i["t"]) > CUT]
                if bigger:
                    K = max(bigger, key=lambda k: info[k]["E"])
                    account("R3", L, K)
    for r, t in T.items():
        if t["n"] == 0:
            print(f"{r}: no cases"); continue
        print(f"{r}: cases {t['n']:5d}, E {t['E']/1e3:6.2f} GeV | purity (home has the cluster's truth time) {100*t['E_pure']/t['E']:5.1f}% | placement disagrees with home: {t['E_dis']/1e3:5.2f} GeV -> fixable {t['E_fix']/1e3:5.2f} GeV ({t['n_fix']}) vs breakable {t['E_break']/1e3:5.2f} GeV ({t['n_break']})")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 39)


def crossing_census(d, max_events):
    """The 2x2 failure mode: a track CROSSING a shower (hits inside it, both endpoints outside)
    sitting at the shower's truth time while the shower is placed elsewhere (flash stolen),
    and the general count of crossing tracks whose time equals the shower's."""
    E_stolen = E_cross_same = E_cross_diff = 0.0; n_stolen = n_same = n_diff = 0
    E_showers_wrong = E_showers = 0.0
    for ff in sorted(glob.glob(f"{d}/full_dumps/*.npz"))[:max_events]:
        z = np.load(ff, allow_pickle=True)
        e = np.clip(z["e"], 0, None).astype(np.float64); tru = z["truth_t0"].astype(np.float64); ts = z["ts_final"].astype(np.float64)
        labs = z["labels"].astype(int); xyz = np.stack([z["x"], z["y"], z["z"]], 1).astype(np.float64)
        ok_t = np.isfinite(tru) & (tru >= 0)
        info = {}
        for L in np.unique(labs[labs >= 0]):
            m = labs == L; E = e[m].sum()
            if E < 10 or m.sum() < 20:
                continue
            v = ts[m]; v = v[np.isfinite(v) & (v >= 0)]; mt = m & ok_t
            if not v.size or not mt.any():
                continue
            lin, e1, e2, length = shape(xyz[m], np.maximum(e[m], 1e-6))
            info[int(L)] = dict(E=E, n=int(m.sum()), t=float(np.median(v)), tru=float(np.median(tru[mt])), lin=lin, e1=e1, e2=e2, len=length, hits=np.flatnonzero(m))
        showers = [L for L, i in info.items() if i["E"] >= 50 and i["n"] >= 100 and i["lin"] < 0.85]
        tracks = [L for L, i in info.items() if i["lin"] >= 0.95 and i["len"] >= 20]
        for S in showers:
            s = info[S]; E_showers += s["E"]
            sh_wrong = abs(s["t"] - s["tru"]) > CUT
            if sh_wrong: E_showers_wrong += s["E"]
            tree = cKDTree(xyz[s["hits"]])
            for T_ in tracks:
                t = info[T_]
                if min(tree.query(t["e1"])[0], tree.query(t["e2"])[0]) < 10.0:
                    continue                                   # endpoint at/in the shower: not a crossing
                inside = np.mean(tree.query(xyz[t["hits"]], distance_upper_bound=5.0)[0] < np.inf)
                if inside < 0.1:
                    continue
                if abs(t["t"] - s["t"]) <= CUT:                # crossing track at the SHOWER'S RECO time
                    if abs(t["tru"] - s["tru"]) <= CUT: E_cross_same += t["E"]; n_same += 1
                    else: E_cross_diff += t["E"]; n_diff += 1
                if sh_wrong and abs(t["t"] - s["tru"]) <= CUT and abs(t["tru"] - s["tru"]) > CUT:
                    E_stolen += t["E"]; n_stolen += 1       # the track sits at the shower's true flash, shower displaced
    print(f"showers (proxy): E {E_showers/1e3:.1f} GeV, wrong {E_showers_wrong/1e3:.2f} GeV | crossing tracks at the shower's reco time: same interaction {n_same} ({E_cross_same/1e3:.2f} GeV), different interaction {n_diff} ({E_cross_diff/1e3:.2f} GeV) | 2x2 failure mode (track at the shower's TRUE flash, shower displaced): {n_stolen} tracks, {E_stolen/1e3:.2f} GeV")


if __name__ == "__main__" and len(sys.argv) > 3 and sys.argv[3] == "cross":
    crossing_census(sys.argv[1], int(sys.argv[2]))
