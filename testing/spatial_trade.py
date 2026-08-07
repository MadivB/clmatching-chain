"""v0.8 Task 2 — offline trading study: can the v4 spatial score MOVE energy
correctly between reco families?

Train the score on TRUTH families of files 0000002+ ; trade on RECO families
of file 0000001 (held out).  Spatial-only here (the light-curve veto is added
in-chain where the a_c(t) curves exist) — so these numbers are the FLOOR:
light veto can only remove bad trades.

    python spatial_trade.py <dumps_dir> [n_train_events]

Reports: reachability ceiling, fixed/broke/net ledger (count + energy) over a
(margin, p_min) sweep.
"""
from __future__ import annotations

import glob
import sys

import numpy as np
from scipy.spatial import cKDTree

from spatial_score import TRUTH_GAP, cluster_shape, emedian
from spatial_score4 import (FEATURES2, RNG, lsq_vertex, member_direction,
                            member_type, pair_features2)

CUT = 10.0
BIG_MEV = 30.0


def load_event(fp):
    d = np.load(fp, allow_pickle=True)
    ev = dict(
        xyz=np.stack([np.asarray(d[k], np.float64) for k in "xyz"], 1),
        e=np.clip(np.asarray(d["e"], np.float64), 0, None),
        lab=np.asarray(d["labels"]),
        tru=np.asarray(d["truth_t0"], np.float64),
        tsf=np.asarray(d["ts_final"], np.float64))
    d.close()
    return ev


def shapes_and_times(ev):
    shp, ttime, atime, cmask = {}, {}, {}, {}
    for l in np.unique(ev["lab"][ev["lab"] >= 0]):
        m = ev["lab"] == l
        if m.sum() < 3:
            continue
        fin_t = m & np.isfinite(ev["tru"])
        fin_a = m & np.isfinite(ev["tsf"])
        shp[int(l)] = cluster_shape(ev["xyz"][m], ev["e"][m])
        cmask[int(l)] = m
        if fin_t.any():
            ttime[int(l)] = emedian(ev["tru"][fin_t], ev["e"][fin_t])
        if fin_a.any():
            atime[int(l)] = emedian(ev["tsf"][fin_a], ev["e"][fin_a])
    return shp, ttime, atime, cmask


def build_families(ev, shp, cmask, times, gap=3.0):
    """Group BIG clusters by `times` (truth or assigned) into family dicts."""
    labs = [l for l in shp if l in times
            and float(ev["e"][cmask[l]].sum()) >= BIG_MEV]
    if not labs:
        return {}
    tt = np.array([times[l] for l in labs])
    fam_id, gid, last = {}, 0, None
    for i in np.argsort(tt):
        if last is not None and tt[i] - last > gap:
            gid += 1
        fam_id[labs[i]] = gid
        last = tt[i]
    fams = {}
    for l, g in fam_id.items():
        fams.setdefault(g, []).append(l)
    F = {}
    for g, mem in fams.items():
        members2, tracks = [], []
        for l in mem:
            sh = shp[l]
            mt = member_type(sh)
            ms, md = member_direction(sh)
            members2.append((sh, cKDTree(sh["xyz"]), mt, ms, md))
            if mt == "track":
                tracks.append(sh)
        vtx = lsq_vertex(tracks)
        vdir = None
        if vtx is not None and tracks:
            v = np.sum([sh["mu"] - vtx for sh in tracks], axis=0)
            n = np.linalg.norm(v)
            vdir = v / n if n > 1e-9 else None
        F[g] = dict(members2=members2, vtx=vtx, vdir=vdir,
                    t0=float(np.median([times[l] for l in mem])),
                    mu=np.mean([shp[l]["mu"] for l in mem], axis=0))
    return F


def main():
    dumps = sys.argv[1] if len(sys.argv) > 1 else "v060dump_out/dumps"
    n_train = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    files = sorted(glob.glob(f"{dumps}/*.npz"))
    test_files = [f for f in files if "0000001_" in f]
    train_files = [f for f in files if "0000001_" not in f][: n_train]

    # ---------------- train the score on TRUTH families ----------------
    X, y = [], []
    for fp in train_files:
        ev = load_event(fp)
        shp, ttime, atime, cmask = shapes_and_times(ev)
        F = build_families(ev, shp, cmask, ttime, TRUTH_GAP)
        if not F:
            continue
        f_ids = sorted(F)
        f_mu = np.stack([F[g]["mu"] for g in f_ids])
        cands = [l for l in shp if l in ttime
                 and float(ev["e"][cmask[l]].sum()) < BIG_MEV]
        if len(cands) > 80:
            cands = list(RNG.choice(cands, size=80, replace=False))
        for c in cands:
            near = [f_ids[i] for i in
                    np.argsort(np.linalg.norm(f_mu - shp[c]["mu"], axis=1))[:4]]
            for g in near:
                ft = pair_features2(shp[c], F[g])
                X.append([ft[k] for k in FEATURES2])
                y.append(1 if abs(ttime[c] - F[g]["t0"]) <= TRUTH_GAP else 0)
    X, y = np.array(X, float), np.array(y, int)
    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(max_iter=600, learning_rate=0.08,
                                         max_depth=6, early_stopping=True,
                                         random_state=0).fit(X, y)
    print(f"trained on {len(y)} truth pairs from {len(train_files)} events "
          f"({y.mean()*100:.0f}% same)\n")

    # ---------------- trade on RECO families of the held-out file ----------------
    ledger = []
    pend = []           # feature rows for one batched predict
    ceil_reach = ceil_wrong = 0.0
    for fp in test_files:
        ev = load_event(fp)
        shp, ttime, atime, cmask = shapes_and_times(ev)
        F = build_families(ev, shp, cmask, atime, 3.0)
        if not F:
            continue
        f_ids = sorted(F)
        f_mu = np.stack([F[g]["mu"] for g in f_ids])
        test_c = [c for c in shp if c in atime and c in ttime
                  and float(ev["e"][cmask[c]].sum()) < BIG_MEV]
        if len(test_c) > 200:
            test_c = list(RNG.choice(test_c, size=200, replace=False))
        for c in test_c:
            e_c = float(ev["e"][cmask[c]].sum())
            t_cur = atime[c]
            was_ok = abs(t_cur - ttime[c]) <= CUT
            # candidate families: current (nearest in time) + 4 nearest in space
            gc = min(f_ids, key=lambda g: abs(F[g]["t0"] - t_cur))
            if abs(F[gc]["t0"] - t_cur) > 5.0:
                gc = None
            near = [f_ids[i] for i in
                    np.argsort(np.linalg.norm(f_mu - shp[c]["mu"], axis=1))[:4]]
            cand_fams = [g for g in dict.fromkeys(([gc] if gc is not None else []) + near)]
            if not was_ok:
                ceil_wrong += e_c
                if any(abs(F[g]["t0"] - ttime[c]) <= CUT for g in cand_fams):
                    ceil_reach += e_c
            rec = dict(e=e_c, was_ok=was_ok, gc=gc, fams=[])
            for g in cand_fams:
                ft = [pair_features2(shp[c], F[g])[k] for k in FEATURES2]
                pend.append(ft)
                rec["fams"].append((g, len(pend) - 1,
                                    abs(F[g]["t0"] - ttime[c]) <= CUT))
            ledger.append(rec)
    probs = clf.predict_proba(np.array(pend, float))[:, 1] if pend else np.zeros(0)
    for rec in ledger:
        ps = {g: float(probs[i]) for g, i, _ in rec["fams"]}
        okmap = {g: ok for g, i, ok in rec["fams"]}
        gc = rec["gc"]
        rec["p_cur"] = ps.get(gc, 0.0) if gc is not None else 0.0
        alts = [g for g, _, _ in rec["fams"] if g != gc]
        if not alts:
            rec["p_best"], rec["now_ok"] = -1.0, False
            continue
        g_best = max(alts, key=lambda g: ps[g])
        rec["p_best"] = ps[g_best]
        rec["now_ok"] = okmap[g_best]
    print(f"test clusters considered: {len(ledger)}; "
          f"reachability ceiling: {100*ceil_reach/max(ceil_wrong,1e-9):.0f}% of "
          f"misplaced small-cluster energy has its TRUE family among candidates\n")
    print(f"{'margin':>7s} {'p_min':>6s} {'trades':>7s} {'FIXED':>6s} {'BROKE':>6s} "
          f"{'net':>5s} {'E_fixed':>8s} {'E_broke':>8s} {'netE[MeV]':>9s}")
    for margin in (0.2, 0.3, 0.4, 0.5, 0.6):
        for p_min in (0.5, 0.7):
            fixed = broke = ef = eb = n = 0
            for r in ledger:
                if r["p_best"] - r["p_cur"] > margin and r["p_best"] > p_min:
                    n += 1
                    if not r["was_ok"] and r["now_ok"]:
                        fixed += 1; ef += r["e"]
                    elif r["was_ok"] and not r["now_ok"]:
                        broke += 1; eb += r["e"]
            print(f"{margin:7.1f} {p_min:6.1f} {n:7d} {fixed:6d} {broke:6d} "
                  f"{fixed-broke:+5d} {ef:8.1f} {eb:8.1f} {ef-eb:+9.1f}")


if __name__ == "__main__":
    main()
