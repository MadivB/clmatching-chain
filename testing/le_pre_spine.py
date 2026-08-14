"""LE (low-energy blip) re-timing — pre-SPINE scorecard + ranker fit + emulation.

The measured anatomy (v0.8.7 dumps, file 1): LE class (noise hits +
sub-1.5 MeV no-template clusters) = 2.1 % of truth energy at 56.4 % accuracy.
Of the wrong hits: correct-time family EXISTS for 93 %, is within 60 cm for
68 %, is among the 3 nearest for 57 % — but is the single nearest only 2 %.
=> candidate RANKING among placed families, not proximity, is the lever.

This module:
  build_blobs      3-cm connected components of LE hits (decide per blob)
  family_summaries placed non-LE families: t0, member tree, charge centroid,
                   total energy  (THE ALREADY-PLACED BACKBONE — nothing new)
  blob_features    per (blob, candidate family): d_min, d_centroid, log fam E,
                   fam size, |dt| to nearest other candidate, same-TPC, blob E
  fit              logistic ranker on truth (train files), saved as json
  emulate          re-time every LE blob by argmax score -> LE accuracy

Usage:
  python le_pre_spine.py fit      # train on v060 dumps files 2-10, save model
  python le_pre_spine.py emulate  # emulate on v096 dumps (current chain, OOS)
"""
import glob
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

CUT = 10.0
LE_EMAX = 1.5          # clusters below this have no light template
BLOB_LINK = 3.0
CAND_R = 60.0          # candidate families within this centroid distance
CAND_K = 8
FEATS = ["d_min", "d_cent", "log_fe", "fam_n", "dt_ambig", "same_tpc",
         "log_be", "d_min_rank", "is_inc"]
_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_JSON = os.path.join(_HERE, "le_ranker.json")


def load_ev(fp):
    d = np.load(fp, allow_pickle=True)
    ev = {k: np.asarray(d[k]) for k in ("x", "y", "z", "e", "tpc", "labels",
                                        "ts_final", "truth_t0")}
    ev["e"] = np.clip(ev["e"].astype(float), 0, None)
    ev["xyz"] = np.stack([ev["x"], ev["y"], ev["z"]], 1).astype(float)
    d.close()
    return ev


def le_mask(ev):
    labs = ev["labels"]
    ce = {}
    for l in np.unique(labs[labs >= 0]):
        ce[l] = ev["e"][labs == l].sum()
    small = np.isin(labs, [l for l, en in ce.items() if en < LE_EMAX])
    return (labs < 0) | small


def build_blobs(xyz, idx):
    """connected components (3 cm) among the LE hits; returns list of index arrays."""
    if idx.size == 0:
        return []
    tree = cKDTree(xyz[idx])
    parent = np.arange(idx.size)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for i, j in tree.query_pairs(BLOB_LINK):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri
    groups = {}
    for k in range(idx.size):
        groups.setdefault(find(k), []).append(idx[k])
    return [np.asarray(g) for g in groups.values()]


def family_summaries(ev, le):
    """placed families from the NON-LE (backbone) assignments only."""
    ts = ev["ts_final"].astype(float)
    donor = np.isfinite(ts) & (ts >= 0) & ~le
    di = np.flatnonzero(donor)
    if di.size == 0:
        return []
    order = di[np.argsort(ts[di])]
    fams, cur = [], [order[0]]
    for i in order[1:]:
        if ts[i] - ts[cur[-1]] > 3.0:
            fams.append(cur)
            cur = []
        cur.append(i)
    fams.append(cur)
    out = []
    for mem in fams:
        mem = np.asarray(mem)
        if mem.size < 2:
            continue
        w = np.maximum(ev["e"][mem], 1e-6)
        out.append(dict(
            t0=float(np.median(ts[mem])),
            tree=cKDTree(ev["xyz"][mem]),
            cent=np.average(ev["xyz"][mem], 0, weights=w),
            e=float(ev["e"][mem].sum()),
            n=int(mem.size),
            tpcs=set(int(t) for t in np.unique(ev["tpc"][mem]))))
    return out


def blob_rows(ev, fams, blob, t_cur=None):
    """candidate rows for one blob: top-k by MEMBER distance, plus the
    INCUMBENT family (t0 nearest the blob's current time) always included."""
    bx = ev["xyz"][blob]
    cent = bx.mean(0)
    be = float(ev["e"][blob].sum())
    btpc = int(np.bincount(ev["tpc"][blob].astype(int)).argmax())
    dm = np.array([float(f["tree"].query(cent)[0]) for f in fams])
    dc = np.array([np.linalg.norm(f["cent"] - cent) for f in fams])
    order = list(np.argsort(dm)[:CAND_K])
    cands = [i for i in order if dm[i] <= CAND_R] or order[:1]
    inc = None
    if t_cur is not None and np.isfinite(t_cur):
        inc = int(np.argmin([abs(f["t0"] - t_cur) for f in fams]))
        if abs(fams[inc]["t0"] - t_cur) > 3.0:
            inc = None
        elif inc not in cands:
            cands.append(inc)
    rk = {i: r for r, i in enumerate(sorted(cands, key=lambda i: dm[i]))}
    rows, t0s = [], []
    for i in cands:
        f = fams[i]
        others = [abs(f["t0"] - fams[j]["t0"]) for j in cands if j != i]
        rows.append([float(dm[i]), float(dc[i]), np.log10(max(f["e"], 1e-3)),
                     float(f["n"]), (min(others) if others else 999.0),
                     float(btpc in f["tpcs"]), np.log10(max(be, 1e-3)),
                     float(rk[i]), float(i == inc)])
        t0s.append(f["t0"])
    return np.array(rows, float), np.array(t0s, float), (cands.index(inc) if inc in cands else -1)


def harvest_file(fp):
    ev = load_ev(fp)
    le = le_mask(ev)
    ok_t = np.isfinite(ev["truth_t0"]) & (ev["truth_t0"] >= 0)
    fams = family_summaries(ev, le)
    if not fams:
        return [], []
    idx = np.flatnonzero(le & ok_t)
    X, Y = [], []
    for blob in build_blobs(ev["xyz"], idx):
        et = ev["e"][blob]
        tr = ev["truth_t0"][blob]
        t_true = float(np.median(tr))
        t_cur = float(np.median(ev["ts_final"][blob])) \
            if np.isfinite(ev["ts_final"][blob]).any() else None
        rows, t0s, _ = blob_rows(ev, fams, blob, t_cur=t_cur)
        for r, t0 in zip(rows, t0s):
            X.append(r)
            Y.append(1.0 if abs(t0 - t_true) <= CUT else 0.0)
    return X, Y


def fit(train_glob="v060dump_out/dumps/00000*_ev*.npz"):
    import multiprocessing as mp
    files = [f for f in sorted(glob.glob(train_glob))
             if not os.path.basename(f).startswith("0000001_")]
    with mp.Pool(16) as pool:
        parts = pool.map(harvest_file, files)
    X = np.array(sum((p[0] for p in parts), []), float)
    y = np.array(sum((p[1] for p in parts), []))
    if os.environ.get("LE_MODEL", "logistic") == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        import joblib
        clf = HistGradientBoostingClassifier(max_iter=300, max_depth=6,
                                             learning_rate=0.08)
        clf.fit(X, y)
        joblib.dump({"model": clf, "feats": FEATS},
                    MODEL_JSON.replace(".json", "_hgb.joblib"))
        print(f"{len(X)} rows (base {y.mean():.3f}) -> HGB saved")
        return
    from sklearn.linear_model import LogisticRegression
    mu, sd = X.mean(0), X.std(0) + 1e-9
    clf = LogisticRegression(max_iter=2000)
    clf.fit((X - mu) / sd, y)
    json.dump(dict(mu=mu.tolist(), sd=sd.tolist(), coef=clf.coef_[0].tolist(),
                   b=float(clf.intercept_[0]), feats=FEATS),
              open(MODEL_JSON, "w"), indent=1)
    print(f"{len(X)} rows (base {y.mean():.3f}) from {len(files)} events "
          f"-> {MODEL_JSON}")
    for f, c in sorted(zip(FEATS, clf.coef_[0]), key=lambda t: -abs(t[1])):
        print(f"  {f:12s} {c:+.2f}")


def score(md, X):
    if "hgb" in md:
        return md["hgb"].predict_proba(np.asarray(X, float))[:, 1]
    z = ((X - np.array(md["mu"])) / np.array(md["sd"])) @ np.array(md["coef"]) + md["b"]
    return 1.0 / (1.0 + np.exp(-z))


def emulate(test_glob="v096EVO_out/dumps/*.npz"):
    if os.environ.get("LE_MODEL", "logistic") == "hgb":
        import joblib
        _h = joblib.load(MODEL_JSON.replace(".json", "_hgb.joblib"))["model"]
        md = {"hgb": _h}
    else:
        md = json.load(open(MODEL_JSON))
    MARGINS = (0.0, 0.1, 0.2, 0.4, 0.6)
    acc = {m: 0.0 for m in MARGINS}
    chg = {m: 0 for m in MARGINS}
    e_tot = e_good_now = 0.0
    n_blob = 0
    for fp in sorted(glob.glob(test_glob)):
        ev = load_ev(fp)
        le = le_mask(ev)
        ok_t = np.isfinite(ev["truth_t0"]) & (ev["truth_t0"] >= 0)
        fams = family_summaries(ev, le)
        if not fams:
            continue
        idx = np.flatnonzero(le & ok_t)
        ts = ev["ts_final"].astype(float)
        for blob in build_blobs(ev["xyz"], idx):
            n_blob += 1
            et = ev["e"][blob]
            tr = ev["truth_t0"][blob]
            e_tot += et.sum()
            e_good_now += et[np.abs(ts[blob] - tr) <= CUT].sum()
            t_cur = float(np.median(ts[blob][np.isfinite(ts[blob])])) \
                if np.isfinite(ts[blob]).any() else None
            rows, t0s, inc_i = blob_rows(ev, fams, blob, t_cur=t_cur)
            sc = score(md, rows)
            k = int(np.argmax(sc))
            s_inc = sc[inc_i] if inc_i >= 0 else -1.0
            for m in MARGINS:
                if inc_i < 0 or sc[k] - s_inc < m:
                    t_pick = t_cur          # keep current (no identified
                                            # incumbent => never force a change)
                else:
                    t_pick = float(t0s[k])
                    if t_cur is None or abs(t_pick - t_cur) > 2.0:
                        chg[m] += 1
                if t_pick is not None:
                    acc[m] += et[np.abs(t_pick - tr) <= CUT].sum()
    print(f"{n_blob} blobs, LE energy {e_tot:.0f} MeV, "
          f"current {100*e_good_now/max(e_tot,1e-9):.1f}%")
    for m in MARGINS:
        print(f"  margin {m:.1f}: re-timed {100*acc[m]/max(e_tot,1e-9):5.1f}%  "
              f"(changed {chg[m]} blobs)")


if __name__ == "__main__":
    (fit if (len(sys.argv) > 1 and sys.argv[1] == "fit") else emulate)()
