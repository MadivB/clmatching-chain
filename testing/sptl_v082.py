"""v0.8.1 spatial score, parallel pipeline — all variants in one harvest.

Variants (doc split files 1-8 train / 9-10 test; targets from SPATIAL_SCORE.md):
  J0 (doc 0.649) -> J1 (+pointing/tpc/triangles, doc 0.720)
  S v0   (18 feats, 45 cm reach)                  doc 0.915
  S v1.1 (+continuous contact topology, shower reach 150 cm)
  S v1.2 (+NEIGHBOR-TPC: tpc_adj, gap-aware axis DOCA across the module
          gap, adjacency-extended reach)  <- the fresh-TPC fragment fix

Phase 1 (parallel): per-event summaries + pair features -> pickle cache
Phase 2 (serial):   train J0/J1, verify vs doc
Phase 3 (parallel): per-event S rows at WIDEST reach with per-variant
                    population flags + slice flags
Phase 4 (serial):   train/test all S variants + deploy models (files != 1)

Usage: python sptl_v082.py [dump_dir] [cache_dir]
"""
import glob
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sptl_v081 import (CONTACT, PAIR_RCENT, PAIR_DMIN, REACH_SHOWER, J0_FEATS,
                       J1_EXTRA, S_FEATS, S11_EXTRA, add_triangles, auc_midrank,
                       calib, contact_topology, fam_summary, fit_logistic,
                       jfeat0, jfeat1, predict, sfeat, summaries, truth_groups)

REACH_ADJ = 60.0
S12_EXTRA = ["tpc_adj", "diff_tpc_nn", "xgap_doca"]
V12_FEATS = S_FEATS + S11_EXTRA + S12_EXTRA


# ---------------------------------------------------------------- phase 1
def _p1(arg):
    f, cache = arg
    out = os.path.join(cache, os.path.basename(f).replace(".npz", ".pkl"))
    if os.path.exists(out):
        return out
    d = np.load(f, allow_pickle=True)
    ev = {k: d[k] for k in ("x", "y", "z", "e", "tpc", "labels",
                            "truth_t0", "ts_final")}
    ev["file"] = os.path.basename(f).split("_")[0]
    S = summaries(ev)
    gid, groups = truth_groups(S)
    labs = sorted(S)
    cents = np.stack([S[l]["mu"] for l in labs]) if labs else np.zeros((0, 3))
    pairs = []
    for a_i in range(len(labs) - 1):
        dc_row = np.linalg.norm(cents[a_i + 1:] - cents[a_i], axis=1)
        for off in np.flatnonzero(dc_row <= REACH_SHOWER):
            b_i = a_i + 1 + int(off)
            a, b = labs[a_i], labs[b_i]
            dc = float(dc_row[off])
            shower = S[a]["typ"] == "shower" or S[b]["typ"] == "shower"
            if dc > (REACH_SHOWER if shower else PAIR_RCENT):
                continue
            ft = jfeat0(S[a], S[b])
            v0_ok = dc <= PAIR_RCENT and ft["d_min"] <= PAIR_DMIN
            if not v0_ok and not (shower and ft["d_min"] <= 60.0):
                continue
            ft = jfeat1(S[a], S[b], ft)
            pairs.append(dict(a=a, b=b, ft=ft, v0=v0_ok,
                              y=float(gid.get(a, -1) == gid.get(b, -2))))
    boxes = {}
    for t in np.unique(ev["tpc"]):
        m = ev["tpc"] == t
        boxes[int(t)] = [float(ev["x"][m].min()), float(ev["x"][m].max()),
                         float(ev["z"][m].min()), float(ev["z"][m].max())]
    with open(out, "wb") as fh:
        pickle.dump(dict(file=ev["file"], name=os.path.basename(f),
                         S=S, gid=gid, groups=groups, pairs=pairs,
                         boxes=boxes), fh, protocol=4)
    return out


def build_adjacency(box_list):
    B = {}
    for boxes in box_list:
        for t, b in boxes.items():
            if t in B:
                B[t] = [min(B[t][0], b[0]), max(B[t][1], b[1]),
                        min(B[t][2], b[2]), max(B[t][3], b[3])]
            else:
                B[t] = list(b)
    adj = {}
    for t in B:
        nb = set()
        for u in B:
            if u == t:
                continue
            dx = max(B[u][0] - B[t][1], B[t][0] - B[u][1])
            dz = max(B[u][2] - B[t][3], B[t][2] - B[u][3])
            if (dx < 0 and 0 <= dz < 20) or (dz < 0 and 0 <= dx < 20) \
               or (dx < 0 and dz < 0):
                nb.add(u)
        adj[t] = nb
    return adj


# ---------------------------------------------------------------- phase 3
def _axis_doca(si, sm):
    """gap-aware: min perp distance of one cluster's hits to the OTHER's
    infinite axis (both directions, best)."""
    da = float(np.linalg.norm(
        np.cross(sm["sub"] - si["mu"], si["v1"]), axis=1).min())
    db = float(np.linalg.norm(
        np.cross(si["sub"] - sm["mu"], sm["v1"]), axis=1).min())
    return min(da, db)


def _p3(arg):
    pkl, j0_md, adj = arg
    E = pickle.load(open(pkl, "rb"))
    S, gid, groups = E["S"], E["gid"], E["groups"]
    j0_of = {}
    if E["pairs"]:
        X = np.array([[pr["ft"][k] for k in J0_FEATS] for pr in E["pairs"]])
        p0 = predict(j0_md, X)
        for pr, v in zip(E["pairs"], p0):
            j0_of.setdefault(pr["a"], {})[pr["b"]] = float(v)
            j0_of.setdefault(pr["b"], {})[pr["a"]] = float(v)
    rows, ys, flags = [], [], []
    fams = {g: [l for l in mem if l in S] for g, mem in enumerate(groups)}
    # family summaries cached ONCE per family; leave-one-out recompute only
    # for the candidate's own family (the expensive SVD was per-row before)
    F_full = {g: fam_summary(mem, S) for g, mem in fams.items() if mem}
    for l, si in S.items():
        for g, mem in fams.items():
            mm = [m for m in mem if m != l]
            if not mm:
                continue
            cds = {m: float(np.linalg.norm(S[m]["mu"] - si["mu"])) for m in mm}
            m_near = min(cds, key=cds.get)
            cd = cds[m_near]
            shower = si["typ"] == "shower" or any(
                S[m]["typ"] == "shower" for m in mm)
            fam_tpcs = set().union(*(S[m]["tpcs"] for m in mm))
            adj_ok = any(u in adj.get(si["tpc"], set()) for u in fam_tpcs)
            in_v0 = cd <= PAIR_RCENT
            in_v11 = in_v0 or (shower and cd <= REACH_SHOWER)
            in_v12 = in_v11 or (adj_ok and cd <= REACH_ADJ)
            if not in_v12:
                continue
            F = fam_summary(mm, S) if l in mem else F_full.get(g)
            if F is None:
                continue
            ft = sfeat(si, F, S, j0_of.get(l, {}), v11=True)
            # ---- v1.2 neighbor-TPC features ----
            sm = S[m_near]
            diff_nn = float(sm["tpc"] != si["tpc"])
            ft["tpc_adj"] = float(si["tpc"] in fam_tpcs or adj_ok)
            ft["diff_tpc_nn"] = diff_nn
            ft["xgap_doca"] = _axis_doca(si, sm) if diff_nn else 0.0
            rows.append([ft[k] for k in V12_FEATS])
            ys.append(float(gid.get(l, -1) == g))
            flags.append((in_v0, in_v11, in_v12, ft["crossing"] > 0,
                          ft["thru_shower"] > 0, ft["d_min"] > CONTACT,
                          ft["tpc_in"] < 0.5, E["file"]))
    return rows, ys, flags


# ---------------------------------------------------------------- main
def main():
    import multiprocessing as mp
    dump_dir = sys.argv[1] if len(sys.argv) > 1 else "v060dump_out/dumps"
    cache = sys.argv[2] if len(sys.argv) > 2 else \
        os.environ.get("SPTL_CACHE", "/tmp/sptl_v082_cache")
    os.makedirs(cache, exist_ok=True)
    files = sorted(glob.glob(os.path.join(dump_dir, "*.npz")))

    print(f"phase 1: {len(files)} events ...", flush=True)
    with mp.Pool(16) as pool:
        pkls = pool.map(_p1, [(f, cache) for f in files])

    # ---------------- phase 2: J models ----------------
    Es = [pickle.load(open(p, "rb")) for p in pkls]
    adj = build_adjacency([E["boxes"] for E in Es])
    nbc = {len(v) for v in adj.values()}
    print(f"adjacency: neighbor counts {sorted(nbc)}", flush=True)

    def pair_mat(sel, feats):
        rows, ys, meta = [], [], []
        for E in Es:
            if not sel(E["file"]):
                continue
            for pr in E["pairs"]:
                if not pr["v0"]:
                    continue
                rows.append([pr["ft"][k] for k in feats])
                ys.append(pr["y"])
                meta.append(pr)
        return np.array(rows, float), np.array(ys), meta

    tr_s = lambda f: f <= "0000008"
    te_s = lambda f: f >= "0000009"
    dep_s = lambda f: f != "0000001"

    X, y, _ = pair_mat(tr_s, J0_FEATS)
    j0 = fit_logistic(X, y)
    Xt, yt, _ = pair_mat(te_s, J0_FEATS)
    print(f"J0: n {len(X)} base {y.mean():.3f} | test AUC "
          f"{auc_midrank(yt, predict(j0, Xt)):.3f} (doc 0.649)", flush=True)
    # triangles need j0 on the v0 graph
    for E in Es:
        j0d = {}
        if E["pairs"]:
            Xa = np.array([[pr["ft"][k] for k in J0_FEATS]
                           for pr in E["pairs"]])
            for pr, v in zip(E["pairs"], predict(j0, Xa)):
                pr["j0"] = float(v)
                if pr["v0"]:
                    j0d[(pr["a"], pr["b"])] = pr["j0"]
        tri = add_triangles(list(j0d), j0d, E["S"])
        for pr in E["pairs"]:
            pr["ft"].update(tri.get((pr["a"], pr["b"]),
                                    dict(tri_max=0.0, tri_n90=0.0,
                                         tri_cross_max=0.0)))
    F1 = J0_FEATS + J1_EXTRA
    X, y, _ = pair_mat(tr_s, F1)
    j1 = fit_logistic(X, y)
    Xt, yt, mt = pair_mat(te_s, F1)
    p = predict(j1, Xt)
    tt = np.array([m["ft"]["both_track"] > 0 for m in mt])
    print(f"J1: test AUC {auc_midrank(yt, p):.3f} (doc 0.720) | "
          f"track-track {auc_midrank(yt[tt], p[tt]):.3f} (doc 0.864)",
          flush=True)
    Xd, yd, _ = pair_mat(dep_s, J0_FEATS)
    j0_deploy = fit_logistic(Xd, yd)
    del Es

    # ---------------- phase 3: S rows (parallel) ----------------
    print("phase 3: S rows ...", flush=True)
    with mp.Pool(16) as pool:
        parts = pool.map(_p3, [(p, j0, adj) for p in pkls])
    X = np.array(sum((p[0] for p in parts), []), float)
    y = np.array(sum((p[1] for p in parts), []))
    fl = sum((p[2] for p in parts), [])
    in0 = np.array([f[0] for f in fl]); in11 = np.array([f[1] for f in fl])
    crs = np.array([f[3] for f in fl]); ths = np.array([f[4] for f in fl])
    ncn = np.array([f[5] for f in fl]); fresh = np.array([f[6] for f in fl])
    fil = np.array([f[7] for f in fl])
    TR, TE, DEP = fil <= "0000008", fil >= "0000009", fil != "0000001"

    cols = {"v0": [V12_FEATS.index(k) for k in S_FEATS],
            "v1.1": [V12_FEATS.index(k) for k in S_FEATS + S11_EXTRA],
            "v1.2": list(range(len(V12_FEATS)))}
    pops = {"v0": in0, "v1.1": in11, "v1.2": np.ones(len(y), bool)}
    models = {}
    for tag in ("v0", "v1.1", "v1.2"):
        c, m = cols[tag], pops[tag]
        md = fit_logistic(X[np.ix_(m & TR, c)], y[m & TR])
        models[tag] = md
        mte = m & TE
        p = predict(md, X[np.ix_(mte, c)])
        yy = y[mte]
        def sauc(extra):
            mm = mte & extra
            pp = predict(md, X[np.ix_(mm, c)])
            return auc_midrank(y[mm], pp)
        doc = " (doc 0.915)" if tag == "v0" else ""
        print(f"S {tag}: n_tr {int((m & TR).sum())} base "
              f"{y[m & TR].mean():.3f} | test AUC {auc_midrank(yy, p):.3f}"
              f"{doc} | crossing {sauc(crs):.3f} | thru-shower "
              f"{sauc(ths):.3f} | non-contact {sauc(ncn):.3f} | "
              f"fresh-TPC {sauc(fresh):.3f} (base {y[mte & fresh].mean():.3f},"
              f" n={int((mte & fresh).sum())})", flush=True)
        for row in calib(yy, p):
            print("   calib", row)

    # deploy (files != 0000001) for the h2h
    dep_md = {}
    for tag in ("v1.1", "v1.2"):
        c, m = cols[tag], pops[tag]
        dep_md[tag] = fit_logistic(X[np.ix_(m & DEP, c)], y[m & DEP])
    art = dict(j0=j0, j1=j1, s0=models["v0"], s11=models["v1.1"],
               s12=models["v1.2"], s11_deploy=dep_md["v1.1"],
               s12_deploy=dep_md["v1.2"], j0_deploy=j0_deploy)
    with open("sptl_v081_models.json", "w") as f:
        json.dump({k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                       for kk, vv in v.items()} for k, v in art.items()},
                  f, indent=1)
    json.dump({str(k): sorted(v) for k, v in adj.items()},
              open("tpc_adjacency.json", "w"))
    np.savez_compressed("sptl_v082_rows.npz", X=X, y=y,
                        in0=in0, in11=in11, fresh=fresh, fil=fil)
    # hybrid: HGB (v4's model class) on the v1.2 feature superset, isotonic-
    # calibrated; deploy = files != 0000001 for the operational h2h
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import HistGradientBoostingClassifier
    import joblib
    hgb = CalibratedClassifierCV(HistGradientBoostingClassifier(
        max_iter=300, max_depth=6, learning_rate=0.08), method="isotonic",
        cv=3)
    hgb.fit(X[TR], y[TR])
    p = hgb.predict_proba(X[TE])[:, 1]
    print(f"S v1.2-HGB: test AUC {auc_midrank(y[TE], p):.3f} | fresh-TPC "
          f"{auc_midrank(y[TE & fresh], hgb.predict_proba(X[TE & fresh])[:, 1]):.3f}",
          flush=True)
    for row in calib(y[TE], p):
        print("   calib", row)
    hgb_dep = CalibratedClassifierCV(HistGradientBoostingClassifier(
        max_iter=300, max_depth=6, learning_rate=0.08), method="isotonic",
        cv=3)
    hgb_dep.fit(X[DEP], y[DEP])
    joblib.dump({"model": hgb_dep, "features": V12_FEATS},
                "sptl_v082_hgb.joblib")
    print("saved sptl_v081_models.json + tpc_adjacency.json + "
          "sptl_v082_rows.npz + sptl_v082_hgb.joblib", flush=True)


if __name__ == "__main__":
    main()
