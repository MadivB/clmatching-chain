"""v0.8.1 head-to-head on file 0000001 (out-of-sample for ALL models):
spatial_score_v4 HGB  vs  S_iF v1.1  vs  S_iF v1.2 (neighbor-TPC)
(deploy S models trained files 2-10; v4 trained files != 0000001).

Identical task, identical rows: every assigned cluster with truth, its OWN
reco family (time-grouped ts_final), and the SAME candidate set as the real
PASS 2 (8 nearest families by centroid).  Report:
  1 misplacement-detection AUC (label: |t_own - truth| > 10)
  2 decision replay at PASS-2 gates (margin 0.3 / p_min 0.6, no light veto):
    proposed -> FIXED (was wrong, destination time correct) / BROKE

Usage: python sptl_v081_h2h.py
"""
import glob
import json
import os
import sys

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sptl_v081 import (CUT, J0_FEATS, PAIR_DMIN, PAIR_RCENT, S11_EXTRA,
                       S_FEATS, auc_midrank, emedian, fam_summary, jfeat0,
                       predict, sfeat, summaries)
from sptl_v082 import S12_EXTRA, V12_FEATS, _axis_doca
from spatial_score import cluster_shape
from spatial_score4 import FEATURES2, pair_features2
from spatial_trade import build_families

MARGIN, P_MIN, N_CAND = 0.3, 0.6, 8
SFX11 = S_FEATS + S11_EXTRA


def np_load(f):
    d = np.load(f, allow_pickle=True)
    return {k: d[k] for k in ("x", "y", "z", "e", "tpc", "labels",
                              "truth_t0", "ts_final")}


def reco_families(S, ev):
    ts, labs = ev["ts_final"].astype(np.float64), ev["labels"]
    at = {}
    for l in S:
        m = (labs == l) & np.isfinite(ts) & (ts >= 0)
        if m.any():
            at[l] = emedian(ts[m], np.maximum(ev["e"][m].astype(np.float64), 1e-6))
    order = sorted(at, key=at.get)
    fams, cur = [], []
    for l in order:
        if cur and at[l] - at[cur[-1]] > 3.0:
            fams.append(cur)
            cur = []
        cur.append(l)
    if cur:
        fams.append(cur)
    own = {l: g for g, mem in enumerate(fams) for l in mem}
    f_t0 = {g: float(np.median([at[l] for l in mem]))
            for g, mem in enumerate(fams)}
    return fams, own, f_t0, at


def s_features(si, mm, S, j0_i, adj, F=None):
    if F is None:
        F = fam_summary(mm, S)
    if F is None:
        return None, None
    ft = sfeat(si, F, S, j0_i, v11=True)
    cds = {m: float(np.linalg.norm(S[m]["mu"] - si["mu"])) for m in mm}
    sm = S[min(cds, key=cds.get)]
    fam_tpcs = set().union(*(S[m]["tpcs"] for m in mm))
    diff_nn = float(sm["tpc"] != si["tpc"])
    ft["tpc_adj"] = float(si["tpc"] in fam_tpcs or any(
        u in adj.get(si["tpc"], set()) for u in fam_tpcs))
    ft["diff_tpc_nn"] = diff_nn
    ft["xgap_doca"] = _axis_doca(si, sm) if diff_nn else 0.0
    return (np.array([[ft[k] for k in SFX11]]),
            np.array([[ft[k] for k in V12_FEATS]]))


_CTX = {}


def _ctx():
    if not _CTX:
        md = json.load(open("sptl_v081_models.json"))
        for k in md:
            md[k] = {kk: (np.array(v) if isinstance(v, list) else v)
                     for kk, v in md[k].items()}
        _CTX["md"] = md
        _CTX["adj"] = {int(k): set(v) for k, v in
                       json.load(open("tpc_adjacency.json")).items()}
        v4 = joblib.load("spatial_score_v4.joblib")
        assert v4["features"] == FEATURES2
        _CTX["clf4"] = v4["model"]
        _CTX["clf12h"] = joblib.load("sptl_v082_hgb.joblib")["model"]
    return _CTX["md"], _CTX["adj"], _CTX["clf4"], _CTX["clf12h"]


def _h2h_event(f):
    md, adj, clf4, clf12h = _ctx()
    rows = []
    if True:
        ev = np_load(f)
        S = summaries(ev)
        fams, own, f_t0, at = reco_families(S, ev)
        if len(fams) < 2:
            return rows
        # deploy-J0 for S features
        labs = sorted(S)
        j0_of = {l: {} for l in labs}
        prs, keys = [], []
        cents = np.stack([S[l]["mu"] for l in labs])
        for i in range(len(labs) - 1):
            dc = np.linalg.norm(cents[i + 1:] - cents[i], axis=1)
            for off in np.flatnonzero(dc <= PAIR_RCENT):
                a, b = labs[i], labs[i + 1 + int(off)]
                ft = jfeat0(S[a], S[b])
                if ft["d_min"] > PAIR_DMIN:
                    continue
                prs.append([ft[k] for k in J0_FEATS])
                keys.append((a, b))
        if prs:
            for (a, b), v in zip(keys, predict(md["j0_deploy"],
                                               np.array(prs, float))):
                j0_of[a][b] = float(v)
                j0_of[b][a] = float(v)
        # v4 machinery
        xyz_all = np.stack([ev["x"], ev["y"], ev["z"]], 1).astype(np.float64)
        e_all = np.clip(ev["e"].astype(np.float64), 0, None)
        shp, cmask = {}, {}
        for l in S:
            m = ev["labels"] == l
            shp[l] = cluster_shape(xyz_all[m], e_all[m])
            cmask[l] = m
        F4 = build_families({"e": e_all}, shp, cmask, at, 3.0)
        fam_mu = {g: np.mean([S[m]["mu"] for m in mem if m in S], axis=0)
                  for g, mem in enumerate(fams)
                  if any(m in S for m in mem)}
        Fc = {g: fam_summary([m for m in mem if m in S], S)
              for g, mem in enumerate(fams) if any(m in S for m in mem)}

        def score_all(l, si, g):
            mm = [m for m in fams[g] if m != l and m in S]
            if not mm:
                return None
            F = fam_summary(mm, S) if l in fams[g] else Fc.get(g)
            x11, x12 = s_features(si, mm, S, j0_of.get(l, {}), adj, F=F)
            if x11 is None:
                return None
            p11 = float(predict(md["s11_deploy"], x11)[0])
            p12 = float(predict(md["s12_deploy"], x12)[0])
            p12h = float(clf12h.predict_proba(x12)[0, 1])
            g4 = (min(F4, key=lambda k: abs(F4[k]["t0"] - f_t0[g]))
                  if F4 else None)
            p4 = None
            if g4 is not None and abs(F4[g4]["t0"] - f_t0[g]) <= 5.0:
                ft4 = pair_features2(shp[l], F4[g4])
                p4 = float(clf4.predict_proba(
                    np.array([[ft4[k] for k in FEATURES2]], float))[0, 1])
            return p11, p12, p12h, p4

        for l, si in S.items():
            if l not in own or si["truth"] is None:
                continue
            g = own[l]
            sc = score_all(l, si, g)
            if sc is None or sc[3] is None:
                continue
            wrong = abs(at[l] - si["truth"]) > CUT
            near = sorted((g2 for g2 in fam_mu if g2 != g),
                          key=lambda g2: float(np.linalg.norm(
                              fam_mu[g2] - si["mu"])))[:N_CAND]
            alts = []
            for g2 in near:
                if abs(f_t0[g2] - at[l]) <= 2.0:
                    continue
                sc2 = score_all(l, si, g2)
                if sc2 is None:
                    continue
                alts.append(dict(p11=sc2[0], p12=sc2[1], p12h=sc2[2],
                                 p4=sc2[3],
                                 ok=abs(f_t0[g2] - si["truth"]) <= CUT))
            rows.append(dict(lab=int(l), e=si["e"], wrong=bool(wrong),
                             p11=sc[0], p12=sc[1], p12h=sc[2], p4=sc[3],
                             alts=alts))
    return rows


MODELS = (("v4 HGB", "p4"), ("S v1.1", "p11"), ("S v1.2", "p12"),
          ("v1.2-HGB", "p12h"))


def main():
    import multiprocessing as mp
    import pickle
    files = sorted(glob.glob("v060dump_out/dumps/0000001_ev*.npz"))
    with mp.Pool(min(13, len(files))) as pool:
        rows = sum(pool.map(_h2h_event, files), [])
    pickle.dump(rows, open("sptl_h2h_rows.pkl", "wb"))

    y = np.array([r["wrong"] for r in rows], float)
    e_arr = np.array([r["e"] for r in rows])
    small = e_arr < 30.0
    print(f"{len(rows)} identical (cluster, own-family) rows | "
          f"wrong rate {y.mean():.3f} | small(<30 MeV) {int(small.sum())} "
          f"(wrong rate {y[small].mean():.3f})")
    for name, kp in MODELS:
        s = 1 - np.array([r[kp] for r in rows])
        print(f"  misplacement AUC {name:8s}: all {auc_midrank(y, s):.3f}  "
              f"small {auc_midrank(y[small], s[small]):.3f}")
    for gate, gname in ((None, "all-E"), (30.0, "prod-gate e<30")):
        for name, kp in MODELS:
            nf = nb = npr = 0
            ef = eb = 0.0
            for r in rows:
                if gate is not None and r["e"] >= gate:
                    continue
                av = [a for a in r["alts"] if a[kp] is not None]
                if not av:
                    continue
                best = max(av, key=lambda a: a[kp])
                if best[kp] - r[kp] > MARGIN and best[kp] > P_MIN:
                    npr += 1
                    if r["wrong"] and best["ok"]:
                        nf += 1
                        ef += r["e"]
                    elif not r["wrong"]:
                        nb += 1
                        eb += r["e"]
            print(f"replay [{gname}] {name:8s}: proposed {npr:4d}  "
                  f"FIXED {nf:3d} ({ef:6.0f} MeV)  BROKE {nb:3d} "
                  f"({eb:6.0f} MeV)  net {ef - eb:+7.0f} MeV")


if __name__ == "__main__":
    main()
