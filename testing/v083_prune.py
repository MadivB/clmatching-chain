"""v0.8.3 study: confidence-based HIT PRUNING.

A supervised "badness" score per ASSIGNED cluster from charge+light signals:
  - the full v4 spatial feature vector of the cluster vs its OWN reco family
    (contact, continuation, crossing, cone, vertex, envelope, ...)
  - light confidence from the in-chain per-cluster metrics (z_min, z_own,
    the regret dz = z_own - z_min, chi2_loc, n_tpc)
  - basics (logE, log n_hits, E/hit, |t_c - t_family|, family size/energy)
HGB (NaN-native), leave-one-file-out over the 126 cached events -> every
cluster scored OUT-OF-SAMPLE.  Prune the worst clusters until X% of assigned
HITS are dropped (t -> -1) and measure the accuracy of the hits that REMAIN
assigned (energy- and hit-weighted, NS band and all-TPC).

Usage: python v083_prune.py [dump_dir]
"""
import glob
import json
import os
import sys

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sptl_v081 import emedian
from spatial_score import cluster_shape
from spatial_score4 import FEATURES2, pair_features2
from spatial_trade import build_families

CUT = 10.0
NS_EMAX = 1400.0
CONF_FEATS = ["z_min", "z_own", "dz", "chi2_loc", "n_tpc"]
BASIC_FEATS = ["logE", "lognh", "e_hit", "dt_fam", "fam_n", "fam_loge"]
ALL_FEATS = FEATURES2 + CONF_FEATS + BASIC_FEATS


_V4 = {}


def _v4_model():
    if not _V4:
        _V4["m"] = joblib.load(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "spatial_score_v4.joblib"))["model"]
    return _V4["m"]


def cluster_rows(ev, conf):
    """Feature rows for every assigned cluster of one event.

    ev: dict with x,y,z,e,tpc,labels,ts_final,truth_t0 (truth may be all-NaN
    at apply time — truth-dependent bookkeeping fields are then 0).
    conf: per-cluster light-confidence dict (str or int label keys).
    """
    labs_a = ev["labels"]
    ts = ev["ts_final"].astype(np.float64)
    tru = ev["truth_t0"].astype(np.float64)
    e = np.clip(ev["e"].astype(np.float64), 0, None)
    xyz = np.stack([ev["x"], ev["y"], ev["z"]], 1).astype(np.float64)
    assigned = np.isfinite(ts) & (ts >= 0)

    # per-TPC truth energy -> NS mask per hit
    tpc_e = {}
    okt = np.isfinite(tru)
    for t in np.unique(ev["tpc"][okt]):
        tpc_e[int(t)] = float(e[okt & (ev["tpc"] == t)].sum())

    shp, cmask, atime = {}, {}, {}
    for l in np.unique(labs_a[labs_a >= 0]):
        m = labs_a == l
        fa = m & assigned
        if m.sum() < 3 or not fa.any():
            continue
        shp[int(l)] = cluster_shape(xyz[m], e[m])
        cmask[int(l)] = m
        atime[int(l)] = emedian(ts[fa], np.maximum(e[fa], 1e-6))
    F = build_families({"e": e}, shp, cmask, atime, 3.0)

    rows = []
    for l in shp:
        m = cmask[l]
        fa = m & assigned
        mt = m & np.isfinite(tru)
        tr = emedian(tru[mt], np.maximum(e[mt], 1e-6)) if mt.any() else None
        ft = dict.fromkeys(ALL_FEATS, np.nan)
        g = min(F, key=lambda k: abs(F[k]["t0"] - atime[l])) if F else None
        if g is not None and abs(F[g]["t0"] - atime[l]) <= 5.0:
            for k, v in pair_features2(shp[l], F[g]).items():
                ft[k] = v
            ft["dt_fam"] = abs(F[g]["t0"] - atime[l])
            ft["fam_n"] = float(len(F[g]["members2"]))
            ft["fam_loge"] = float(np.log10(max(
                sum(float(np.clip(m2[0]["e"], 0, None).sum())
                    for m2 in F[g]["members2"]), 1e-3)))
        c = conf.get(str(int(l))) or conf.get(int(l)) or {}
        for k in ("z_min", "z_own", "chi2_loc", "n_tpc"):
            if k in c:
                ft[k] = float(c[k])
        if "z_own" in c and "z_min" in c:
            ft["dz"] = float(c["z_own"]) - float(c["z_min"])
        e_c = float(e[fa].sum())
        ft["logE"] = float(np.log10(max(e_c, 1e-3)))
        ft["lognh"] = float(np.log10(max(int(fa.sum()), 1)))
        ft["e_hit"] = e_c / max(int(fa.sum()), 1)
        tpcs = np.unique(ev["tpc"][m])
        ns_ok = all(tpc_e.get(int(t), 0.0) <= NS_EMAX for t in tpcs)
        rows.append(dict(
            label=int(l),
            x=[ft[k] for k in ALL_FEATS],
            wrong=(bool(abs(atime[l] - tr) > CUT) if tr is not None else False),
            e=e_c, nh=int(fa.sum()), ns=bool(ns_ok),
            e_good=float(e[fa & (np.abs(ts - tru) <= CUT)
                          & np.isfinite(tru)].sum()),
            nh_good=int((fa & np.isfinite(tru)
                         & (np.abs(ts - tru) <= CUT)).sum()),
            e_tr=float(e[fa & np.isfinite(tru)].sum()),
            nh_tr=int((fa & np.isfinite(tru)).sum())))
    return rows


def _proc(fp):
    d = np.load(fp, allow_pickle=True)
    ev = {k: d[k] for k in ("x", "y", "z", "e", "tpc", "labels",
                            "ts_final", "truth_t0")}
    try:
        conf = json.loads(str(d["cluster_conf"]))
    except Exception:
        conf = {}
    rows = [r for r in cluster_rows(ev, conf) if r["nh_tr"] > 0]
    ftag = os.path.basename(fp).split("_")[0]
    for r in rows:
        r["file"] = ftag
    return rows


_PRUNE_MD = {}


def prune_event(ev, conf, model_path, frac):
    """In-chain post-filter: score every assigned cluster, prune those over
    the calibrated global threshold for `frac`.  Returns stats + the pruned
    label list (the caller decides whether to actually blank ts)."""
    if model_path not in _PRUNE_MD:
        _PRUNE_MD[model_path] = joblib.load(model_path)
    bd = _PRUNE_MD[model_path]
    rows = cluster_rows(ev, conf)
    if not rows:
        return {"n_clusters": 0, "n_pruned": 0, "pruned_labels": []}
    X = np.array([r["x"] for r in rows], float)
    p = bd["model"].predict_proba(X)[:, 1]
    fkey = min(bd["thresholds"], key=lambda k: abs(float(k) - float(frac)))
    thr = float(bd["thresholds"][fkey])
    keep = [r for r, pi in zip(rows, p) if pi < thr]
    prun = [r for r, pi in zip(rows, p) if pi >= thr]

    def acc(rs):
        te = sum(r["e_tr"] for r in rs)
        return round(100 * sum(r["e_good"] for r in rs) / te, 3) if te > 0 else None

    return {
        "n_clusters": len(rows), "n_pruned": len(prun),
        "frac_hits_pruned": round(sum(r["nh"] for r in prun)
                                  / max(sum(r["nh"] for r in rows), 1), 5),
        "e_pruned": round(sum(r["e"] for r in prun), 1),
        "thr": thr, "thr_frac_key": float(fkey),
        "acc_before_all": acc(rows), "acc_after_all": acc(keep),
        "acc_before_ns": acc([r for r in rows if r["ns"]]),
        "acc_after_ns": acc([r for r in keep if r["ns"]]),
        "prune_purity": (round(sum(r["e_tr"] - r["e_good"] for r in prun)
                               / max(sum(r["e_tr"] for r in prun), 1e-9), 3)
                         if prun else None),
        "pruned_labels": [r["label"] for r in prun][:300]}


def main():
    import multiprocessing as mp
    dump_dir = sys.argv[1] if len(sys.argv) > 1 else "v060dump_out/dumps"
    files = sorted(glob.glob(os.path.join(dump_dir, "*.npz")))
    with mp.Pool(16) as pool:
        rows = sum(pool.map(_proc, files), [])
    X = np.array([r["x"] for r in rows], float)
    y = np.array([r["wrong"] for r in rows], float)
    fil = np.array([r["file"] for r in rows])
    print(f"{len(rows)} assigned clusters, wrong rate {y.mean():.3f} "
          f"(by energy {sum(r['e'] for r in rows if r['wrong'])/max(sum(r['e'] for r in rows),1e-9):.3f})")

    # leave-one-file-out out-of-sample badness
    from sklearn.ensemble import HistGradientBoostingClassifier
    p_bad = np.zeros(len(rows))
    for f in np.unique(fil):
        tr_m = fil != f
        clf = HistGradientBoostingClassifier(max_iter=300, max_depth=6,
                                             learning_rate=0.08)
        clf.fit(X[tr_m], y[tr_m])
        te_m = ~tr_m
        p_bad[te_m] = clf.predict_proba(X[te_m])[:, 1]
    from sptl_v081 import auc_midrank
    print(f"OOS wrong-cluster AUC: {auc_midrank(y, p_bad):.3f}")

    # pruning curve: drop worst clusters until N% of assigned truth-hits pruned
    order = np.argsort(-p_bad)
    nh_tr = np.array([r["nh_tr"] for r in rows], float)
    tot_nh = nh_tr.sum()

    def metrics(keep_mask):
        out = {}
        for tag, sel in (("NS", [r["ns"] for r in rows]),
                         ("all", [True] * len(rows))):
            sel = np.array(sel)
            ge = sum(r["e_good"] for r, k, s in zip(rows, keep_mask, sel) if k and s)
            te = sum(r["e_tr"] for r, k, s in zip(rows, keep_mask, sel) if k and s)
            gh = sum(r["nh_good"] for r, k, s in zip(rows, keep_mask, sel) if k and s)
            th = sum(r["nh_tr"] for r, k, s in zip(rows, keep_mask, sel) if k and s)
            out[tag] = (100 * ge / max(te, 1e-9), 100 * gh / max(th, 1))
        return out

    keep = np.ones(len(rows), bool)
    base = metrics(keep)
    print(f"\n{'pruned hits':>11} {'NS E-acc':>9} {'NS hit-acc':>10} "
          f"{'all E-acc':>9} {'all hit-acc':>11} {'prune purity':>12}")
    print(f"{'0.00%':>11} {base['NS'][0]:9.2f} {base['NS'][1]:10.2f} "
          f"{base['all'][0]:9.2f} {base['all'][1]:11.2f} {'':>12}")
    targets = [0.0025, 0.005, 0.01, 0.02, 0.03, 0.05]
    ti = 0
    pruned_nh = 0.0
    pr_wrong_e = pr_e = 0.0
    for i in order:
        keep[i] = False
        pruned_nh += nh_tr[i]
        pr_e += rows[i]["e_tr"]
        pr_wrong_e += rows[i]["e_tr"] - rows[i]["e_good"]
        while ti < len(targets) and pruned_nh >= targets[ti] * tot_nh:
            mm = metrics(keep)
            print(f"{100*targets[ti]:10.2f}% {mm['NS'][0]:9.2f} "
                  f"{mm['NS'][1]:10.2f} {mm['all'][0]:9.2f} "
                  f"{mm['all'][1]:11.2f} {100*pr_wrong_e/max(pr_e,1e-9):11.1f}%")
            ti += 1
        if ti >= len(targets):
            break
    np.savez_compressed("v083_prune_rows.npz", X=X, y=y, fil=fil, p_bad=p_bad,
                        e=[r["e"] for r in rows], nh=nh_tr,
                        ns=[r["ns"] for r in rows])
    print("\nsaved v083_prune_rows.npz")


if __name__ == "__main__":
    main()
