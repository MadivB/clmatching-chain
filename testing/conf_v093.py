"""v0.9.3 cluster-level calibrated confidence — the study3 winner baked.

Combines THREE evidence families per assigned cluster (LOO AUC 0.850 count /
0.970 E-weighted on file 0000001; subtick_conf_study3.py):
  - 40 v0.8.3 cluster features (29 spatial orphan-ness + anchored-covariance
    swap-aware block + basics)
  - 10 v0.9.2 t0-group features (quadratic fit + waveform comparison)
  - 8 scan-landscape features + has_scan (Delta-chi2-unit depth / competitor
    margin / PDA null posterior ... — scan_margin.event_scan_feats)
Model: HistGradientBoosting + isotonic calibration (subtick_conf_model3.joblib).

Serving-time feature construction mirrors subtick_conf_study2.load /
subtick_conf_study3 exactly (nan_to_num, majority-group join, hstack order
[X40 | G10 | S8+has_scan]).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import v083_prune as vp
from scan_margin import event_scan_feats

DEFAULT_MODEL = os.path.join(HERE, "subtick_conf_model3.joblib")
_MODEL = None


def _get_model(path):
    global _MODEL
    if _MODEL is None:
        import joblib
        _MODEL = joblib.load(path)
    return _MODEL


def score_event(x, y, z, e, tpc, labels, ts_final, actual, base_image,
                veto_mask, image_maps, ns, n_light, model_path=None):
    """Returns (lab_conf {label: conf 0-1}, hit_conf ndarray, stats dict)."""
    mdl = _get_model(model_path or DEFAULT_MODEL)
    e_pos = np.clip(np.asarray(e, np.float64), 0, None)
    base = np.asarray(base_image, np.float64)
    act = np.asarray(actual, np.float64)
    ts = np.asarray(ts_final, np.float64)

    from confidence_cluster import cluster_scores
    conf = cluster_scores(labels, tpc, np.asarray(y, np.float64), e_pos,
                          ts, act, base)
    ev = {"x": np.asarray(x), "y": np.asarray(y), "z": np.asarray(z),
          "e": np.asarray(e), "tpc": tpc, "labels": labels,
          "ts_final": ts, "truth_t0": np.full(ts.size, np.nan)}
    crows = vp.cluster_rows(ev, conf)

    cf, cm, gid = ns["subtick_group"]
    gf = {int(m[0]): f for f, m in zip(cf, cm)}

    al, at, ar = [], [], []
    for (lab, c), im in image_maps.items():
        if 0 <= int(c) < n_light:
            al.append(int(lab))
            at.append(int(c))
            ar.append(np.asarray(im, np.float64).max(axis=1))
    al = np.asarray(al, np.int32)
    at = np.asarray(at, np.int32)
    ar = (np.asarray(ar) if len(ar) else np.zeros((0, 120)))

    ts_bef = np.asarray(ns["ts_before_subtick"], np.float64)
    lab_t0 = {}
    for L in np.unique(al):
        m = (labels == L) & np.isfinite(ts_bef) & (ts_bef >= 0)
        if m.any():
            v, c = np.unique(np.round(ts_bef[m]).astype(int), return_counts=True)
            lab_t0[int(L)] = int(v[np.argmax(c)])
    sfeats = event_scan_feats(base, act, veto_mask, al, at, ar, lab_t0)

    rows, labs_scored = [], []
    n_scan = 0
    for r in crows:
        l = int(r["label"])
        m = labels == l
        gis = gid[m & (gid >= 0)]
        grow = None
        if gis.size:
            vals, cnt = np.unique(gis, return_counts=True)
            grow = gf.get(int(vals[np.argmax(cnt)]))
        X = np.nan_to_num(np.asarray(r["x"], np.float64), nan=0.0)
        G = np.nan_to_num(np.asarray(grow, np.float64) if grow is not None
                          else np.full(10, np.nan), nan=0.0)
        S = np.zeros(9)
        sf = sfeats.get(l)
        if sf is not None:
            S[:-1] = np.nan_to_num(sf[0], nan=0.0, posinf=1e9, neginf=-1e9)
            S[-1] = 1.0
            n_scan += 1
        rows.append(np.concatenate([X, G, S]))
        labs_scored.append(l)

    lab_conf = {}
    hit_conf = np.full(ts.size, np.nan, np.float32)
    stats = {"n_clusters": len(rows), "n_with_scan": n_scan}
    if rows:
        F = np.vstack(rows)
        from subtick_conf_study2 import score_model
        p = mdl["iso"].predict(score_model(mdl["model"], F))
        for l, pv in zip(labs_scored, p):
            lab_conf[l] = float(pv)
            hit_conf[labels == l] = np.float32(pv)
        stats["conf_med"] = round(float(np.median(p)), 3)
        stats["conf_frac_lo"] = round(float(np.mean(p < 0.5)), 3)
    return lab_conf, hit_conf, stats
