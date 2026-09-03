"""v0.10.1 cluster-level confidence - RULE-BASED (user 2026-09-03: no
MC-trained ML in the data path).

conf_rule_v0101.json: 4 transparent checks on the scan-landscape block
(rel_margin, pcu_vote10, pcu_sig_med, c_cosres - all computed from light
data only; thresholds tuned once on file-0000001 MC truth). Confidence
level = number of passed checks; the delivered 0-1 value is the level's
measured E-weighted purity (5-entry lookup). Clusters with no scan support
(light-blind) fail all checks -> level 0, same as at tuning time.

E-wt AUC 0.961 (vs 0.975 for the retired 79-feature HGB); needs ONLY
scan_margin.event_scan_feats -> the stage costs ~0.3 s/event on GPU.

The ML path (subtick_conf_model*.joblib via model_path) is kept for
development comparisons only.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from scan_margin import event_scan_feats, SCAN_NAMES

DEFAULT_RULE = os.path.join(HERE, "conf_rule_v0102.json")
_RULE = None
_MODEL = None


def _get_rule(path):
    global _RULE
    if _RULE is None:
        _RULE = json.load(open(path))
        for c in _RULE.get("checks", []) + _RULE.get("penalties", []):
            c["idx"] = SCAN_NAMES.index(c["feature"])
        if "backbone" in _RULE:
            _RULE["backbone_idx"] = SCAN_NAMES.index(_RULE["backbone"])
    return _RULE


def _get_model(path):
    global _MODEL
    if _MODEL is None:
        import joblib
        _MODEL = joblib.load(path)
    return _MODEL


def _scan(base_image, actual, veto_mask, image_maps, labels, ns, n_light):
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
    try:
        import torch
        gpu = torch.cuda.is_available()
    except Exception:
        gpu = False
    return event_scan_feats(np.asarray(base_image, np.float64),
                            np.asarray(actual, np.float64),
                            veto_mask, al, at, ar, lab_t0, use_gpu=gpu)


def score_event(x, y, z, e, tpc, labels, ts_final, actual, base_image,
                veto_mask, image_maps, ns, n_light, model_path=None):
    """Returns (lab_conf {label: conf 0-1}, hit_conf ndarray, stats dict).

    Default: the 4-check rule (data-path safe). model_path -> dev-only ML.
    """
    ts = np.asarray(ts_final, np.float64)
    sfeats = _scan(base_image, actual, veto_mask, image_maps, labels, ns,
                   n_light)
    hit_conf = np.full(ts.size, np.nan, np.float32)
    lab_conf = {}

    if model_path:                                   # dev-only ML path
        return _score_ml(x, y, z, e, tpc, labels, ts, actual, base_image,
                         ns, sfeats, model_path)

    rule = _get_rule(DEFAULT_RULE)
    n_scan = 0
    all_labs = np.unique(labels[(labels >= 0)])
    fin = np.isfinite(ts) & (ts >= 0)
    if rule.get("mode") == "product":
        # v0.10.2 user design: conf = pcu_vote10 * penalty factors
        for L in all_labs:
            L = int(L)
            if not (fin & (labels == L)).any():
                continue
            sf = sfeats.get(L)
            if sf is None:
                # light-blind: fixed measured-prior value, no calibration
                conf = float(rule.get("no_scan_conf", 0.5))
                lab_conf[L] = conf
                hit_conf[labels == L] = np.float32(conf)
                continue
            if True:
                n_scan += 1
                conf = float(sf[0][rule["backbone_idx"]])
                for c in rule["penalties"]:
                    if float(sf[0][c["idx"]]) < c["below"]:
                        conf *= c["factor"]
            cal = rule.get("calibration")
            if cal is not None:
                conf = float(np.interp(conf, cal["knots"], cal["values"]))
            lab_conf[L] = conf
            hit_conf[labels == L] = np.float32(conf)
        stats = {"n_clusters": len(lab_conf), "n_with_scan": n_scan,
                 "mode": "product"}
        if lab_conf:
            vals = np.array(list(lab_conf.values()))
            stats["conf_med"] = round(float(np.median(vals)), 3)
            stats["conf_frac_lo"] = round(float((vals < 0.25).mean()), 3)
        return lab_conf, hit_conf, stats
    purity = {int(k): float(v) for k, v in rule["level_purity_E"].items()}
    levels = []
    for L in all_labs:
        L = int(L)
        if not (fin & (labels == L)).any():
            continue
        sf = sfeats.get(L)
        lv = 0
        if sf is not None:
            n_scan += 1
            for c in rule["checks"]:
                v = float(sf[0][c["idx"]])
                if (v >= c["threshold"]) if c["op"] == ">=" else (v <= c["threshold"]):
                    lv += 1
        conf = purity.get(lv, purity[0])
        lab_conf[L] = conf
        hit_conf[labels == L] = np.float32(conf)
        levels.append(lv)
    stats = {"n_clusters": len(lab_conf), "n_with_scan": n_scan,
             "mode": "rule4"}
    if levels:
        lv = np.asarray(levels)
        stats["level_frac"] = [round(float((lv == k).mean()), 3)
                               for k in range(5)]
        stats["conf_med"] = round(float(np.median(list(lab_conf.values()))), 3)
    return lab_conf, hit_conf, stats


def _score_ml(x, y, z, e, tpc, labels, ts, actual, base_image, ns, sfeats,
              model_path):
    """Retired 79-feature HGB path, kept for development comparisons."""
    import v083_prune as vp
    from confidence_cluster import cluster_scores
    from subtick_conf_study2 import score_model
    mdl = _get_model(model_path)
    e_pos = np.clip(np.asarray(e, np.float64), 0, None)
    base = np.asarray(base_image, np.float64)
    act = np.asarray(actual, np.float64)
    conf = cluster_scores(labels, tpc, np.asarray(y, np.float64), e_pos,
                          ts, act, base)
    ev = {"x": np.asarray(x), "y": np.asarray(y), "z": np.asarray(z),
          "e": np.asarray(e), "tpc": tpc, "labels": labels,
          "ts_final": ts, "truth_t0": np.full(ts.size, np.nan)}
    crows = vp.cluster_rows(ev, conf)
    cf, cm, gid = ns["subtick_group"]
    gf = {int(m[0]): f for f, m in zip(cf, cm)}
    n_sf = len(mdl.get("scan_feats", [])) or 8
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
        S = np.zeros(n_sf + 1)
        sf = sfeats.get(l)
        if sf is not None:
            S[:-1] = np.nan_to_num(sf[0][:n_sf], nan=0.0,
                                   posinf=1e9, neginf=-1e9)
            S[-1] = 1.0
            n_scan += 1
        if mdl.get("scan_only"):
            rows.append(S)
        elif mdl.get("no_group"):
            rows.append(np.concatenate([X, S]))
        else:
            rows.append(np.concatenate([X, G, S]))
        labs_scored.append(l)
    lab_conf = {}
    hit_conf = np.full(ts.size, np.nan, np.float32)
    stats = {"n_clusters": len(rows), "n_with_scan": n_scan, "mode": "ml"}
    if rows:
        F = np.vstack(rows)
        if mdl.get("feat_cols") is not None:
            F = F[:, mdl["feat_cols"]]
        p = mdl["iso"].predict(score_model(mdl["model"], F))
        for l, pv in zip(labs_scored, p):
            lab_conf[l] = float(pv)
            hit_conf[labels == l] = np.float32(pv)
        stats["conf_med"] = round(float(np.median(p)), 3)
        stats["conf_frac_lo"] = round(float(np.mean(p < 0.5)), 3)
    return lab_conf, hit_conf, stats
