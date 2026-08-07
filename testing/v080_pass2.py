"""v0.8 PASS 2 (in-chain): spatial cluster trading between reco families,
light-curve veto (the recorded a_c(t) prior).  Q2 of the dual-question loop:
"is cluster i eating from another family's thing?" — answered spatially,
conditioned on i's light loss at all candidate family times.

Operates on the batch namespace after the light stages (joint/matrix), before
swap-fix/force.  Trades are NOT 1:1; margin-ordered; every trade truth-audited
by the caller via truth_fn.
"""
from __future__ import annotations

import os
from typing import Any, Callable

import joblib
import numpy as np
from scipy.spatial import cKDTree

from spatial_score import cluster_shape, emedian
from spatial_score4 import FEATURES2, pair_features2
from spatial_trade import build_families
from v030_expansion import GAIN, SEARCH, _shift, weighted_scan

CUT = 10.0
BIG_MEV = 30.0
_MODEL = {}


def _load_model(path):
    if path not in _MODEL:
        _MODEL[path] = joblib.load(path)
    return _MODEL[path]


def run_pass2_trading(ns: dict[str, Any], *, model_path: str,
                      margin: float = 0.3, p_min: float = 0.6,
                      veto_frac: float = 0.25, n_cand_fams: int = 8,
                      light_nominate: bool = False, nom_frac: float = 0.15,
                      nom_flag_frac: float = 0.4, nom_max: int = 4,
                      truth_fn: Callable | None = None,
                      verbose: bool = False) -> dict[str, Any]:
    md = _load_model(model_path)
    clf, feat_names = md["model"], md["features"]
    assert feat_names == FEATURES2, "model/feature version mismatch"

    image_maps = ns["imageMaps"]
    base = np.asarray(ns["baseImage"], dtype=np.float32)
    actual = np.asarray(ns["fullLightWaveform"], dtype=np.float64)
    n_light, n_ch, n_ticks = actual.shape
    labels = np.asarray(ns["labels_global"])
    ts = np.asarray(ns["hit_timestamps"], dtype=np.float64).copy()
    xyz = np.stack([np.asarray(ns["xset"], np.float64), np.asarray(ns["yset"], np.float64),
                    np.asarray(ns["zset"], np.float64)], axis=1)
    e_pos = np.clip(np.asarray(ns["Eset"], dtype=np.float64), 0.0, None)
    w_all = 1.0 / (GAIN * np.maximum(actual, 100.0))
    assigned = np.isfinite(ts) & (ts >= 0)

    stats = {"n_considered": 0, "n_scored": 0, "n_proposed": 0, "n_vetoed": 0,
             "n_traded": 0, "e_traded": 0.0, "n_light_flagged": 0,
             "n_light_nom": 0, "n_traded_via_light": 0, "audit": []}

    # ---------- shapes + assigned times ----------
    shp, atime, cmask = {}, {}, {}
    for l in np.unique(labels[labels >= 0]):
        m = labels == l
        if m.sum() < 3:
            continue
        fa = m & assigned
        if not fa.any():
            continue
        shp[int(l)] = cluster_shape(xyz[m], e_pos[m])
        atime[int(l)] = emedian(ts[fa], e_pos[fa])
        cmask[int(l)] = m

    ev = dict(e=e_pos)
    F = build_families(ev, shp, cmask, atime, 3.0)
    if len(F) < 2:
        return stats
    f_ids = sorted(F)
    f_mu = np.stack([F[g]["mu"] for g in f_ids])
    # one KD tree of member hits per family (for the cheap prefilter)
    f_tree = {}
    for g in f_ids:
        hx = np.concatenate([m[0]["xyz"] for m in F[g]["members2"]])
        f_tree[g] = cKDTree(hx)

    # ---------- candidates: small assigned clusters ----------
    small = [l for l in shp if float(e_pos[cmask[l]].sum()) < BIG_MEV]
    stats["n_considered"] = len(small)

    # ---------- v0.8.1: light-nominated candidates ----------
    # 35% of wrong small-cluster energy has its correct family in the event
    # but spatially FAR (beyond 8-nearest) — only the cluster's own light
    # curve can name it.  Compute each cluster's self-removed curve once
    # (same definition as the veto) and nominate every family whose t0 the
    # light finds near-optimal.
    nom_cache = {}
    if light_nominate:
        for c in small:
            t_cur = atime[c]
            prs, brs, wrs = [], [], []
            for tc in range(n_light):
                if (int(c), tc) not in image_maps:
                    continue
                img = np.asarray(image_maps[(int(c), tc)], np.float64)
                ch = np.flatnonzero(img.max(axis=1) > 1.0)
                if not ch.size:
                    continue
                own = _shift(img, t_cur, n_ticks)
                prs.append(img[ch])
                brs.append(np.asarray(base[tc], np.float64)[ch] - own[ch]
                           - actual[tc][ch])
                wrs.append(w_all[tc][ch])
            if prs:
                nom_cache[c] = weighted_scan(np.concatenate(prs),
                                             np.concatenate(brs),
                                             np.concatenate(wrs))

    shortlist = []
    for c in small:
        t_cur = atime[c]
        gc = min(f_ids, key=lambda g: abs(F[g]["t0"] - t_cur))
        if abs(F[gc]["t0"] - t_cur) > 5.0:
            gc = None
        d_cur = float(f_tree[gc].query(shp[c]["mu"])[0]) if gc is not None else 1e9
        near = [f_ids[i] for i in
                np.argsort(np.linalg.norm(f_mu - shp[c]["mu"], axis=1))[: n_cand_fams]]
        d_alt = min(float(f_tree[g].query(shp[c]["mu"])[0])
                    for g in near if g != gc) if any(g != gc for g in near) else 1e9
        light_noms, light_flag = [], False
        curve = nom_cache.get(c)
        if curve is not None:
            rng = float(np.ptp(curve))
            if rng > 0:
                cm = float(curve.min())
                vals = []
                for g in f_ids:
                    if g == gc:
                        continue
                    ig = int(round(min(max(F[g]["t0"], 0), SEARCH)))
                    v = float(curve[ig])
                    if v - cm <= nom_frac * rng:
                        vals.append((v, g))
                vals.sort()
                light_noms = [g for _, g in vals[:nom_max]]
                i_cur = int(round(min(max(t_cur, 0), SEARCH)))
                light_flag = bool(light_noms) and \
                    (float(curve[i_cur]) - cm) > nom_flag_frac * rng
        stats["n_light_nom"] += len(light_noms)
        if light_flag:
            stats["n_light_flagged"] += 1
        # prefilter: current family not clearly closest, OR the light
        # strongly prefers a nominated family's time
        if gc is None or d_cur > 5.0 or d_alt < 0.8 * d_cur or light_flag:
            spatial_set = set(near) | ({gc} if gc is not None else set())
            cands = near + [g for g in light_noms if g not in spatial_set]
            shortlist.append((c, gc, cands, set(light_noms) - spatial_set))
    stats["n_scored"] = len(shortlist)
    if not shortlist:
        return stats

    # ---------- batched spatial scoring ----------
    pend, recs = [], []
    for c, gc, cands, light_only in shortlist:
        fams = [g for g in dict.fromkeys(([gc] if gc is not None else []) + cands)]
        rec = dict(c=c, gc=gc, rows=[], light_only=light_only)
        for g in fams:
            ft = pair_features2(shp[c], F[g])
            pend.append([ft[k] for k in FEATURES2])
            rec["rows"].append((g, len(pend) - 1))
        recs.append(rec)
    probs = clf.predict_proba(np.array(pend, float))[:, 1]

    # ---------- propose + light veto + apply (margin-ordered) ----------
    proposals = []
    for rec in recs:
        ps = {g: float(probs[i]) for g, i in rec["rows"]}
        gc = rec["gc"]
        p_cur = ps.get(gc, 0.0) if gc is not None else 0.0
        alts = [g for g, _ in rec["rows"] if g != gc]
        if not alts:
            continue
        g_best = max(alts, key=lambda g: ps[g])
        if ps[g_best] - p_cur > margin and ps[g_best] > p_min:
            proposals.append((ps[g_best] - p_cur, rec["c"], gc, g_best,
                              g_best in rec["light_only"]))
    proposals.sort(reverse=True)
    stats["n_proposed"] = len(proposals)

    for gain, c, gc, g_new, via_light in proposals:
        t_cur, t_new = atime[c], F[g_new]["t0"]
        if abs(t_new - t_cur) <= 2.0:
            continue
        tpcs = [tc for tc in range(n_light) if (int(c), tc) in image_maps]
        # light veto from the cluster's own curve with itself removed
        if tpcs:
            prs, brs, wrs = [], [], []
            for tc in tpcs:
                img = np.asarray(image_maps[(int(c), tc)], np.float64)
                ch = np.flatnonzero(img.max(axis=1) > 1.0)
                if not ch.size:
                    continue
                own = _shift(img, t_cur, n_ticks)
                prs.append(img[ch])
                brs.append(np.asarray(base[tc], np.float64)[ch] - own[ch] - actual[tc][ch])
                wrs.append(w_all[tc][ch])
            if prs:
                curve = weighted_scan(np.concatenate(prs), np.concatenate(brs),
                                      np.concatenate(wrs))
                i_new = int(round(min(max(t_new, 0), SEARCH)))
                i_cur = int(round(min(max(t_cur, 0), SEARCH)))
                if curve[i_new] - curve[i_cur] > veto_frac * float(np.ptp(curve)):
                    stats["n_vetoed"] += 1
                    continue
        # apply the trade: move light + hits
        for tc in tpcs:
            img = np.asarray(image_maps[(int(c), tc)], np.float64)
            base[tc] += (_shift(img, t_new, n_ticks)
                         - _shift(img, t_cur, n_ticks)).astype(np.float32)
        np.clip(base, 0.0, None, out=base)
        ts[cmask[c] & assigned] = np.float32(t_new)
        stats["n_traded"] += 1
        if via_light:
            stats["n_traded_via_light"] += 1
        e_c = float(e_pos[cmask[c]].sum())
        stats["e_traded"] += e_c
        rec = {"label": int(c), "from_t": float(t_cur), "to_t": float(t_new),
               "e": e_c, "gain": float(gain), "via_light": bool(via_light)}
        if truth_fn is not None:
            tr = truth_fn(int(c))
            rec["truth"] = tr
            rec["was_ok"] = (tr is not None and abs(t_cur - tr) <= CUT)
            rec["now_ok"] = (tr is not None and abs(t_new - tr) <= CUT)
        stats["audit"].append(rec)

    ns["baseImage"] = base
    ns["hit_timestamps"] = ts.astype(np.float32)
    return stats
