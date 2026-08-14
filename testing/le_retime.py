"""Stage 5 — LE (low-energy blip) re-timing on the placed backbone.

Runs AFTER force coverage.  Fast by construction: KD-tree geometry against
the already-placed families + one linear ranker — no light scans, no
rendering.  Re-times every LE blob (3-cm component of noise hits +
sub-1.5 MeV no-template clusters) by ranked candidate families
(le_ranker.json, trained on truth from the cached dumps).

Motivation (measured): the borrowed LE class sits at 56 %; for wrong LE
hits the correct-time family exists 93 % of the time but is the nearest
one only 2 % — proximity borrowing is the wrong selector.  Ranking over
k=8 candidates within 60 cm reaches the rank-3 ceiling (~57 % of wrong).
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable

import numpy as np
from scipy.spatial import cKDTree

from le_pre_spine import (BLOB_LINK, CAND_K, CAND_R, CUT, LE_EMAX,
                          blob_rows, build_blobs, score)

_HERE = os.path.dirname(os.path.abspath(__file__))
_MD = {}


def run_le_retime(ns: dict[str, Any], *, model_path: str | None = None,
                  min_margin: float = 0.0,
                  truth_fn_hit=None, verbose: bool = False) -> dict[str, Any]:
    if model_path is None:
        hgb = os.path.join(_HERE, "le_ranker_hgb.joblib")
        model_path = hgb if os.path.exists(hgb) else os.path.join(_HERE, "le_ranker.json")
    if model_path not in _MD:
        if model_path.endswith(".joblib"):
            import joblib
            _MD[model_path] = {"hgb": joblib.load(model_path)["model"]}
        else:
            _MD[model_path] = json.load(open(model_path))
    md = _MD[model_path]

    labels = np.asarray(ns["labels_global"])
    ts = np.asarray(ns["hit_timestamps"], dtype=np.float64).copy()
    e = np.clip(np.asarray(ns["Eset"], dtype=np.float64), 0.0, None)
    ev = dict(xyz=np.stack([np.asarray(ns["xset"], np.float64),
                            np.asarray(ns["yset"], np.float64),
                            np.asarray(ns["zset"], np.float64)], 1),
              e=e, tpc=np.asarray(ns["hitTPCid"], dtype=np.int64),
              ts_final=ts, labels=labels)

    # LE mask: noise labels + clusters without a light template
    ce = {}
    for l in np.unique(labels[labels >= 0]):
        ce[l] = e[labels == l].sum()
    small = np.isin(labels, [l for l, en in ce.items() if en < LE_EMAX])
    le = (labels < 0) | small

    # families from the placed non-LE backbone (nothing re-scanned)
    donor = np.isfinite(ts) & (ts >= 0) & ~le
    di = np.flatnonzero(donor)
    stats = {"n_blobs": 0, "n_changed": 0, "e_le": 0.0, "e_changed": 0.0,
             "n_fams": 0}
    if di.size < 10:
        return stats
    order = di[np.argsort(ts[di])]
    fams, cur = [], [order[0]]
    for i in order[1:]:
        if ts[i] - ts[cur[-1]] > 3.0:
            fams.append(cur)
            cur = []
        cur.append(i)
    fams.append(cur)
    F = []
    for mem in fams:
        mem = np.asarray(mem)
        if mem.size < 3:
            continue
        w = np.maximum(e[mem], 1e-6)
        F.append(dict(t0=float(np.median(ts[mem])), tree=cKDTree(ev["xyz"][mem]),
                      cent=np.average(ev["xyz"][mem], 0, weights=w),
                      e=float(e[mem].sum()), n=int(mem.size),
                      tpcs=set(int(t) for t in np.unique(ev["tpc"][mem]))))
    stats["n_fams"] = len(F)
    if not F:
        return stats

    idx = np.flatnonzero(le)
    audit_was = audit_now = 0
    e_tr = 0.0
    # batched scoring: gather every blob's candidate rows, ONE model call
    blobs = build_blobs(ev["xyz"], idx)
    pend, meta = [], []
    for blob in blobs:
        t_cur = float(np.median(ts[blob][np.isfinite(ts[blob])])) \
            if np.isfinite(ts[blob]).any() else None
        rows, t0s, inc_i = blob_rows(ev, F, blob, t_cur=t_cur)
        meta.append((blob, t_cur, t0s, inc_i, len(pend), len(pend) + len(rows)))
        pend.extend(rows)
    sc_all = score(md, np.asarray(pend, float)) if pend else np.zeros(0)
    for blob, t_cur, t0s, inc_i, a, b in meta:
        stats["n_blobs"] += 1
        eb = float(e[blob].sum())
        stats["e_le"] += eb
        sc = sc_all[a:b]
        k = int(np.argmax(sc))
        # incumbent gating: only leave the current family if the challenger
        # beats it by min_margin; NO identified incumbent => keep current
        # (never force a change — the v2-emulation lesson)
        if inc_i < 0 or sc[k] - sc[inc_i] < min_margin:
            continue
        t_new = float(t0s[k])
        if t_cur is None:
            t_cur = -1.0
        if truth_fn_hit is not None:
            tr = truth_fn_hit(blob)
            if tr is not None:
                m = np.isfinite(tr) & (tr >= 0)
                if m.any():
                    e_tr += float(e[blob][m].sum())
                    audit_was += float(e[blob][m][np.abs(ts[blob][m] - tr[m]) <= CUT].sum())
                    audit_now += float(e[blob][m][np.abs(t_new - tr[m]) <= CUT].sum())
        if abs(t_new - t_cur) > 2.0:
            stats["n_changed"] += 1
            stats["e_changed"] += eb
        ts[blob] = t_new
    if e_tr > 0:
        stats["le_acc_before"] = round(100 * audit_was / e_tr, 2)
        stats["le_acc_after"] = round(100 * audit_now / e_tr, 2)
    ns["hit_timestamps"] = ts.astype(np.float32)
    return stats
