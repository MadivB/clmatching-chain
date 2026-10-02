"""Stage 4.9 variant: cluster-density vote x LIGHT for isolated unclustered hits (user 2026-09-30).

Unassigned noise hits (label < 0, not isolated):
  d1 < contact_cm                 : time of the nearest assigned hit (unchanged);
  d1 >= contact_cm                : candidates = family times of the hit's TPC (rounded final times of assigned
                                    clustered hits); every assigned cluster within r_max votes for the candidate
                                    within +-bin_ticks of its time with w = exp(-(d_c - d1)/lam) x (big_w if E >= big_mev);
  d1 > light_cm (default 10 cm)   : the vote of each candidate is multiplied by (1 + res_f / median_res), where res_f is
                                    the positive light RESIDUAL (actual - everything rendered) at the candidate's time,
                                    averaged over +-2 ticks around the template peak, on the K=6 non-vetoed channels
                                    nearest to the hit, weighted 1/d^2 (floor 5 cm) - the un-rendered light of the
                                    parent's own low-energy deposits near the hit.  Spatial information alone is weak
                                    out there (nearest-hit 40-49 % correct); the vote x light rule measured +6.5-7 pp.
Unassigned clusters and the isolation veto exactly as force_simple.  No truth, no ML; ~0.1 s per spill.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import cKDTree

PEAK = 105; HW = 2; K_CH = 6; D_FLOOR = 5.0


def light_residual_scores(R, veto_t, sipm_t, xyz_h, cands):
    """(n_h, F) positive residual at each candidate time on the K nearest non-vetoed channels, 1/d^2-weighted."""
    chan_ok = np.flatnonzero(~veto_t)
    if chan_ok.size == 0 or cands.size == 0 or xyz_h.shape[0] == 0:
        return np.zeros((xyz_h.shape[0], cands.size))
    idx = np.clip(np.round(cands).astype(int) + PEAK, HW, R.shape[1] - HW - 1)
    rw = np.stack([R[:, i - HW:i + HW + 1].mean(1) for i in idx], 1)          # (120, F)
    rw = np.maximum(rw, 0.0)
    d = np.linalg.norm(xyz_h[:, None, :] - sipm_t[chan_ok][None, :, :], axis=2)
    k = min(K_CH, chan_ok.size)
    near = np.argpartition(d, k - 1, axis=1)[:, :k] if k < chan_ok.size else np.tile(np.arange(chan_ok.size), (xyz_h.shape[0], 1))
    w = 1.0 / np.maximum(np.take_along_axis(d, near, 1), D_FLOOR) ** 2
    w /= w.sum(1, keepdims=True)
    return np.einsum("hk,hkf->hf", w, rw[chan_ok[near]])


def vote_pairs(xh, d1, xyz_a, lab_a, ts_a, E_a, *, lam, big_w, big_mev, r_max, r_exact, vox_cm=3.0):
    """ONE pass for all far hits: (hit index, cluster time, vote weight) over clusters within r_max
    (exact member distances within r_exact, 2-cm voxel representatives beyond)."""
    if xh.shape[0] == 0 or xyz_a.shape[0] == 0:
        return np.zeros(0, int), np.zeros(0), np.zeros(0)
    tn = cKDTree(xh); ta = cKDTree(xyz_a)
    M = tn.sparse_distance_matrix(ta, r_exact, output_type="coo_matrix")
    i, j, d = M.row, M.col, M.data
    if r_max > r_exact:
        vox = np.floor(xyz_a / vox_cm).astype(np.int64)
        _, rep = np.unique(lab_a * 1000003 + (vox[:, 0] * 73856093 ^ vox[:, 1] * 19349663 ^ vox[:, 2] * 83492791), return_index=True)
        M2 = tn.sparse_distance_matrix(cKDTree(xyz_a[rep]), r_max, output_type="coo_matrix")
        k = M2.data > r_exact
        i = np.r_[i, M2.row[k]]; j = np.r_[j, rep[M2.col[k]]]; d = np.r_[d, M2.data[k]]
    if i.size == 0:
        return np.zeros(0, int), np.zeros(0), np.zeros(0)
    key = i.astype(np.int64) * (int(lab_a.max()) + 2) + lab_a[j]
    o = np.lexsort((d, key)); key, i, j, d = key[o], i[o], j[o], d[o]
    f = np.r_[True, key[1:] != key[:-1]]
    i, j, d = i[f], j[f], d[f]
    w = np.exp(-(d - d1[i]) / lam) * np.where(E_a[j] >= big_mev, big_w, 1.0)
    return i, ts_a[j], w


def vote_matrix_from_pairs(pi, pt, pw, n_h, cands, bin_ticks):
    """(n_h, F): add each pair's weight to the candidates within +-bin_ticks of its time."""
    V = np.zeros((n_h, cands.size))
    if pi.size == 0 or cands.size == 0:
        return V
    lo = np.searchsorted(cands, pt - bin_ticks, "left"); hi = np.searchsorted(cands, pt + bin_ticks, "right")
    for s in range(int((hi - lo).max())):
        m = lo + s < hi
        np.add.at(V, (pi[m], lo[m] + s), pw[m])
    return V


def force_light_vote_coverage(ns: dict[str, Any], *, sipm: dict, isolate_cm: float = 25.0, energy=None,
                              light_cm: float = 10.0, contact_cm: float = 3.0, lam: float = 14.0, big_w: float = 0.5,
                              big_mev: float = 20.0, bin_ticks: float = 6.0, r_max: float = 60.0, r_exact: float = 15.0,
                              actual=None, base=None, veto=None) -> dict[str, Any]:
    import time
    t0 = time.time()
    ts = np.asarray(ns["hit_timestamps"], dtype=np.float64).copy()
    xyz = np.stack([np.asarray(ns["xset"], np.float64), np.asarray(ns["yset"], np.float64), np.asarray(ns["zset"], np.float64)], axis=1)
    labels = np.asarray(ns["labels_global"]).astype(np.int64); tpc = np.asarray(ns["hitTPCid"]).astype(np.int64)
    E = np.clip(np.asarray(energy if energy is not None else ns["Eset"], np.float64), 0, None)
    n = ts.size
    assigned = np.isfinite(ts) & (ts >= 0)
    isolated = np.zeros(n, bool)
    if isolate_cm > 0 and n > 1:
        d2, _ = cKDTree(xyz).query(xyz, k=2); isolated = d2[:, 1] > isolate_cm
    ns["isolated_mask"] = isolated
    todo = ~assigned & ~isolated
    st = {"n_isolated": int(isolated.sum()), "n_todo": int(todo.sum()), "n_cluster": 0, "n_noise": 0, "n_light": 0, "n_vote": 0}
    if not assigned.any() or not todo.any():
        ts[isolated] = np.nan; ns["hit_timestamps"] = ts; st["time_s"] = time.time() - t0; return st
    xyz_a = xyz[assigned]; ts_a = ts[assigned]; lab_a = labels[assigned]; tpc_a = tpc[assigned]
    tree_a = cKDTree(xyz_a)
    # unassigned clusters -> one coherent time (as force_simple)
    for lab in np.unique(labels[todo & (labels >= 0)]):
        m = todo & (labels == lab); d, j = tree_a.query(xyz[m], k=1); ts[m] = ts_a[j[np.argmin(d)]]; st["n_cluster"] += 1
    mnoise = np.flatnonzero(todo & (labels < 0))
    if mnoise.size:
        d1, j1 = tree_a.query(xyz[mnoise], k=1)
        out = ts_a[j1].copy()
        labE = np.bincount(labels[labels >= 0], weights=E[labels >= 0], minlength=int(labels.max()) + 1) if (labels >= 0).any() else np.zeros(1)
        E_a = labE[lab_a]
        far = np.flatnonzero(d1 >= contact_cm)
        pi, pt, pw = vote_pairs(xyz[mnoise[far]], d1[far], xyz_a, lab_a, ts_a, E_a, lam=lam, big_w=big_w, big_mev=big_mev, r_max=r_max, r_exact=r_exact)
        tpc_far = tpc[mnoise[far]]
        for t_ in np.unique(tpc_far):
            loc = np.flatnonzero(tpc_far == t_); sel = far[loc]
            cands = np.unique(np.round(ts_a[tpc_a == t_]))
            if cands.size == 0:
                continue
            pm = np.isin(pi, loc)
            remap = np.full(far.size, -1); remap[loc] = np.arange(loc.size)
            V = vote_matrix_from_pairs(remap[pi[pm]], pt[pm], pw[pm], loc.size, cands, bin_ticks)
            use_light = (d1[sel] > light_cm) & (actual is not None) & (int(t_) in sipm) & (int(t_) < (actual.shape[0] if actual is not None else 0))
            if use_light.any():
                R = np.asarray(actual[int(t_)], np.float32) - np.asarray(base[int(t_)], np.float32)
                res = light_residual_scores(R, np.asarray(veto[int(t_)], bool), sipm[int(t_)], xyz[mnoise[sel[use_light]]], cands)
                pos = res[res > 0]; med = np.median(pos) if pos.size else 1.0
                V[use_light] *= (1.0 + res / med)
                st["n_light"] += int(use_light.sum())
            has = V.max(1) > 0
            pick = cands[np.argmax(V, 1)]
            out[sel[has]] = pick[has]; st["n_vote"] += int(has.sum())
        ts[mnoise] = out; st["n_noise"] = int(mnoise.size)
    ts[isolated] = np.nan
    ns["hit_timestamps"] = ts
    st["time_s"] = round(time.time() - t0, 3)
    return st
