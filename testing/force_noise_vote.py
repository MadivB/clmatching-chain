"""force_simple with a cluster-density vote for noise hits (drop-in candidate for testing/force_simple.py).

Clusters and isolation exactly as force_simple.  Noise hits (label < 0, not isolated):
  - d1 = distance to the nearest assigned hit.  If d1 < contact_cm: take that hit's time (unchanged rule).
  - else: every assigned cluster within r_max votes for its time (time of its member nearest to the hit)
    with weight  w = exp(-(d_c - d1) / lam)  x (big_w if cluster E >= big_mev else 1),
    d_c = min distance hit->cluster.  Votes are summed over clusters whose times lie within +-bin_ticks
    of each candidate; the hit takes the time of the strongest cluster in the winning bin.
No truth, no light.  Cluster geometry: exact hits within r_exact, 2 cm voxel representatives beyond.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def _noise_vote(xyz_n, xyz_a, lab_a, ts_a, E_a, *, lam, big_w, big_mev, bin_ticks, r_max, r_exact,
                contact_cm, vox_cm=2.0):
    n = xyz_n.shape[0]
    ta = cKDTree(xyz_a)
    d1, j1 = ta.query(xyz_n, k=1)
    out = ts_a[j1].copy()
    far = np.where(d1 >= contact_cm)[0]
    if far.size == 0:
        return out
    tn = cKDTree(xyz_n[far])
    M = tn.sparse_distance_matrix(ta, r_exact, output_type="coo_matrix")
    i, j, d = M.row, M.col, M.data
    i = np.r_[i, np.arange(far.size)]; j = np.r_[j, j1[far]]; d = np.r_[d, d1[far]]
    if r_max > r_exact:
        vox = np.floor(xyz_a / vox_cm).astype(np.int64)
        _, rep = np.unique(np.c_[lab_a, vox], axis=0, return_index=True)
        M2 = tn.sparse_distance_matrix(cKDTree(xyz_a[rep]), r_max, output_type="coo_matrix")
        k = M2.data > r_exact
        i = np.r_[i, M2.row[k]]; j = np.r_[j, rep[M2.col[k]]]; d = np.r_[d, M2.data[k]]
    # min distance per (hit, cluster)
    key = i.astype(np.int64) * (int(lab_a.max()) + 2) + lab_a[j]
    o = np.lexsort((d, key)); key, i, j, d = key[o], i[o], j[o], d[o]
    f = np.r_[True, key[1:] != key[:-1]]
    i, j, d = i[f], j[f], d[f]
    t = ts_a[j]
    w = np.exp(-(d - d1[far][i]) / lam) * np.where(E_a[j] >= big_mev, big_w, 1.0)
    kk = i * 10000.0 + np.round(t)
    o = np.argsort(kk, kind="stable"); kk, i, w, t = kk[o], i[o], w[o], t[o]
    cw = np.r_[0.0, np.cumsum(w)]
    S = cw[np.searchsorted(kk, kk + bin_ticks, "right")] - cw[np.searchsorted(kk, kk - bin_ticks, "left")]
    o = np.lexsort((-w, -S, i)); i2 = i[o]
    first = np.r_[True, i2[1:] != i2[:-1]]
    out[far[i2[first]]] = t[o][first]
    return out


def force_noise_vote_coverage(ns: dict, *, isolate_cm: float = 25.0, energy=None, lam: float = 14.0,
                              big_w: float = 0.5, big_mev: float = 20.0, bin_ticks: float = 6.0,
                              r_max: float = 100.0, r_exact: float = 15.0, contact_cm: float = 3.0,
                              vox_cm: float = 2.0) -> dict:
    ts = np.asarray(ns["hit_timestamps"], dtype=np.float64).copy()
    xyz = np.stack([np.asarray(ns["xset"], np.float64), np.asarray(ns["yset"], np.float64),
                    np.asarray(ns["zset"], np.float64)], axis=1)
    labels = np.asarray(ns["labels_global"]).astype(np.int64)
    e = np.asarray(energy, np.float64)
    n = ts.size
    assigned = np.isfinite(ts) & (ts >= 0)
    isolated = np.zeros(n, bool)
    if isolate_cm > 0 and n > 1:
        d2, _ = cKDTree(xyz).query(xyz, k=2)
        isolated = d2[:, 1] > isolate_cm
    ns["isolated_mask"] = isolated
    todo = ~assigned & ~isolated
    stats = {"n_isolated": int(isolated.sum()), "n_todo": int(todo.sum()), "n_cluster": 0, "n_noise": 0}
    if not assigned.any() or not todo.any():
        ts[isolated] = np.nan
        ns["hit_timestamps"] = ts
        return stats
    tree_a = cKDTree(xyz[assigned]); ts_a = ts[assigned]
    for lab in np.unique(labels[todo & (labels >= 0)]):
        m = todo & (labels == lab)
        d, j = tree_a.query(xyz[m], k=1)
        ts[m] = ts_a[j[np.argmin(d)]]
        stats["n_cluster"] += 1
    mnoise = todo & (labels < 0)
    if mnoise.any():
        ul, inv = np.unique(labels, return_inverse=True)
        cE = np.bincount(inv, weights=e)[inv]
        A = np.where(assigned)[0]
        ts[mnoise] = _noise_vote(xyz[mnoise], xyz[A], labels[A], ts_a, cE[A], lam=lam, big_w=big_w,
                                 big_mev=big_mev, bin_ticks=bin_ticks, r_max=r_max, r_exact=r_exact,
                                 contact_cm=contact_cm, vox_cm=vox_cm)
        stats["n_noise"] = int(mnoise.sum())
    ts[isolated] = np.nan
    ns["hit_timestamps"] = ts
    return stats
