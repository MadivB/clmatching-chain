"""Simple force coverage (user 2026-09-04: stop 'messing around' with the
low-energy hits).

Replaces the geodesic force_full_coverage + ranked le_retime pair with ONE
KD-tree pass, plus the isolation veto:

  - hits whose nearest OTHER hit is > isolate_cm away get NO t0 (NaN) - at
    any stage; they are physically untimeable singles.
  - remaining unassigned CLUSTERS (label >= 0): one coherent t0 = t0 of the
    nearest assigned hit to any member (Euclidean).
  - remaining unassigned noise hits: t0 of the Euclidean-nearest assigned hit.

No graphs, no geodesics, no rankers, no ML. O(N log N).
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import cKDTree


def force_simple_coverage(ns: dict[str, Any], *, isolate_cm: float = 25.0,
                          verbose: bool = False) -> dict[str, Any]:
    ts = np.asarray(ns["hit_timestamps"], dtype=np.float64).copy()
    xyz = np.stack([np.asarray(ns["xset"], np.float64),
                    np.asarray(ns["yset"], np.float64),
                    np.asarray(ns["zset"], np.float64)], axis=1)
    labels = np.asarray(ns["labels_global"])
    n = ts.size
    assigned = np.isfinite(ts) & (ts >= 0)

    # isolation veto: nearest-neighbour distance over ALL hits
    isolated = np.zeros(n, bool)
    if isolate_cm > 0 and n > 1:
        tree_all = cKDTree(xyz)
        d2, _ = tree_all.query(xyz, k=2)
        isolated = d2[:, 1] > isolate_cm
    ns["isolated_mask"] = isolated

    todo = ~assigned & ~isolated
    stats = {"n_isolated": int(isolated.sum()),
             "n_todo": int(todo.sum()), "n_cluster": 0, "n_noise": 0}
    if not assigned.any() or not todo.any():
        ts[isolated] = np.nan
        ns["hit_timestamps"] = ts
        return stats

    tree_a = cKDTree(xyz[assigned])
    ts_a = ts[assigned]

    # unassigned clusters -> one coherent t0 (nearest assigned hit to any member)
    for lab in np.unique(labels[todo & (labels >= 0)]):
        m = todo & (labels == lab)
        d, j = tree_a.query(xyz[m], k=1)
        ts[m] = ts_a[j[np.argmin(d)]]
        stats["n_cluster"] += 1
    # remaining noise hits -> per-hit nearest assigned
    mnoise = todo & (labels < 0)
    if mnoise.any():
        _, j = tree_a.query(xyz[mnoise], k=1)
        ts[mnoise] = ts_a[j]
        stats["n_noise"] = int(mnoise.sum())

    ts[isolated] = np.nan          # veto wins over any earlier assignment
    ns["hit_timestamps"] = ts
    return stats
