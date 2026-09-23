"""v1.1 stage 3.98: final-state COMPONENT re-decision (--redecide).

After every assignment stage (pool, corrections, joint tail, pass2 trading)
the residual is as clean as the chain can make it.  Light-coupled components
of PLACED imaged clusters (>= e_min, non-shower TPCs) are then re-decided
JOINTLY on the leave-component-out base: each member's candidate times are
the local minima of its own curve on that base plus its current time plus
the greedy path, all combinations are scored on the exact pairwise-quadratic
loss (v110_components.search_chunk) and the winner replaces the incumbent
only if its exact clipped union loss is strictly lower.

Why joint and why components only: an ISOLATED single-cluster move to the
argmin of its own final-state curve was measured harmful three times (the
isolated curve locks onto other interactions' residual template systematics).
Here singletons are never touched; only clusters that have a look-alike in the
same TPC (the population that holds 84 % of the >= 5 MeV wrong energy) are
re-decided, and only together with that look-alike, so a member that moves
must beat the incumbent WITH its twins' light accounted for.  With amplitude
profiling on (v1.1 pool), the under-prediction deficits that used to attract
twins are absorbed by their owners first.
"""
from __future__ import annotations

import time

import numpy as np

from v110_components import build_components, search_chunk, add_label, PlacedContacts

CUT = 10.0
TPC_E_MAX = 1400.0     # the chain's shower-TPC boundary (v0.7 hybrid split)


def run_redecide(ns, *, veto_mask, support_cache, label_info, ti, pick=None,
                 e_min=5.0, cos_min=0.80, k=6, n_max=8, n_exact=8,
                 truth_fn=None, gain_band=0.0, spatial_frac=0.0):
    t_start = time.time()
    image_maps = ns["imageMaps"]
    base32 = np.asarray(ns["baseImage"], dtype=np.float32)
    actual = np.asarray(ns["fullLightWaveform"], dtype=np.float64)
    n_light, n_ch, n_ticks = actual.shape
    labels = np.asarray(ns["labels_global"])
    ts = np.asarray(ns["hit_timestamps"], dtype=np.float64).copy()
    e_pos = np.clip(np.asarray(ns["Eset"], dtype=np.float64), 0.0, None)
    hit_tpc = np.asarray(ns["hitTPCid"], dtype=np.int64)
    w_full = 1.0 / (200.0 * np.maximum(actual, 100.0))
    xyz, pc, placed_t = None, None, {}
    if spatial_frac > 0:
        xyz = np.stack([np.asarray(ns["xset"], np.float64), np.asarray(ns["yset"], np.float64),
                        np.asarray(ns["zset"], np.float64)], axis=1)
        pc = PlacedContacts(xyz, labels)
        for lab in np.unique(labels[labels >= 0]):
            v = ts[labels == lab]; v = v[np.isfinite(v) & (v >= 0)]
            if v.size:
                placed_t[int(lab)] = float(np.median(v))
    baseA = base32.astype(np.float64)
    assigned = np.isfinite(ts) & (ts >= 0)
    stats = {"n_participants": 0, "n_components": 0, "n_multi": 0, "n_chunks": 0,
             "n_moved": 0, "e_moved": 0.0, "n_fixed": 0, "n_broke": 0,
             "e_fixed": 0.0, "e_broke": 0.0, "n_scans": 0, "n_leaves": 0,
             "loss_gain_sum": 0.0, "audit": []}

    tpc_e = np.zeros(int(hit_tpc.max()) + 2)
    np.add.at(tpc_e, hit_tpc, e_pos)
    info = {}
    for lab in np.unique(labels[labels >= 0]):
        m = labels == lab
        if not np.all(assigned[m]):
            continue
        e_c = float(e_pos[m].sum())
        if e_c < e_min:
            continue
        if np.any(tpc_e[np.unique(hit_tpc[m])] > TPC_E_MAX):
            continue
        tpcs = [c for c in range(n_light) if (int(lab), int(c)) in image_maps]
        if not tpcs:
            continue
        t0v = ts[m]
        t0 = float(np.median(t0v))
        if np.ptp(t0v) > 0.5:          # split cluster (parts at different times): leave alone
            continue
        lab_type = str(label_info.get(int(lab), {}).get("type", "track")).lower()
        sup = {}
        for c in tpcs:
            entry = support_cache.get((int(lab), int(c))) if lab_type == "track" else None
            if entry is not None:
                ch = np.asarray(entry["channel_indices"], dtype=int)
            else:
                ch = np.flatnonzero(np.asarray(image_maps[(int(lab), int(c))]).max(axis=1) > 1.0)
            ch = ch[~veto_mask[int(c), ch]]
            if ch.size:
                sup[int(c)] = np.sort(ch)
        if not sup:
            continue
        info[int(lab)] = {"t0": t0, "tpcs": tpcs, "sup": sup, "type": lab_type,
                          "energy": e_c, "hits": np.flatnonzero(m)}
    stats["n_participants"] = len(info)
    if len(info) < 2:
        stats["time_s"] = time.time() - t_start
        return stats

    comps, n_edges = build_components(list(info), info, image_maps, veto_mask, cos_min)
    stats["n_components"] = len(comps)
    stats["n_edges"] = int(n_edges)
    pool_stats = {"n_scans": 0, "n_leaves": 0, "n_exact_checks": 0, "perm_total": 0,
                  "perm_better": 0, "perm_rel_gain_sum": 0.0, "t_search": 0.0}
    for comp in comps:
        if len(comp) < 2:
            continue
        stats["n_multi"] += 1
        remaining = list(comp)
        while remaining:
            B = remaining[:n_max]
            remaining = remaining[n_max:]
            if len(B) < 2:
                break
            stats["n_chunks"] += 1
            # leave-chunk-out base
            for lab in B:
                add_label(baseA, lab, info[lab]["t0"], info, image_maps, ti, sign=-1.0)
            extra = {lab: [int(round(info[lab]["t0"]))] for lab in B}
            ts_new, diag = search_chunk(B, baseA, info, image_maps, actual, w_full, ti,
                                        pick, k=k, n_exact=n_exact, stats=pool_stats,
                                        extra_cands=extra, incumbent=extra,
                                        spatial_frac=spatial_frac,
                                        hits_of=(lambda l: info[l]["hits"]), xyz=xyz,
                                        placed_contacts=pc, placed_t=placed_t)
            stats["n_contacts"] = stats.get("n_contacts", 0) + int(diag.get("n_contacts", 0))
            better = diag.get("loss_incumbent") is not None and \
                diag["loss_best"] < diag["loss_incumbent"] - 1e-9
            chosen = ts_new if better else {lab: info[lab]["t0"] for lab in B}
            for lab in B:
                add_label(baseA, lab, chosen[lab], info, image_maps, ti)
            if better:
                stats["loss_gain_sum"] += float(diag["loss_incumbent"] - diag["loss_best"])
                for lab in B:
                    if abs(chosen[lab] - info[lab]["t0"]) > 0.5:
                        old = info[lab]["t0"]
                        info[lab]["t0"] = float(chosen[lab])
                        placed_t[int(lab)] = float(chosen[lab])
                        ts[info[lab]["hits"]] = np.float32(chosen[lab])
                        stats["n_moved"] += 1
                        stats["e_moved"] += info[lab]["energy"]
                        rec = {"label": int(lab), "from_t": float(old), "to_t": float(chosen[lab]),
                               "e": float(info[lab]["energy"]), "comp": len(B),
                               "loss_inc": float(diag["loss_incumbent"]), "loss_best": float(diag["loss_best"]),
                               "rel_gain": float((diag["loss_incumbent"] - diag["loss_best"])
                                                 / max(abs(diag["loss_incumbent"]), 1e-12)),
                               "n_contacts": int(diag.get("n_contacts", 0)),
                               "n_unary_contact": int(diag.get("n_unary_contact", 0))}
                        if truth_fn is not None:
                            tr = truth_fn(int(lab))
                            if tr is not None and np.isfinite(tr):
                                was = abs(old - tr) <= CUT
                                now = abs(chosen[lab] - tr) <= CUT
                                rec.update({"truth": float(tr), "was_ok": bool(was), "now_ok": bool(now)})
                                if not was and now:
                                    stats["n_fixed"] += 1
                                    stats["e_fixed"] += info[lab]["energy"]
                                elif was and not now:
                                    stats["n_broke"] += 1
                                    stats["e_broke"] += info[lab]["energy"]
                        stats["audit"].append(rec)
    stats["n_scans"] = pool_stats["n_scans"]
    stats["n_leaves"] = pool_stats["n_leaves"]
    stats["n_better"] = pool_stats["perm_better"]
    ns["baseImage"] = baseA.astype(np.float32)
    ns["hit_timestamps"] = ts.astype(np.float32)
    stats["time_s"] = time.time() - t_start
    return stats
