"""v0.7-b joint small-cluster assignment (annealed mean-field on the exact
pairwise light energy + geodesic-connectivity prior).  See V070_DESIGN.md.

Replaces expansion+matrix for the IMAGED unassigned tail (--phase3 joint);
backbone/placed clusters are FROZEN in v0.7-b (they define the fixed residual
and the spatial anchors).  No-template clusters and noise keep the force path.

Energy (exact up to the saturation clip, Poisson weights fixed by the data):

    E(T) = const + sum_i a_i(t_i) + sum_{i<j} C_ij(t_i, t_j) - priors

    a_i(t)     : from ONE FFT scan per cluster (curve minus its constant)
    C_ij(t,t') : 2 sum w * s_i(t) * s_j(t') over shared support channels
    prior      : +LAMBDA_NATS on slots geodesically CONNECTED to already-
                 placed hits at that time (calibrated 2026-08-06: P(same
                 interaction | connected within 30 cm) ~ 0.99, distance-flat;
                 ~100% of tail pairs are disconnected -> binary certificate)

Mean-field with damping + geometric annealing; harden by argmax; light added
at the winning slots.  All decisions live in chi2 units; the prior enters as
a temperature-free log-prior (nats * T would anneal it away).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from v030_expansion import (CLIP, GAIN, SEARCH, _shift, build_hit_graph,
                            family_geodesics, weighted_scan)

CUT = 10.0

# ---- tunables ----
K_SLOTS = 20            # max candidate slots per cluster
OWN_MINIMA = 3          # own scan minima added as slots
SLOT_DEDUP = 2.0        # ticks: merge closer slot candidates
D_GEO_MAX = 30.0        # connectivity reach through the 3-cm hit graph [cm]
R_LINK = 3.0
LAMBDA_NATS = 4.6       # log-odds of the certificate (kept for reference)
PRIOR_FRAC = 1.0        # connectivity prior in ENERGY units: PRIOR_FRAC * median
                        # unary spread — dominates when light is degenerate,
                        # yields when light is decisive, at EVERY temperature
ANCHOR_DT = 2.0         # placed hits within ~+-3 ticks of a slot count as anchors
T_SWEEPS = 25
T_DECAY = 0.72
DAMP = 0.5
C_MAX_DT = 250.0        # slot pairs farther than this never couple (template len)
PAIR_CAP = 20000        # safety cap on coupled (i,j) pairs per event
TPC_E_MAX = None        # if set: clusters touching TPCs with more charge
                        # energy than this [MeV] are LEFT for the matrix
                        # (shower TPCs; the tuned matrix wins there)


def _own_minima(curve, n=OWN_MINIMA, sep=10):
    order = np.argsort(curve)
    out = []
    for t in order:
        if all(abs(int(t) - o) > sep for o in out):
            out.append(int(t))
            if len(out) >= n:
                break
    return out


def run_joint_assignment(ns: dict[str, Any], *, verbose: bool = False,
                         params: dict[str, float] | None = None) -> dict[str, Any]:
    p = {k: v for k, v in globals().items() if k.isupper() and not k.startswith("_")}
    if params:
        p.update(params)

    image_maps = ns["imageMaps"]
    base = np.asarray(ns["baseImage"], dtype=np.float32)
    actual = np.asarray(ns["fullLightWaveform"], dtype=np.float64)
    n_light, n_ch, n_ticks = actual.shape
    labels = np.asarray(ns["labels_global"])
    ts = np.asarray(ns["hit_timestamps"], dtype=np.float64).copy()
    xyz = np.stack([np.asarray(ns["xset"], np.float64), np.asarray(ns["yset"], np.float64),
                    np.asarray(ns["zset"], np.float64)], axis=1)
    e_pos = np.clip(np.asarray(ns["Eset"], dtype=np.float64), 0.0, None)
    hit_tpc = np.asarray(ns["hitTPCid"], dtype=np.int64)
    w_all = 1.0 / (GAIN * np.maximum(actual, 100.0))
    assigned = np.isfinite(ts) & (ts >= 0)

    stats = {"n_cands": 0, "n_assigned": 0, "e_assigned": 0.0, "n_slots": 0,
             "n_pairs": 0, "n_connected_slots": 0, "sweeps": 0, "assigned": []}

    # ---------- participants: unassigned clusters with templates ----------
    tpc_e = np.zeros(int(hit_tpc.max()) + 2, dtype=np.float64)
    np.add.at(tpc_e, hit_tpc, e_pos)
    cands = {}
    for lab in np.unique(labels[labels >= 0]):
        m = labels == lab
        if np.any(assigned[m]):
            continue
        if p.get("TPC_E_MAX") is not None and \
                np.any(tpc_e[np.unique(hit_tpc[m])] > float(p["TPC_E_MAX"])):
            continue          # shower-TPC cluster: leave for the matrix
        tpcs = [c for c in range(n_light) if (int(lab), int(c)) in image_maps]
        if not tpcs:
            continue
        e_c = float(e_pos[m].sum())
        if e_c <= 0:
            continue
        sup = {}
        for c in tpcs:
            img = np.asarray(image_maps[(int(lab), int(c))])
            ch = np.flatnonzero(img.max(axis=1) > 1.0)
            if ch.size:
                sup[int(c)] = ch
        if not sup:
            continue
        cands[int(lab)] = {"hits": np.flatnonzero(m), "e": e_c,
                           "tpcs": sorted(sup), "tpcs_all": tpcs, "sup": sup}
    stats["n_cands"] = len(cands)
    if not cands:
        return stats

    # ---------- unary light term from ONE FFT scan per cluster ----------
    slot_pool = {c: sorted(float(t) for t in ns["t0Candidates"][c]
                           if 0 <= float(t) <= SEARCH) for c in range(n_light)}
    slots, a_un = {}, {}
    for lab, cd in cands.items():
        prs = [np.asarray(image_maps[(lab, c)], np.float64)[ch]
               for c, ch in cd["sup"].items()]
        brs = [np.asarray(base[c], np.float64)[ch] - actual[c][ch]
               for c, ch in cd["sup"].items()]
        wrs = [w_all[c][ch] for c, ch in cd["sup"].items()]
        curve = weighted_scan(np.concatenate(prs), np.concatenate(brs),
                              np.concatenate(wrs))
        cand_t = list(_own_minima(curve, int(p["OWN_MINIMA"])))
        for c in cd["tpcs"]:
            cand_t += [t for t in slot_pool[c]]
        cand_t.sort()
        sl = []
        for t in cand_t:
            if sl and t - sl[-1] <= p["SLOT_DEDUP"]:
                if curve[int(round(t))] < curve[int(round(sl[-1]))]:
                    sl[-1] = float(t)          # keep the better of the merged pair
            else:
                sl.append(float(t))
        if len(sl) > int(p["K_SLOTS"]):
            # keep the best by unary term
            sl = sorted(sl, key=lambda t: curve[int(round(t))])[: int(p["K_SLOTS"])]
            sl.sort()
        slots[lab] = np.asarray(sl)
        a_un[lab] = np.asarray([curve[int(round(t))] for t in sl], np.float64)
        a_un[lab] -= a_un[lab].min()          # per-cluster offset is irrelevant
    stats["n_slots"] = int(sum(len(s) for s in slots.values()))

    # ---------- spatial prior: connectivity to placed mass at each slot ----------
    graph, _ = build_hit_graph(xyz, p["R_LINK"])
    # group placed times into interaction-scale anchors (gap <= ANCHOR_DT) so the
    # dijkstra count is ~#interactions, and only keep groups near some slot
    all_slot_t = np.unique(np.concatenate([slots[l] for l in cands]))
    at_sorted = np.unique(np.round(ts[assigned]))
    groups = []
    for at in at_sorted:
        if groups and at - groups[-1][-1] <= p["ANCHOR_DT"]:
            groups[-1].append(float(at))
        else:
            groups.append([float(at)])
    S = {lab: np.zeros(len(slots[lab])) for lab in cands}
    geo_cache = {}
    for grp in groups:
        lo, hi = grp[0] - 1.0, grp[-1] + 1.0
        if not np.any((all_slot_t >= lo) & (all_slot_t <= hi)):
            continue
        src = np.flatnonzero(assigned)[(ts[assigned] >= grp[0] - p["ANCHOR_DT"])
                                       & (ts[assigned] <= grp[-1] + p["ANCHOR_DT"])]
        if src.size == 0:
            continue
        geo_cache[(lo, hi)] = family_geodesics(graph, src, p["D_GEO_MAX"])
    for lab, cd in cands.items():
        for si, t in enumerate(slots[lab]):
            for (lo, hi), g in geo_cache.items():
                if lo <= t <= hi and np.isfinite(np.min(g[cd["hits"]])):
                    S[lab][si] = 1.0          # scaled to energy units below
                    stats["n_connected_slots"] += 1
                    break

    # ---------- pairwise light coupling ----------
    by_tpc = {}
    for lab, cd in cands.items():
        for c in cd["tpcs"]:
            by_tpc.setdefault(c, []).append(lab)
    pairs = {}
    n_pairs = 0
    labs_list = sorted(cands)
    idx_of = {l: i for i, l in enumerate(labs_list)}
    for c, ll in by_tpc.items():
        for ai in range(len(ll)):
            for bi in range(ai + 1, len(ll)):
                i, j = ll[ai], ll[bi]
                if n_pairs >= int(p["PAIR_CAP"]):
                    break
                chi = cands[i]["sup"].get(c)
                chj = cands[j]["sup"].get(c)
                if chi is None or chj is None:
                    continue
                ch = np.intersect1d(chi, chj)
                if ch.size == 0:
                    continue
                ti, tj = slots[i], slots[j]
                close = np.abs(ti[:, None] - tj[None, :]) < p["C_MAX_DT"]
                if not close.any():
                    continue
                Pi = np.asarray(image_maps[(i, c)], np.float64)[ch]
                Pj = np.asarray(image_maps[(j, c)], np.float64)[ch]
                W = w_all[c][ch]
                Si = np.stack([_shift(Pi, t, n_ticks).ravel() for t in ti])
                Sj = np.stack([(W * _shift(Pj, t, n_ticks)).ravel() for t in tj])
                Cij = 2.0 * Si @ Sj.T
                key = (i, j)
                pairs[key] = pairs.get(key, 0.0) + Cij
                n_pairs += 1
    stats["n_pairs"] = n_pairs

    # ---------- annealed damped mean-field ----------
    spread = np.median([max(float(np.ptp(a_un[l])), 1.0) for l in labs_list])
    prior_e = float(p["PRIOR_FRAC"]) * spread
    for lab in labs_list:
        S[lab] = S[lab] * prior_e             # certificate as an ENERGY bonus
    q = {}
    for lab in labs_list:
        e0 = a_un[lab] - S[lab]
        logit0 = -(e0 - e0.min()) / max(spread, 1e-3)
        q[lab] = np.exp(logit0 - logit0.max())
        q[lab] /= q[lab].sum()
    T = float(params.get("T0", spread)) if params and "T0" in (params or {}) else float(spread)
    nbrs = {l: [] for l in labs_list}
    for (i, j), Cij in pairs.items():
        nbrs[i].append((j, Cij))
        nbrs[j].append((i, Cij.T))
    for sweep in range(int(p["T_SWEEPS"])):
        moved = 0.0
        for lab in labs_list:
            E = a_un[lab].copy()
            for j, Cij in nbrs[lab]:
                E += Cij @ q[j]
            E = E - S[lab]                 # prior lives in the energy itself
            logit = -(E - E.min()) / max(T, 1e-3)
            qn = np.exp(logit - logit.max())
            qn /= qn.sum()
            q_new = (1 - p["DAMP"]) * q[lab] + p["DAMP"] * qn
            moved += float(np.abs(q_new - q[lab]).sum())
            q[lab] = q_new
        T *= p["T_DECAY"]
        stats["sweeps"] = sweep + 1
        if moved / max(len(labs_list), 1) < 1e-4 and sweep > 5:
            break

    # ---------- harden + commit ----------
    order = sorted(labs_list, key=lambda l: -cands[l]["e"])
    for lab in order:
        si = int(np.argmax(q[lab]))
        t0 = float(slots[lab][si])
        for c in cands[lab]["tpcs_all"]:
            img = np.asarray(image_maps[(lab, c)], np.float64)
            base[c] += _shift(img, t0, n_ticks).astype(np.float32)
        ts[cands[lab]["hits"]] = np.float32(t0)
        stats["n_assigned"] += 1
        stats["e_assigned"] += cands[lab]["e"]
        conn = bool(S[lab][si] > 0)
        if conn:
            stats["n_conn_assigned"] = stats.get("n_conn_assigned", 0) + 1
            stats["e_conn_assigned"] = stats.get("e_conn_assigned", 0.0) + cands[lab]["e"]
        stats["assigned"].append({"label": int(lab), "t0": t0,
                                  "e": cands[lab]["e"],
                                  "q_max": float(q[lab][si]),
                                  "connected": conn})
    ns["baseImage"] = base
    ns["hit_timestamps"] = ts.astype(np.float32)
    stats["member_labels"] = sorted(int(l) for l in labs_list)
    return stats
