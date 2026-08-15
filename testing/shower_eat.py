"""Eager shower absorption (user scheme 2026-08-15; SHOWER_EAT_DESIGN.md).

Once a shower is recognized AND placed (explicitly typed by the clustering,
or emergent: several co-located big clusters committed at one t0), it
immediately absorbs surrounding unplaced clusters until 60 % of its light
at that t0 is fulfilled.  Ranking is by LIGHT PREFERENCE (the candidate's
own best/second-best loss ratio at the shower t0), not distance; guards:
two-shower contact veto, overflow, amplitude fit.

UNVERSIONED until measured better than v0.8.8.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy.spatial import cKDTree

from v030_expansion import _shift, weighted_scan

PEAK = 105
CLIP = 60780.0
GAIN = 200.0
CUT = 10.0
P2_BOUND = 30.0


def _label_hits(labels_global, lab):
    return np.flatnonzero(labels_global == int(lab))


def emergent_groups(placed_map, info_b, label_info, labels_global, xyz, e_pos,
                    dt=3.0, link_cm=30.0):
    """Groups of same-t0, co-located placed big clusters that imply a shower.

    Keep a group iff >= 2 members within link_cm chains AND
    (any non-track member OR >= 3 members OR a non-collinear pair) —
    NOT two distant/collinear tracks that merely share a time.
    """
    labs = sorted(placed_map)
    cents = {}
    for l in labs:
        h = _label_hits(labels_global, l)
        if h.size:
            w = np.maximum(e_pos[h], 1e-6)
            cents[l] = np.average(xyz[h], 0, weights=w)
    labs = [l for l in labs if l in cents]
    used, groups = set(), []
    order = sorted(labs, key=lambda l: -e_pos[labels_global == l].sum())
    for seed in order:
        if seed in used:
            continue
        grp = [seed]
        used.add(seed)
        changed = True
        while changed:
            changed = False
            for l in labs:
                if l in used:
                    continue
                if any(abs(placed_map[l] - placed_map[g]) <= dt
                       and np.linalg.norm(cents[l] - cents[g]) <= link_cm
                       for g in grp):
                    grp.append(l)
                    used.add(l)
                    changed = True
        if len(grp) < 2:
            continue
        types = [str(label_info.get(int(l), {}).get("type", "track")).lower()
                 for l in grp]
        showery = any(t != "track" for t in types) or len(grp) >= 3
        if not showery and len(grp) == 2:
            h = np.concatenate([_label_hits(labels_global, l) for l in grp])
            pts = xyz[h]
            mu = pts.mean(0)
            try:
                _, sv, _ = np.linalg.svd(pts - mu, full_matrices=False)
                lat = float(np.sqrt((sv[1] ** 2 + sv[2] ** 2) / max(len(pts), 1)))
                showery = lat > 4.0            # non-collinear pair
            except np.linalg.LinAlgError:
                pass
        if showery:
            t0g = float(np.median([placed_map[l] for l in grp]))
            groups.append(dict(t0=t0g, labels=list(grp)))
    return groups


def run_shower_eat(*, seeds, b_full, hit_t0, image_maps, actual, w_full, ti,
                   labels_global, e_pos, xyz, cluster_to_tpcs, label_info,
                   n_light, placed, fulfill=0.60, r_geo=60.0, s_min=1.02,
                   truth_fn: Callable | None = None) -> tuple[dict, dict]:
    """Absorb around each seed group. Returns (eaten {lab:(t0,[tpcs])}, stats)."""
    n_ticks = actual.shape[2]
    stats = {"n_seeds": len(seeds), "n_cand": 0, "n_eaten": 0, "e_eaten": 0.0,
             "n_veto_contact": 0, "n_misfit": 0, "n_weak_want": 0,
             "fulfill_start": [], "fulfill_end": [], "audit": []}
    eaten: dict[int, tuple] = {}
    if not seeds:
        return eaten, stats

    # non-track hit cloud for the two-shower contact veto
    nt_labs = [int(l) for l in np.unique(labels_global[labels_global >= 0])
               if str(label_info.get(int(l), {}).get("type", "track")).lower() != "track"]
    nt_idx = np.flatnonzero(np.isin(labels_global, nt_labs))
    nt_tree = cKDTree(xyz[nt_idx]) if nt_idx.size else None
    nt_owner = labels_global[nt_idx] if nt_idx.size else None

    # global candidate list: unplaced, imaged, below the P2 boundary
    cands_all = []
    for lab in np.unique(labels_global[labels_global >= 0]):
        lab = int(lab)
        if lab in placed or lab in eaten:
            continue
        tpcs = [c for c in cluster_to_tpcs.get(lab, [])
                if int(c) < n_light and (lab, int(c)) in image_maps]
        if not tpcs:
            continue
        e_c = float(e_pos[labels_global == lab].sum())
        if e_c >= P2_BOUND or e_c <= 0:
            continue
        cands_all.append((lab, tpcs, e_c))
    stats["n_cand"] = len(cands_all)

    def sup_channels(lab_list):
        sup = {}
        for l in lab_list:
            for c in cluster_to_tpcs.get(int(l), []):
                c = int(c)
                if c >= n_light or (int(l), c) not in image_maps:
                    continue
                img = np.asarray(image_maps[(int(l), c)])
                ch = np.flatnonzero(img.max(axis=1) > 1.0)
                sup.setdefault(c, set()).update(int(x) for x in ch)
        return {c: np.array(sorted(s), np.int64) for c, s in sup.items() if s}

    def fulfillment(sup, tick):
        num = den = 0.0
        for c, ch in sup.items():
            if not (0 <= tick < n_ticks):
                continue
            a = actual[c][ch, tick]
            b = np.minimum(b_full[c][ch, tick], CLIP)
            m = a > 200.0
            num += float(np.minimum(b[m], a[m]).sum())
            den += float(a[m].sum())
        return (num / den) if den > 0 else 1.0

    for sd in seeds:
        t0_s = float(sd["t0"])
        tick = int(round(t0_s)) + PEAK
        seed_labs = [int(l) for l in sd["labels"]]
        sup = sup_channels(seed_labs)
        if not sup:
            continue
        F = fulfillment(sup, tick)
        stats["fulfill_start"].append(round(F, 3))
        if F >= fulfill:
            stats["fulfill_end"].append(round(F, 3))
            continue
        # seed hit cloud for the surrounding gate
        sh = np.concatenate([_label_hits(labels_global, l) for l in seed_labs])
        if sh.size > 500:
            sh = sh[:: sh.size // 500 + 1]
        seed_tree = cKDTree(xyz[sh])

        scored = []
        for lab, tpcs, e_c in cands_all:
            if lab in eaten:
                continue
            hidx = _label_hits(labels_global, lab)
            hs = hidx if hidx.size <= 40 else hidx[:: hidx.size // 40 + 1]
            d_seed = float(seed_tree.query(xyz[hs])[0].min())
            if d_seed > r_geo:
                continue
            # two-shower veto: touches a FOREIGN non-track cluster
            if nt_tree is not None:
                dd, jj = nt_tree.query(xyz[hs], k=1)
                near = dd < 3.0
                if near.any():
                    owners = set(int(o) for o in np.asarray(nt_owner)[jj[near]])
                    owners -= set(seed_labs) | {lab}
                    if owners:
                        stats["n_veto_contact"] += 1
                        continue
            # own light preference curve (candidate absent from the base)
            prs, brs, wrs = [], [], []
            for c in tpcs:
                img = np.asarray(image_maps[(lab, int(c))], np.float64)
                ch = np.flatnonzero(img.max(axis=1) > 1.0)
                if not ch.size:
                    continue
                prs.append(img[ch])
                brs.append(np.asarray(b_full[int(c)], np.float64)[ch]
                           - actual[int(c)][ch])
                wrs.append(w_full[int(c)][ch])
            if not prs:
                continue
            curve = weighted_scan(np.concatenate(prs), np.concatenate(brs),
                                  np.concatenate(wrs))
            i_s = int(round(t0_s))
            if not (0 <= i_s < curve.size):
                continue
            i_b = int(np.argmin(curve))
            c2 = curve.copy()
            c2[max(0, i_b - 15):i_b + 16] = np.inf
            i_2 = int(np.argmin(c2))
            if abs(i_b - i_s) <= 3:
                s = float(curve[i_2] / max(curve[i_s], 1e-9))
            else:
                s = float(curve[i_b] / max(curve[i_s], 1e-9))
            scored.append((s, lab, tpcs, e_c, d_seed))
        scored.sort(reverse=True)

        for s, lab, tpcs, e_c, d_seed in scored:
            if F >= fulfill:
                break
            if s < s_min:
                stats["n_weak_want"] += 1
                continue
            # tentative absorption + amplitude-fit guard on own channels
            added = []
            for c in tpcs:
                img = np.asarray(image_maps[(lab, int(c))], np.float64)
                sh_img = _shift(img, t0_s, n_ticks).astype(np.float32)
                b_full[int(c)] += sh_img
                added.append((int(c), sh_img))
            bad = 0
            for c, _ in added:
                img = np.asarray(image_maps[(lab, c)])
                ch = np.flatnonzero(img.max(axis=1) > 1.0)
                if not (0 <= tick < n_ticks) or not ch.size:
                    continue
                a = actual[c][ch, tick]
                p = np.minimum(b_full[c][ch, tick], CLIP)
                ex = p - a
                z = ex / np.sqrt(GAIN * np.maximum(a, 100.0))
                bad += int(((ex >= 500.0) & (z > 2.0)).sum())
            if bad >= 2:
                for c, sh_img in added:
                    b_full[c] -= sh_img
                stats["n_misfit"] += 1
                continue
            hit_t0[labels_global == lab] = np.float32(t0_s)
            eaten[lab] = (t0_s, sorted(int(c) for c in tpcs))
            stats["n_eaten"] += 1
            stats["e_eaten"] += e_c
            if truth_fn is not None:
                tr = truth_fn(lab)
                stats["audit"].append(
                    {"label": lab, "e": e_c, "t0": t0_s, "want": round(s, 2),
                     "d": round(d_seed, 1), "truth": tr,
                     "ok": (tr is not None and abs(t0_s - tr) <= CUT)})
            F = fulfillment(sup, tick)
        stats["fulfill_end"].append(round(F, 3))
    return eaten, stats
