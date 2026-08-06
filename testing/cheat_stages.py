"""Three-stage cheating study (v0.4.0 decision tool).

Cluster categories (per user definition, by CLUSTER properties, not by stage):
  type 1: first-stage objects — showers and tracks crossing >= 3 TPCs
  type 2: all other BIG clusters — multi-TPC, or total energy > 30 MeV
  type 3: everything else (small single-TPC clusters; noise hits)

Trials:
  --cheat type1  : type-1 clusters placed at their MC truth t0 (rounded);
                   the rest of the chain runs normally -> measures how well
                   types 2+3 do when the backbone seeds are perfect.
  --cheat type12 : types 1 AND 2 placed at truth (in the chain's natural
                   order — big backbone objects in stage 1, Phase-2-class
                   clusters pinned right before Phase 2 so type-3 placement
                   sees exactly the same intermediate states as v0.4.0).

Impure clusters (hits from several interactions): the cluster is SPLIT by
truth-t0 groups (gap > 3 ticks), each part gets its OWN perceiver prediction
(fresh forward pass on the hit subset) and is added to the base image at its
own rounded truth time.

Cheated labels are truth-LOCKED: after every downstream stage a re-pin pass
restores any cheated cluster that a black-box stage moved (base image delta
corrected via its templates).
"""
from __future__ import annotations

from typing import Any

import numpy as np

CUT = 10.0
SPLIT_GAP = 3.0          # ticks: truth groups farther apart than this are split
SPLIT_MIN_MEV = 3.0      # a truth group must carry this much energy to split off


def classify_clusters(labels_global, split_index, label_info, cluster_to_tpcs,
                      e_pos, n_light):
    types = {}
    for lab in np.unique(labels_global[labels_global >= 0]):
        lab = int(lab)
        m = labels_global == lab
        e = float(e_pos[m].sum())
        tpcs = [int(t) for t in cluster_to_tpcs.get(lab, []) if int(t) < n_light]
        ltype = str(label_info.get(lab, {}).get("type", "")).lower()
        if lab < split_index and (ltype == "shower" or len(tpcs) >= 3):
            types[lab] = 1
        elif len(tpcs) >= 2 or e > 30.0:
            types[lab] = 2
        else:
            types[lab] = 3
    return types


def truth_groups(truth_vals, e_vals):
    """Split a cluster's finite truth times into groups separated by > SPLIT_GAP."""
    order = np.argsort(truth_vals)
    tv, ev = truth_vals[order], e_vals[order]
    groups, start = [], 0
    for i in range(1, tv.size + 1):
        if i == tv.size or tv[i] - tv[i - 1] > SPLIT_GAP:
            groups.append((order[start:i], float(ev[start:i].sum())))
            start = i
    return groups


def split_and_predict(S, cheat_labels, predict_fn, models, cfg):
    """Split impure cheated clusters by truth group; fresh predictions per part.

    Mutates labels arrays / cluster_to_tpcs / label_info / image_maps /
    support_cache in place.  Returns {label: rounded_truth_t0} for every
    cheated (post-split) label."""
    labels = S["labels_global"]
    lg_left = np.asarray(S["fs"].track_stage.labels_global_with_leftovers)
    cheat_t0: dict[int, int] = {}
    next_label = int(labels.max()) + 1
    new_parts = []          # labels needing fresh predictions

    for lab in sorted(cheat_labels):
        m = np.flatnonzero(labels == lab)
        tt = S["truth_t0"][m]
        fin = np.isfinite(tt)
        if not fin.any():
            continue
        groups = [g for g in truth_groups(tt[fin], S["e_pos"][m][fin])]
        big = [g for g in groups if g[1] >= SPLIT_MIN_MEV] or \
              [max(groups, key=lambda g: g[1])]
        # dominant group keeps the original label; others get new ids
        big.sort(key=lambda g: -g[1])
        idx_fin = m[fin]
        part_of = np.full(m.size, 0, int)          # default: dominant part
        for gi, (gidx, _ge) in enumerate(big):
            part_of[np.isin(m, idx_fin[gidx])] = gi
        for gi, (gidx, ge) in enumerate(big):
            part_hits = m[part_of == gi] if gi > 0 else m[part_of == 0]
            tmed = float(np.median(S["truth_t0"][idx_fin[gidx]]))
            if gi == 0:
                cheat_t0[lab] = int(round(tmed))
                continue
            newlab = next_label
            next_label += 1
            labels[part_hits] = newlab
            lg_left[part_hits] = newlab
            S["cluster_to_tpcs"][newlab] = sorted(
                int(t) for t in np.unique(S["hit_tpc"][part_hits])
                if int(t) < S["n_light"])
            li = dict(S["label_info"].get(int(lab), {}))
            li["n_hits"] = int(part_hits.size)
            li["tpcs"] = S["cluster_to_tpcs"][newlab]
            li["truth_split_from"] = int(lab)
            S["label_info"][newlab] = li
            S["fs"].clustering.cluster_labels.append(int(newlab))
            cheat_t0[newlab] = int(round(tmed))
            new_parts.append(newlab)
        if len(big) > 1:
            # dominant part shrank: its template must be re-predicted too
            new_parts.append(lab)
            S["cluster_to_tpcs"][int(lab)] = sorted(
                int(t) for t in np.unique(S["hit_tpc"][labels == lab])
                if int(t) < S["n_light"])

    if new_parts:
        # fresh perceiver predictions for split parts only: synthetic labels
        syn = np.full(labels.shape, -1, dtype=labels.dtype)
        for lab in new_parts:
            syn[labels == lab] = lab
        bundle = predict_fn(
            x=np.asarray(S["fs"].event.x), y=np.asarray(S["fs"].event.y),
            z=np.asarray(S["fs"].event.z), energy=np.asarray(S["fs"].event.energy),
            hit_tpc_id=S["hit_tpc"], labels_global=syn,
            split_index=int(max(new_parts) + 1), label_info=S["label_info"],
            full_light_waveform=S["actual"].astype(np.float32),
            models=models, config=cfg)
        for lab in new_parts:
            for c in range(S["n_light"]):
                S["image_maps"].pop((int(lab), c), None)
                S["support_cache"].pop((int(lab), c), None)
        for (l, c), img in bundle.image_maps.items():
            S["image_maps"][(int(l), int(c))] = img
        for (l, c), entry in bundle.cluster_channel_support_cache.items():
            S["support_cache"][(int(l), int(c))] = entry
    return cheat_t0


def make_repin(S, cheat_t0, ti_fn):
    """Return repin(get_base, ts_array, tag): restore truth-locked clusters."""
    def repin(base_arr, ts, tag=""):
        n_fix = 0
        for lab, tc in cheat_t0.items():
            m = S["labels_global"] == lab
            if not m.any():
                continue
            cur = ts[m]
            fin = np.isfinite(cur) & (cur >= 0)
            med = float(np.median(cur[fin])) if fin.any() else None
            if med is not None and abs(med - tc) <= 0.5 and fin.all():
                continue
            for c in S["cluster_to_tpcs"].get(int(lab), []):
                key = (int(lab), int(c))
                if key not in S["image_maps"] or int(c) >= S["n_light"]:
                    continue
                img = np.asarray(S["image_maps"][key], np.float64)
                if med is not None:
                    base_arr[c] -= ti_fn(img, med).astype(base_arr.dtype)
                base_arr[c] += ti_fn(img, float(tc)).astype(base_arr.dtype)
            ts[m] = np.float32(tc)
            n_fix += 1
        if n_fix:
            np.clip(base_arr, 0.0, None, out=base_arr)
        return n_fix
    return repin


def hit_types(S, types):
    ht = np.full(S["labels_global"].shape, 3, np.int8)
    for lab, ty in types.items():
        ht[S["labels_global"] == lab] = ty
    ht[S["labels_global"] < 0] = 3
    return ht


def cheat_metrics(S, ht, ts):
    out = {}
    for ty in (1, 2, 3):
        m = (ht == ty) & np.isfinite(S["truth_t0"])
        ok = m & np.isfinite(ts) & (ts >= 0)
        res = np.abs(ts[ok] - S["truth_t0"][ok])
        good = res <= CUT
        e_ok = S["e_pos"][ok]
        out[f"type{ty}"] = {
            "e_truth": float(S["e_pos"][m].sum()),
            "coverage": float(e_ok.sum() / max(S["e_pos"][m].sum(), 1e-9)),
            "eff_e": float(e_pos_sum(e_ok, good) / max(e_ok.sum(), 1e-9)),
            "eff_n": float(good.mean()) if good.size else None,
            "mean_abs_res": float(np.average(res, weights=e_ok)) if res.size else None,
            "median_abs_res": float(np.median(res)) if res.size else None,
        }
    return out


def e_pos_sum(e, mask):
    return float(e[mask].sum())
