"""Dedicated shower-TPC algorithm (TPC total energy > 1.5 GeV). Target: 99%.

Design (user 2026-07-09, both proposed approaches merged):
  Stage 1  FLASH TABLE   per shower TPC: the t0 candidates + their recorded
           amplitudes; the strongest flash t* is the shower flash.
  Stage 2  ANCHOR        the most energetic cluster-part in the TPC is matched
           to t* DIRECTLY (no scan — the monster owns the max flash).
  Stage 3  70% FILL      grow shower membership from the anchor outward in
           GEODESIC order (serious spatial containment: only parts within
           reach of the current shower body may join), rendering each part's
           light at t*, until the flash is filled to fill_frac (0.70) —
           deliberately underfilled so "everything wants in" cannot happen.
  Stage 4  THE REST      remaining parts full-scan against the residual, with
           a SPATIAL-DISLIKE penalty on choosing t*: parts geodesically far
           from the shower body prefer their own flash.
  Stage 5  PRUNE & GROW  boundary refinement by differential light yield:
           an outer member whose removal improves the t* fit AND whose own
           scan prefers another flash is pruned; a nearby non-member is
           grabbed when t* still shows a deficit and adding it improves.

Runs at the very end of the chain (post phase-3/swap) and REDOES the flagged
TPCs from scratch, per-TPC parts (cross-TPC seams accepted, as in swap-fix).
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from v030_expansion import build_hit_graph, family_geodesics, weighted_scan

PEAK = 105
GAIN = 200.0
CLIP = 60780.0
SEARCH = 895


def run_shower_tpc_pass(ns: dict[str, Any], *, hit_tpc: np.ndarray,
                        e_thresh_mev: float = 1500.0,
                        fill_frac: float = 0.70,
                        grow_reach_cm: float = 12.0,
                        dislike_start_cm: float = 15.0,
                        dislike_per_cm: float = 0.03,
                        prune_outer_frac: float = 0.25,
                        prune_iters: int = 2,
                        mode: str = "redo",
                        min_part_mev: float = 1.0,
                        truth_fn: Optional[Callable] = None,
                        verbose: bool = False) -> dict[str, Any]:
    image_maps = ns["imageMaps"]
    base = np.asarray(ns["baseImage"], dtype=np.float32)
    actual = np.asarray(ns["fullLightWaveform"], dtype=np.float64)
    n_light, n_ch, n_ticks = actual.shape
    labels = np.asarray(ns["labels_global"])
    ts = np.asarray(ns["hit_timestamps"], dtype=np.float64).copy()
    xyz = np.stack([np.asarray(ns["xset"], np.float64), np.asarray(ns["yset"], np.float64),
                    np.asarray(ns["zset"], np.float64)], axis=1)
    e_pos = np.clip(np.asarray(ns["Eset"], dtype=np.float64), 0.0, None)
    hit_tpc = np.asarray(hit_tpc, dtype=np.int64)
    w_all = 1.0 / (GAIN * np.maximum(actual, 100.0))

    tpc_e = np.zeros(n_light + 1)
    valid = hit_tpc < n_light
    np.add.at(tpc_e, np.where(valid, hit_tpc, n_light), e_pos)
    shower_tpcs = [int(t) for t in np.flatnonzero(tpc_e[:n_light] > e_thresh_mev)]

    stats = {"tpcs": [], "n_tpcs": len(shower_tpcs)}

    for t in shower_tpcs:
        m_t = (hit_tpc == t)
        A = actual[t]
        W = w_all[t]

        # ---------- parts: per-TPC cluster fragments with templates ----------
        parts = {}
        for lab in np.unique(labels[m_t & (labels >= 0)]):
            if (int(lab), t) not in image_maps:
                continue
            hidx = np.flatnonzero(m_t & (labels == lab))
            e_c = float(e_pos[hidx].sum())
            if e_c < min_part_mev:
                continue
            img = np.asarray(image_maps[(int(lab), t)], np.float64)
            sup = np.flatnonzero(img.max(axis=1) > 1.0)
            if not sup.size:
                continue
            parts[int(lab)] = {"hits": hidx, "e": e_c, "img": img, "sup": sup,
                               "centroid": xyz[hidx].mean(axis=0)}
        if not parts:
            continue

        # ---------- stage 1: flash table ----------
        cand = sorted(set(int(round(v)) for v in ns["t0Candidates"][t]
                          if 0 <= round(v) <= SEARCH))
        if not cand:
            continue
        summed = A.sum(axis=0)
        amps = {c: float(summed[min(c + PEAK, n_ticks - 1)]) for c in cand}
        t_star = max(cand, key=lambda c: amps[c])

        # ---------- stage 2: anchor ----------
        anchor = max(parts, key=lambda l: parts[l]["e"])
        cur_t0 = {}
        for l in parts:
            v = ts[parts[l]["hits"]]
            v = v[np.isfinite(v) & (v >= 0)]
            cur_t0[l] = float(np.median(v)) if v.size else None

        # geodesics within this TPC's hit cloud
        tidx = np.flatnonzero(m_t)
        sub_xyz = xyz[tidx]
        graph, _ = build_hit_graph(sub_xyz, 3.0)
        pos_in_sub = {int(g): i for i, g in enumerate(tidx)}

        def part_subidx(lab):
            return np.asarray([pos_in_sub[int(h)] for h in parts[lab]["hits"]])

        # flash channels of t*: channels with meaningful light at the peak
        tick_s = min(t_star + PEAK, n_ticks - 1)
        fch = np.flatnonzero(A[:, tick_s] > max(200.0, 0.02 * A[:, tick_s].max()))
        denom = float(A[fch, tick_s].sum())

        def fill_of(members):
            # a template shifted to t* peaks at tick t*+PEAK with pattern img[:, PEAK]
            pred = np.zeros(fch.size)
            for l in members:
                pred += parts[l]["img"][fch, PEAK]
            return float(np.minimum(pred, A[fch, tick_s]).sum()) / max(denom, 1e-9)

        if mode == "refine":
            # keep the chain's assignments as the prior; only prune & grow
            assign = {l: (cur_t0[l] if cur_t0[l] is not None else float(t_star))
                      for l in parts}
            members = [l for l, tv in assign.items() if abs(tv - t_star) <= 2] or [anchor]
            src_idx = np.concatenate([part_subidx(l) for l in members])
            geo = family_geodesics(graph, src_idx, limit=1e9)
            Bt = np.zeros_like(A)
            for l, tv in assign.items():
                Bt += _shift(parts[l]["img"], tv, n_ticks)
            fill = fill_of(members)
        # ---------- stages 3+4 (full REDO mode only) ----------
        if mode != "refine":
            members = [anchor]
            geo = family_geodesics(graph, part_subidx(anchor), limit=1e9)
            pool = [l for l in parts if l != anchor]
            fill = fill_of(members)
            n_since_refresh = 0
            while fill < fill_frac and pool:
                dists = [(float(np.min(geo[part_subidx(l)])), l) for l in pool]
                dists.sort()
                d_next, l_next = dists[0]
                if not np.isfinite(d_next) or d_next > grow_reach_cm:
                    break
                members.append(l_next)
                pool.remove(l_next)
                fill = fill_of(members)
                n_since_refresh += 1
                if n_since_refresh >= 5:      # refresh geodesics every 5 additions
                    src = np.concatenate([part_subidx(l) for l in members])
                    geo = family_geodesics(graph, src, limit=1e9)
                    n_since_refresh = 0

            # ---------- stage 4: the rest, spatial-dislike scans ----------
            Bt = np.zeros_like(A)
            for l in members:
                Bt += _shift(parts[l]["img"], t_star, n_ticks)
            assign = {l: float(t_star) for l in members}
            for l in sorted(pool, key=lambda x: -parts[x]["e"]):
                P = parts[l]["img"][parts[l]["sup"]]
                resid = (Bt - A)[parts[l]["sup"]]
                curve = weighted_scan(P, resid, W[parts[l]["sup"]])
                d_sh = float(np.min(geo[part_subidx(l)]))
                pen = np.zeros_like(curve)
                if np.isfinite(d_sh) and d_sh > dislike_start_cm:
                    # spatial dislike: far parts pay a fraction of the curve's
                    # dynamic range for choosing t*
                    frac = min(1.0, dislike_per_cm * (d_sh - dislike_start_cm))
                    lo = max(0, t_star - 10)
                    hi = min(curve.size, t_star + 11)
                    pen[lo:hi] = frac * float(curve.max() - curve.min())
                tl = int(np.argmin(curve + pen))
                assign[l] = float(tl)
                Bt += _shift(parts[l]["img"], tl, n_ticks)

        # ---------- stage 5: prune & grow ----------
        for _ in range(int(prune_iters)):
            changed = False
            mem_now = [l for l, tv in assign.items() if abs(tv - t_star) <= 2]
            if len(mem_now) > 1:
                d_mem = sorted(((float(np.min(geo[part_subidx(l)])), l) for l in mem_now
                                if l != anchor), reverse=True)
                outer = [l for _, l in d_mem[: max(1, int(prune_outer_frac * len(d_mem)))]]
                for l in outer:
                    sup = parts[l]["sup"]
                    sh = _shift(parts[l]["img"], t_star, n_ticks)
                    without = Bt - sh
                    d_now = float(np.sum(W[sup] * (np.minimum(Bt[sup], CLIP) - A[sup]) ** 2))
                    d_wo = float(np.sum(W[sup] * (np.minimum(without[sup], CLIP) - A[sup]) ** 2))
                    curve = weighted_scan(parts[l]["img"][sup], (without - A)[sup], W[sup])
                    t_alt = int(np.argmin(curve))
                    gain_alt = float(curve[t_star] - curve[t_alt])
                    if d_wo < d_now and abs(t_alt - t_star) > 10 and gain_alt > 0:
                        Bt = without + _shift(parts[l]["img"], t_alt, n_ticks)
                        assign[l] = float(t_alt)
                        changed = True
            # grow: t* deficit and a touching non-member helps
            deficit = float(np.sum(np.clip(A[fch, tick_s] - Bt[fch, tick_s], 0, None))) \
                / max(denom, 1e-9)
            if deficit > 1.0 - fill_frac:
                near = sorted(((float(np.min(geo[part_subidx(l)])), l)
                               for l, tv in assign.items() if abs(tv - t_star) > 2))
                for d_l, l in near[:5]:
                    if d_l > grow_reach_cm:
                        break
                    sup = parts[l]["sup"]
                    old = _shift(parts[l]["img"], assign[l], n_ticks)
                    new = _shift(parts[l]["img"], t_star, n_ticks)
                    d_now = float(np.sum(W[sup] * (np.minimum(Bt[sup], CLIP) - A[sup]) ** 2))
                    Bt2 = Bt - old + new
                    d_new = float(np.sum(W[sup] * (np.minimum(Bt2[sup], CLIP) - A[sup]) ** 2))
                    if d_new < d_now:
                        Bt = Bt2
                        assign[l] = float(t_star)
                        changed = True
            if not changed:
                break

        # ---------- write back ----------
        rec = {"tpc": t, "e_tot": float(tpc_e[t]), "t_star": int(t_star),
               "n_parts": len(parts), "n_shower": sum(1 for v in assign.values()
                                                      if abs(v - t_star) <= 2),
               "fill": round(fill, 3)}
        n_moved = 0
        for l, tv in assign.items():
            hidx = parts[l]["hits"]
            old = ts[hidx]
            if not np.all(np.isfinite(old)) or abs(float(np.median(old[np.isfinite(old)]))
                                                   - tv) > 1e-6:
                n_moved += 1
            ts[hidx] = np.float32(tv)
        rec["n_moved"] = n_moved
        if truth_fn is not None:
            good = tot = 0.0
            for l, tv in assign.items():
                tr = truth_fn(int(l), t)
                if tr is None:
                    continue
                tot += parts[l]["e"]
                if abs(tv - tr) <= 10:
                    good += parts[l]["e"]
            rec["part_eff"] = round(good / max(tot, 1e-9), 4)
        stats["tpcs"].append(rec)
        if verbose:
            print(f"[showerTPC] {rec}", flush=True)

    ns["hit_timestamps"] = ts.astype(np.float32)
    return stats


def _shift(img, t0, n_ticks):
    out = np.zeros_like(img, dtype=np.float64)
    s = int(round(t0))
    if 0 <= s < n_ticks:
        out[:, s:] = img[:, : n_ticks - s]
    return out


__all__ = ["run_shower_tpc_pass"]
