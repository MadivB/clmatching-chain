"""v0.8.4 side path: DECLINE-OR-REPAIR for the small-cluster tail.

After force coverage, audit every small (< e_max) imaged assigned cluster:
if placing it ANYWHERE only worsens the light loss by more than a threshold
(r = best-case delta-loss / its own template norm; r ~ 1 means "the light
sees nothing of this cluster anywhere", r ~ 0 neutral, r < 0 improves):

  1. MITIGATE: maybe a mis-placed neighbor is blocking its spot.  Find
     placed small clusters with similar support channels (allowing channel
     shifts of 0 / +-6 / +-12 — the light-readout geometry), and try the
     small permutation pool over {subject + partners} placements, exactly
     like the large-energy scanning: commit the arrangement with the best
     joint loss.
  2. DECLINE: if after mitigation the subject still worsens the light
     everywhere beyond the threshold (or it has no partners), unassign it:
     hits -> -10000, template removed from the base.

Truth-audited per operation.  Tunable: r_max (fail threshold), e_max.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from spatial_score import emedian
from v030_expansion import GAIN, SEARCH, _shift, weighted_scan

CUT = 10.0
UNASSIGNED = -10000.0


def run_decline(ns: dict[str, Any], *, r_max: float = 0.5,
                e_max: float = 30.0, shifts=(0, 6, -6, 12, -12),
                max_partners: int = 3, cand_per_part: int = 3,
                truth_fn: Callable | None = None,
                verbose: bool = False) -> dict[str, Any]:
    image_maps = ns["imageMaps"]
    base = np.asarray(ns["baseImage"], dtype=np.float32)
    actual = np.asarray(ns["fullLightWaveform"], dtype=np.float64)
    n_light, n_ch, n_ticks = actual.shape
    labels = np.asarray(ns["labels_global"])
    ts = np.asarray(ns["hit_timestamps"], dtype=np.float64).copy()
    e_pos = np.clip(np.asarray(ns["Eset"], dtype=np.float64), 0.0, None)
    w_all = 1.0 / (GAIN * np.maximum(actual, 100.0))

    stats = {"n_subjects": 0, "n_triggered": 0, "n_mitigated": 0,
             "n_recovered": 0, "n_declined": 0, "e_declined": 0.0,
             "audit": []}

    def assigned_mask():
        return np.isfinite(ts) & (ts >= 0)

    # ---------- per-cluster info: small, imaged, assigned ----------
    def build_info():
        info = {}
        A = assigned_mask()
        for l in np.unique(labels[(labels >= 0)]):
            m = (labels == l) & A
            if not m.any():
                continue
            e_c = float(e_pos[m].sum())
            tpcs = [c for c in range(n_light) if (int(l), c) in image_maps]
            if not tpcs or e_c >= e_max:
                continue
            supp = {}
            for c in tpcs:
                img = np.asarray(image_maps[(int(l), c)], np.float64)
                ch = set(int(x) for x in np.flatnonzero(img.max(axis=1) > 5.0))
                if ch:
                    supp[c] = ch
            if not supp:
                continue
            info[int(l)] = dict(e=e_c, tpcs=tpcs, supp=supp,
                                t=emedian(ts[m], np.maximum(e_pos[m], 1e-6)))
        return info

    def curve_and_r(l, info):
        """Self-removed scan; returns (curve, loss_empty, t*, r)."""
        prs, brs, wrs = [], [], []
        for c in info[l]["tpcs"]:
            img = np.asarray(image_maps[(int(l), c)], np.float64)
            ch = np.flatnonzero(img.max(axis=1) > 1.0)
            if not ch.size:
                continue
            own = _shift(img, info[l]["t"], n_ticks)
            prs.append(img[ch])
            brs.append(np.asarray(base[c], np.float64)[ch] - own[ch]
                       - actual[c][ch])
            wrs.append(w_all[c][ch])
        if not prs:
            return None
        P_, B_, W_ = (np.concatenate(prs), np.concatenate(brs),
                      np.concatenate(wrs))
        curve = weighted_scan(P_, B_, W_)
        loss_empty = float((W_ * B_ ** 2).sum())
        i_best = int(np.argmin(curve))
        own_b = np.zeros_like(P_)
        s0 = i_best
        if 0 <= s0 < P_.shape[1]:
            own_b[:, s0:] = P_[:, :P_.shape[1] - s0]
        Pnorm = float((W_ * own_b ** 2).sum())
        r = (float(curve[i_best]) - loss_empty) / max(Pnorm, 1e-9)
        return curve, loss_empty, i_best, r

    def local_minima(curve, k=2):
        i = int(np.argmin(curve))
        out = [i]
        c = curve.copy()
        for _ in range(k - 1):
            lo, hi = max(0, out[-1] - 15), min(len(c), out[-1] + 15)
            c[lo:hi] = np.inf
            out.append(int(np.argmin(c)))
        return out

    def move(l, info, t_new):
        for c in info[l]["tpcs"]:
            img = np.asarray(image_maps[(int(l), c)], np.float64)
            base[c] += (_shift(img, t_new, n_ticks)
                        - _shift(img, info[l]["t"], n_ticks)).astype(np.float32)
        np.clip(base, 0.0, None, out=base)
        m = (labels == l) & assigned_mask()
        ts[m] = float(t_new)
        info[l]["t"] = float(t_new)

    def unassign(l, info):
        for c in info[l]["tpcs"]:
            img = np.asarray(image_maps[(int(l), c)], np.float64)
            base[c] -= _shift(img, info[l]["t"], n_ticks).astype(np.float32)
        np.clip(base, 0.0, None, out=base)
        m = (labels == l) & assigned_mask()
        ts[m] = UNASSIGNED

    info = build_info()
    stats["n_subjects"] = len(info)
    # first cheap pass: r for everyone, process worst first
    order = []
    for l in info:
        cr = curve_and_r(l, info)
        if cr is not None:
            order.append((cr[3], l))
    order.sort(reverse=True)

    for r0, l in order:
        if l not in info:
            continue
        cr = curve_and_r(l, info)     # fresh (base may have changed)
        if cr is None:
            continue
        curve, loss_empty, i_best, r = cr
        if r <= r_max:
            continue
        stats["n_triggered"] += 1
        tr = truth_fn(int(l)) if truth_fn else None
        was_ok = tr is not None and abs(info[l]["t"] - tr) <= CUT
        rec = {"label": int(l), "e": info[l]["e"], "r_before": round(r, 3),
               "t_before": info[l]["t"], "truth": tr, "was_ok": was_ok}

        # ---- 1. mitigation: permute with support-channel neighbours ----
        partners = []
        for l2, inf2 in info.items():
            if l2 == l:
                continue
            score = 0.0
            for c, S in info[l]["supp"].items():
                S2 = inf2["supp"].get(c)
                if not S2:
                    continue
                for sh in shifts:
                    Ssh = {x + sh for x in S}
                    score = max(score, len(S2 & Ssh) / max(len(S), 1))
            if score >= 0.5:
                partners.append((score, l2))
        partners.sort(reverse=True)
        parts = [l] + [l2 for _, l2 in partners[:max_partners]]
        rec["n_partners"] = len(parts) - 1

        if len(parts) > 1:
            # candidate times: each participant's current t + curve minima
            cands = {}
            for p in parts:
                cp = curve_and_r(p, info)
                mins = local_minima(cp[0], cand_per_part - 1) if cp else []
                cands[p] = list(dict.fromkeys(
                    [int(round(info[p]["t"]))] + mins))[:cand_per_part]
            # union top-channel rows across participants (subject TPCs)
            rows = []
            for p in parts:
                for c, S in info[p]["supp"].items():
                    for ch in S:
                        rows.append((c, ch))
            rows = list(dict.fromkeys(rows))[:80]
            tmpl = {}
            for p in parts:
                tmpl[p] = {}
                for c, ch in rows:
                    if (int(p), c) in image_maps:
                        tmpl[p][(c, ch)] = np.asarray(
                            image_maps[(int(p), c)], np.float64)[ch]
            b0 = {}
            for c, ch in rows:
                v = np.asarray(base[c], np.float64)[ch].copy()
                for p in parts:
                    if (c, ch) in tmpl[p]:
                        sft = np.zeros(n_ticks)
                        s0 = int(round(info[p]["t"]))
                        if 0 <= s0 < n_ticks:
                            sft[s0:] = tmpl[p][(c, ch)][:n_ticks - s0]
                        v -= sft
                b0[(c, ch)] = v

            def arr_loss(assign):
                tot = 0.0
                for c, ch in rows:
                    v = b0[(c, ch)].copy()
                    for p, tp in assign.items():
                        if (c, ch) in tmpl[p]:
                            s0 = int(round(tp))
                            if 0 <= s0 < n_ticks:
                                v[s0:] += tmpl[p][(c, ch)][:n_ticks - s0]
                    tot += float((w_all[c][ch]
                                  * (np.minimum(v, 60780.0)
                                     - actual[c][ch]) ** 2).sum())
                return tot

            import itertools
            cur_assign = {p: info[p]["t"] for p in parts}
            best_assign, best_loss = cur_assign, arr_loss(cur_assign)
            for combo in itertools.product(*(cands[p] for p in parts)):
                a = dict(zip(parts, combo))
                lo = arr_loss(a)
                if lo < best_loss - 1e-9:
                    best_assign, best_loss = a, lo
            if any(abs(best_assign[p] - cur_assign[p]) > 2.0 for p in parts):
                stats["n_mitigated"] += 1
                rec["mitigated"] = {int(p): (round(cur_assign[p], 1),
                                             round(float(best_assign[p]), 1))
                                    for p in parts
                                    if abs(best_assign[p] - cur_assign[p]) > 2.0}
                for p in parts:
                    if abs(best_assign[p] - cur_assign[p]) > 2.0:
                        move(p, info, float(best_assign[p]))
                cr2 = curve_and_r(l, info)
                r = cr2[3] if cr2 is not None else r
        rec["r_after"] = round(r, 3)

        # ---- 2. decline if still unexplained ----
        if r > r_max:
            unassign(l, info)
            stats["n_declined"] += 1
            stats["e_declined"] += info[l]["e"]
            rec["declined"] = True
            del info[l]
        else:
            stats["n_recovered"] += 1
            rec["declined"] = False
            rec["now_ok"] = (tr is not None
                             and abs(info[l]["t"] - tr) <= CUT)
        stats["audit"].append(rec)

    ns["baseImage"] = base
    ns["hit_timestamps"] = ts.astype(np.float32)
    return stats
