"""v0.8 PASS 3 (Q1, in-chain): family-level light — MOVE whole families to
better flash times, and CLOSE ridge-artifact families (a registered time that
is only the shoulder of a neighboring real flash: dissolve it, let the
neighbor absorb members and light).

Conservative by construction: rigid-family rescans with a ratio gate (a la
swap-fix), truth-audited per operation.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from spatial_score import emedian
from v030_expansion import GAIN, SEARCH, _shift, weighted_scan

CUT = 10.0


def run_pass3_family_light(ns: dict[str, Any], *,
                           move_ratio: float = 0.85,
                           swap_ratio: float = 0.85,
                           close_ratio: float = 1.05,
                           close_dt_max: float = 12.0,
                           min_move_ticks: float = 3.0,
                           max_stack_channels: int = 40,
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
    assigned = np.isfinite(ts) & (ts >= 0)

    stats = {"n_families": 0, "n_move_tried": 0, "n_moved": 0,
             "n_swap_tried": 0, "n_swapped": 0,
             "n_close_tried": 0, "n_closed": 0, "audit": []}

    # ---------- families = time groups of assigned clusters ----------
    lab_t, lab_e, lab_tpcs = {}, {}, {}
    for l in np.unique(labels[labels >= 0]):
        m = (labels == l) & assigned
        if not m.any():
            continue
        lab_t[int(l)] = emedian(ts[m], np.maximum(e_pos[m], 1e-6))
        lab_e[int(l)] = float(e_pos[m].sum())
    order = sorted(lab_t, key=lambda l: lab_t[l])
    fams, cur = [], []
    for l in order:
        if cur and lab_t[l] - lab_t[cur[-1]] > 3.0:
            fams.append(cur)
            cur = []
        cur.append(l)
    if cur:
        fams.append(cur)
    stats["n_families"] = len(fams)

    def fam_stack(mem):
        """Summed member templates per TPC, aligned at shift 0 (members of a
        family share the same t within <= 3 ticks, so a rigid stack is fair)."""
        stk = {}
        for l in mem:
            for c in range(n_light):
                if (int(l), c) in image_maps:
                    img = np.asarray(image_maps[(int(l), c)], np.float64)
                    stk.setdefault(c, np.zeros((n_ch, n_ticks)))
                    stk[c] += img
        return stk

    def top_channels(stk):
        rows = []
        for c, img in stk.items():
            for ch in np.argsort(img.max(axis=1))[::-1][:8]:
                if img[ch].max() > 5.0:
                    rows.append((float(img[ch].max()), c, int(ch)))
        rows.sort(reverse=True)
        return rows[:max_stack_channels]

    fam_info = []
    for mem in fams:
        t0 = float(np.median([lab_t[l] for l in mem]))
        e_tot = sum(lab_e[l] for l in mem)
        stk = fam_stack(mem)
        fam_info.append(dict(mem=mem, t0=t0, e=e_tot, stk=stk))

    def truth_of(mem):
        if truth_fn is None:
            return None
        ws = [(lab_e[l], truth_fn(int(l))) for l in mem]
        ws = [(w, t) for w, t in ws if t is not None]
        if not ws:
            return None
        ws.sort(key=lambda x: -x[0])
        return ws[0][1]

    def apply_move(fi, t_new):
        dt = t_new - fi["t0"]
        for c, img in fi["stk"].items():
            base[c] += (_shift(img, t_new, n_ticks)
                        - _shift(img, fi["t0"], n_ticks)).astype(np.float32)
        np.clip(base, 0.0, None, out=base)
        for l in fi["mem"]:
            m = (labels == l) & assigned
            ts[m] = ts[m] + dt
            lab_t[l] = lab_t[l] + dt
        fi["t0"] = t_new

    # ---------- Q1-MOVE: rigid rescan of each family ----------
    for fi in fam_info:
        rows = top_channels(fi["stk"])
        if not rows or fi["e"] < 20.0:
            continue
        stats["n_move_tried"] += 1
        prs = np.stack([fi["stk"][c][ch] for _, c, ch in rows])
        # residual with this family removed
        brs = np.stack([np.asarray(base[c], np.float64)[ch]
                        - _shift(fi["stk"][c], fi["t0"], n_ticks)[ch]
                        - actual[c][ch] for _, c, ch in rows])
        wrs = np.stack([w_all[c][ch] for _, c, ch in rows])
        curve = weighted_scan(prs, brs, wrs)
        i_cur = int(round(min(max(fi["t0"], 0), SEARCH)))
        i_new = int(np.argmin(curve))
        if abs(i_new - i_cur) < min_move_ticks:
            continue
        if curve[i_new] < move_ratio * curve[i_cur]:
            tr = truth_of(fi["mem"])
            was_ok = tr is not None and abs(fi["t0"] - tr) <= CUT
            apply_move(fi, float(i_new))
            stats["n_moved"] += 1
            stats["audit"].append({"op": "move", "e": fi["e"], "truth": tr,
                                   "was_ok": was_ok,
                                   "now_ok": (tr is not None
                                              and abs(fi["t0"] - tr) <= CUT)})

    # ---------- Q1-SWAP: rigid family-pair t0 exchange ----------
    # the 91% failure mode: two co-located families traded flashes wholesale.
    # A single-family rescan cannot fix it (the partner's light blocks the
    # destination); the pairwise exchange evaluates both moves JOINTLY.
    def chan_sig(fi):
        return set((c, int(ch)) for _, c, ch in top_channels(fi["stk"]))
    sigs = {id(fi): chan_sig(fi) for fi in fam_info}
    cand_pairs = []
    for i in range(len(fam_info)):
        for j in range(i + 1, len(fam_info)):
            fi, fj = fam_info[i], fam_info[j]
            if fi["e"] < 20.0 or fj["e"] < 20.0:
                continue
            dt = abs(fi["t0"] - fj["t0"])
            if dt < min_move_ticks or dt > 250.0:
                continue
            shared = len(sigs[id(fi)] & sigs[id(fj)])
            if shared >= 2:
                cand_pairs.append((shared, i, j))
    cand_pairs.sort(reverse=True)
    swapped = set()
    for shared, i, j in cand_pairs[:200]:
        fi, fj = fam_info[i], fam_info[j]
        if id(fi) in swapped or id(fj) in swapped:
            continue
        stats["n_swap_tried"] += 1
        rows = list(dict.fromkeys(top_channels(fi["stk"]) + top_channels(fj["stk"])))[:60]
        chans = [(c, ch) for _, c, ch in rows]
        ti0, tj0 = fi["t0"], fj["t0"]
        def stack_at(fk, t):
            return {(c, ch): _shift(fk["stk"][c], t, n_ticks)[ch]
                    for c, ch in chans if c in fk["stk"]}
        si_cur, sj_cur = stack_at(fi, ti0), stack_at(fj, tj0)
        si_new, sj_new = stack_at(fi, tj0), stack_at(fj, ti0)
        l_cur = l_new = 0.0
        for c, ch in chans:
            b0 = (np.asarray(base[c], np.float64)[ch]
                  - si_cur.get((c, ch), 0.0) - sj_cur.get((c, ch), 0.0))
            w = w_all[c][ch]
            a = actual[c][ch]
            l_cur += float(np.sum(w * (np.minimum(
                b0 + si_cur.get((c, ch), 0.0) + sj_cur.get((c, ch), 0.0), 60780.0) - a) ** 2))
            l_new += float(np.sum(w * (np.minimum(
                b0 + si_new.get((c, ch), 0.0) + sj_new.get((c, ch), 0.0), 60780.0) - a) ** 2))
        if l_new < swap_ratio * l_cur:
            tri, trj = truth_of(fi["mem"]), truth_of(fj["mem"])
            wi = tri is not None and abs(ti0 - tri) <= CUT
            wj = trj is not None and abs(tj0 - trj) <= CUT
            apply_move(fi, tj0)
            apply_move(fj, ti0)
            swapped.add(id(fi)); swapped.add(id(fj))
            stats["n_swapped"] += 1
            for fk, tr, was in ((fi, tri, wi), (fj, trj, wj)):
                stats["audit"].append({"op": "swap", "e": fk["e"], "truth": tr,
                                       "was_ok": was,
                                       "now_ok": (tr is not None
                                                  and abs(fk["t0"] - tr) <= CUT)})

    # ---------- Q1-CLOSE: dissolve ridge families into a near neighbor ----------
    fam_info.sort(key=lambda f: f["t0"])
    for i, fi in enumerate(fam_info):
        if fi.get("closed") or fi["e"] > 100.0:
            continue
        nbrs = [fj for fj in fam_info
                if fj is not fi and not fj.get("closed")
                and 0 < abs(fj["t0"] - fi["t0"]) <= close_dt_max
                and fj["e"] > 2.0 * fi["e"]]
        if not nbrs:
            continue
        fj = min(nbrs, key=lambda f: abs(f["t0"] - fi["t0"]))
        rows = top_channels(fi["stk"])
        if not rows:
            continue
        stats["n_close_tried"] += 1
        prs = np.stack([fi["stk"][c][ch] for _, c, ch in rows])
        brs = np.stack([np.asarray(base[c], np.float64)[ch]
                        - _shift(fi["stk"][c], fi["t0"], n_ticks)[ch]
                        - actual[c][ch] for _, c, ch in rows])
        wrs = np.stack([w_all[c][ch] for _, c, ch in rows])
        curve = weighted_scan(prs, brs, wrs)
        i_cur = int(round(min(max(fi["t0"], 0), SEARCH)))
        i_nb = int(round(min(max(fj["t0"], 0), SEARCH)))
        # dissolve if the neighbor's time explains this family's light nearly
        # as well (the family's own slot was a ridge artifact)
        if curve[i_nb] < close_ratio * curve[i_cur]:
            tr = truth_of(fi["mem"])
            was_ok = tr is not None and abs(fi["t0"] - tr) <= CUT
            apply_move(fi, float(i_nb))
            fi["closed"] = True
            stats["n_closed"] += 1
            stats["audit"].append({"op": "close", "e": fi["e"], "truth": tr,
                                   "was_ok": was_ok,
                                   "now_ok": (tr is not None
                                              and abs(fi["t0"] - tr) <= CUT)})

    ns["baseImage"] = base
    ns["hit_timestamps"] = ts.astype(np.float32)
    return stats
