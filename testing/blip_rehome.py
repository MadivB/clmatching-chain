"""Stage 4.7 blip re-homing (flag --blip-rehome): two operators on 0.5-20 MeV clusters, run after
swap-fix (4.5) and before force coverage (4.9).  Clusters >= 20 MeV are never touched.

(1) CONTACT certificate.  A small cluster whose hits touch (<= contact_cm) hits of placed clusters
    >= big_mev, all of which share one time (spread <= 3 ticks), takes that time.  This is the joint's
    connectivity certificate applied everywhere, including the shower TPCs that the joint never sees
    (there the 4.1 matrix places embedded fragments with no spatial prior).
    Offline on the v1.3 cert dumps (63 events): fixed 2.23 / broken 0.49 GeV at 2 cm.

(2) ISOLATED re-homing.  A displaced deposit whose parent left nothing big in this TPC at that time
    is pulled by the chi2 to a bright slot (a big family's 17 % model error pays as much as its own
    quiet hole).  Candidates are the times of placed big clusters in OTHER TPCs within cand_cm of the
    cluster at which its OWN TPC holds no big family (quiet there).  Decision on the local channels of
    the summed template of the small clusters within group_cm (scale-free pattern match):
      move iff  cos(best candidate) >= cos_min  and  > cos(current) + dcos_min,
                amplitude ratio (template / positive residual) in [0.5, 2],
                distance to the current host family >= host_cm (spray sits close to its parent),
                own light >= own_min x (0.17 x others' predicted light) at the current slot
                (a deposit buried deep in its host's glare is the host's spray).
    Decisions are taken on the incoming state (static), then committed together.
    Offline (candidate gates chosen on files 1-3, checked on files 4-5): ~1.6-2.5 fixed per broken.

(1b) NEAR-contact (near=True, flag --blip-near).  No placed big hit within contact_cm, but the nearest
    placed big hit is within near_cm; nothing big at the current time within near_host_cm (any TPC); the
    light does not prefer the current slot (local-channel cos margin <= near_dcos) and the deposit would
    be buried in glare at the new slot (own light <= near_own_max x 0.17 x others) -> nearest big time.
(3) RE-ATTACH strays (reattach=True, flag --blip-reattach), decided on the state after (1)/(1b)/(2) are
    committed.  A templated small cluster whose current slot has no co-timed big host nearby (own TPC:
    none within host_cm; any TPC: none within host_any_cm) takes the time of the placed big cluster of its
    own TPC (other time, within ra_cand_cm, not in contact) minimising own(t) x d, own(t) = own light /
    (0.17 x base light on its support channels at t), iff that score <= od_max [cm].
Offline on files 1-3 (thresholds) / 4-5 (holdout), each over the certified stage: near +0.054/+0.047 pp,
re-attach +0.093/+0.082 pp all-TPC.

No truth enters any decision; truth_t0 (optional) only fills the audit.
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
from scipy.spatial import cKDTree

PEAK = 105
T_MAX = 894
SYS_FRAC = 0.17          # measured per-channel light-model systematic (V200 s.6)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def run_blip_rehome(*, ts: np.ndarray, labels: np.ndarray, energy: np.ndarray, xyz: np.ndarray,
                    tpc: np.ndarray, amp: dict, actual: np.ndarray, base: np.ndarray,
                    render: Callable[[int, int, float, float], None],
                    contact: bool = True, isolated: bool = True,
                    small_min: float = 0.5, small_max: float = 20.0, big_mev: float = 20.0,
                    contact_cm: float = 2.0, cand_cm: float = 150.0, group_cm: float = 30.0,
                    cos_min: float = 0.9, dcos_min: float = 0.1, host_cm: float = 20.0,
                    own_min: float = 0.1, truth_t0: np.ndarray | None = None,
                    near: bool = False, near_cm: float = 15.0, near_host_cm: float = 20.0,
                    near_dcos: float = 0.2, near_own_max: float = 0.3,
                    reattach: bool = False, host_any_cm: float = 10.0, ra_cand_cm: float = 40.0,
                    od_max: float = 1.25) -> dict:
    """ts is modified in place (hit times); render(L, c, t_old, t_new) must move cluster L's light in
    TPC c from t_old to t_new in `base` (t_old/t_new may be None for add-only / remove-only).
    amp[(L, c)] = per-channel predicted peak amplitude (120,) for templated (label, TPC) pairs."""
    t0 = time.time()
    labels = np.asarray(labels).astype(np.int64)
    ul, inv = np.unique(labels, return_inverse=True)
    labE = np.bincount(inv, weights=np.asarray(energy, np.float64))
    cE = labE[inv]
    lab_hits = {int(L): np.where(inv == k)[0] for k, L in enumerate(ul) if L >= 0}
    Emap = {int(L): float(labE[k]) for k, L in enumerate(ul)}
    assigned = np.isfinite(ts) & (ts >= 0)

    def lab_time(L):
        idx = lab_hits[L]
        v = ts[idx]; v = v[np.isfinite(v) & (v >= 0)]
        return float(np.median(v)) if v.size else None

    big = assigned & (cE >= big_mev) & (labels >= 0)
    bxyz = xyz[big]; bts = np.round(ts[big]); btpc = tpc[big]; blab = labels[big]
    btree = cKDTree(bxyz) if big.any() else None
    small_labels = [L for L in lab_hits if small_min <= Emap[L] < small_max]
    stats = {"n_small": len(small_labels), "contact_moved": 0, "contact_agree": 0, "contact_ambiguous": 0,
             "iso_eligible": 0, "iso_moved": 0, "e_contact": 0.0, "e_iso": 0.0,
             "near_moved": 0, "e_near": 0.0, "reattach_moved": 0, "e_reattach": 0.0}
    audit = []
    anchored = set()
    touched = set()
    moves = []   # (L, t_old, t_new, op)
    tmpl_tpcs: dict[int, list] = {}
    for (L, c) in amp:
        tmpl_tpcs.setdefault(int(L), []).append(int(c))

    # ---------------- (1) contact certificate ----------------
    if contact and btree is not None:
        for L in small_labels:
            idx = lab_hits[L]
            lst = btree.query_ball_point(xyz[idx], contact_cm)
            hit = sorted({j for l in lst for j in l})
            if not hit:
                continue
            touched.add(L)
            times = bts[hit]
            if np.ptp(times) > 3:
                stats["contact_ambiguous"] += 1
                continue
            anchored.add(L)
            t_new = float(np.median(times))
            t_old = lab_time(L)
            if t_old is not None and abs(t_new - t_old) <= 3:
                stats["contact_agree"] += 1
                continue
            moves.append((L, t_old, t_new, "contact"))

    # ---------------- (1b) near-contact ----------------
    if near and btree is not None:
        for L in small_labels:
            if L in touched:
                continue
            cs = tmpl_tpcs.get(L, [])
            t_cur = lab_time(L)
            if t_cur is None or len(cs) != 1:
                continue
            X = xyz[lab_hits[L]]
            dd, jj = btree.query(X, k=1, distance_upper_bound=near_cm)
            if not np.isfinite(dd).any():
                continue
            t_new = float(bts[jj[int(np.argmin(dd))]])
            if abs(t_new - t_cur) <= 3:
                continue
            lst = btree.query_ball_point(X, near_host_cm)
            cand = [j for l in lst for j in l]
            if cand and np.any(np.abs(bts[np.asarray(cand)] - round(t_cur)) <= 3):
                continue
            T = cs[0]; s_ = np.asarray(amp[(L, T)], np.float64)
            jc = int(round(t_cur)) + PEAK; jn = int(round(t_new)) + PEAK
            if not (0 <= jc < base.shape[2] and 0 <= jn < base.shape[2]):
                continue
            loc = s_ >= 0.1 * s_.max()
            if loc.sum() >= 2:
                rc = np.clip(actual[T, loc, jc] - base[T, loc, jc] + s_[loc], 0, None)
                rn = np.clip(actual[T, loc, jn] - base[T, loc, jn], 0, None)
                if _cos(s_[loc], rc) - _cos(s_[loc], rn) > near_dcos:
                    continue
            sup = s_ > 0
            if s_[sup].sum() / max(SYS_FRAC * float(base[T, sup, jn].sum()), 1.0) > near_own_max:
                continue
            anchored.add(L)
            moves.append((L, t_cur, t_new, "near"))

    # ---------------- (2) isolated re-homing (static decisions on the incoming state) ----------------
    if isolated and btree is not None:
        # templated single-TPC small clusters, per TPC
        per_tpc: dict[int, list] = {}
        for L in small_labels:
            cs = tmpl_tpcs.get(L, [])
            if len(cs) != 1 or L in anchored:
                continue
            t_cur = lab_time(L)
            if t_cur is None or not (0 <= t_cur <= T_MAX):
                continue
            idx = lab_hits[L]
            per_tpc.setdefault(cs[0], []).append(dict(L=L, t=t_cur, s=np.asarray(amp[(L, cs[0])], np.float64),
                                                      c=xyz[idx].mean(axis=0)))
        reco_big_by_tpc = {int(T): np.unique(bts[btpc == T]) for T in np.unique(btpc)}
        for T, cl in per_tpc.items():
            own_big = reco_big_by_tpc.get(T, np.zeros(0))
            ownmask = btpc == T
            cen = np.array([c["c"] for c in cl]); tree = cKDTree(cen)
            for k, c in enumerate(cl):
                jR = int(round(c["t"])) + PEAK
                sup = c["s"] > 0
                S = c["s"][sup].sum()
                others = max(float(base[T, sup, jR].sum()) - S, 0.0)
                own_ratio = S / max(SYS_FRAC * others, 1.0)
                hm = ownmask & (np.abs(bts - round(c["t"])) <= 3)
                d_host = float(np.min(np.linalg.norm(bxyz[hm] - c["c"], axis=1))) if hm.any() else 1e9
                if d_host < host_cm or own_ratio < own_min:
                    continue
                idx0 = np.asarray(btree.query_ball_point(c["c"], cand_cm), dtype=np.int64)
                if idx0.size == 0:
                    continue
                idx0 = idx0[btpc[idx0] != T]
                cand = []
                for t in (np.unique(bts[idx0]) if idx0.size else []):
                    if abs(t - c["t"]) <= 10 or not (0 <= t <= T_MAX):
                        continue
                    if own_big.size and np.min(np.abs(own_big - t)) <= 5:
                        continue
                    cand.append(int(t))
                if not cand:
                    continue
                stats["iso_eligible"] += 1
                g = list(tree.query_ball_point(c["c"], group_cm))
                Ssum = np.sum([cl[j]["s"] for j in g], axis=0)
                loc = Ssum >= 0.1 * Ssum.max()
                if loc.sum() < 2:
                    continue
                sv = Ssum[loc]

                def resid(t):
                    jj = int(round(t)) + PEAK
                    r = actual[T, loc, jj] - base[T, loc, jj]
                    for q in g:
                        if abs(cl[q]["t"] - t) <= 1.5:
                            r = r + cl[q]["s"][loc]
                    return np.clip(r, 0, None)

                ccur = _cos(sv, resid(c["t"]))
                best = None
                for t in cand:
                    rp = resid(t)
                    ratio = sv.sum() / max(float(rp.sum()), 1.0)
                    if not (0.5 <= ratio <= 2.0):
                        continue
                    cp = _cos(sv, rp)
                    if best is None or cp > best[0]:
                        best = (cp, t)
                if best is None:
                    continue
                cp, t_new = best
                if cp >= cos_min and cp > ccur + dcos_min:
                    moves.append((c["L"], c["t"], float(t_new), "iso"))

    # ---------------- commit (1)/(1b)/(2), then (3) on the committed state ----------------
    tr = None if truth_t0 is None else np.asarray(truth_t0, np.float64)
    moves = _commit(moves, ts, lab_hits, tmpl_tpcs, render)
    if reattach and btree is not None:
        moves += _commit(_reattach_moves(ts=ts, labels=labels, cE=cE, xyz=xyz, tpc=tpc, amp=amp, base=base,
                                         lab_hits=lab_hits, Emap=Emap, small_min=small_min, small_max=small_max,
                                         big_mev=big_mev, contact_cm=contact_cm, host_cm=host_cm,
                                         host_any_cm=host_any_cm, cand_cm=ra_cand_cm, od_max=od_max),
                         ts, lab_hits, tmpl_tpcs, render)
    for L, t_old, t_new, op in moves:
        idx = lab_hits[L]
        e = Emap[L]
        stats[f"{op}_moved"] += 1
        stats[f"e_{op}"] += e
        row = {"label": int(L), "op": op, "e": e, "t_old": t_old, "t_new": t_new}
        if tr is not None:
            m = np.isfinite(tr[idx])
            if m.any():
                tt = float(np.median(tr[idx][m]))
                row["was_ok"] = bool(t_old is not None and abs(t_old - tt) <= 10)
                row["now_ok"] = bool(abs(t_new - tt) <= 10)
        audit.append(row)
    for op in ("contact", "near", "iso", "reattach"):
        rows = [a for a in audit if a["op"] == op and "was_ok" in a]
        stats[f"{op}_fixed_e"] = float(sum(a["e"] for a in rows if a["now_ok"] and not a["was_ok"]))
        stats[f"{op}_broke_e"] = float(sum(a["e"] for a in rows if a["was_ok"] and not a["now_ok"]))
    stats["time_s"] = time.time() - t0
    stats["audit"] = audit
    return stats


def _commit(moves, ts, lab_hits, tmpl_tpcs, render):
    """Write the new times and move the templated light in base."""
    for L, t_old, t_new, op in moves:
        ts[lab_hits[L]] = t_new
        for c in tmpl_tpcs.get(L, []):
            render(L, c, t_old, t_new)
    return list(moves)


def _reattach_moves(*, ts, labels, cE, xyz, tpc, amp, base, lab_hits, Emap, small_min, small_max, big_mev,
                    contact_cm, host_cm, host_any_cm, cand_cm, od_max):
    """Operator (3): stray re-attachment, static decisions on the (committed) incoming state."""
    assigned = np.isfinite(ts) & (ts >= 0)
    big = assigned & (cE >= big_mev) & (labels >= 0)
    bidx = np.where(big)[0]
    if bidx.size == 0:
        return []
    bts = np.round(ts[bidx]); btpc = tpc[bidx]; blab = labels[bidx]
    btree = cKDTree(xyz[bidx])
    rmax = max(cand_cm, host_cm, host_any_cm)
    moves = []
    for L, idx in lab_hits.items():
        E = Emap[L]
        if not (small_min <= E < small_max):
            continue
        T = int(np.bincount(tpc[idx]).argmax())
        if (L, T) not in amp:
            continue
        v = ts[idx]; v = v[np.isfinite(v) & (v >= 0)]
        if v.size == 0:
            continue
        tcur = float(np.median(np.round(v)))
        P = xyz[idx]; cen = P.mean(axis=0)
        ext = float(np.sqrt(((P - cen) ** 2).sum(axis=1).max()))
        cand = np.asarray(btree.query_ball_point(cen, rmax + ext), dtype=np.int64)
        if cand.size == 0:
            continue
        dd = cKDTree(P).query(xyz[bidx[cand]])[0]
        keep = dd <= rmax
        cand, dd = cand[keep], dd[keep]
        if cand.size == 0:
            continue
        co = np.abs(bts[cand] - tcur) <= 3
        own = btpc[cand] == T
        if np.any(co & own & (dd < host_cm)) or np.any(co & (dd < host_any_cm)):
            continue
        sel = own & ~co & (dd <= cand_cm)
        if not sel.any() or np.min(dd[sel]) <= contact_cm:
            continue
        c_s, d_s = cand[sel], dd[sel]
        o = np.lexsort((d_s, blab[c_s]))
        c_s, d_s = c_s[o], d_s[o]
        first = np.r_[True, blab[c_s][1:] != blab[c_s][:-1]]
        s_ = np.asarray(amp[(L, T)], np.float64); sup = s_ > 0; S = float(s_.sum())
        best = None
        for j, dj in zip(c_s[first], d_s[first]):
            t = float(bts[j])
            if not (0 <= t <= T_MAX):
                continue
            b = float(base[T, sup, int(t) + PEAK].sum())
            sc = S / max(SYS_FRAC * b, 1.0) * dj
            if best is None or sc < best[0]:
                best = (sc, t)
        if best is not None and best[0] <= od_max:
            moves.append((L, float(np.median(v)), best[1], "reattach"))   # t_old = rendered time (unrounded)
    return moves
