"""v0.3.0 family expansion: seeded absorption of small clusters into placed
interactions (see V030_DESIGN.md).

Operates on the batch namespace AFTER Phase 2 + V2 (+police), where Phase 3
would run.  Mutates ns["baseImage"], ns["hit_timestamps"], ns["t0Candidates"].
Pure numpy/scipy; the geometry helpers are synthetic-testable.
"""
from __future__ import annotations

from typing import Any

import os

import numpy as np
from scipy.signal import fftconvolve
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

PEAK = 105
GAIN = 200.0
CLIP = 60780.0
SEARCH = 895
CUT = 10.0

# ---- tunables (fragmentation-/purity-biased defaults) ----
FAM_DT = float(os.environ.get('V04_FAM_DT', '2.0'))    # family time window; env override for assignment tuning
R_LINK = 3.0            # hit-graph edge radius [cm]
D_GEO_MAX = float(os.environ.get('V04_D_GEO_MAX', '20.0'))   # expansion reach [cm]; env override
BETA_GEO = 0.05         # score per cm of geodesic distance
TAU_ACCEPT = float(os.environ.get('V04_TAU_ACCEPT', '0.0'))  # assign only if score < TAU; env override
DISSENT_MARGIN = 0.5    # light-preference margin that counts as a dissent
DISSENT_FREEZE = 2      # user Q1: freeze expansion after 2 consecutive dissents
EJECT_DEPTH = 2
TABU_ITERS = 5
NEWT0_MIN_MEV = 15.0
NEWT0_KAPPA = 0.5       # label cost (per-MeV loss units) to open a new t0
NEWT0_SEP = 10.0
NOISE_R = 3.0           # final noise-hit attach radius [cm]
BAND_SPLIT_MEV = 10.0   # process >=10 MeV candidates first
REFRESH_GROWTH = 0.20   # re-run a family's Dijkstra after 20% hit growth
PLAIN_SCORE = False     # True: score = raw clipped-Poisson delta-chi2 only
                        # (no per-MeV normalisation, no geodesic term in the
                        # score; geodesic reach still gates eligibility)
LOSS_MODE = "chi2"      # v0.5.0: "cubic"/"signed2" signed residual loss;
                        # accept primitive becomes |L_after| - |L_before|
                        # (drive toward balance); see v050_loss.py


EDGE_PENALTY = False    # 2x2 boundary-hiding guard (HANDOFF_ND_threshold_association)
EDGE_TAIL_TICKS = 75


def _edge_penalty_vec(pred, w, max_shift):
    """chi2 charge for predicted light shifted past the window end.

    A cluster must not hide its light by sliding it out of the digitizer
    window: hidden template columns are scored against each channel's
    quiet-tail weight (mean w over the last EDGE_TAIL_TICKS). Suffix-summed
    once per scan; interior shifts (tail fully visible) get zero penalty.
    """
    n = pred.shape[1]
    wt = w[:, -EDGE_TAIL_TICKS:].mean(axis=1)
    suf = np.cumsum((pred * pred)[:, ::-1], axis=1)[:, ::-1]
    A = wt @ suf
    pen = np.zeros(max_shift + 1)
    s = np.arange(1, max_shift + 1)
    idx = n - s
    m = (idx >= 0) & (idx < n)
    pen[1:][m] = A[idx[m]]
    return pen


SCAN_TEMPLATE = None    # rank-1 fast path: set to the shared pulse template
                        # (avg_pulse/999) to collapse multi-channel scans to
                        # one 1-D correlation - exact same math, the channel
                        # sum factors out of the chi2. Verified per call.


SCAN_TCOLS = None       # 3 spot-check columns, set with SCAN_TEMPLATE


FS_STAT = np.zeros(4, np.int64)   # calls, full-fast, split, rows_bad


def _rank1_amp(pred, tmpl, tmax):
    """Per-ROW rank-1 check (clipped rows are flat-topped and fail; they get
    the generic path while conforming rows take the collapsed path)."""
    amp = pred.max(axis=1) / tmax
    cols = SCAN_TCOLS
    exp = amp[:, None] * tmpl[cols][None, :]
    ok_rows = np.all(np.abs(pred[:, cols] - exp) <= 1.0 + 2e-3 * np.abs(exp),
                     axis=1)
    return amp, ok_rows


def weighted_scan(pred, resid, w, max_shift=SEARCH):
    n = pred.shape[1]
    if SCAN_TEMPLATE is not None and pred.shape[1] == SCAN_TEMPLATE.size:
        tmpl = SCAN_TEMPLATE
        predf = np.asarray(pred, np.float64)
        amp, ok_rows = _rank1_amp(predf, tmpl, float(tmpl.max()))
        FS_STAT[0] += 1
        FS_STAT[3] += int((~ok_rows).sum())
        if ok_rows.all():
            FS_STAT[1] += 1
        else:
            FS_STAT[2] += 1
        if ok_rows.any():
            wo, ro, ao = w[ok_rows], resid[ok_rows], amp[ok_rows]
            w2 = (wo * np.square(ao)[:, None]).sum(axis=0)[None, :]
            wr = (wo * ro * ao[:, None]).sum(axis=0)[None, :]
            t1 = fftconvolve(w2, np.square(tmpl)[None, ::-1], mode="full", axes=1)[0, n - 1: n - 1 + max_shift + 1]
            t2 = fftconvolve(wr, tmpl[None, ::-1], mode="full", axes=1)[0, n - 1: n - 1 + max_shift + 1]
            out = t1 + 2.0 * t2 + float(np.sum(wo * ro * ro))
            if EDGE_PENALTY:
                wt = wo[:, -EDGE_TAIL_TICKS:].mean(axis=1)
                suf = np.cumsum(np.square(tmpl)[::-1])[::-1]
                A = float((wt * np.square(ao)).sum()) * suf
                pen = np.zeros(max_shift + 1)
                sh = np.arange(1, max_shift + 1)
                idx = n - sh
                m = (idx >= 0) & (idx < n)
                pen[1:][m] = A[idx[m]]
                out = out + pen
            if not ok_rows.all():
                pb, rb, wb = predf[~ok_rows], resid[~ok_rows], w[~ok_rows]
                t1b = fftconvolve(wb, (pb * pb)[:, ::-1], mode="full", axes=1)[:, n - 1: n - 1 + max_shift + 1].sum(axis=0)
                t2b = fftconvolve(wb * rb, pb[:, ::-1], mode="full", axes=1)[:, n - 1: n - 1 + max_shift + 1].sum(axis=0)
                out = out + t1b + 2.0 * t2b + float(np.sum(wb * rb * rb))
                if EDGE_PENALTY:
                    out = out + _edge_penalty_vec(pb, wb, max_shift)
            return out
    t1 = fftconvolve(w, (pred * pred)[:, ::-1], mode="full", axes=1)[:, n - 1: n - 1 + max_shift + 1].sum(axis=0)
    t2 = fftconvolve(w * resid, pred[:, ::-1], mode="full", axes=1)[:, n - 1: n - 1 + max_shift + 1].sum(axis=0)
    out = t1 + 2.0 * t2 + float(np.sum(w * resid * resid))
    if EDGE_PENALTY:
        out = out + _edge_penalty_vec(pred, w, max_shift)
    return out


# --------------------------------------------------------------------------
# geometry: radius graph + multi-source geodesics (synthetic-testable)
# --------------------------------------------------------------------------

def build_hit_graph(xyz: np.ndarray, r_link: float = R_LINK):
    """Sparse symmetric radius graph over hit positions (edge weight = cm)."""
    tree = cKDTree(xyz)
    pairs = tree.query_pairs(r_link, output_type="ndarray")
    if pairs.size == 0:
        n = xyz.shape[0]
        return coo_matrix((n, n)).tocsr(), tree
    d = np.linalg.norm(xyz[pairs[:, 0]] - xyz[pairs[:, 1]], axis=1)
    n = xyz.shape[0]
    g = coo_matrix((np.concatenate([d, d]),
                    (np.concatenate([pairs[:, 0], pairs[:, 1]]),
                     np.concatenate([pairs[:, 1], pairs[:, 0]]))), shape=(n, n))
    return g.tocsr(), tree

def family_geodesics(graph, source_idx: np.ndarray, limit: float = D_GEO_MAX):
    """Geodesic distance from a family's hits to every hit (inf beyond limit)."""
    if source_idx.size == 0:
        return np.full(graph.shape[0], np.inf)
    return dijkstra(graph, directed=False, indices=source_idx,
                    min_only=True, limit=limit)


# --------------------------------------------------------------------------
# main pass
# --------------------------------------------------------------------------

def run_family_expansion(ns: dict[str, Any], *, verbose: bool = False,
                         params: dict[str, float] | None = None) -> dict[str, Any]:
    p = {k: v for k, v in globals().items() if k.isupper()}
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

    # ---------- families: union-find of placed labels over (dt<=2, shared TPC)
    lab_t0, lab_tpcs = {}, {}
    for lab in np.unique(labels[assigned & (labels >= 0)]):
        m = assigned & (labels == lab)
        lab_t0[int(lab)] = float(np.median(ts[m]))
        lab_tpcs[int(lab)] = set(int(t) for t in np.unique(hit_tpc[m]) if t < n_light)
    placed = sorted(lab_t0)
    parent = {l: l for l in placed}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, a in enumerate(placed):
        for b in placed[i + 1:]:
            if abs(lab_t0[a] - lab_t0[b]) <= p["FAM_DT"] and lab_tpcs[a] & lab_tpcs[b]:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra
    fam_members: dict[int, list[int]] = {}
    for l in placed:
        fam_members.setdefault(find(l), []).append(l)

    families = []
    for root, mem in fam_members.items():
        hits = np.flatnonzero(assigned & np.isin(labels, mem))
        w = np.array([e_pos[labels == l].sum() for l in mem])
        t0v = np.array([lab_t0[l] for l in mem])
        families.append({
            "id": len(families), "t0": float(np.average(t0v, weights=np.maximum(w, 1e-9))),
            "labels": set(int(l) for l in mem), "hits": hits,
            "tpcs": set().union(*(lab_tpcs[l] for l in mem)),
            "seed_labels": set(int(l) for l in mem),   # never evicted
            "dissent": 0, "frozen": False, "n0_hits": hits.size,
            "geo": None,
        })

    # ---------- candidates: unassigned clusters with images
    cands = {}
    for lab in np.unique(labels[labels >= 0]):
        m = labels == lab
        if np.any(assigned[m]):
            continue
        tpcs = [c for c in range(n_light) if (int(lab), int(c)) in image_maps]
        if not tpcs:
            continue
        e_c = float(e_pos[m].sum())
        if e_c <= 0:
            continue
        if e_c > p.get("E_MAX_CAND", np.inf):
            continue          # early-expansion mode: big clusters only absorb SMALLER ones
        sup = {}
        for c in tpcs:
            img = np.asarray(image_maps[(int(lab), int(c))])
            ch = np.flatnonzero(img.max(axis=1) > 1.0)
            if ch.size:
                sup[int(c)] = ch
        if not sup:
            continue
        cands[int(lab)] = {"hits": np.flatnonzero(m), "e": e_c, "tpcs": tpcs, "sup": sup}

    graph, _ = build_hit_graph(xyz, p["R_LINK"])
    for f in families:
        f["geo"] = family_geodesics(graph, f["hits"], p["D_GEO_MAX"])

    def d_geo(c, f):
        g = f["geo"][cands[c]["hits"]]
        return float(np.min(g)) if g.size else np.inf

    def dlight(c, f):
        """delta chi2 (clipped, Poisson-weighted) of adding c at f's t0.

        Regular mode (PLAIN_SCORE): the raw delta chi2 — the pipeline's
        standard loss.  Legacy mode: normalised per MeV of candidate energy."""
        t0 = f["t0"]
        if p.get("LOSS_MODE", "chi2") != "chi2":
            import v050_loss as _v050
            la = lb = 0.0
            for tc, ch in cands[c]["sup"].items():
                img = np.asarray(image_maps[(int(c), int(tc))], np.float64)[ch]
                sh = _shift(img, t0, n_ticks)
                B = np.asarray(base[tc], np.float64)[ch]
                A = actual[tc][ch]
                W = w_all[tc][ch]
                la += _v050.point_loss(B + sh, A, W, p["LOSS_MODE"])
                lb += _v050.point_loss(B, A, W, p["LOSS_MODE"])
            tot = abs(la) - abs(lb)      # improvement toward balance
            if p["PLAIN_SCORE"]:
                return tot
            return tot / max(cands[c]["e"], 1e-9)
        tot = 0.0
        for tc, ch in cands[c]["sup"].items():
            img = np.asarray(image_maps[(int(c), int(tc))], np.float64)[ch]
            sh = _shift(img, t0, n_ticks)
            B = np.asarray(base[tc], np.float64)[ch]
            A = actual[tc][ch]
            W = w_all[tc][ch]
            tot += float(np.sum(W * ((np.minimum(B + sh, CLIP) - A) ** 2
                                     - (np.minimum(B, CLIP) - A) ** 2)))
        if p["PLAIN_SCORE"]:
            return tot
        return tot / max(cands[c]["e"], 1e-9)

    def overflow_fires(c, f):
        tick = int(round(f["t0"])) + PEAK
        if not (0 <= tick < n_ticks):
            return False
        for tc, ch in cands[c]["sup"].items():
            img = np.asarray(image_maps[(int(c), int(tc))], np.float64)
            own = _shift(img, f["t0"], n_ticks)[:, tick]
            pcol = np.minimum(np.asarray(base[tc], np.float64)[:, tick] + own, CLIP)
            a = actual[tc][:, tick]
            ex = pcol - a
            z = ex / np.sqrt(GAIN * np.maximum(a, 100.0))
            if int(np.sum((ex >= 500.0) & (z > 2.0))) >= 3:
                return True
        return False

    def score(c, f):
        dg = d_geo(c, f)
        if dg > p["D_GEO_MAX"]:
            return np.inf, dg
        if p["PLAIN_SCORE"]:
            return dlight(c, f), dg
        return dlight(c, f) + p["BETA_GEO"] * dg, dg

    def add_to_base(c, t0, sign=1.0):
        for tc in cands[c]["tpcs"]:
            img = np.asarray(image_maps[(int(c), int(tc))], np.float64)
            base[tc] += (sign * _shift(img, t0, n_ticks)).astype(np.float32)

    stats = {"n_families0": len(families), "n_cands": len(cands), "assigned": [],
             "dissents": 0, "freezes": 0, "ejections": 0, "new_t0": 0,
             "noise_hits": 0, "left_for_matrix": 0}
    tabu: dict[tuple[int, int], int] = {}
    it = 0
    member_of: dict[int, int] = {}          # candidate label -> family id
    member_score: dict[int, float] = {}

    pool_all = sorted(cands, key=lambda c: -cands[c]["e"])
    bands = [[c for c in pool_all if cands[c]["e"] >= p["BAND_SPLIT_MEV"]],
             [c for c in pool_all if cands[c]["e"] < p["BAND_SPLIT_MEV"]]]

    banned: set[tuple[int, int]] = set()   # (candidate, family) pairs vetoed by dissent
    for band in bands:
        pool = set(band)
        while pool:
            it += 1
            best_pick = None
            for c in pool:
                rows = []
                for f in families:
                    if f["frozen"] or (c, f["id"]) in banned:
                        continue
                    s, dg = score(c, f)
                    if np.isfinite(s):
                        rows.append((s, f["id"], dg))
                if not rows:
                    continue
                rows.sort()
                s1 = rows[0][0]
                if s1 >= p["TAU_ACCEPT"]:
                    continue
                s2 = rows[1][0] if len(rows) > 1 else p["TAU_ACCEPT"]
                regret = s2 - s1
                if best_pick is None or regret > best_pick[0]:
                    best_pick = (regret, c, rows)
            if best_pick is None:
                break
            _, c, rows = best_pick
            fbest = families[rows[0][1]]

            # dissent check: light term alone prefers a different family clearly.
            # The candidate is BANNED from this family (stays available to its
            # light-preferred one); 2 consecutive dissents freeze the family.
            if len(rows) > 1:
                lbest = sorted((dlight(c, families[fid]), fid) for _, fid, _ in rows[:3])
                _marg = p["DISSENT_MARGIN"] * (cands[c]["e"] if p["PLAIN_SCORE"] else 1.0)
                if lbest[0][1] != fbest["id"] and \
                        dlight(c, fbest) - lbest[0][0] > _marg:
                    fbest["dissent"] += 1
                    stats["dissents"] += 1
                    banned.add((c, fbest["id"]))
                    if fbest["dissent"] >= p["DISSENT_FREEZE"]:
                        fbest["frozen"] = True
                        stats["freezes"] += 1
                    continue

            # overflow gate -> bounded ejection attempt (depth<=2), else skip
            if overflow_fires(c, fbest):
                did = _try_ejection(c, fbest, families, cands, member_of, member_score,
                                    tabu, it, score, dlight, add_to_base, ts, labels,
                                    p, stats)
                pool.discard(c)
                if not did:
                    continue
            else:
                add_to_base(c, fbest["t0"])
                ts[cands[c]["hits"]] = np.float32(fbest["t0"])
                fbest["hits"] = np.concatenate([fbest["hits"], cands[c]["hits"]])
                fbest["tpcs"] |= set(cands[c]["tpcs"])
                member_of[c] = fbest["id"]
                member_score[c] = rows[0][0]
                fbest["dissent"] = 0
                stats["assigned"].append({"label": int(c), "fam": fbest["id"],
                                          "t0": fbest["t0"], "e": cands[c]["e"],
                                          "score": rows[0][0], "d_geo": rows[0][2]})
                pool.discard(c)
                if fbest["hits"].size > (1 + p["REFRESH_GROWTH"]) * fbest["n0_hits"]:
                    fbest["geo"] = family_geodesics(graph, fbest["hits"], p["D_GEO_MAX"])
                    fbest["n0_hits"] = fbest["hits"].size

            # new-t0 opening for energetic, family-rejected candidates
        # after each band, try opening new t0s from the still-unassigned bright tail
        for c in sorted(pool, key=lambda cc: -cands[cc]["e"]) if p.get("OPEN_NEW_T0", True) else []:
            if cands[c]["e"] < p["NEWT0_MIN_MEV"]:
                break
            t_new, gain = _own_scan_best(c, cands, image_maps, base, actual, w_all,
                                         n_ticks, per_mev=not p["PLAIN_SCORE"],
                                         loss_mode=p.get("LOSS_MODE", "chi2"))
            if t_new is None:
                continue
            fam_scores = [score(c, f)[0] for f in families]
            s_best = min([s for s in fam_scores if np.isfinite(s)], default=np.inf)
            _kappa = p["NEWT0_KAPPA"] * (cands[c]["e"] if p["PLAIN_SCORE"] else 1.0)
            if gain < min(s_best, 0.0) - _kappa and \
                    all(abs(t_new - f["t0"]) > p["NEWT0_SEP"] for f in families):
                add_to_base(c, t_new)
                ts[cands[c]["hits"]] = np.float32(t_new)
                families.append({"id": len(families), "t0": float(t_new),
                                 "labels": {int(c)}, "hits": cands[c]["hits"].copy(),
                                 "tpcs": set(cands[c]["tpcs"]), "seed_labels": set(),
                                 "dissent": 0, "frozen": False,
                                 "n0_hits": cands[c]["hits"].size, "geo": None})
                families[-1]["geo"] = family_geodesics(graph, families[-1]["hits"],
                                                       p["D_GEO_MAX"])
                member_of[c] = families[-1]["id"]
                stats["new_t0"] += 1
                pool.discard(c)

    # ---------- noise hits: pure-geometry attach (no image)
    noise = np.flatnonzero((labels < 0) & ~assigned) if p.get("ASSIGN_NOISE", True) \
        else np.empty(0, dtype=np.int64)
    if noise.size:
        best_d = np.full(noise.size, np.inf)
        best_t = np.full(noise.size, np.nan)
        for f in families:
            g = f["geo"][noise]
            upd = g < best_d
            best_d[upd] = g[upd]
            best_t[upd] = f["t0"]
        ok = best_d <= p["NOISE_R"]
        ts[noise[ok]] = best_t[ok].astype(np.float32)
        stats["noise_hits"] = int(ok.sum())

    stats["left_for_matrix"] = int(len(cands) - len(member_of))
    stats["n_assigned"] = len(member_of)
    stats["e_assigned"] = float(sum(cands[c]["e"] for c in member_of))
    stats["member_labels"] = sorted(int(k) for k in member_of)
    ns["hit_timestamps"] = ts.astype(np.float32)
    ns["baseImage"] = base
    if verbose:
        print(f"[v030] fams {stats['n_families0']} cands {stats['n_cands']} "
              f"assigned {stats['n_assigned']} ({stats['e_assigned']:.0f} MeV) "
              f"ejections {stats['ejections']} new_t0 {stats['new_t0']} "
              f"noise {stats['noise_hits']}", flush=True)
    return stats


def _shift(img, t0, n_ticks):
    out = np.zeros_like(img, dtype=np.float64)
    s = int(round(t0))
    if 0 <= s < n_ticks:
        out[:, s:] = img[:, : n_ticks - s]
    return out


def _own_scan_best(c, cands, image_maps, base, actual, w_all, n_ticks,
                   per_mev=True, loss_mode="chi2"):
    prs, rrs, wrs = [], [], []
    for tc, ch in cands[c]["sup"].items():
        prs.append(np.asarray(image_maps[(int(c), int(tc))], np.float64)[ch])
        rrs.append(np.asarray(base[tc], np.float64)[ch] - actual[tc][ch])
        wrs.append(w_all[tc][ch])
    if loss_mode != "chi2":
        import v050_loss as _v050
        Pn, Rn, Wn = (np.concatenate(prs), np.concatenate(rrs), np.concatenate(wrs))
        curve = _v050.scan_curve(Pn, Rn, Wn, mode=loss_mode)
        t = int(np.argmin(np.abs(curve)))
        # gain toward balance vs not placing at all (coarse, unmasked)
        gain = float(abs(curve[t]) - abs(float(np.sum(Wn * Rn ** 3))
                                         if loss_mode == "cubic"
                                         else float(np.sum(Wn * np.sign(Rn) * Rn ** 2))))
    else:
        curve = weighted_scan(np.concatenate(prs), np.concatenate(rrs), np.concatenate(wrs))
        t = int(np.argmin(curve))
        # gain relative to not placing at all (per MeV in legacy mode)
        gain = float(curve[t] - np.sum(np.concatenate(wrs) * np.concatenate(rrs) ** 2))
    if per_mev:
        gain /= max(cands[c]["e"], 1e-9)
    return float(t), gain


def _try_ejection(c, f, families, cands, member_of, member_score, tabu, it,
                  score, dlight, add_to_base, ts, labels, p, stats):
    """Depth<=2 ejection: evict the most reluctant expansion member(s) of f,
    insert c, re-place evictees at their 2nd family or unassign.  Accept only
    if the summed per-MeV scores improve.  Tabu prevents comebacks."""
    exp_members = [m for m, fid in member_of.items()
                   if fid == f["id"] and tabu.get((m, f["id"]), -99) + p["TABU_ITERS"] < it]
    if not exp_members:
        return False
    reluct = sorted(exp_members, key=lambda m: -(member_score.get(m, 0.0)))[: p["EJECT_DEPTH"]]
    s_c, _ = score(c, f)
    if not np.isfinite(s_c):
        return False
    for m in reluct:
        # tentative: remove m, check c no longer overflows and total improves
        add_to_base(m, f["t0"], sign=-1.0)
        alt = sorted((score(m, g)[0], g["id"]) for g in families if g["id"] != f["id"])
        alt_s, alt_f = (alt[0] if alt and np.isfinite(alt[0][0]) else (0.0, None))
        old = member_score.get(m, 0.0)
        if s_c + min(alt_s, 0.0) < old - 1e-9:
            ts[cands[m]["hits"]] = np.nan
            del member_of[m]
            tabu[(m, f["id"])] = it
            f["hits"] = f["hits"][~np.isin(f["hits"], cands[m]["hits"])]
            add_to_base(c, f["t0"])
            ts[cands[c]["hits"]] = np.float32(f["t0"])
            f["hits"] = np.concatenate([f["hits"], cands[c]["hits"]])
            member_of[c] = f["id"]
            member_score[c] = s_c
            if alt_f is not None and alt_s < 0.0:
                g = families[alt_f]
                add_to_base(m, g["t0"])
                ts[cands[m]["hits"]] = np.float32(g["t0"])
                g["hits"] = np.concatenate([g["hits"], cands[m]["hits"]])
                member_of[m] = alt_f
                member_score[m] = alt_s
            stats["ejections"] += 1
            return True
        add_to_base(m, f["t0"])   # rollback
    return False


def force_full_coverage(ns: dict[str, Any], *, d_geo_family: float = 30.0,
                        noise_geo: float = 15.0, verbose: bool = False) -> dict[str, Any]:
    """Coverage >= 99% mandate: assign EVERYTHING still unassigned.

    Per remaining cluster (energy-descending):
      - with a light image: place at the better (absolute loss) of the nearest
        families' t0 vs its OWN full-range scan minimum — a loss INCREASE is
        accepted (correct placements can raise chi2 under the ML amplitude
        systematics).  The base image is updated so later choices see it.
      - without an image (below the prediction threshold): pure geometry —
        the geodesically nearest family's t0, Euclidean-nearest assigned hit
        as fallback.
    Noise hits: nearest family within ``noise_geo`` through the cloud, else
    the t0 of the Euclidean-nearest assigned hit."""
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
    # families over current assignments
    lab_t0, lab_tpcs = {}, {}
    for lab in np.unique(labels[assigned & (labels >= 0)]):
        m = assigned & (labels == lab)
        lab_t0[int(lab)] = float(np.median(ts[m]))
        lab_tpcs[int(lab)] = set(int(t) for t in np.unique(hit_tpc[m]) if t < n_light)
    placed = sorted(lab_t0)
    parent = {l: l for l in placed}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, a in enumerate(placed):
        for b in placed[i + 1:]:
            if abs(lab_t0[a] - lab_t0[b]) <= FAM_DT and lab_tpcs[a] & lab_tpcs[b]:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra
    fam_members: dict[int, list[int]] = {}
    for l in placed:
        fam_members.setdefault(find(l), []).append(l)
    graph, _ = build_hit_graph(xyz, R_LINK)
    fams = []
    for mem in fam_members.values():
        hits = np.flatnonzero(assigned & np.isin(labels, mem))
        w = np.array([e_pos[labels == l].sum() for l in mem])
        t0v = np.array([lab_t0[l] for l in mem])
        fams.append({"t0": float(np.average(t0v, weights=np.maximum(w, 1e-9))),
                     "geo": family_geodesics(graph, hits, max(d_geo_family, noise_geo))})

    stats = {"n_img": 0, "n_img_own": 0, "n_noimg": 0, "n_noise": 0,
             "e_img": 0.0, "e_noimg": 0.0, "e_noise": 0.0, "n_unreached": 0}
    todo = []
    for lab in np.unique(labels[labels >= 0]):
        m = labels == lab
        if np.any(assigned[m]):
            continue
        todo.append((float(e_pos[m].sum()), int(lab), np.flatnonzero(m)))
    todo.sort(reverse=True)

    asg_idx = np.flatnonzero(assigned)
    tree_assigned = cKDTree(xyz[asg_idx]) if asg_idx.size else None

    for e_c, lab, hidx in todo:
        tpcs = [c for c in range(n_light) if (int(lab), int(c)) in image_maps]
        sup = {}
        for c in tpcs:
            img = np.asarray(image_maps[(int(lab), int(c))])
            ch = np.flatnonzero(img.max(axis=1) > 1.0)
            if ch.size:
                sup[int(c)] = ch
        # nearest families through the cloud
        fam_d = [(float(np.min(f["geo"][hidx])) if hidx.size else np.inf, fi)
                 for fi, f in enumerate(fams)]
        fam_d.sort()
        if sup:
            # absolute loss of placing at t: sum w*(clip(B+sh)-A)^2 - (clip(B)-A)^2
            def loss_at(t0):
                tot = 0.0
                for tc, ch in sup.items():
                    img = np.asarray(image_maps[(int(lab), int(tc))], np.float64)[ch]
                    sh = _shift(img, t0, n_ticks)
                    B = np.asarray(base[tc], np.float64)[ch]
                    A = actual[tc][ch]
                    W = w_all[tc][ch]
                    tot += float(np.sum(W * ((np.minimum(B + sh, CLIP) - A) ** 2
                                             - (np.minimum(B, CLIP) - A) ** 2)))
                return tot
            t_own, _g = _own_scan_best_abs(lab, sup, image_maps, base, actual, w_all)
            options = [(loss_at(t_own), t_own, "own")]
            for d, fi in fam_d[:3]:
                if d <= d_geo_family:
                    options.append((loss_at(fams[fi]["t0"]), fams[fi]["t0"], "fam"))
            options.sort()
            _, t_pick, mode = options[0]
            for tc in tpcs:
                img = np.asarray(image_maps[(int(lab), int(tc))], np.float64)
                base[tc] += _shift(img, t_pick, n_ticks).astype(np.float32)
            ts[hidx] = np.float32(t_pick)
            stats["n_img"] += 1
            stats["e_img"] += e_c
            if mode == "own":
                stats["n_img_own"] += 1
        else:
            # no image: geometry only
            if fam_d and np.isfinite(fam_d[0][0]):
                ts[hidx] = np.float32(fams[fam_d[0][1]]["t0"])
            elif tree_assigned is not None:
                _, nn = tree_assigned.query(xyz[hidx], k=1)
                ts[hidx] = ts[asg_idx[nn]].astype(np.float32)
            else:
                stats["n_unreached"] += 1
                continue
            stats["n_noimg"] += 1
            stats["e_noimg"] += e_c

    # noise hits
    noise = np.flatnonzero((labels < 0) & ~(np.isfinite(ts) & (ts >= 0)))
    if noise.size:
        best_d = np.full(noise.size, np.inf)
        best_t = np.full(noise.size, np.nan)
        for f in fams:
            g = f["geo"][noise]
            upd = g < best_d
            best_d[upd] = g[upd]
            best_t[upd] = f["t0"]
        near = best_d <= noise_geo
        ts[noise[near]] = best_t[near].astype(np.float32)
        far = noise[~near]
        if far.size and tree_assigned is not None:
            _, nn = tree_assigned.query(xyz[far], k=1)
            ts[far] = ts[asg_idx[nn]].astype(np.float32)
        stats["n_noise"] = int(noise.size)
        stats["e_noise"] = float(e_pos[noise].sum())

    ns["hit_timestamps"] = ts.astype(np.float32)
    ns["baseImage"] = base
    if verbose:
        print(f"[force] img {stats['n_img']} ({stats['e_img']:.0f} MeV, "
              f"{stats['n_img_own']} at own scan) | no-img {stats['n_noimg']} "
              f"({stats['e_noimg']:.0f} MeV) | noise {stats['n_noise']}", flush=True)
    return stats


def _own_scan_best_abs(lab, sup, image_maps, base, actual, w_all):
    prs, rrs, wrs = [], [], []
    for tc, ch in sup.items():
        prs.append(np.asarray(image_maps[(int(lab), int(tc))], np.float64)[ch])
        rrs.append(np.asarray(base[tc], np.float64)[ch] - actual[tc][ch])
        wrs.append(w_all[tc][ch])
    curve = weighted_scan(np.concatenate(prs), np.concatenate(rrs), np.concatenate(wrs))
    t = int(np.argmin(curve))
    return float(t), float(curve[t])


__all__ = ["run_family_expansion", "force_full_coverage", "build_hit_graph",
           "family_geodesics", "weighted_scan"]
