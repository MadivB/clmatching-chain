"""Stage 4.85 (ND): delayed-daughter / Michel re-timing, piece-first (kernel shared by the harness and the offline study
testing/v14_michel_nd_fast.py; see that file's docstring and VERSION_HISTORY 2026-10-02 for the design and numbers).

Physics: a Michel electron is a short branch (5-55 MeV, <= 16 cm) LEAVING the Bragg peak at the end of a stopping muon,
with its own flash 12-900 ticks later.  Steps per spill:
  1. dense points (Bragg peaks / vertices) in every cluster >= dense_cluster_min_mev;
  2. candidate pieces: free tips (>= 3 cm from any dense point) of clusters >= host_min_mev whose geodesic growth is
     stopped by a dense ball within 16 cm; whole templated clusters of 5-80 MeV;
  3. one batched perceiver call for the probe pieces (L <= 5 cm) through predict_fn;
  4. per piece: matched-filter scan of its light over all later times (t_c + 12 .. t_c + delay_max) on the residual with
     its prompt light removed; the 2 best minima are tested: mf, alpha, apc (prompt deficit), cov, sat;
  5. passing probes refined over all lengths (second batch); one move per cluster; base updated.
predict_fn(items) with items = [(key, hit_index_array), ...] must return {key: (n_ch,) amplitude pattern in the
piece's light TPC} (vetoed channels zeroed by the caller).  Times in ticks (16 ns), template peak at PEAK.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Tuple

import numpy as np
from scipy.signal import fftconvolve
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

PEAK = 105; GAIN = 200.0; CUT = 10.0


class MichelNDConfig:
    late_min_ticks = 12.0; delay_max = 300.0; host_min_mev = 10.0; whole_min_mev = 5.0; whole_max_mev = 80.0
    link_cm = 3.0; tip_min_cm = 2.0; dense_r = 1.5; dense_sep = 3.0; dense_min_mev = 5.0; dense_rel = 1.5; dense_cluster_min_mev = 5.0
    free_cm = 3.0; dense_leak_mev = 2.0; piece_cm = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0); piece_cm_probe = 5.0
    piece_min_hits = 3; piece_min_mev = 5.0; piece_max_mev = 1e9; max_frac = 0.5; mf_window = (-3, 45); n_minima = 2
    mf_min = 0.8; mf_tol = 0.02; apc_max = -0.5; sat_max = 0.5; cov_min = 0.2; alpha_lo = 0.8; alpha_hi = 6.0
    support_frac = 0.9; require_bragg = True; min_tpc_hits = 6; host_extent_min_cm = 0.0; bragg_end_cm = 1e9; vox_cm = 1.0
    unexpl_gain_tol = 0.3; unexpl_min_frac = 0.1   # light at the destination must stay >= this fraction of the actual after the placed families may scale up by gain_tol
    whole_alpha_hi = 2.0                            # whole-cluster moves: amplitude ratio cap (bad whole moves read alpha 2.3-3.3)
    host_dedx_max = 1e9        # mean dE/dx of the host in the TPC (E / geodesic extent, MeV/cm); Michel hosts read 1.9-3.6 (stopping mu / pi)
    tip_face_min_cm = 0.0      # a tip within this distance of an OUTER face of the active volume is an exit, not a Michel end
    box = (-347.0, 347.0, -215.0, 82.0, 418.0, 913.0)


# ---------------------------------------------------------------- geometry
def skeleton_tips(X, link_cm, tip_min_cm):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree
    n = X.shape[0]
    pairs = cKDTree(X).query_pairs(link_cm, output_type="ndarray")
    if pairs.size == 0:
        return [], None
    d = np.linalg.norm(X[pairs[:, 0]] - X[pairs[:, 1]], axis=1) + 1e-6
    G = csr_matrix((np.concatenate([d, d]), (np.concatenate([pairs[:, 0], pairs[:, 1]]), np.concatenate([pairs[:, 1], pairs[:, 0]]))), shape=(n, n))
    d0 = dijkstra(G, indices=[0])[0]; d0[~np.isfinite(d0)] = -1; a = int(np.argmax(d0))
    da = dijkstra(G, indices=[a])[0]; da[~np.isfinite(da)] = -1; b = int(np.argmax(da))
    tips = {a, b}
    T = minimum_spanning_tree(G).tocoo()
    deg = np.bincount(np.concatenate([T.row, T.col]), minlength=n)
    adj: Dict[int, List[Tuple[int, float]]] = {}
    for i, j, wgt in zip(T.row, T.col, T.data):
        adj.setdefault(int(i), []).append((int(j), float(wgt))); adj.setdefault(int(j), []).append((int(i), float(wgt)))
    for leaf in np.flatnonzero(deg == 1):
        leaf = int(leaf); prev, cur, length = -1, leaf, 0.0
        while deg[cur] <= 2 and length < 100.0:
            nxt = [(j, wgt) for j, wgt in adj.get(cur, []) if j != prev]
            if not nxt:
                break
            prev, cur = cur, nxt[0][0]; length += nxt[0][1]
            if deg[cur] >= 3:
                break
        if length >= tip_min_cm:
            tips.add(leaf)
    return sorted(tips), G


def skeleton_vox(Xh, link_cm, tip_min_cm, vox):
    key = np.floor(Xh / vox).astype(np.int64); _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel(); cent = np.zeros((cnt.size, 3)); np.add.at(cent, inv, Xh); cent /= cnt[:, None]
    d2 = ((Xh - cent[inv]) ** 2).sum(1); rep = np.full(cnt.size, -1)
    for i in np.argsort(d2)[::-1]:
        rep[inv[i]] = i
    tips_r, G = skeleton_tips(Xh[rep], link_cm, tip_min_cm)
    if not tips_r:
        return [], None
    D_r = dijkstra(G, indices=tips_r); D_r[~np.isfinite(D_r)] = -1
    return [int(rep[t]) for t in tips_r], D_r[:, inv]


def dense_points(X, E, r, sep, min_mev, rel):
    tree = cKDTree(X); nb = tree.query_ball_point(X, r)
    ball = np.fromiter((E[n].sum() for n in nb), float, count=len(nb)); thr = max(min_mev, rel * float(np.median(ball)))
    order = np.argsort(-ball); order = order[ball[order] >= thr]
    supp = np.zeros(X.shape[0], bool); chosen = []
    for i in order:
        if supp[i]:
            continue
        chosen.append(int(i)); supp[tree.query_ball_point(X[i], sep)] = True
    return chosen


# ---------------------------------------------------------------- light helpers
def render(amp, t, n, T):
    s = np.arange(n); sh = np.interp(s - t, np.arange(T.size), T, left=0.0, right=0.0)
    return amp[:, None] * sh[None, :]


def support(amp, frac):
    o = np.argsort(-amp); c = np.cumsum(amp[o])
    if c[-1] <= 0:
        return np.zeros(0, int)
    k = int(np.searchsorted(c, frac * c[-1])) + 1
    return o[:max(k, 3)]


def wcos(a, b, w):
    a = a * w; b = np.clip(b, 0, None) * w
    da = float(np.sqrt((a * a).sum())); db = float(np.sqrt((b * b).sum()))
    return float((a * b).sum() / (da * db)) if da > 0 and db > 0 else 0.0


# ---------------------------------------------------------------- the stage
def run_michel_nd(*, xyz, e, tpc, labels, ts, amp, actual, base, veto, ceil, template, predict_fn: Callable, cfg: MichelNDConfig = None,
                  truth_t0=None, apply=True, debug_tag=None, debug_out=None):
    """xyz (n,3); e, tpc, labels (n); ts (n) final times, MODIFIED in place when apply; amp {(label, light_tpc): (n_ch,)};
    actual/base (n_light, n_ch, n_s) float32, base MODIFIED in place when apply; veto (n_light, n_ch) bool; ceil (n_light, n_ch);
    template (NT,) unit-peak pulse; predict_fn as in the module docstring.  Returns stats dict with 'moves' (per move:
    cluster, tpc, piece hit indices, t_from, t_to, scores, and truth energies when truth_t0 is given)."""
    cfg = cfg or MichelNDConfig(); t_start = time.time(); tim = {}
    T = np.asarray(template, np.float64); T2 = T ** 2; NT = T.size; kS = NT - 1
    X = np.asarray(xyz, np.float64); e = np.clip(np.asarray(e, np.float64), 0, None); tpc = np.asarray(tpc); labels = np.asarray(labels)
    n_light, n_ch, n_s = actual.shape
    ok = np.isfinite(truth_t0) if truth_t0 is not None else None
    lab_hits = {int(L): np.flatnonzero(labels == L) for L in np.unique(labels[labels >= 0])}
    E_of = {L: float(e[h].sum()) for L, h in lab_hits.items()}
    t_of = {}
    for L, h in lab_hits.items():
        v = ts[h]; v = v[np.isfinite(v) & (v >= 0)]
        if v.size:
            t_of[L] = float(np.median(v))
    # 1. dense points
    t0 = time.time(); dense_xyz = []; dense_of = {}
    for L, h in lab_hits.items():
        if E_of[L] < cfg.dense_cluster_min_mev or h.size < 3:
            continue
        ch = dense_points(X[h], e[h], cfg.dense_r, cfg.dense_sep, cfg.dense_min_mev, cfg.dense_rel)
        dense_of[L] = [int(h[i]) for i in ch]; dense_xyz.extend(X[dense_of[L]])
    dense_xyz = np.array(dense_xyz) if dense_xyz else np.zeros((0, 3)); dense_tree = cKDTree(dense_xyz) if dense_xyz.shape[0] else None
    tim["dense"] = time.time() - t0
    # 2. candidate pieces
    t0 = time.time(); cands = []
    for L, h_all in lab_hits.items():
        if E_of[L] < cfg.host_min_mev or L not in t_of:
            continue
        for c in np.unique(tpc[h_all]):
            c = int(c)
            if c >= n_light or (L, c) not in amp:
                continue
            h = h_all[tpc[h_all] == c]
            if h.size < max(2 * cfg.piece_min_hits, cfg.min_tpc_hits):
                continue
            if cfg.vox_cm > 0:
                tips, D = skeleton_vox(X[h], cfg.link_cm, cfg.tip_min_cm, cfg.vox_cm)
            else:
                tips, G = skeleton_tips(X[h], cfg.link_cm, cfg.tip_min_cm)
                if tips:
                    D = dijkstra(G, indices=tips); D[~np.isfinite(D)] = -1
            if not tips:
                continue
            excl = np.zeros(h.size, bool)
            if dense_of.get(L):
                tr = cKDTree(X[h])
                for hd in dense_of[L]:
                    excl[tr.query_ball_point(X[hd], cfg.dense_r)] = True
            free = [bool(d > cfg.free_cm) for d in dense_tree.query(X[h][tips])[0]] if dense_tree is not None else [True] * len(tips)
            i_max = int(np.argmax(D.max(1))); j_max = int(np.argmax(D[i_max])); extent = float(D[i_max, j_max]); ends_xyz = X[h][[tips[i_max], j_max]]
            if extent < cfg.host_extent_min_cm:
                continue
            if float(e[h].sum()) / max(extent, 1.0) > cfg.host_dedx_max:          # proton-like / shower-like host: no Michel expected
                continue
            for ti, tip in enumerate(tips):
                if excl[tip] or not free[ti]:
                    continue
                if cfg.tip_face_min_cm > 0:
                    pt = X[h][tip]; bx = cfg.box
                    if min(pt[0] - bx[0], bx[1] - pt[0], pt[1] - bx[2], bx[3] - pt[1], pt[2] - bx[4], bx[5] - pt[2]) < cfg.tip_face_min_cm:
                        continue
                masks = {}; bragg = False; stop_xyz = None
                for Lc in cfg.piece_cm:
                    mk = (D[ti] >= 0) & (D[ti] <= Lc)
                    if e[h][mk & excl].sum() > cfg.dense_leak_mev:
                        bragg = True; stop_xyz = X[h][mk & excl].mean(0); break
                    if mk.sum() > h.size - cfg.piece_min_hits or e[h][mk].sum() > cfg.max_frac * E_of[L] or e[h][mk].sum() > cfg.piece_max_mev:
                        break
                    if mk.sum() >= cfg.piece_min_hits and e[h][mk].sum() >= cfg.piece_min_mev:
                        masks[Lc] = mk
                if not masks or (cfg.require_bragg and not bragg):
                    continue
                if bragg and cfg.bragg_end_cm < 1e8 and np.min(np.linalg.norm(ends_xyz - stop_xyz[None, :], axis=1)) > cfg.bragg_end_cm:
                    continue
                probe = max(Lc for Lc in masks if Lc <= cfg.piece_cm_probe) if any(Lc <= cfg.piece_cm_probe for Lc in masks) else min(masks)
                cands.append(dict(L=L, c=c, tip=int(tip), hits=h, masks=masks, bragg=bragg, probe=probe))
    wholes = [(L, c) for (L, c) in amp if cfg.whole_min_mev <= E_of.get(L, 0) <= cfg.whole_max_mev and L in t_of and c < n_light]
    tim["geom"] = time.time() - t0
    if debug_out is not None and debug_tag is not None:
        cov_tag = {}
        for cd in cands:
            for Lc, mk in cd["masks"].items():
                hp = cd["hits"][mk]; tg = debug_tag[hp]
                for tv in np.unique(tg[tg >= 0]):
                    cov_tag[int(tv)] = max(cov_tag.get(int(tv), 0.0), float(e[hp][tg == tv].sum()))
        debug_out.append(dict(kind="geometry", covered=cov_tag))
    # 3. perceiver (probe pieces)
    pred = {}

    def predict(items):
        todo = [(k, h) for k, h in items if k not in pred]
        if todo:
            out = predict_fn(todo)
            for k, _ in todo:
                ap_ = np.asarray(out.get(k, np.zeros(n_ch)), np.float64).copy(); ap_[veto[k[1]]] = 0.0; pred[k] = ap_
    t0 = time.time(); predict([((cd["L"], cd["c"], cd["tip"], cd["probe"]), cd["hits"][cd["masks"][cd["probe"]]]) for cd in cands]); tim["pred1"] = time.time() - t0
    n_pred1 = len(pred)
    # 4. scans and scores
    w_cache = {}

    def weights(c):
        if c not in w_cache:
            w_cache[c] = 1.0 / (GAIN * np.maximum(actual[c].astype(np.float64), 100.0))
        return w_cache[c]

    def scan(a_p, c, R, t_c):
        w = weights(c); q = (a_p ** 2) @ w; r = a_p @ (w * R)
        curve = fftconvolve(q, T2[::-1], mode="full")[kS: kS + 886] - 2.0 * fftconvolve(r, T[::-1], mode="full")[kS: kS + 886]
        lo = int(t_c + cfg.late_min_ticks) + 1; hi = min(int(t_c + cfg.delay_max), 885)
        if hi <= lo:
            return []
        seg = curve[lo: hi + 1]
        mins = [i for i in range(1, seg.size - 1) if seg[i] <= seg[i - 1] and seg[i] <= seg[i + 1] and seg[i] < 0]
        mins.sort(key=lambda i: seg[i]); out = []
        for i in mins:
            if all(abs(i - j) > 6 for j in out):
                out.append(i)
            if len(out) >= cfg.n_minima:
                break
        return [float(lo + i) for i in out]

    ch_cache = {}

    def chans_ok(c):
        if c not in ch_cache:
            ch_cache[c] = np.flatnonzero(~veto[c])
        return ch_cache[c]

    def score(a_p, c, L, hits_p, tf, own_c, rest, A, B, w, cl, whole=False):
        t_c = t_of[L]; sup = support(a_p, cfg.support_frac)
        Sg = render(a_p, tf, n_s, T); Sp = render(a_p, t_c, n_s, T); r = A - (B - own_c + rest)
        num = float((r * Sg * w).sum()); den = float((Sg * Sg * w).sum()); alpha = num / max(den, 1e-9)
        k_f = int(round(tf)) + PEAK; sl_f = slice(max(0, k_f + cfg.mf_window[0]), min(n_s, k_f + cfg.mf_window[1]))
        ww = 1.0 / np.sqrt(GAIN * np.maximum(A[sup][:, sl_f], 100.0)); mf = wcos(Sg[sup][:, sl_f], r[sup][:, sl_f], ww)
        kp = int(round(t_c)) + PEAK; sl_p = slice(max(0, kp - 5), min(n_s, kp + 60))
        Spw = Sp[sup][:, sl_p]; m_cur = B[sup][:, sl_p]; wwp = w[sup][:, sl_p]; r_c = A[sup][:, sl_p] - np.minimum(m_cur, cl[sup][:, None])
        dc = float((Spw * Spw * wwp).sum()); apc = float((r_c * Spw * wwp).sum()) / dc if dc > 0 else np.nan
        sat = float((m_cur.max(1) >= 0.95 * cl[sup]).mean()) if sup.size else 1.0
        rl = np.clip(A[sup][:, sl_f] - B[sup][:, sl_f], 0, None); cov = float(np.minimum(Sg[sup][:, sl_f], rl).sum() / max(rl.sum(), 1e-9))
        # gain-corrected unexplained light at the destination (all non-vetoed channels, peak window): the placed families
        # may be under-predicted by up to unexpl_gain_tol; what remains must be >= unexpl_min_frac of the actual
        chv = chans_ok(c); sl_u = slice(max(0, k_f - 3), min(n_s, k_f + 6))
        a_u = np.clip(A[chv][:, sl_u], 0, None).max(1).sum(); r_u = np.clip(A[chv][:, sl_u] - B[chv][:, sl_u], 0, None).max(1); F_u = np.clip(B[chv][:, sl_u], 0, None).max(1)
        den_u = float(F_u @ F_u); g_u = min(max(float(r_u @ F_u) / den_u, 0.0), cfg.unexpl_gain_tol) if den_u > 0 else 0.0
        unexpl = float(np.clip(r_u - g_u * F_u, 0, None).sum() / max(a_u, 1e-9))
        E_p = float(e[hits_p].sum())
        passes = (mf >= cfg.mf_min and cfg.alpha_lo <= alpha <= (cfg.whole_alpha_hi if whole else cfg.alpha_hi) and np.isfinite(apc) and apc <= cfg.apc_max
                  and sat <= cfg.sat_max and cov >= cfg.cov_min and E_p >= cfg.piece_min_mev and unexpl >= cfg.unexpl_min_frac)
        return dict(mf=mf, alpha=alpha, apc=apc, sat=sat, cov=cov, unexpl=unexpl, E_p=E_p, ok=passes, Sg=Sg, Sp=Sp, tf=tf, hits=hits_p)

    def dbg(sc, kind):
        """record a scored hypothesis: tag composition (energy per tag value) + scores"""
        if debug_out is None or debug_tag is None:
            return
        hp = sc["hits"]; tg = debug_tag[hp]; comp = {}
        for tv in np.unique(tg[tg >= 0]):
            comp[int(tv)] = float(e[hp][tg == tv].sum())
        debug_out.append(dict(kind=kind, L=int(sc["L"]), c=int(sc["c"]), tip=int(sc.get("tip", -1)), Lc=float(sc.get("Lc", -1)), tf=float(sc["tf"]), t_c=float(t_of[sc["L"]]),
                              E_p=sc["E_p"], mf=sc["mf"], alpha=sc["alpha"], apc=sc["apc"], sat=sc["sat"], cov=sc["cov"], unexpl=sc["unexpl"], ok=bool(sc["ok"]), comp=comp))

    t0 = time.time(); res = []
    for cd in cands:
        L, c, tip = cd["L"], cd["c"], cd["tip"]; a_p = pred.get((L, c, tip, cd["probe"]))
        if a_p is None or a_p.max() <= 0:
            continue
        A = actual[c].astype(np.float64); B = base[c].astype(np.float64); w = weights(c); cl = np.asarray(ceil[c], np.float64)
        own_c = render(amp[(L, c)] * (~veto[c]), t_of[L], n_s, T); Sp = render(a_p, t_of[L], n_s, T); rest = np.clip(own_c - Sp, 0, None)
        R = A - (B - own_c + rest)
        for tf in scan(a_p, c, R, t_of[L]):
            sc = score(a_p, c, L, cd["hits"][cd["masks"][cd["probe"]]], tf, own_c, rest, A, B, w, cl); sc.update(L=L, c=c, tip=tip, Lc=cd["probe"], whole=False, cd=cd); res.append(sc); dbg(sc, "probe")
    for (L, c) in wholes:
        a_p = amp[(L, c)] * (~veto[c])
        if a_p.max() <= 0:
            continue
        A = actual[c].astype(np.float64); B = base[c].astype(np.float64); w = weights(c); cl = np.asarray(ceil[c], np.float64)
        own_c = render(a_p, t_of[L], n_s, T); R = A - (B - own_c)
        for tf in scan(a_p, c, R, t_of[L]):
            sc = score(a_p, c, L, lab_hits[L], tf, own_c, np.zeros_like(own_c), A, B, w, cl, whole=True); sc.update(L=L, c=c, tip=-1, Lc=-1, whole=True, cd=None); res.append(sc); dbg(sc, "whole")
    tim["scan"] = time.time() - t0
    # 5. refinement + selection
    t0 = time.time(); passing = [r for r in res if r["ok"] and not r["whole"]]
    predict([((r["L"], r["c"], r["tip"], Lc), r["cd"]["hits"][mk]) for r in passing for Lc, mk in r["cd"]["masks"].items()]); tim["pred2"] = time.time() - t0
    t0 = time.time(); final = [r for r in res if r["ok"] and r["whole"]]
    for r in passing:
        cd = r["cd"]; L, c = r["L"], r["c"]; A = actual[c].astype(np.float64); B = base[c].astype(np.float64); w = weights(c); cl = np.asarray(ceil[c], np.float64)
        own_c = render(amp[(L, c)] * (~veto[c]), t_of[L], n_s, T); best = None
        for Lc, mk in cd["masks"].items():
            a_p = pred.get((L, c, r["tip"], Lc))
            if a_p is None or a_p.max() <= 0:
                continue
            Sp = render(a_p, t_of[L], n_s, T); rest = np.clip(own_c - Sp, 0, None)
            for dt in (-2.0, -1.0, 0.0, 1.0, 2.0):
                sc = score(a_p, c, L, cd["hits"][mk], r["tf"] + dt, own_c, rest, A, B, w, cl)
                if not sc["ok"]:
                    continue
                sc.update(L=L, c=c, tip=r["tip"], Lc=Lc, whole=False); dbg(sc, "refine")
                if best is None or sc["mf"] > best["mf"] + cfg.mf_tol or (sc["mf"] >= best["mf"] - cfg.mf_tol and sc["E_p"] > best["E_p"]):
                    best = sc
        if best is not None:
            final.append(best)
    final.sort(key=lambda r: -r["mf"]); moved = set(); moves = []
    for r in final:
        L, c = r["L"], r["c"]
        if L in moved:
            continue
        hits = r["hits"]; rec = dict(cluster=int(L), tpc=int(c), tip=int(r["tip"]), L_cm=float(r["Lc"]), whole=bool(r["whole"]), E=r["E_p"], t_from=t_of[L], t_to=float(r["tf"]),
                                     mf=r["mf"], alpha=r["alpha"], apc=r["apc"], cov=r["cov"], unexpl=r["unexpl"], hits=hits)
        if truth_t0 is not None:
            okh = ok[hits]; rec["was"] = float((e[hits][okh] * (np.abs(t_of[L] - truth_t0[hits][okh]) <= CUT)).sum()); rec["now"] = float((e[hits][okh] * (np.abs(r["tf"] - truth_t0[hits][okh]) <= CUT)).sum())
        moves.append(rec); moved.add(L)
        if apply:
            ts[hits] = r["tf"]; base[c] = (base[c].astype(np.float64) + r["Sg"] - r["Sp"]).astype(base.dtype)
    tim["refine"] = time.time() - t0
    st = dict(n_cand=len(cands), n_whole=len(wholes), n_pred=len(pred), n_pred_probe=n_pred1, n_hyp=len(res), n_moves=len(moves), moves=moves, tim={k: round(v, 3) for k, v in tim.items()},
              time_s=round(time.time() - t_start, 3), e_moved=float(sum(m["E"] for m in moves)))
    if truth_t0 is not None:
        st["fixed_e"] = float(sum(max(m["now"] - m["was"], 0) for m in moves)); st["broke_e"] = float(sum(max(m["was"] - m["now"], 0) for m in moves))
    return st
