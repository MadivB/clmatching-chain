"""ND Michel stage, PIECE-FIRST (fast).  Physics injected (user 2026-10-02: Michels are critical, inject physics):
a Michel electron is a short branch (<= ~16 cm, 5-50 MeV) that LEAVES the Bragg peak at the end of a stopping muon,
and it has its own flash 12-900 ticks later with the muon lifetime (tau = 137 ticks).  So instead of enumerating
unexplained flashes and testing every cluster against each one (the 2x2 stage M order; 100 s per ND spill):
  1. DENSE POINTS (Bragg peaks / vertices) in every cluster >= dense_cluster_min_mev, event-wide.
  2. CANDIDATE PIECES: for every cluster >= host_min_mev, per TPC: free tips (MST leaves >= 2 cm, not within 3 cm of a
     dense point); the probe piece = hits within geodesic L <= 5 cm of the tip; the piece is grown over L = 2..16 cm and
     stops when it would swallow > 2 MeV of a dense ball ("bragg" = the growth was stopped by a dense ball within L_max,
     i.e. the branch leaves a Bragg peak) or > 50 % of the cluster.  Whole templated clusters of 5-80 MeV are candidates too.
  3. ONE batched perceiver call for all probe pieces of the spill (whole clusters use their existing amp pattern).
  4. For every piece a MATCHED-FILTER SCAN of its light pattern over all later times in its TPC (t > t_c + 12, FFT), on the
     residual with the piece's prompt light removed; the best 2 minima are the candidate flashes.  At each: mf (pattern
     cosine on the support channels, window -3..+45), alpha, cov, apc (prompt deficit at t_c), sat.  Gates (dev-chosen):
     mf >= 0.8, 0.8 <= alpha <= 6, apc <= -0.5, sat <= 0.5, cov >= 0.2, piece >= 5 MeV, delay <= delay_max.
  5. Passing probe pieces are refined: the full L list predicted in a second batched call, scanned at the chosen flash
     +- 3 ticks; the fullest piece within mf_tol of the best wins; move, update the base; a cluster moves once.
Truth only labels.  python v14_michel_nd_fast.py <dump_dir> [--events N] [--files ...] [--apply] [--no-bragg] [--out JSON]"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, "..", "CLMatching_AlphaRelease"))
for p in (BASE, os.path.join(BASE, "M5p1"), os.path.join(BASE, "NewMLSection"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
from resid_fill import render, _template  # noqa: E402
from v14_michel_nd import skeleton_tips, support, wcos, SMALL, PULSE, CUT, PEAK, GAIN, TMP0  # noqa: E402


def dense_points(X, E, r, sep, min_mev, rel):
    """local energy maxima (ball r) >= max(min_mev, rel x median), greedy with separation sep; identical to the 2x2 rule"""
    tree = cKDTree(X); nb = tree.query_ball_point(X, r)
    ball = np.fromiter((E[n].sum() for n in nb), float, count=len(nb)); thr = max(min_mev, rel * float(np.median(ball)))
    order = np.argsort(-ball); order = order[ball[order] >= thr]
    supp = np.zeros(X.shape[0], bool); chosen = []
    for i in order:
        if supp[i]:
            continue
        chosen.append(int(i)); supp[tree.query_ball_point(X[i], sep)] = True
    return chosen


def skeleton_vox(Xh, link_cm, tip_min_cm, vox):
    """skeleton on 1 voxel representative per vox-cm cell (the hit nearest the cell centroid); tips and geodesic
    distances are mapped back to every hit through its cell.  Returns (tips as hit indices, D (n_tips, n_hits))."""
    from scipy.sparse.csgraph import dijkstra
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


class Cfg:
    late_min_ticks = 12.0; delay_max = 300.0; host_min_mev = 10.0; whole_min_mev = 5.0; whole_max_mev = 80.0
    link_cm = 3.0; tip_min_cm = 2.0; dense_r = 1.5; dense_sep = 3.0; dense_min_mev = 5.0; dense_rel = 1.5; dense_cluster_min_mev = 5.0
    free_cm = 3.0; dense_leak_mev = 2.0; piece_cm = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0); piece_cm_probe = 5.0
    piece_min_hits = 3; piece_min_mev = 5.0; max_frac = 0.5; mf_window = (-3, 45); n_minima = 2
    mf_min = 0.8; mf_tol = 0.02; apc_max = -0.5; sat_max = 0.5; cov_min = 0.2; alpha_lo = 0.8; alpha_hi = 6.0; alpha_min = 0.65
    support_frac = 0.9; require_bragg = True; min_tpc_hits = 6
    host_extent_min_cm = 0.0     # geodesic diameter of the host's hits in the TPC (a stopping muon is long)
    bragg_end_cm = 1e9           # the dense ball that stops the piece must lie within this of a diameter end of the host
    piece_max_mev = 1e9          # a Michel carries <= ~55 MeV
    vox_cm = 1.0                 # skeleton on 1-cm voxel representatives (0 = every hit)
    max_tips = 1000              # hosts with more tips (showers) are skipped


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dump_dir"); ap.add_argument("--events", type=int, default=0); ap.add_argument("--files", default=None)
    ap.add_argument("--apply", action="store_true"); ap.add_argument("--no-bragg", action="store_true"); ap.add_argument("--batch", type=int, default=16); ap.add_argument("--out", default=os.path.join(HERE, "michel_nd_out", "fast.json"))
    for k in ("mf_min", "apc_max", "cov_min", "alpha_lo", "alpha_hi", "delay_max", "piece_min_mev", "host_min_mev", "host_extent_min_cm", "bragg_end_cm", "piece_max_mev", "vox_cm", "max_tips"):
        ap.add_argument("--" + k.replace("_", "-"), type=float, default=None)
    a = ap.parse_args(); cfg = Cfg()
    for k in ("mf_min", "apc_max", "cov_min", "alpha_lo", "alpha_hi", "delay_max", "piece_min_mev", "host_min_mev", "host_extent_min_cm", "bragg_end_cm", "piece_max_mev", "vox_cm", "max_tips"):
        if getattr(a, k) is not None:
            setattr(cfg, k, getattr(a, k))
    if a.no_bragg:
        cfg.require_bragg = False
    files = sorted(glob.glob(os.path.join(a.dump_dir, "full_dumps", "*.npz")))
    if a.files:
        tags = a.files.split(","); files = [f for f in files if any(os.path.basename(f).startswith(t) for t in tags)]
    if a.events:
        files = files[: a.events]
    from M5p1.first_stage_matching import load_first_stage_models
    from M5p1.first_stage_matching.config import FirstStageConfig
    from M5p1.first_stage_matching.streaming_prediction import process_clusters_to_imageMaps_streaming
    fcfg = FirstStageConfig(); fcfg.model.light_checkpoint = SMALL
    models = load_first_stage_models(fcfg); tmpl = np.asarray(models.waveform_template, np.float64); T = _template(PULSE)
    T2 = T ** 2; NT = T.size
    print("models on", models.device, flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    all_hyps, all_moves, ev_stats = [], [], []; t_begin = time.time()

    for f in files:
        t_ev = time.time(); tim = {}
        z = np.load(f, allow_pickle=True); tag = os.path.basename(f)[:-4]
        e = np.clip(z["e"], 0, None).astype(np.float64); ts = z["ts_final"].astype(np.float64); tru = z["truth_t0"].astype(np.float64)
        labs = z["labels"].astype(int); tpc = z["tpc"].astype(int); x, y, zz = (z[k].astype(np.float32) for k in ("x", "y", "z"))
        X = np.stack([x, y, zz], 1).astype(np.float64)
        A_all = z["full_light"].astype(np.float32); B_all = z["base_image"].astype(np.float32).copy(); veto = z["veto_mask"]; ceil_all = z["sat_ceiling"].astype(np.float32)
        al, at, ar = z["amp_lab"].astype(int), z["amp_tpc"].astype(int), z["amp_row"].astype(np.float64)
        amp = {(int(L), int(c)): ar[i] for i, (L, c) in enumerate(zip(al, at))}
        n_light, n_ch, n_s = A_all.shape; ok = np.isfinite(tru)
        lab_hits = {int(L): np.flatnonzero(labs == L) for L in np.unique(labs[labs >= 0])}
        E_of = {L: float(e[h].sum()) for L, h in lab_hits.items()}
        t_of = {}
        for L, h in lab_hits.items():
            v = ts[h]; v = v[np.isfinite(v) & (v >= 0)]
            if v.size:
                t_of[L] = float(np.median(v))
        # delayed class bookkeeping
        dmask = np.zeros(labs.size, bool)
        for L, h in lab_hits.items():
            m = h[ok[h] & (tru[h] >= 0) & (tru[h] <= 885)]
            if E_of[L] < 10 or m.size < 5:
                continue
            tmaj = np.median(tru[m]); dmask[m[tru[m] - tmaj > 12]] = True
        E_del = float(e[dmask].sum()); E_del_before = float((e[dmask] * (np.abs(ts[dmask] - tru[dmask]) <= CUT)).sum()); ts_before = ts.copy()
        # ---- 1. dense points
        t0 = time.time(); dense_xyz = []; dense_of = {}
        for L, h in lab_hits.items():
            if E_of[L] < cfg.dense_cluster_min_mev or h.size < 3:
                continue
            ch = dense_points(X[h], e[h], cfg.dense_r, cfg.dense_sep, cfg.dense_min_mev, cfg.dense_rel)
            dense_of[L] = [int(h[i]) for i in ch]; dense_xyz.extend(X[dense_of[L]])
        dense_xyz = np.array(dense_xyz) if dense_xyz else np.zeros((0, 3)); dense_tree = cKDTree(dense_xyz) if dense_xyz.shape[0] else None
        tim["dense"] = time.time() - t0
        # ---- 2. candidate pieces (geometry)
        t0 = time.time(); cands = []       # dict(L, c, tip, hits(local idx array of cluster-in-tpc), masks {Lc: mask}, bragg, probe_L)
        from scipy.sparse.csgraph import dijkstra
        for L, h_all in lab_hits.items():
            if E_of[L] < cfg.host_min_mev or L not in t_of:
                continue
            for c in np.unique(tpc[h_all]):
                c = int(c)
                if c >= n_light or (L, c) not in amp:
                    continue
                h = h_all[tpc[h_all] == c]
                if h.size < 2 * cfg.piece_min_hits or h.size < cfg.min_tpc_hits:
                    continue
                if cfg.vox_cm > 0:
                    tips, D = skeleton_vox(X[h], cfg.link_cm, cfg.tip_min_cm, cfg.vox_cm)
                else:
                    tips, G = skeleton_tips(X[h], cfg.link_cm, cfg.tip_min_cm)
                    if tips:
                        D = dijkstra(G, indices=tips); D[~np.isfinite(D)] = -1
                if not tips or len(tips) > cfg.max_tips:
                    continue
                excl = np.zeros(h.size, bool)
                if dense_of.get(L):
                    tr = cKDTree(X[h])
                    for hd in dense_of[L]:
                        excl[tr.query_ball_point(X[hd], cfg.dense_r)] = True
                free = [bool(d > cfg.free_cm) for d in dense_tree.query(X[h][tips])[0]] if dense_tree is not None else [True] * len(tips)
                # host extent = geodesic diameter (max over tip rows); diameter ends = the two tips realising it
                i_max = int(np.argmax(D.max(1))); j_max = int(np.argmax(D[i_max])); extent = float(D[i_max, j_max])
                ends_xyz = X[h][[tips[i_max], j_max]]
                if extent < cfg.host_extent_min_cm:
                    continue
                for ti, tip in enumerate(tips):
                    if excl[tip] or not free[ti]:
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
                    cands.append(dict(L=L, c=c, tip=int(tip), hits=h, masks=masks, bragg=bragg, probe=probe, extent=extent))
        # whole-cluster candidates
        wholes = [(L, c) for (L, c) in amp if cfg.whole_min_mev <= E_of.get(L, 0) <= cfg.whole_max_mev and L in t_of and c < n_light]
        tim["geom"] = time.time() - t0
        # ---- 3. perceiver: probe pieces in one (or a few) passes
        pred = {}

        def predict(keys_masks):
            todo = [(k, mk, h) for k, mk, h in keys_masks if k not in pred]
            if not todo:
                return
            passes = []
            for j, (k, mk, h) in enumerate(todo):
                hh = h[mk]
                for pm in passes:
                    if not pm["used"][hh].any():
                        pm["lab"][hh] = TMP0 + j; pm["used"][hh] = True; pm["want"].add(TMP0 + j); pm["keys"].append((j, k)); break
                else:
                    pm = {"lab": np.full(labs.size, -1, np.int64), "used": np.zeros(labs.size, bool), "want": set(), "keys": []}
                    pm["lab"][hh] = TMP0 + j; pm["used"][hh] = True; pm["want"].add(TMP0 + j); pm["keys"].append((j, k)); passes.append(pm)
            for pm in passes:
                im, _ = process_clusters_to_imageMaps_streaming(x, y, zz, e.astype(np.float32), tpc, pm["lab"], model=models.light_model, template=tmpl,
                                                                batch_size=a.batch, restrict_clusters=pm["want"], device_policy="auto")
                for j, k in pm["keys"]:
                    img = im.get((TMP0 + j, k[1])); ap_ = np.asarray(img, np.float64).max(1) if img is not None else np.zeros(n_ch)
                    ap_[veto[k[1]]] = 0.0; pred[k] = ap_
        t0 = time.time()
        predict([((cd["L"], cd["c"], cd["tip"], cd["probe"]), cd["masks"][cd["probe"]], cd["hits"]) for cd in cands])
        tim["pred1"] = time.time() - t0; n_pred1 = len(pred)
        # ---- 4. scans + scores
        w_all = {}; kS = NT - 1

        def weights(c):
            if c not in w_all:
                w_all[c] = 1.0 / (GAIN * np.maximum(A_all[c].astype(np.float64), 100.0))
            return w_all[c]

        def scan(a_p, c, R, t_c):
            """chi2 gain curve of placing a_p x T at t over the residual R; returns up to n_minima best later times"""
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

        def score(a_p, c, L, mk_hits, tf, own_c, rest, A, B, w, ceil):
            t_c = t_of[L]; sup = support(a_p, cfg.support_frac)
            Sg = render(a_p, tf, n_s, T); Sp = render(a_p, t_c, n_s, T)
            r = A - (B - own_c + rest)
            num = float((r * Sg * w).sum()); den = float((Sg * Sg * w).sum()); alpha = num / max(den, 1e-9)
            k_f = int(round(tf)) + PEAK; sl_f = slice(max(0, k_f + cfg.mf_window[0]), min(n_s, k_f + cfg.mf_window[1]))
            ww = 1.0 / np.sqrt(GAIN * np.maximum(A[sup][:, sl_f], 100.0)); mf = wcos(Sg[sup][:, sl_f], r[sup][:, sl_f], ww)
            kp = int(round(t_c)) + PEAK; sl_p = slice(max(0, kp - 5), min(n_s, kp + 60))
            Spw = Sp[sup][:, sl_p]; m_cur = B[sup][:, sl_p]; wwp = w[sup][:, sl_p]; r_c = A[sup][:, sl_p] - np.minimum(m_cur, ceil[sup][:, None])
            dc = float((Spw * Spw * wwp).sum()); apc = float((r_c * Spw * wwp).sum()) / dc if dc > 0 else np.nan
            sat = float((m_cur.max(1) >= 0.95 * ceil[sup]).mean()) if sup.size else 1.0
            rl = np.clip(A[sup][:, sl_f] - B[sup][:, sl_f], 0, None); cov = float(np.minimum(Sg[sup][:, sl_f], rl).sum() / max(rl.sum(), 1e-9))
            E_p = float(e[mk_hits].sum()); okh = ok[mk_hits]
            true_frac = float((e[mk_hits][okh] * (np.abs(tru[mk_hits][okh] - tf) <= CUT)).sum() / max(E_p, 1e-9))
            cur_frac = float((e[mk_hits][okh] * (np.abs(tru[mk_hits][okh] - t_c) <= CUT)).sum() / max(E_p, 1e-9))
            passes_ = (mf >= cfg.mf_min and cfg.alpha_lo <= alpha <= cfg.alpha_hi and np.isfinite(apc) and apc <= cfg.apc_max and sat <= cfg.sat_max and cov >= cfg.cov_min and E_p >= cfg.piece_min_mev)
            return dict(mf=mf, alpha=alpha, apc=apc, sat=sat, cov=cov, E_p=E_p, ok=passes_, true_frac=true_frac, cur_frac=cur_frac, Sg=Sg, Sp=Sp, tf=tf)

        t0 = time.time(); res = []       # probe-level results
        for cd in cands:
            L, c, tip = cd["L"], cd["c"], cd["tip"]; a_p = pred.get((L, c, tip, cd["probe"]))
            if a_p is None or a_p.max() <= 0:
                continue
            A = A_all[c].astype(np.float64); B = B_all[c].astype(np.float64); w = weights(c); ceil = ceil_all[c].astype(np.float64)
            own_c = render(amp[(L, c)] * (~veto[c]), t_of[L], n_s, T); Sp = render(a_p, t_of[L], n_s, T); rest = np.clip(own_c - Sp, 0, None)
            R = A - (B - own_c + rest)
            for tf in scan(a_p, c, R, t_of[L]):
                sc = score(a_p, c, L, cd["hits"][cd["masks"][cd["probe"]]], tf, own_c, rest, A, B, w, ceil)
                sc.update(L=L, c=c, tip=tip, Lc=cd["probe"], whole=False, bragg=cd["bragg"], E_L=E_of[L], t_c=t_of[L]); res.append(sc)
        for (L, c) in wholes:
            a_p = amp[(L, c)] * (~veto[c])
            if a_p.max() <= 0:
                continue
            A = A_all[c].astype(np.float64); B = B_all[c].astype(np.float64); w = weights(c); ceil = ceil_all[c].astype(np.float64)
            own_c = render(a_p, t_of[L], n_s, T); R = A - (B - own_c)
            for tf in scan(a_p, c, R, t_of[L]):
                sc = score(a_p, c, L, lab_hits[L], tf, own_c, np.zeros_like(own_c), A, B, w, ceil)
                sc.update(L=L, c=c, tip=-1, Lc=-1, whole=True, bragg=False, E_L=E_of[L], t_c=t_of[L]); res.append(sc)
        tim["scan"] = time.time() - t0
        # ---- 5. refinement of passing probe pieces: all lengths, scanned at the chosen flash +-3
        t0 = time.time()
        passing = [r for r in res if r["ok"] and not r["whole"]]
        keys = []
        for r in passing:
            cd = next(cd for cd in cands if cd["L"] == r["L"] and cd["c"] == r["c"] and cd["tip"] == r["tip"])
            for Lc, mk in cd["masks"].items():
                keys.append(((r["L"], r["c"], r["tip"], Lc), mk, cd["hits"]))
        predict(keys); tim["pred2"] = time.time() - t0; n_pred2 = len(pred) - n_pred1
        t0 = time.time(); final = list(r for r in res if r["ok"] and r["whole"])
        for r in passing:
            cd = next(cd for cd in cands if cd["L"] == r["L"] and cd["c"] == r["c"] and cd["tip"] == r["tip"])
            L, c = r["L"], r["c"]; A = A_all[c].astype(np.float64); B = B_all[c].astype(np.float64); w = weights(c); ceil = ceil_all[c].astype(np.float64)
            own_c = render(amp[(L, c)] * (~veto[c]), t_of[L], n_s, T)
            best = None
            for Lc, mk in cd["masks"].items():
                a_p = pred.get((L, c, r["tip"], Lc))
                if a_p is None or a_p.max() <= 0:
                    continue
                Sp = render(a_p, t_of[L], n_s, T); rest = np.clip(own_c - Sp, 0, None)
                for dt in (-2.0, -1.0, 0.0, 1.0, 2.0):
                    sc = score(a_p, c, L, cd["hits"][mk], r["tf"] + dt, own_c, rest, A, B, w, ceil)
                    if not sc["ok"]:
                        continue
                    sc.update(L=L, c=c, tip=r["tip"], Lc=Lc, whole=False, bragg=cd["bragg"], E_L=E_of[L], t_c=t_of[L], mk=mk)
                    if best is None or sc["mf"] > best["mf"] + cfg.mf_tol or (sc["mf"] >= best["mf"] - cfg.mf_tol and sc["E_p"] > best["E_p"]):
                        best = sc
            if best is not None:
                final.append(best)
        # per cluster one move (best mf); apply
        final.sort(key=lambda r: -r["mf"]); moved = set(); ev_moves = []
        for r in final:
            L, c = r["L"], r["c"]
            if L in moved:
                continue
            hits = lab_hits[L] if r["whole"] else next(cd for cd in cands if cd["L"] == L and cd["c"] == c and cd["tip"] == r["tip"])["hits"][r["mk"]]
            okh = ok[hits]; was = float((e[hits][okh] * (np.abs(t_of[L] - tru[hits][okh]) <= CUT)).sum()); now = float((e[hits][okh] * (np.abs(r["tf"] - tru[hits][okh]) <= CUT)).sum())
            rec = dict(ev=tag, tpc=c, L=L, tip=r["tip"], Lc=r["Lc"], whole=r["whole"], bragg=r["bragg"], E_L=r["E_L"], E_p=r["E_p"], t_c=r["t_c"], tf=r["tf"], mf=r["mf"], alpha=r["alpha"], apc=r["apc"], cov=r["cov"], was=was, now=now)
            ev_moves.append(rec); moved.add(L)
            if a.apply:
                ts[hits] = r["tf"]; B_all[c] = (B_all[c].astype(np.float64) + r["Sg"] - r["Sp"]).astype(np.float32)
        tim["refine"] = time.time() - t0
        for r in res:
            all_hyps.append(dict(ev=tag, tpc=r["c"], L=r["L"], tip=r["tip"], Lc=r["Lc"], whole=r["whole"], bragg=r["bragg"], E_L=r["E_L"], E_p=r["E_p"], t_c=r["t_c"], tf=r["tf"],
                                 mf=r["mf"], alpha=r["alpha"], apc=r["apc"], sat=r["sat"], cov=r["cov"], ok=bool(r["ok"]), true_frac=r["true_frac"], cur_frac=r["cur_frac"]))
        all_moves += ev_moves
        E_del_after = float((e[dmask] * (np.abs(ts[dmask] - tru[dmask]) <= CUT)).sum()) if a.apply else E_del_before
        fixed = sum(max(m["now"] - m["was"], 0) for m in ev_moves); broke = sum(max(m["was"] - m["now"], 0) for m in ev_moves); dt = time.time() - t_ev
        ev_stats.append(dict(ev=tag, n_cand=len(cands), n_whole=len(wholes), n_pred=len(pred), n_hyp=len(res), n_moves=len(ev_moves), fixed=fixed, broke=broke, t=dt,
                             E_del=E_del, E_del_before=E_del_before, E_del_after=E_del_after, tim=tim))
        print(f"{tag}: pieces {len(cands)} (bragg {sum(cd['bragg'] for cd in cands)}), wholes {len(wholes)}, predictions {n_pred1}+{n_pred2}, hypotheses {len(res)}, moves {len(ev_moves)} fixed {fixed:.0f} broke {broke:.0f} MeV, "
              f"{dt:.1f}s [dense {tim['dense']:.1f} geom {tim['geom']:.1f} pred {tim['pred1']:.1f}+{tim['pred2']:.1f} scan {tim['scan']:.1f} refine {tim['refine']:.1f}]", flush=True)
    json.dump(dict(cfg={k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("_")}, apply=a.apply, hyps=all_hyps, moves=all_moves, events=ev_stats), open(a.out, "w"), default=float)
    M = all_moves
    if M:
        d = np.array([m["now"] - m["was"] for m in M]); fx = sum(e_["fixed"] for e_ in ev_stats); br = sum(e_["broke"] for e_ in ev_stats)
        print(f"\n{len(files)} events: moves {len(M)} (bragg {sum(m['bragg'] for m in M)}, whole {sum(m['whole'] for m in M)}); fixed {fx/1e3:.2f} GeV broke {br/1e3:.2f} GeV net {(fx-br)/1e3:+.2f}; purity {fx/max(fx+br,1e-9):.2f}; right {np.mean(d>0.5):.2f} wrong {np.mean(d<-0.5):.2f}; time/ev {np.mean([e_['t'] for e_ in ev_stats]):.1f} s")
        Ed = sum(e_["E_del"] for e_ in ev_stats); Eb = sum(e_["E_del_before"] for e_ in ev_stats); Ea = sum(e_["E_del_after"] for e_ in ev_stats)
        print(f"delayed-daughter class {Ed/1e3:.2f} GeV: correct {Eb/max(Ed,1e-9):.3f} -> {Ea/max(Ed,1e-9):.3f}; events better/worse {sum(e_['fixed']>e_['broke']+0.5 for e_ in ev_stats)}/{sum(e_['broke']>e_['fixed']+0.5 for e_ in ev_stats)}")
        for lab_, m in (("bragg pieces", [m for m in M if m["bragg"]]), ("non-bragg pieces", [m for m in M if not m["bragg"] and not m["whole"]]), ("whole clusters", [m for m in M if m["whole"]])):
            if m:
                fx = sum(max(q["now"] - q["was"], 0) for q in m); br = sum(max(q["was"] - q["now"], 0) for q in m); print(f"   {lab_:18s} n {len(m):3d} fixed {fx:6.0f} broke {br:5.0f} purity {fx/max(fx+br,1e-9):.2f}")
    tt = {k: np.mean([e_["tim"][k] for e_ in ev_stats]) for k in ev_stats[0]["tim"]} if ev_stats else {}
    print("mean time per event by part:", {k: round(v, 2) for k, v in tt.items()})


if __name__ == "__main__":
    main()
