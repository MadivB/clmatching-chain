"""ND port of the 2x2 stage M (v2.0_alpha_12, assoc_michel_2x2.py): delayed deposits (Michel electrons and other
daughters) that the clustering merged into a prompt cluster, or that were parked at the prompt flash, get their own
late time.  Offline emulation on chain dumps with the perceiver predicting the pieces (GPU).

Per event:
  1. LATE FLASHES per light TPC: candidate times = the TPC's family times + peaks of the summed residual (k_prom_late
     sigma), merged within 2 ticks, kept when the light there is unexplained (2x2 test: residual >= unexpl_frac of the
     actual at the peak and >= unexpl_min_adc).  (The stage-4.8 finder, which rejects tail bumps and family times,
     reached only 9 % of the ND delayed-daughter energy: 65 % of those pieces' flashes coincide with a family time.)
  2. CANDIDATES per flash: clusters with hits in the TPC placed > late_min_ticks EARLIER whose whole-cluster pattern
     (amp_row) matches the residual at the flash with cosine >= cluster_cos_min (top max_clusters_per_flash).
  3. HYPOTHESES: whole cluster (E <= whole_max_mev); END PIECES of clusters >= piece_min_cluster_mev: hits within
     geodesic distance L of a FREE TIP (MST leaves with a branch >= tip_min_cm + graph-diameter ends; not within
     free_cm of any DENSE POINT = local energy maximum >= max(dense_min_mev, dense_rel x median), found event-wide
     in clusters >= dense_cluster_min_mev), the piece holding <= dense_leak_mev of dense-ball energy, <= max_frac of
     the cluster, >= piece_min_hits and >= piece_min_mev.  Probe pass (one piece per tip, L <= piece_cm_probe), then
     the full L scan for tips whose probe pattern reaches mf_probe.
  4. SCORE (piece light = perceiver amplitude pattern x template; rest = clip(own - piece, 0)):
       mf    pattern cosine of the moved light at the flash vs the hypothesis residual (window mf_window around the
             flash peak, moved light's 90 % support channels, 1/sigma weights);
       alpha amplitude ratio of the moved light at the flash (all channels, full waveform);
       apc   PROMPT DEFICIT: projection of the moved light's pattern at the cluster time on the CURRENT residual
             (actual - clipped model) in the prompt window; ~ -1 absent, ~ 0 present, > 0 under-prediction;
       sat   fraction of support channels saturated at the prompt;  cov  share of the flash's unexplained light explained.
     Gates (2x2 v2 defaults): mf >= 0.8, 0.8 <= alpha <= 3, L <= 16 cm, apc <= -0.5, sat <= 0.5, cov >= 0.1, whole >= 5 MeV.
  5. SELECTION per flash: fullest passing piece per (cluster, tip), best mf wins (overlap tie-break towards the fuller
     piece), clipped amplitude of both groups >= alpha_min; move, update the base; a cluster moves at most once.
Truth is used only to LABEL hypotheses and moves (energy within 10 ticks of the flash / of the cluster time).
python v14_michel_nd.py <dump_dir> [--events N] [--files 0000001,0000002] [--out JSON] [--mf-min 0.8 ...]"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, "..", "CLMatching_AlphaRelease"))
for p in (BASE, os.path.join(BASE, "M5p1"), os.path.join(BASE, "NewMLSection"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
from resid_fill import residual_flashes, render, _template  # noqa: E402

PULSE = os.path.join(BASE, "assets", "avg_pulse.npy")
SMALL = "/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/NewMLSection/runs/ndfull_small4x/best_model_arch.pt"
CUT = 10.0; PEAK = 105; GAIN = 200.0; TMP0 = 1_000_000


class Cfg:
    late_min_ticks = 12.0; unexpl_frac = 0.2; unexpl_min_adc = 5000.0; tail = 0.5; k_prom = 10.0; min_adc = 600.0; k_prom_late = 10.0; family_times = True; gain_tol = 0.3; max_flash_per_tpc = 8
    cluster_min_mev = 1.0; whole_max_mev = 80.0; whole_min_mev = 5.0; piece_min_cluster_mev = 10.0
    cluster_cos_min = 0.4; max_clusters_per_flash = 3
    link_cm = 3.0; tip_min_cm = 2.0; dense_r = 1.5; dense_sep = 3.0; dense_min_mev = 5.0; dense_rel = 1.5
    dense_cluster_min_mev = 5.0; free_cm = 3.0; dense_leak_mev = 2.0
    piece_cm = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0); piece_cm_probe = 5.0; mf_probe = 0.5
    piece_min_hits = 3; piece_min_mev = 1.5; max_frac = 0.5; mf_window = (-3, 45)
    mf_min = 0.8; mf_min_whole = 0.8; mf_tol = 0.02; apc_max = -0.5; sat_max = 0.5; cov_min = 0.1
    alpha_sel_lo = 0.8; alpha_sel_hi = 3.0; alpha_min = 0.65; support_frac = 0.9


def skeleton_tips(X, link_cm, tip_min_cm):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra, minimum_spanning_tree
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
    adj = {}
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


def dense_points(X, E, r, sep, min_mev, rel):
    tree = cKDTree(X); nb = tree.query_ball_point(X, r)
    ball = np.array([E[n].sum() for n in nb]); thr = max(min_mev, rel * float(np.median(ball)))
    chosen = []
    for i in np.argsort(-ball):
        if ball[i] < thr:
            break
        if all(np.linalg.norm(X[i] - X[j]) > sep for j in chosen):
            chosen.append(int(i))
    return chosen


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


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dump_dir"); ap.add_argument("--events", type=int, default=0); ap.add_argument("--files", default=None)
    ap.add_argument("--out", default=os.path.join(HERE, "michel_nd_out", "michel_nd.json"))
    for k in ("mf_min", "mf_min_whole", "apc_max", "sat_max", "cov_min", "alpha_sel_lo", "alpha_sel_hi", "tail", "unexpl_frac", "cluster_cos_min", "unexpl_min_adc", "k_prom_late", "gain_tol"):
        ap.add_argument("--" + k.replace("_", "-"), type=float, default=None)
    ap.add_argument("--apply", action="store_true", help="apply the gates and move (default: label only, no moves)")
    ap.add_argument("--no-family-times", action="store_true", help="candidate late times from residual peaks only")
    a = ap.parse_args()
    cfg = Cfg()
    for k in ("mf_min", "mf_min_whole", "apc_max", "sat_max", "cov_min", "alpha_sel_lo", "alpha_sel_hi", "tail", "unexpl_frac", "cluster_cos_min", "unexpl_min_adc", "k_prom_late", "gain_tol"):
        v = getattr(a, k)
        if v is not None:
            setattr(cfg, k, v)
    if a.no_family_times:
        cfg.family_times = False
    files = sorted(glob.glob(os.path.join(a.dump_dir, "full_dumps", "*.npz")))
    if a.files:
        tags = a.files.split(","); files = [f for f in files if any(os.path.basename(f).startswith(t) for t in tags)]
    if a.events:
        files = files[: a.events]
    from M5p1.first_stage_matching import load_first_stage_models
    from M5p1.first_stage_matching.config import FirstStageConfig
    from M5p1.first_stage_matching.streaming_prediction import process_clusters_to_imageMaps_streaming
    fcfg = FirstStageConfig(); fcfg.model.light_checkpoint = SMALL
    models = load_first_stage_models(fcfg); tmpl = np.asarray(models.waveform_template, np.float64)
    T = _template(PULSE)
    print("models on", models.device, flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    all_hyps = []; all_moves = []; ev_stats = []; t_begin = time.time()

    for f in files:
        t_ev = time.time()
        z = np.load(f, allow_pickle=True); tag = os.path.basename(f)[:-4]
        e = np.clip(z["e"], 0, None).astype(np.float64); ts = z["ts_final"].astype(np.float64); tru = z["truth_t0"].astype(np.float64)
        labs = z["labels"].astype(int); tpc = z["tpc"].astype(int); x, y, zz = (z[k].astype(np.float32) for k in ("x", "y", "z"))
        X = np.stack([x, y, zz], 1).astype(np.float64)
        A_all = z["full_light"].astype(np.float32); B_all = z["base_image"].astype(np.float32).copy(); veto = z["veto_mask"]
        ceil_all = z["sat_ceiling"].astype(np.float32)
        al, at, ar = z["amp_lab"].astype(int), z["amp_tpc"].astype(int), z["amp_row"].astype(np.float64)
        amp = {(int(L), int(c)): ar[i] for i, (L, c) in enumerate(zip(al, at))}
        n_light, n_ch, n_s = A_all.shape
        asg = np.isfinite(ts) & (ts >= 0); ok = np.isfinite(tru)
        lab_hits = {int(L): np.flatnonzero(labs == L) for L in np.unique(labs[labs >= 0])}
        E_of = {L: float(e[h].sum()) for L, h in lab_hits.items()}
        t_of = {}
        for L, h in lab_hits.items():
            v = ts[h]; v = v[np.isfinite(v) & (v >= 0)]
            if v.size:
                t_of[L] = float(np.median(v))
        # dense points event-wide (clusters >= dense_cluster_min_mev)
        dense_xyz = []; dense_of = {}
        for L, h in lab_hits.items():
            if E_of[L] < cfg.dense_cluster_min_mev or h.size < 3:
                continue
            ch = dense_points(X[h], e[h], cfg.dense_r, cfg.dense_sep, cfg.dense_min_mev, cfg.dense_rel)
            dense_of[L] = [int(h[i]) for i in ch]; dense_xyz.extend(X[dense_of[L]])
        dense_xyz = np.array(dense_xyz) if dense_xyz else np.zeros((0, 3)); dense_tree = cKDTree(dense_xyz) if dense_xyz.shape[0] else None
        # ---- 1. late flashes per TPC: candidate times = family times (a delayed daughter's flash is often already a
        #         family: 65 % of the ND delayed pieces sit within 5 ticks of one) + residual peaks (k_prom_late sigma),
        #         kept when the light there is UNEXPLAINED (2x2 test: residual >= unexpl_frac of the actual and >= unexpl_min_adc)
        from scipy.signal import find_peaks
        flashes = []
        for c in range(n_light):
            mc = tpc == c
            if not mc.any():
                continue
            fam_t = np.unique(np.round(ts[mc & asg & (labs >= 0)]))
            R = A_all[c].astype(np.float64) - B_all[c].astype(np.float64)
            chans = np.flatnonzero(~veto[c])
            if chans.size == 0:
                continue
            rs = R[chans].sum(0); sig = max(float(rs[:90].std()), 1.0)
            pk, _ = find_peaks(rs, prominence=cfg.k_prom_late * sig, distance=4); t_pk = (pk - PEAK).astype(float); t_pk = t_pk[(t_pk >= 0) & (t_pk <= 885)]
            cand_t = np.concatenate([fam_t, t_pk]) if cfg.family_times else t_pk
            cand_t = np.unique(np.round(cand_t))
            keep = []
            for tv in cand_t:                      # merge within 2 ticks
                if keep and tv - keep[-1] <= 2.0:
                    continue
                keep.append(float(tv))
            fl_c = []
            for tf in keep:
                k0 = int(round(tf)) + PEAK; sl = slice(max(0, k0 - 3), min(n_s, k0 + 6))
                a_ = np.clip(A_all[c][chans][:, sl], 0, None).max(1).sum()
                r_ch = np.clip(R[chans][:, sl], 0, None).max(1); F_ch = np.clip(B_all[c][chans][:, sl], 0, None).max(1)
                # gain-corrected unexplained light: the placed families may be under-predicted by up to gain_tol
                # (ND light-model systematics); only the residual beyond that counts as unexplained
                den = float(F_ch @ F_ch); g = min(max(float(r_ch @ F_ch) / den, 0.0), cfg.gain_tol) if den > 0 else 0.0
                r2 = float(np.clip(r_ch - g * F_ch, 0, None).sum())
                if a_ > 0 and r2 / a_ >= cfg.unexpl_frac and r2 >= cfg.unexpl_min_adc:
                    fl_c.append((float(tf), int(c), r2, float(r2 / a_)))
            fl_c.sort(key=lambda t: -t[2]); flashes += fl_c[: cfg.max_flash_per_tpc]
        flashes.sort(key=lambda t: -t[2])
        # ---- geometry per (cluster, tpc)
        geom = {}

        def cluster_geom(L, c):
            key = (L, c)
            if key in geom:
                return geom[key]
            from scipy.sparse.csgraph import dijkstra
            h = lab_hits[L]; h = h[tpc[h] == c]; g = dict(hits=h, tips=[], D=None, excl=None, free=[])
            if E_of[L] >= cfg.piece_min_cluster_mev and h.size >= 2 * cfg.piece_min_hits:
                tips, G = skeleton_tips(X[h], cfg.link_cm, cfg.tip_min_cm)
                if tips:
                    excl = np.zeros(h.size, bool)
                    if L in dense_of and dense_of[L]:
                        tr = cKDTree(X[h])
                        for hd in dense_of[L]:
                            excl[tr.query_ball_point(X[hd], cfg.dense_r)] = True
                    free = [bool(d > cfg.free_cm) for d in dense_tree.query(X[h][tips])[0]] if dense_tree is not None else [True] * len(tips)
                    D = dijkstra(G, indices=tips); D[~np.isfinite(D)] = -1
                    g.update(tips=tips, D=D, excl=excl, free=free)
            geom[key] = g; return g

        def piece_masks(L, c, ti, Ls):
            g = cluster_geom(L, c); h = g["hits"]; D, excl = g["D"], g["excl"]; out = []
            for Lc in Ls:
                mk = (D[ti] >= 0) & (D[ti] <= Lc)
                if e[h][mk & excl].sum() > cfg.dense_leak_mev or mk.sum() > h.size - cfg.piece_min_hits or e[h][mk].sum() > cfg.max_frac * E_of[L]:
                    break
                if mk.sum() >= cfg.piece_min_hits and e[h][mk].sum() >= cfg.piece_min_mev:
                    out.append((Lc, mk))
            return out

        # ---- piece predictions (perceiver), cached per (cluster, tpc, tip, L)
        pred = {}

        def predict_missing(keys_masks):
            todo = [(k, mk) for k, mk in keys_masks if k not in pred]
            if not todo:
                return
            passes = []
            for j, (k, mk) in enumerate(todo):
                h = geom[(k[0], k[1])]["hits"][mk]
                for pm in passes:
                    if not pm["used"][h].any():
                        pm["lab"][h] = TMP0 + j; pm["used"][h] = True; pm["want"].add(TMP0 + j); pm["keys"].append((j, k)); break
                else:
                    pm = {"lab": np.full(labs.size, -1, np.int64), "used": np.zeros(labs.size, bool), "want": set(), "keys": []}
                    pm["lab"][h] = TMP0 + j; pm["used"][h] = True; pm["want"].add(TMP0 + j); pm["keys"].append((j, k)); passes.append(pm)
            for pm in passes:
                im, _ = process_clusters_to_imageMaps_streaming(x, y, zz, e.astype(np.float32), tpc, pm["lab"], model=models.light_model, template=tmpl,
                                                                batch_size=8, restrict_clusters=pm["want"], device_policy="auto")
                for j, k in pm["keys"]:
                    img = im.get((TMP0 + j, k[1]))
                    ap_ = np.asarray(img, np.float64).max(1) if img is not None else np.zeros(n_ch)
                    ap_[veto[k[1]]] = 0.0; pred[k] = ap_

        moved = set(); n_hyp = 0; ev_moves = []; ev_hyps = []
        # delayed-daughter class (truth > 12 ticks after the host cluster's majority truth, hosts >= 10 MeV, in window)
        dmask = np.zeros(labs.size, bool)
        for L, h in lab_hits.items():
            m = h[ok[h] & (tru[h] >= 0) & (tru[h] <= 885)]
            if E_of[L] < 10 or m.size < 5:
                continue
            tmaj = np.median(tru[m]); dmask[m[tru[m] - tmaj > 12]] = True
        E_del = float(e[dmask].sum()); E_del_before = float((e[dmask] * (np.abs(ts[dmask] - tru[dmask]) <= CUT)).sum())
        ts_before = ts.copy()
        w_all = {}
        for c in range(n_light):
            w_all[c] = 1.0 / (GAIN * np.maximum(A_all[c].astype(np.float64), 100.0))
        for tf, c, famp, unexpl in flashes:
            A = A_all[c].astype(np.float64); B = B_all[c].astype(np.float64); w = w_all[c]; ceil = ceil_all[c].astype(np.float64)
            R = A - B
            k_f = int(round(tf)) + PEAK; sl_f = slice(max(0, k_f + cfg.mf_window[0]), min(n_s, k_f + cfg.mf_window[1]))
            pat_f = R[:, max(0, k_f - 2):k_f + 3].mean(1)
            # ---- 2. candidates
            scored = []
            for L in np.unique(labs[(tpc == c) & (labs >= 0)]):
                L = int(L)
                if L in moved or L not in t_of or E_of[L] < cfg.cluster_min_mev or tf - t_of[L] <= cfg.late_min_ticks or (L, c) not in amp:
                    continue
                a_c = amp[(L, c)].copy(); a_c[veto[c]] = 0; sup = support(a_c, cfg.support_frac)
                if sup.size == 0:
                    continue
                cs = float(a_c[sup] @ np.clip(pat_f[sup], 0, None) / (np.linalg.norm(a_c[sup]) * np.linalg.norm(np.clip(pat_f[sup], 0, None)) + 1e-9))
                if cs >= cfg.cluster_cos_min:
                    scored.append((cs, L))
            cands = [L for _, L in sorted(scored, reverse=True)[: cfg.max_clusters_per_flash]]
            if not cands:
                continue
            own = {L: render(amp[(L, c)] * (~veto[c]), t_of[L], n_s, T) for L in cands}

            def evaluate(hyps):
                predict_missing([((L, c, tip, Lc), mk) for L, mk, tip, Lc in hyps if tip >= 0])
                out = []
                for L, mk, tip, Lc in hyps:
                    t_c = t_of[L]; h = geom[(L, c)]["hits"] if tip >= 0 else lab_hits[L]
                    a_p = pred[(L, c, tip, Lc)] if tip >= 0 else amp[(L, c)] * (~veto[c])
                    if a_p.max() <= 0:
                        continue
                    E_p = float(e[h][mk].sum()); sup = support(a_p, cfg.support_frac)
                    Sg = render(a_p, tf, n_s, T); Sp = render(a_p, t_c, n_s, T)
                    rest = np.clip(own[L] - Sp, 0, None) if tip >= 0 else np.zeros_like(Sp)
                    m_rest = B - own[L] + rest; r = A - m_rest
                    num = float((r * Sg * w).sum()); den = float((Sg * Sg * w).sum()); alpha = num / max(den, 1e-9)
                    ww = 1.0 / np.sqrt(GAIN * np.maximum(A[sup][:, sl_f], 100.0))
                    mf = wcos(Sg[sup][:, sl_f], r[sup][:, sl_f], ww)
                    kp = int(round(t_c)) + PEAK; sl_p = slice(max(0, kp - 5), min(n_s, kp + 60))
                    Spw = Sp[sup][:, sl_p]; m_cur = B[sup][:, sl_p]; wwp = w[sup][:, sl_p]
                    r_c = A[sup][:, sl_p] - np.minimum(m_cur, ceil[sup][:, None])
                    dc = float((Spw * Spw * wwp).sum()); apc = float((r_c * Spw * wwp).sum()) / dc if dc > 0 else np.nan
                    sat = float((m_cur.max(1) >= 0.95 * ceil[sup]).mean()) if sup.size else 1.0
                    rl = np.clip(A[sup][:, sl_f] - B[sup][:, sl_f], 0, None); cov = float(np.minimum(Sg[sup][:, sl_f], rl).sum() / max(rl.sum(), 1e-9))
                    whole = tip < 0
                    passes_ = (mf >= (cfg.mf_min_whole if whole else cfg.mf_min) and cfg.alpha_sel_lo <= alpha <= cfg.alpha_sel_hi and np.isfinite(apc)
                               and apc <= cfg.apc_max and sat <= cfg.sat_max and cov >= cfg.cov_min and (not whole or E_p >= cfg.whole_min_mev))
                    hp = h[mk]; okh = ok[hp]
                    true_frac = float((e[hp][okh] * (np.abs(tru[hp][okh] - tf) <= CUT)).sum() / max(E_p, 1e-9))
                    cur_frac = float((e[hp][okh] * (np.abs(tru[hp][okh] - t_c) <= CUT)).sum() / max(E_p, 1e-9))
                    out.append(dict(L=L, mk=mk, tip=tip, Lc=Lc, t_c=t_c, E_p=E_p, n_p=int(mk.sum()), mf=mf, alpha=alpha, apc=apc, sat=sat, cov=cov, ok=passes_,
                                    true_frac=true_frac, cur_frac=cur_frac, a_p=a_p, Sg=Sg, Sp=Sp, rest=rest, E_L=E_of[L]))
                return out

            # ---- 3. hypotheses
            hyps = []
            for L in cands:
                if cfg.whole_min_mev <= E_of[L] <= cfg.whole_max_mev:
                    hyps.append((L, np.ones(lab_hits[L].size, bool), -1, float("inf")))
                g = cluster_geom(L, c)
                for ti, tip in enumerate(g["tips"]):
                    if g["excl"][tip] or not g["free"][ti]:
                        continue
                    pcs = piece_masks(L, c, ti, [Lc for Lc in cfg.piece_cm if Lc <= cfg.piece_cm_probe])
                    if pcs:
                        Lc, mk = pcs[-1]; hyps.append((L, mk, int(tip), Lc))
            if not hyps:
                continue
            res = evaluate(hyps)
            good_tips = {(r["L"], r["tip"]) for r in res if r["tip"] >= 0 and r["mf"] >= cfg.mf_probe}
            hyps2 = []
            for L in cands:
                g = cluster_geom(L, c)
                for ti, tip in enumerate(g["tips"]):
                    if (L, int(tip)) not in good_tips:
                        continue
                    done = {r["Lc"] for r in res if r["L"] == L and r["tip"] == tip}
                    for Lc, mk in piece_masks(L, c, ti, cfg.piece_cm):
                        if Lc not in done:
                            hyps2.append((L, mk, int(tip), Lc))
            if hyps2:
                res += evaluate(hyps2)
            n_hyp += len(res)
            for r in res:
                ev_hyps.append(dict(ev=tag, tpc=c, tf=tf, unexpl=unexpl, L=r["L"], E_L=r["E_L"], tip=r["tip"], Lc=(r["Lc"] if np.isfinite(r["Lc"]) else -1), E_p=r["E_p"], n_p=r["n_p"], t_c=r["t_c"],
                                    mf=r["mf"], alpha=r["alpha"], apc=r["apc"], sat=r["sat"], cov=r["cov"], ok=bool(r["ok"]), true_frac=r["true_frac"], cur_frac=r["cur_frac"]))
            if not a.apply:
                continue
            # ---- 5. selection
            passing = [r for r in res if r["ok"]]
            if not passing:
                continue
            reps = {}
            for r in passing:
                key = (r["L"], r["tip"])
                if key not in reps or r["E_p"] > reps[key]["E_p"]:
                    reps[key] = r
            top = max(reps.values(), key=lambda r: r["mf"]); best = top
            for r in reps.values():
                if r is not top and r["L"] == top["L"] and r["mf"] >= top["mf"] - cfg.mf_tol and r["E_p"] > best["E_p"] and r["tip"] >= 0 and top["tip"] >= 0:
                    inter = int((r["mk"] & top["mk"]).sum())
                    if inter >= 0.5 * min(int(r["mk"].sum()), int(top["mk"].sum())):
                        best = r
            L = best["L"]; t_c = t_of[L]
            r_p = A - (B - own[L]) - best["rest"]; num = float((r_p * best["Sg"] * w).sum()); den = float((best["Sg"] ** 2 * w).sum())
            r_r = A - (B - own[L]) - best["Sg"]; num_r = float((r_r * best["rest"] * w).sum()); den_r = float((best["rest"] ** 2 * w).sum())
            a_piece = num / max(den, 1e-9) if den > 0 else 1.0; a_rest = num_r / max(den_r, 1e-9) if den_r > 0 else 1.0
            if min(a_piece, a_rest) < cfg.alpha_min:
                continue
            h = (geom[(L, c)]["hits"] if best["tip"] >= 0 else lab_hits[L])[best["mk"]]
            okh = ok[h]; was = float((e[h][okh] * (np.abs(t_c - tru[h][okh]) <= CUT)).sum()); now = float((e[h][okh] * (np.abs(tf - tru[h][okh]) <= CUT)).sum())
            ts[h] = tf; moved.add(L)
            B_all[c] = (B + best["Sg"] - best["Sp"]).astype(np.float32)
            ev_moves.append(dict(ev=tag, tpc=c, tf=tf, t_c=t_c, L=L, tip=best["tip"], Lc=(best["Lc"] if np.isfinite(best["Lc"]) else -1), E_p=best["E_p"], mf=best["mf"], alpha=best["alpha"],
                                 apc=best["apc"], cov=best["cov"], was=was, now=now))
        dt = time.time() - t_ev
        all_hyps += ev_hyps; all_moves += ev_moves
        fixed = sum(max(m["now"] - m["was"], 0) for m in ev_moves); broke = sum(max(m["was"] - m["now"], 0) for m in ev_moves)
        E_del_after = float((e[dmask] * (np.abs(ts[dmask] - tru[dmask]) <= CUT)).sum())
        chg = ts != ts_before
        E_moved_del = float(e[chg & dmask].sum()); E_moved_nondel = float(e[chg & ~dmask].sum())
        ev_stats.append(dict(ev=tag, n_flash=len(flashes), n_hyp=n_hyp, n_pred=len(pred), n_moves=len(ev_moves), fixed=fixed, broke=broke, t=dt,
                             E_del=E_del, E_del_before=E_del_before, E_del_after=E_del_after, E_moved_del=E_moved_del, E_moved_nondel=E_moved_nondel))
        print(f"{tag}: late flashes {len(flashes)}, hypotheses {n_hyp}, piece predictions {len(pred)}, moves {len(ev_moves)} fixed {fixed:.0f} broke {broke:.0f} MeV, {dt:.0f}s  (total {time.time()-t_begin:.0f}s)", flush=True)
    json.dump(dict(cfg={k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("_")}, apply=a.apply, hyps=all_hyps, moves=all_moves, events=ev_stats), open(a.out, "w"), default=float)
    # ---- summary
    H = all_hyps
    if H:
        tr = np.array([h["true_frac"] for h in H]); cu = np.array([h["cur_frac"] for h in H]); okk = np.array([h["ok"] for h in H]); Ep = np.array([h["E_p"] for h in H])
        true = tr >= 0.5; false = cu >= 0.5
        print(f"\n{len(files)} events, {len(H)} hypotheses: true (>= 50 % of the piece energy at the flash time) {true.sum()} = {Ep[true].sum()/1e3:.2f} GeV; currently-right pieces {false.sum()} = {Ep[false].sum()/1e3:.2f} GeV")
        for k in ("mf", "alpha", "apc", "sat", "cov"):
            v = np.array([h[k] for h in H], float)
            print(f"   {k:5s} true p10/50/90 {np.nanpercentile(v[true], [10,50,90]).round(2)}   currently-right {np.nanpercentile(v[false], [10,50,90]).round(2)}")
        print(f"   gates pass: true {np.mean(okk[true]):.3f} ({Ep[true & okk].sum()/1e3:.2f} GeV)   currently-right {np.mean(okk[false]):.3f} ({Ep[false & okk].sum()/1e3:.2f} GeV)")
    if all_moves:
        was = sum(m["was"] for m in all_moves); now = sum(m["now"] for m in all_moves)
        print(f"moves {len(all_moves)}: energy right before {was:.0f} -> after {now:.0f} MeV; fixed {sum(e_['fixed'] for e_ in ev_stats)/1e3:.2f} GeV broke {sum(e_['broke'] for e_ in ev_stats)/1e3:.2f} GeV; time/ev {np.mean([e_['t'] for e_ in ev_stats]):.1f} s")
    Ed = sum(e_["E_del"] for e_ in ev_stats); Eb = sum(e_["E_del_before"] for e_ in ev_stats); Ea = sum(e_["E_del_after"] for e_ in ev_stats)
    print(f"delayed-daughter class: {Ed/1e3:.2f} GeV in {len(ev_stats)} events; correctly timed before {Eb/max(Ed,1e-9):.3f} -> after {Ea/max(Ed,1e-9):.3f}; moved energy in the class {sum(e_['E_moved_del'] for e_ in ev_stats)/1e3:.2f} GeV, outside it {sum(e_['E_moved_nondel'] for e_ in ev_stats)/1e3:.2f} GeV")


if __name__ == "__main__":
    main()
