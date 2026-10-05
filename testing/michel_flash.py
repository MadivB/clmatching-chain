"""Stage 4.86 (ND): Michel re-timing from UNEXPLAINED FLASHES AT TRACK ENDS (user's request 2026-10-05; the light study
showed the Michels the piece-first stage never sees are bright, median 1.7k ADC on their best channel, and that 131 per
63 spills sit at a time where no family is placed: the skeleton rules, not the light, were the ceiling).

Per spill:
  1. residual flashes per light TPC: positive residual R = actual - chain model on non-vetoed channels; a flash is a
     local maximum of the summed residual with sum >= flash_sum_min, >= flash_nch_min channels where the residual is
     >= flash_ch_min ADC AND >= flash_ch_frac of the actual, a rise (S[t] - S[t-6]) / S[t] >= flash_rise_min (tails of
     big flashes rise slowly), and no placed family within fam_excl_ticks; at most flash_max_per_tpc per TPC.
     The lit channels give the flash a y range (ND light channels sit on the +-z faces of the TPC, sorted in y).
  2. candidate pieces per flash: EVERY skeleton tip of every cluster >= host_min_mev in that TPC whose current time is
     at least late_min_ticks before the flash and whose tip lies within y_window_cm of the flash's lit y range (no Bragg,
     extent or dE/dx rules), grown 2-16 cm along the geodesic (piece 5-60 MeV, host remainder >= remainder_min_mev);
     plus whole templated clusters 5-80 MeV in the y window (earlier than the flash).
  3. one batched perceiver call for the unique probe pieces (<= 5 cm);
  4. each (flash, probe) scored AT THE FLASH TIME with the piece-first stage's tests (mf on the support channels, alpha,
     apc = prompt deficit at the host time, cov, sat, gain-corrected unexplained light);
  5. passing probes refined over lengths (second batch); best hypothesis per flash (max mf), one move per cluster;
     ts and base updated.
predict_fn / inputs as in michel_nd.run_michel_nd, plus ch_pos (n_light, n_ch, 3) channel positions (NaN where unknown).
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np

from michel_nd import PEAK, GAIN, CUT, skeleton_vox, render, support, wcos


class MichelFlashConfig:
    flash_sum_min = 10000.0; flash_ch_min = 300.0; flash_ch_frac = 0.3; flash_nch_min = 5; flash_rise_min = 0.2; fam_excl_ticks = 10.0
    local_baseline_ticks = 12   # scoring residual is measured above the per-channel median residual over the (k_f-30 .. k_f-4) ticks before the flash (over-predicted tails of big families otherwise swallow a small flash); 0 = off
    probe_long = False          # 2026-10-05 S3: scoring the longest mask as well as the 5 cm probe changed no move on 63 spills and cost 1.7 s per spill; off
    flash_max_per_tpc = 8; flash_merge_ticks = 6
    lit_unexpl_min = 0.3      # on the lit channels, residual / actual at the peak (a Michel is ~half of what is there on its channels)
    top_frac = 0.5; y_span_max = 120.0   # channels above top_frac x max residual define the flash's y range; a whole-wall residual (big-flash mismatch) is not a Michel
    tips_per_flash = 8
    late_min_ticks = 12.0; host_min_mev = 10.0; whole_min_mev = 5.0; whole_max_mev = 80.0; y_window_cm = 30.0
    link_cm = 3.0; tip_min_cm = 2.0; vox_cm = 1.0; piece_cm = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0); piece_cm_probe = 5.0
    piece_min_hits = 3; piece_min_mev = 5.0; piece_max_mev = 60.0; remainder_min_mev = 10.0; min_tpc_hits = 6
    mf_window = (-3, 45); mf_min = 0.8; mf_tol = 0.02; apc_max = -0.5; sat_max = 0.5; cov_min = 0.2; alpha_lo = 0.8; alpha_hi = 6.0
    whole_mf_min = 0.8; whole_apc_max = -0.5; whole_alpha_hi = 2.0; support_frac = 0.9; unexpl_gain_tol = 0.3; unexpl_min_frac = 0.1
    tip_face_min_cm = 5.0; box = (-347.0, 347.0, -215.0, 82.0, 418.0, 913.0)
    max_probes = 600


def find_flashes(A, B, veto, fam_times, cfg, ch_y):
    """residual flashes of one TPC: list of dict(tf, S, lit (channel idx), y_lo, y_hi)"""
    g = ~veto
    if g.sum() < 10:
        return []
    R = np.clip(A - B, 0, None); R[~g] = 0.0; S = R.sum(0); n_s = S.size
    fam = np.asarray(fam_times, float)
    out = []
    for t in range(PEAK + 6, min(885 + PEAK, n_s - 1)):
        s = S[t]
        if s < cfg.flash_sum_min or s < S[t - cfg.flash_merge_ticks: t + cfg.flash_merge_ticks + 1].max() or (s - S[t - 6]) < cfg.flash_rise_min * s:
            continue
        lit = np.flatnonzero((R[:, t] >= cfg.flash_ch_min) & (R[:, t] >= cfg.flash_ch_frac * A[:, t]))
        if lit.size < cfg.flash_nch_min:
            continue
        tf = float(t - PEAK)
        if fam.size and np.min(np.abs(fam - tf)) <= cfg.fam_excl_ticks:
            continue
        if R[lit, t].sum() < cfg.lit_unexpl_min * A[lit, t].sum():
            continue
        top = np.flatnonzero(R[:, t] >= cfg.top_frac * R[:, t].max()); ys = ch_y[top]; ys = ys[np.isfinite(ys)]
        if ys.size and (ys.max() - ys.min()) > cfg.y_span_max:
            continue
        w = R[top, t]; fin = np.isfinite(ch_y[top])
        out.append(dict(tf=tf, S=float(s), lit=lit, y_lo=float(ys.min()) if ys.size else -1e9, y_hi=float(ys.max()) if ys.size else 1e9,
                        y_c=float(np.average(ch_y[top][fin], weights=w[fin])) if fin.any() else np.nan))
    out.sort(key=lambda d: -d["S"])
    return out[: cfg.flash_max_per_tpc]


def run_michel_flash(*, xyz, e, tpc, labels, ts, amp, actual, base, veto, ceil, template, predict_fn: Callable, ch_pos, cfg: MichelFlashConfig = None,
                     truth_t0=None, apply=True, debug_tag=None, debug_out=None):
    cfg = cfg or MichelFlashConfig(); t_start = time.time(); tim = {}
    T = np.asarray(template, np.float64)
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
    # 1. flashes
    t0 = time.time(); flashes = []
    for c in range(n_light):
        hc = np.flatnonzero(tpc == c)
        if hc.size == 0:
            continue
        fam = [t_of[L] for L in np.unique(labels[hc]) if L >= 0 and L in t_of]
        ch_y = np.asarray(ch_pos[c][:, 1], float) if ch_pos is not None else np.full(n_ch, np.nan)
        for fl in find_flashes(actual[c].astype(np.float64), base[c].astype(np.float64), veto[c], fam, cfg, ch_y):
            fl["c"] = c; flashes.append(fl)
    tim["flash"] = time.time() - t0
    # 2. candidates: tips per (L, c) cached; pieces per tip cached; pair with flashes by time and y
    t0 = time.time(); skel = {}; tips_of = {}; cands = {}; pairs = []; wholes = []
    bx = cfg.box
    for fl in flashes:
        c = fl["c"]; tf = fl["tf"]; y_lo = fl["y_lo"] - cfg.y_window_cm; y_hi = fl["y_hi"] + cfg.y_window_cm
        for L in np.unique(labels[(tpc == c) & (labels >= 0)]):
            L = int(L)
            if L not in t_of or t_of[L] > tf - cfg.late_min_ticks or (L, c) not in amp:
                continue
            h = lab_hits[L]; h = h[tpc[h] == c]
            if cfg.whole_min_mev <= E_of[L] <= cfg.whole_max_mev:
                yy = X[h, 1]
                if yy.max() >= y_lo and yy.min() <= y_hi:
                    wholes.append((fl, L))
            if E_of[L] < cfg.host_min_mev or h.size < max(2 * cfg.piece_min_hits, cfg.min_tpc_hits):
                continue
            if X[h, 1].max() < y_lo or X[h, 1].min() > y_hi:
                continue
            key = (L, c)
            if key not in skel:
                tips, D = skeleton_vox(X[h], cfg.link_cm, cfg.tip_min_cm, cfg.vox_cm); skel[key] = (h, tips, D)
            h, tips, D = skel[key]
            if not tips:
                continue
            E_h = float(e[h].sum())
            for ti, tip in enumerate(tips):
                pt = X[h][tip]
                if pt[1] < y_lo or pt[1] > y_hi:
                    continue
                if cfg.tip_face_min_cm > 0 and min(pt[0] - bx[0], bx[1] - pt[0], pt[1] - bx[2], bx[3] - pt[1], pt[2] - bx[4], bx[5] - pt[2]) < cfg.tip_face_min_cm:
                    continue
                ck = (L, c, int(tip))
                if ck not in cands:
                    masks = {}
                    for Lc in cfg.piece_cm:
                        mk = (D[ti] >= 0) & (D[ti] <= Lc); Ep = float(e[h][mk].sum())
                        if mk.sum() > h.size - cfg.piece_min_hits or Ep > cfg.piece_max_mev or (E_h - Ep) < cfg.remainder_min_mev:
                            break
                        if mk.sum() >= cfg.piece_min_hits and Ep >= cfg.piece_min_mev:
                            masks[Lc] = mk
                    if not masks:
                        cands[ck] = None; continue
                    probe = max(Lc for Lc in masks if Lc <= cfg.piece_cm_probe) if any(Lc <= cfg.piece_cm_probe for Lc in masks) else min(masks)
                    cands[ck] = dict(L=L, c=c, tip=int(tip), hits=h, masks=masks, probe=probe)
                if cands[ck] is not None:
                    pairs.append((fl, ck, abs(float(pt[1]) - fl["y_c"]) if np.isfinite(fl["y_c"]) else 0.0))
    cands = {k: v for k, v in cands.items() if v is not None}
    by_flash = {}
    for fl, ck, dy in pairs:
        by_flash.setdefault(id(fl), []).append((dy, fl, ck))
    pairs = []
    for lst in by_flash.values():
        lst.sort(key=lambda t: t[0])
        pairs += [(fl, ck) for _, fl, ck in lst[: cfg.tips_per_flash]]
    tim["geom"] = time.time() - t0
    if debug_out is not None and debug_tag is not None:
        debug_out.append(dict(kind="flashes", flashes=[dict(c=f["c"], tf=f["tf"], S=f["S"], n_lit=int(f["lit"].size), y_lo=f["y_lo"], y_hi=f["y_hi"]) for f in flashes]))
        cov_tag = {}
        for cd in cands.values():
            for Lc, mk in cd["masks"].items():
                hp = cd["hits"][mk]; tg = debug_tag[hp]
                for tv in np.unique(tg[tg >= 0]):
                    cov_tag[int(tv)] = max(cov_tag.get(int(tv), 0.0), float(e[hp][tg == tv].sum()))
        debug_out.append(dict(kind="geometry", covered=cov_tag, n_cand=len(cands), n_pairs=len(pairs), n_whole=len(wholes)))
    # 3. perceiver for probes (cap by flash brightness order: pairs are already in flash order within TPC)
    pred = {}

    def predict(items):
        todo = [(k, h) for k, h in items if k not in pred]
        if todo:
            out = predict_fn(todo)
            for k, _ in todo:
                ap_ = np.asarray(out.get(k, np.zeros(n_ch)), np.float64).copy(); ap_[veto[k[1]]] = 0.0; pred[k] = ap_
    t0 = time.time()
    probe_items = []
    seen = set()
    for fl, ck in pairs:
        cd = cands[ck]; lens = [cd["probe"]] + ([max(cd["masks"])] if cfg.probe_long and max(cd["masks"]) != cd["probe"] else [])
        for Lc in lens:
            pk = (cd["L"], cd["c"], cd["tip"], Lc)
            if pk not in seen:
                seen.add(pk); probe_items.append((pk, cd["hits"][cd["masks"][Lc]]))
        if len(probe_items) >= cfg.max_probes:
            break
    predict(probe_items); tim["pred1"] = time.time() - t0; n_pred1 = len(pred)
    # 4. scoring at the flash time
    w_cache = {}

    def weights(c):
        if c not in w_cache:
            w_cache[c] = 1.0 / (GAIN * np.maximum(actual[c].astype(np.float64), 100.0))
        return w_cache[c]

    ch_cache = {}

    def chans_ok(c):
        if c not in ch_cache:
            ch_cache[c] = np.flatnonzero(~veto[c])
        return ch_cache[c]

    def score(a_p, c, L, hits_p, tf, own_c, rest, A, B, w, cl, whole=False):
        t_c = t_of[L]; sup = support(a_p, cfg.support_frac)
        Sg = render(a_p, tf, n_s, T); Sp = render(a_p, t_c, n_s, T); r = A - (B - own_c + rest)
        k_f = int(round(tf)) + PEAK
        if cfg.local_baseline_ticks > 0 and k_f - 4 > 0:
            r = r - np.median(r[:, max(0, k_f - cfg.local_baseline_ticks): k_f - 4], axis=1, keepdims=True)
        sl_f = slice(max(0, k_f + cfg.mf_window[0]), min(n_s, k_f + cfg.mf_window[1]))
        # amplitude scale IN THE FLASH WINDOW (the full-trace product multiplies the template's long tail with the decaying
        # residual of earlier big flashes and goes negative)
        num = float((r[:, sl_f] * Sg[:, sl_f] * w[:, sl_f]).sum()); den = float((Sg[:, sl_f] * Sg[:, sl_f] * w[:, sl_f]).sum()); alpha = num / max(den, 1e-9)
        ww = 1.0 / np.sqrt(GAIN * np.maximum(A[sup][:, sl_f], 100.0)); mf = wcos(Sg[sup][:, sl_f], r[sup][:, sl_f], ww)
        kp = int(round(t_c)) + PEAK; sl_p = slice(max(0, kp - 5), min(n_s, kp + 60))
        Spw = Sp[sup][:, sl_p]; m_cur = B[sup][:, sl_p]; wwp = w[sup][:, sl_p]; r_c = A[sup][:, sl_p] - np.minimum(m_cur, cl[sup][:, None])
        dc = float((Spw * Spw * wwp).sum()); apc = float((r_c * Spw * wwp).sum()) / dc if dc > 0 else np.nan
        sat = float((m_cur.max(1) >= 0.95 * cl[sup]).mean()) if sup.size else 1.0
        rl = np.clip(r[sup][:, sl_f], 0, None); cov = float(np.minimum(Sg[sup][:, sl_f], rl).sum() / max(rl.sum(), 1e-9))
        chv = chans_ok(c); sl_u = slice(max(0, k_f - 3), min(n_s, k_f + 6))
        a_u = np.clip(A[chv][:, sl_u], 0, None).max(1).sum(); r_u = np.clip(A[chv][:, sl_u] - B[chv][:, sl_u], 0, None).max(1); F_u = np.clip(B[chv][:, sl_u], 0, None).max(1)
        den_u = float(F_u @ F_u); g_u = min(max(float(r_u @ F_u) / den_u, 0.0), cfg.unexpl_gain_tol) if den_u > 0 else 0.0
        unexpl = float(np.clip(r_u - g_u * F_u, 0, None).sum() / max(a_u, 1e-9))
        E_p = float(e[hits_p].sum())
        mf_min_ = cfg.whole_mf_min if whole else cfg.mf_min; apc_max_ = cfg.whole_apc_max if whole else cfg.apc_max
        passes = (mf >= mf_min_ and cfg.alpha_lo <= alpha <= (cfg.whole_alpha_hi if whole else cfg.alpha_hi) and np.isfinite(apc) and apc <= apc_max_
                  and sat <= cfg.sat_max and cov >= cfg.cov_min and E_p >= cfg.piece_min_mev and unexpl >= cfg.unexpl_min_frac)
        return dict(mf=mf, alpha=alpha, apc=apc, sat=sat, cov=cov, unexpl=unexpl, E_p=E_p, ok=passes, Sg=Sg, Sp=Sp, tf=tf, hits=hits_p)

    def dbg(sc, kind, fl):
        if debug_out is None or debug_tag is None:
            return
        hp = sc["hits"]; tg = debug_tag[hp]; comp = {int(tv): float(e[hp][tg == tv].sum()) for tv in np.unique(tg[tg >= 0])}
        rec = dict(kind=kind, L=int(sc["L"]), c=int(sc["c"]), tip=int(sc.get("tip", -1)), Lc=float(sc.get("Lc", -1)), tf=float(sc["tf"]), t_c=float(t_of[sc["L"]]), flash_S=fl["S"],
                   E_p=sc["E_p"], mf=sc["mf"], alpha=sc["alpha"], apc=sc["apc"], sat=sc["sat"], cov=sc["cov"], unexpl=sc["unexpl"], ok=bool(sc["ok"]), comp=comp)
        if truth_t0 is not None:
            okh = ok[hp]; Et = float(e[hp][okh].sum())
            if Et > 0:
                rec["right_now"] = float((e[hp][okh] * (np.abs(sc["tf"] - truth_t0[hp][okh]) <= CUT)).sum() / Et)
                rec["right_before"] = float((e[hp][okh] * (np.abs(t_of[sc["L"]] - truth_t0[hp][okh]) <= CUT)).sum() / Et)
        debug_out.append(rec)

    t0 = time.time(); res = []; env = {}

    def tpc_env(c):
        if c not in env:
            env[c] = (actual[c].astype(np.float64), base[c].astype(np.float64), weights(c), np.asarray(ceil[c], np.float64))
        return env[c]

    for fl, ck in pairs:
        cd = cands[ck]; L, c, tip = cd["L"], cd["c"], cd["tip"]
        lens = [cd["probe"]] + ([max(cd["masks"])] if cfg.probe_long and max(cd["masks"]) != cd["probe"] else [])
        for Lc in lens:
            a_p = pred.get((L, c, tip, Lc))
            if a_p is None or a_p.max() <= 0:
                continue
            A, B, w, cl = tpc_env(c)
            own_c = render(amp[(L, c)] * (~veto[c]), t_of[L], n_s, T); Sp = render(a_p, t_of[L], n_s, T); rest = np.clip(own_c - Sp, 0, None)
            hp = cd["hits"][cd["masks"][Lc]]
            sc = score(a_p, c, L, hp, fl["tf"], own_c, rest, A, B, w, cl); sc.update(L=L, c=c, tip=tip, Lc=Lc, whole=False, cd=cd, fl=fl); res.append(sc); dbg(sc, "probe", fl)
    for fl, L in wholes:
        c = fl["c"]; a_p = amp[(L, c)] * (~veto[c])
        if a_p.max() <= 0:
            continue
        A, B, w, cl = tpc_env(c); own_c = render(a_p, t_of[L], n_s, T)
        sc = score(a_p, c, L, lab_hits[L], fl["tf"], own_c, np.zeros_like(own_c), A, B, w, cl, whole=True); sc.update(L=L, c=c, tip=-1, Lc=-1, whole=True, cd=None, fl=fl); res.append(sc); dbg(sc, "whole", fl)
    tim["score"] = time.time() - t0
    # 5. refine + select
    t0 = time.time(); passing = []; seen_p = set()
    for r in sorted([r for r in res if r["ok"] and not r["whole"]], key=lambda r: -r["mf"]):
        kk = (r["c"], r["fl"]["tf"], r["L"], r["tip"])
        if kk not in seen_p:
            seen_p.add(kk); passing.append(r)
    predict([((r["L"], r["c"], r["tip"], Lc), r["cd"]["hits"][mk]) for r in passing for Lc, mk in r["cd"]["masks"].items()]); tim["pred2"] = time.time() - t0
    t0 = time.time(); final = [r for r in res if r["ok"] and r["whole"]]
    for r in passing:
        cd = r["cd"]; L, c = r["L"], r["c"]; A, B, w, cl = tpc_env(c)
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
                sc.update(L=L, c=c, tip=r["tip"], Lc=Lc, whole=False, cd=cd, fl=r["fl"]); dbg(sc, "refine", r["fl"])
                if best is None or sc["mf"] > best["mf"] + cfg.mf_tol or (sc["mf"] >= best["mf"] - cfg.mf_tol and sc["E_p"] > best["E_p"]):
                    best = sc
        if best is not None:
            final.append(best)
    final.sort(key=lambda r: -r["mf"]); moved = set(); used_flash = set(); moves = []
    for r in final:
        L, c = r["L"], r["c"]; fid = (c, r["fl"]["tf"])
        if L in moved or fid in used_flash:
            continue
        hits = r["hits"]; rec = dict(cluster=int(L), tpc=int(c), tip=int(r["tip"]), L_cm=float(r["Lc"]), whole=bool(r["whole"]), E=r["E_p"], t_from=t_of[L], t_to=float(r["tf"]),
                                     mf=r["mf"], alpha=r["alpha"], apc=r["apc"], cov=r["cov"], unexpl=r["unexpl"], flash_S=r["fl"]["S"], hits=hits, stage="flash")
        if truth_t0 is not None:
            okh = ok[hits]; rec["was"] = float((e[hits][okh] * (np.abs(t_of[L] - truth_t0[hits][okh]) <= CUT)).sum()); rec["now"] = float((e[hits][okh] * (np.abs(r["tf"] - truth_t0[hits][okh]) <= CUT)).sum())
        moves.append(rec); moved.add(L); used_flash.add(fid)
        if apply:
            ts[hits] = r["tf"]; base[c] = (base[c].astype(np.float64) + r["Sg"] - r["Sp"]).astype(base.dtype)
    tim["refine"] = time.time() - t0
    st = dict(n_flash=len(flashes), n_cand=len(cands), n_pairs=len(pairs), n_whole=len(wholes), n_pred=len(pred), n_pred_probe=n_pred1, n_hyp=len(res), n_moves=len(moves), moves=moves,
              tim={k: round(v, 3) for k, v in tim.items()}, time_s=round(time.time() - t_start, 3), e_moved=float(sum(m["E"] for m in moves)))
    if truth_t0 is not None:
        st["fixed_e"] = float(sum(max(m["now"] - m["was"], 0.0) for m in moves)); st["broke_e"] = float(sum(max(m["was"] - m["now"], 0.0) for m in moves))
    return st


def channel_positions(h5):
    """(n_light, n_ch, 3) channel positions in charge-TPC indexing (NaN where unknown), from the flow geometry via the chain's LUT"""
    import os, sys
    HERE = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.abspath(os.path.join(HERE, "..", "CLMatching_AlphaRelease"))
    for p in (BASE, os.path.join(BASE, "M5p1"), os.path.join(BASE, "NewMLSection")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from M5p1.first_stage_matching import io as fio
    from M5p1.first_stage_matching.io import LUT
    from ML_NDfull_perceiver import light_tpc_to_charge_tpc
    tpc_to_channels, n_lt = fio.build_light_channel_order(h5)
    pos = LUT.from_array(h5["geometry_info/sipm_abs_pos"].attrs["meta"], h5["geometry_info/sipm_abs_pos/data"])
    P = np.full((n_lt, 120, 3), np.nan)
    for lt, lst in tpc_to_channels.items():
        ct = light_tpc_to_charge_tpc(lt)
        for k, (adc, ch) in enumerate(lst):
            try:
                v = np.asarray(pos[(adc, ch)]).ravel()
                if v.size >= 3:
                    P[ct, k] = v[:3]
            except Exception:
                pass
    return P
