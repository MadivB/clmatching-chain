"""Stages 6 (E) + 6.5 (F): continuous t0s (2x2 handoff port).

Stage E (--subtick): interaction-level re-prediction per final-t0 group
(one collective render - the v097-validated form) + 0.25-tick grid +-3 +
analytic quadratic minimum. Membership NEVER touched.

Stage F (--subtick-hit, EXPERIMENTAL - parked on the git branch
`intra-hit-development`; NOT part of v1): per-HIT t0s. Every light channel of a group x TPC
gets its OWN grid+quadratic fit (chi2 restricted to that channel, other
groups as base); channels filtered by predicted peak amplitude
(chan_min_amp, topped up to >= chan_knn by amplitude); each member hit
takes the inverse-distance-weighted mean of its chan_knn nearest fitted
channels (3D hit -> SiPM position from geometry_info/sipm_abs_pos, rows
ordered side0-then-side1 ascending y as in the light image).

NO calibration constant anywhere (user: a global bias is fine - absolute
time is not needed).

ND adaptation: groups are exact final-t0 values (merge 0.5 tick), NOT the
2x2's 10-tick merge - distinct ND families can sit 2-10 ticks apart.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "CLMatching_AlphaRelease")
DEFAULT_CKPT = ("/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/NewMLSection/"
                "runs/ndfull_small4x/best_model_arch.pt")
GAIN, CLIP = 200.0, 60780.0
S2, S4 = 1.75, 0.765625          # sums over x in {-0.75..0.75} step 0.25
_MODEL = {}
_SIPM = {}


def _paths():
    for p in (_BASE, os.path.join(_BASE, "NewMLSection"), os.path.join(_BASE, "M5p1")):
        if p not in sys.path:
            sys.path.insert(0, p)


def _get_model(ckpt):
    if ckpt not in _MODEL:
        _paths()
        from ML_NDfull_perceiver import load_perceiver_model
        model, _meta = load_perceiver_model(ckpt)
        template = np.load(os.path.join(_BASE, "assets", "avg_pulse.npy")).astype(np.float32) / np.float32(999.0)
        _MODEL[ckpt] = (model, template)
    return _MODEL[ckpt]


def build_sipm_positions(flow_path):
    """{light_tpc: (n_rows, 3) abs positions}, rows in light-image order."""
    if flow_path not in _SIPM:
        _paths()
        import h5py
        from first_stage_matching.io import build_light_channel_order
        from lut import LUT
        with h5py.File(flow_path, "r") as h5:
            order, n_light = build_light_channel_order(h5)
            ap = LUT.from_array(h5["geometry_info/sipm_abs_pos"].attrs["meta"],
                                h5["geometry_info/sipm_abs_pos/data"])
            pos = {}
            for t in range(n_light):
                rows = order.get(t) or []
                if rows:
                    pos[t] = np.array([np.asarray(ap[(a, c)][0], np.float64)
                                       for a, c in rows])
        _SIPM[flow_path] = pos
    return _SIPM[flow_path]


TOP_ROWS = 30        # brightest predicted channels used per group x TPC
QUAD_HW = 4          # fit window half-width in 0.25-tick steps: 4 -> +-1 tick
QUAD_GUARD = 0.75    # vertex acceptance [ticks] (user 2026-08-23: widen from
                     # the 2x2's 7-point/0.5 - the vertex sometimes lands out)


def _quad(L, hw=None, guard=None):
    """Analytic quadratic minimum from 2*hw+1 losses around the argmin (ticks)."""
    hw = QUAD_HW if hw is None else hw
    guard = QUAD_GUARD if guard is None else guard
    x = np.arange(-hw, hw + 1) * 0.25
    n = float(x.size)
    s2 = float((x * x).sum())
    s4 = float((x ** 4).sum())
    b = float((x * L).sum()) / s2
    a = float(((x * x * L).sum() - (s2 / n) * L.sum())) / (s4 - s2 * s2 / n)
    if a <= 0:
        return 0.0, "nonconvex", 0.0
    xs = -b / (2.0 * a)
    if abs(xs) > guard:
        return 0.0, "range", a
    return float(xs), "ok", a


def run_subtick_refine(ns, *, ckpt=None, truth_t0=None, half_ticks=3.0,
                       step=0.25, group_merge=0.5, min_group_mev=0.0,
                       hit_refine=None, flow_path=None, chan_min_amp=400.0,
                       chan_amp_floor=300.0, chan_knn=12, dump_path=None):
    # chan_knn=12 + 1/d^2: best of the 60-config offline sweep (2026-08-23,
    # file 1): per-hit sigma 7.79 ns, mechanism corr 0.346. The sweep is
    # FLAT (corr 0.25-0.35 everywhere) - ND's per-channel information
    # ceiling, not a tuning shortfall; see subtick_sweep.py.
    _paths()
    from pulse_shapes import timeinterpolation
    from ML_NDfull_perceiver import process_clusters_to_imageMaps
    model, template = _get_model(ckpt or DEFAULT_CKPT)
    sipm_pos = build_sipm_positions(flow_path) if (hit_refine and flow_path) else {}

    ts = np.asarray(ns["hit_timestamps"], dtype=np.float64)
    ts_before = ts.copy()
    e = np.clip(np.asarray(ns["Eset"], dtype=np.float64), 0.0, None)
    x = np.asarray(ns["xset"], dtype=np.float64)
    y = np.asarray(ns["yset"], dtype=np.float64)
    z = np.asarray(ns["zset"], dtype=np.float64)
    tpc = np.asarray(ns["hitTPCid"], dtype=np.int64)
    actual = np.asarray(ns["fullLightWaveform"], dtype=np.float64)
    n_light, n_ch, n_ticks = actual.shape
    w_full = 1.0 / (GAIN * np.maximum(actual, 100.0))

    ok = np.isfinite(ts) & (ts >= 0) & (tpc >= 0) & (tpc < n_light)
    st = {"n_groups": 0, "n_refined": 0, "n_nonconvex": 0, "n_range": 0,
          "n_edge": 0, "dt_max": 0.0, "dt_rms": 0.0,
          "f_ch_fit": 0, "f_ch_dropped": 0, "f_hits": 0, "f_dt_rms": 0.0}
    if not ok.any():
        return st

    vals = np.sort(np.unique(ts[ok]))
    groups, cur = [], [vals[0]]
    for v in vals[1:]:
        if v - cur[-1] <= group_merge:
            cur.append(v)
        else:
            groups.append(cur)
            cur = [v]
    groups.append(cur)

    ginfo = []
    for gv in groups:
        m = ok & np.isin(ts, gv)
        eg = float(e[m].sum())
        if eg <= min_group_mev or not m.any():
            continue
        t_g = float(np.average(ts[m], weights=np.maximum(e[m], 1e-6)))
        ginfo.append({"mask": m, "t": t_g, "e": eg})
    st["n_groups"] = len(ginfo)
    if not ginfo:
        return st

    all_m = np.zeros(ts.size, bool)
    glab = np.full(ts.size, -1, np.int64)
    for gi, g in enumerate(ginfo):
        all_m |= g["mask"]
        glab[g["mask"]] = gi
    imgs_all, _ = process_clusters_to_imageMaps(
        x[all_m], y[all_m], z[all_m], e[all_m], tpc[all_m], glab[all_m],
        model=model, template=template)          # ONE batched forward pass
    for gi, g in enumerate(ginfo):
        g["imgs"] = {int(c): np.asarray(v, np.float64)
                     for (l, c), v in imgs_all.items()
                     if int(l) == gi and int(c) < n_light}
        g["sup"] = {c: np.flatnonzero(img.max(axis=1) > 1.0)
                    for c, img in g["imgs"].items()}

    total = np.zeros_like(actual)
    for g in ginfo:
        for c, img in g["imgs"].items():
            total[c] += timeinterpolation(img, shift=g["t"], baseline=0)

    # single CENTERED window (user design 2026-08-24): the chain time is the
    # full integer scan's argmin, so the sub-tick minimum is within a tick -
    # no 25-point pre-scan needed. 2*QUAD_HW+1 shifts total (9 at hw=4, 7 at
    # hw=3), quadratic fit directly on the window; vertex beyond the guard ->
    # keep the chain time. Channels capped at the TOP_ROWS brightest.
    offs = np.arange(-QUAD_HW, QUAD_HW + 1) * step
    group_id = np.full(ts.size, -1, np.int32)

    # ---- stage E: one time per group ----
    dts = []
    for gi, g in enumerate(ginfo):
        group_id[g["mask"]] = gi
        curve = np.zeros(offs.size)
        any_sup = False
        for c, img in g["imgs"].items():
            ch = g["sup"][c]
            if not ch.size:
                continue
            if ch.size > TOP_ROWS:
                amp_r = img[ch].max(axis=1)
                ch = ch[np.argsort(amp_r)[::-1][:TOP_ROWS]]
            any_sup = True
            img_s = img[ch]
            base = total[c][ch] - timeinterpolation(img_s, shift=g["t"], baseline=0)
            a_c, w_c = actual[c][ch], w_full[c][ch]
            for k, dt in enumerate(offs):
                pred = np.minimum(base + timeinterpolation(img_s, shift=g["t"] + dt, baseline=0), CLIP)
                curve[k] += float((w_c * np.square(pred - a_c)).sum())
        g["t_new"] = g["t"]
        if not any_sup:
            continue
        xs, why, a_curv = _quad(curve)
        if why == "nonconvex":
            st["n_nonconvex"] += 1
        elif why == "range":
            st["n_range"] += 1
        # confidence foundation (v0.9): chi2(t) ~ min + a (t-t*)^2, so
        # delta-chi2 = 1 gives sigma_t = 1/sqrt(a) [ticks]
        g["sig_t"] = float(1.0 / np.sqrt(a_curv)) if a_curv > 0 else np.inf
        dt = float(xs)              # NO calibration (user: global bias is fine)
        g["t_new"] = g["t"] + dt
        ts[g["mask"]] = ts[g["mask"]] + dt
        dts.append(dt)
        st["n_refined"] += 1
    ts_groupE = ts.copy()

    # ---- stage F: per-channel fits -> per-hit inverse-distance weighting ----
    chdump = [] if dump_path is not None else None
    if hit_refine:
        xyz = np.stack([x, y, z], 1)
        f_moves = []
        for gi_f, g in enumerate(ginfo):
            tg = g["t_new"]
            for c, img in g["imgs"].items():
                if c not in sipm_pos or sipm_pos[c].shape[0] != n_ch:
                    continue
                amp = img.max(axis=1)
                if chdump is not None:
                    keep = np.flatnonzero(amp > 1.0)          # fit ALL lit channels for the offline sweep
                else:
                    keep = np.flatnonzero(amp >= chan_min_amp)
                    if keep.size < chan_knn:
                        extra = np.flatnonzero((amp >= chan_amp_floor) & (amp < chan_min_amp))
                        extra = extra[np.argsort(amp[extra])[::-1]]
                        keep = np.concatenate([keep, extra[:chan_knn - keep.size]])
                if not keep.size:
                    continue
                st["f_ch_dropped"] += int((amp > 1.0).sum() - keep.size)
                img_k = img[keep]
                base = total[c][keep] - timeinterpolation(img_k, shift=tg, baseline=0)
                a_c, w_c = actual[c][keep], w_full[c][keep]
                Lrc = np.zeros((keep.size, offs.size))
                for k, dt in enumerate(offs):
                    pred = np.minimum(base + timeinterpolation(img_k, shift=tg + dt, baseline=0), CLIP)
                    Lrc[:, k] = (w_c * np.square(pred - a_c)).sum(axis=1)
                t_ch = np.full(keep.size, np.nan)
                for r in range(keep.size):
                    i0 = int(np.argmin(Lrc[r]))
                    if i0 < QUAD_HW or i0 > offs.size - QUAD_HW - 1:
                        continue
                    xs, why, _ac = _quad(Lrc[r, i0 - QUAD_HW: i0 + QUAD_HW + 1])
                    t_ch[r] = tg + float(offs[i0]) + (xs if why == "ok" else 0.0)
                fit = np.isfinite(t_ch)
                if chdump is not None and fit.any():
                    chdump.append((gi_f, c, tg, keep[fit], amp[keep[fit]], t_ch[fit]))
                if chdump is not None:
                    sel = fit & (amp[keep] >= chan_min_amp)
                    if sel.sum() < min(chan_knn, int(fit.sum())):
                        pool = fit & (amp[keep] >= chan_amp_floor)
                        order_a = np.argsort(np.where(pool, amp[keep], -1))[::-1]
                        add = order_a[:chan_knn]
                        sel = np.zeros_like(fit)
                        sel[add] = True
                        sel &= pool
                    fit = sel
                if fit.sum() < 2:
                    continue
                st["f_ch_fit"] += int(fit.sum())
                rows, tvals = keep[fit], t_ch[fit]
                ppos = sipm_pos[c][rows]
                hidx = np.flatnonzero(g["mask"] & (tpc == c))
                if not hidx.size:
                    continue
                d = np.linalg.norm(xyz[hidx][:, None, :] - ppos[None, :, :], axis=2)
                kk = min(chan_knn, rows.size)
                near = np.argsort(d, axis=1)[:, :kk]
                dn = np.take_along_axis(d, near, axis=1)
                wn = 1.0 / np.square(np.maximum(dn, 1.0))
                tn = tvals[near]
                t_hit = (wn * tn).sum(axis=1) / wn.sum(axis=1)
                f_moves.append(t_hit - ts[hidx])
                ts[hidx] = t_hit
                st["f_hits"] += int(hidx.size)
        if f_moves:
            mv = np.concatenate(f_moves)
            st["f_dt_rms"] = round(float(np.sqrt(np.square(mv).mean())), 3)

    hit_sigt = np.full(ts.size, np.nan, np.float32)
    sigs = []
    for g in ginfo:
        sg = g.get("sig_t")
        if sg is not None and np.isfinite(sg):
            hit_sigt[g["mask"]] = np.float32(sg)
            sigs.append(sg)
    ns["hit_t0_sigma"] = hit_sigt          # ticks; NaN = no quadratic fit
    if sigs:
        q = np.percentile(sigs, [50, 90])
        st["sigt_med_ns"] = round(float(q[0]) * 16.0, 2)
        st["sigt_p90_ns"] = round(float(q[1]) * 16.0, 2)

    ns["hit_timestamps"] = ts
    if dts:
        d = np.abs(np.array(dts))
        st["dt_max"] = round(float(d.max()), 3)
        st["dt_rms"] = round(float(np.sqrt(np.square(d).mean())), 3)
    if dump_path is not None:
        np.savez_compressed(
            dump_path, ts_before=ts_before.astype(np.float32),
            ts_groupE=ts_groupE.astype(np.float32),
            ts_after=ts.astype(np.float32), e=e.astype(np.float32),
            group_id=group_id, hit_sigt=hit_sigt,
            ch_gi=np.array([t[0] for t in chdump or []], np.int32),
            ch_tpc=np.array([t[1] for t in chdump or []], np.int32),
            ch_tg=np.array([t[2] for t in chdump or []], np.float32),
            ch_rows=np.array([r for t in chdump or [] for r in t[3]], np.int32),
            ch_amp=np.array([a for t in chdump or [] for a in t[4]], np.float32),
            ch_t=np.array([v for t in chdump or [] for v in t[5]], np.float32),
            ch_n=np.array([len(t[3]) for t in chdump or []], np.int32),
            hit_x=x.astype(np.float32), hit_y=y.astype(np.float32),
            hit_z=z.astype(np.float32), hit_tpc=tpc.astype(np.int32),
            truth=(np.asarray(truth_t0, np.float32) if truth_t0 is not None
                   else np.full(ts.size, np.nan, np.float32)))
    return st
