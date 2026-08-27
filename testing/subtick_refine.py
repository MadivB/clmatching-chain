"""Stage 6 (--subtick): continuous t0s — interaction-level re-prediction +
0.25-tick grid + analytic quadratic refinement (stage E of the 2x2 handoff,
HANDOFF_ND_threshold_association.md).

Membership is NEVER touched: each final-t0 group keeps its hits; only the
group time moves (grid +-3 ticks, quadratic |x*| <= 0.5). Each group's light
is re-predicted from scratch as ONE bag of hits (collective render - the
v097-validated form); other groups' re-predicted light forms the base.
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


def _get_model(ckpt):
    if ckpt not in _MODEL:
        for p in (_BASE, os.path.join(_BASE, "NewMLSection"), os.path.join(_BASE, "M5p1")):
            if p not in sys.path:
                sys.path.insert(0, p)
        from ML_NDfull_perceiver import load_perceiver_model
        model, _meta = load_perceiver_model(ckpt)
        template = np.load(os.path.join(_BASE, "assets", "avg_pulse.npy")).astype(np.float32) / np.float32(999.0)
        _MODEL[ckpt] = (model, template)
    return _MODEL[ckpt]


def _quad_refine(L):
    """Analytic quadratic minimum over 7 grid losses around the argmin.

    L: losses at x = {-0.75,-0.5,-0.25,0,0.25,0.5,0.75} ticks around the
    discrete argmin. Returns dt in ticks (0.0 on any guard)."""
    x = np.arange(-3, 4) * 0.25
    b = float((x * L).sum()) / S2
    a = float(((x * x * L).sum() - (S2 / 7.0) * L.sum())) / (S4 - S2 * S2 / 7.0)
    if a <= 0:
        return 0.0, "nonconvex"
    xs = -b / (2.0 * a)
    if abs(xs) > 0.5:
        return 0.0, "range"
    return float(xs), "ok"


CAL_TICKS = 0.304   # per-detector timing calibration, MEASURED on ND file
                    # 0000001 (2026-08-23): the refined chi2 minimum sits
                    # +4.86 ns late of truth (template-shape systematic,
                    # analog of the 2x2's -3.56 ns). Subtracted from every
                    # refined time; re-measure per detector/template.


def run_subtick_refine(ns, *, ckpt=None, truth_t0=None, half_ticks=3.0,
                       step=0.25, group_merge=0.5, min_group_mev=0.0,
                       cal_ticks=CAL_TICKS, dump_path=None):
    from pulse_shapes import timeinterpolation
    from ML_NDfull_perceiver import process_clusters_to_imageMaps
    model, template = _get_model(ckpt or DEFAULT_CKPT)

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
          "n_edge": 0, "dt_max": 0.0, "dt_rms": 0.0}
    if not ok.any():
        return st

    # exact-value groups (merge within group_merge ticks)
    vals = np.sort(np.unique(ts[ok]))
    groups, cur = [], [vals[0]]
    for v in vals[1:]:
        if v - cur[-1] <= group_merge:
            cur.append(v)
        else:
            groups.append(cur); cur = [v]
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

    # collective re-prediction per group (one bag of hits, one forward pass)
    for g in ginfo:
        m = g["mask"]
        imgs, _ = process_clusters_to_imageMaps(
            x[m], y[m], z[m], e[m], tpc[m], np.zeros(int(m.sum()), np.int64),
            model=model, template=template)
        g["imgs"] = {int(c): np.asarray(v, np.float64)
                     for (_l, c), v in imgs.items() if int(c) < n_light}
        g["sup"] = {c: np.flatnonzero(img.max(axis=1) > 1.0)
                    for c, img in g["imgs"].items()}

    # total re-predicted field at current times
    total = np.zeros_like(actual)
    for g in ginfo:
        for c, img in g["imgs"].items():
            total[c] += timeinterpolation(img, shift=g["t"], baseline=0)

    offs = np.arange(-half_ticks, half_ticks + 1e-9, step)
    dts = []
    for g in ginfo:
        own = {c: timeinterpolation(img, shift=g["t"], baseline=0)
               for c, img in g["imgs"].items()}
        curve = np.zeros(offs.size)
        any_sup = False
        for c, img in g["imgs"].items():
            ch = g["sup"][c]
            if not ch.size:
                continue
            any_sup = True
            base = total[c][ch] - own[c][ch]
            a_c, w_c = actual[c][ch], w_full[c][ch]
            for k, dt in enumerate(offs):
                pred = np.minimum(base + timeinterpolation(img, shift=g["t"] + dt, baseline=0)[ch], CLIP)
                curve[k] += float((w_c * (pred - a_c) ** 2).sum())
        if not any_sup:
            continue
        i0 = int(np.argmin(curve))
        if i0 < 3 or i0 > offs.size - 4:
            st["n_edge"] += 1
            continue
        xs, why = _quad_refine(curve[i0 - 3: i0 + 4])
        if why == "nonconvex":
            st["n_nonconvex"] += 1
        elif why == "range":
            st["n_range"] += 1
        dt = float(offs[i0] + xs) - float(cal_ticks)
        ts[g["mask"]] = ts[g["mask"]] + dt      # per-hit shift keeps intra-group offsets
        dts.append(dt)
        st["n_refined"] += 1

    ns["hit_timestamps"] = ts
    if dts:
        d = np.abs(np.array(dts))
        st["dt_max"] = round(float(d.max()), 3)
        st["dt_rms"] = round(float(np.sqrt((d ** 2).mean())), 3)
    if dump_path is not None:
        np.savez_compressed(
            dump_path, ts_before=ts_before.astype(np.float32),
            ts_after=ts.astype(np.float32), e=e.astype(np.float32),
            truth=(np.asarray(truth_t0, np.float32) if truth_t0 is not None
                   else np.full(ts.size, np.nan, np.float32)))
    return st
