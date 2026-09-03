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
_CONF = {}


def _get_conf_model():
    if "m" not in _CONF:
        import json as _json
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "subtick_conf_model.json")
        _CONF["m"] = _json.load(open(p)) if os.path.exists(p) else None
    return _CONF["m"]


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
                       chan_amp_floor=300.0, chan_knn=12, amp_mode="sum",
                       dump_path=None):
    # amp_mode "sum" (default, user 2026-09-02: slight precision loss
    # accepted for speed): group amplitude = sum of the members' stage-1
    # per-cluster amplitudes on support channels - no re-voxelize/model.
    # "model": collective re-render (superposition-exact, +~17 s/ev).
    # chan_knn=12 + 1/d^2: best of the 60-config offline sweep (2026-08-23,
    # file 1): per-hit sigma 7.79 ns, mechanism corr 0.346. The sweep is
    # FLAT (corr 0.25-0.35 everywhere) - ND's per-channel information
    # ceiling, not a tuning shortfall; see subtick_sweep.py.
    _paths()
    from pulse_shapes import timeinterpolation
    from ML_NDfull_perceiver import process_clusters_to_imageMaps
    import time as _time
    _t0 = _time.time()
    model, template = _get_model(ckpt or DEFAULT_CKPT)
    _t_load = _time.time() - _t0
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

    _t0 = _time.time()
    all_m = np.zeros(ts.size, bool)
    glab = np.full(ts.size, -1, np.int64)
    for gi, g in enumerate(ginfo):
        all_m |= g["mask"]
        glab[g["mask"]] = gi
    tmpl64 = np.asarray(template, np.float64)
    _t_vox = _t_model = 0.0
    for gi, g in enumerate(ginfo):
        g["amp"] = {}
        g["sup"] = {}
    if amp_mode == "model":
        # rank-1 collective render: superposition-correct amplitudes from
        # one voxelize+model pass, no (120,1000) materialization
        from ML_NDfull_perceiver import group_voxelize_pairs, predict_phi
        maps_4d, gcls, gtpcs = group_voxelize_pairs(
            x[all_m], y[all_m], z[all_m], e[all_m], tpc[all_m], glab[all_m])
        _t_vox = _time.time() - _t0
        _t0 = _time.time()
        amps = predict_phi(maps_4d, model, gtpcs, target_scale=1e-3,
                           batch_size=32, raw_clip=(0.0, CLIP),
                           min_prediction_threshold=100.0)
        _t_model = _time.time() - _t0
        _t0 = _time.time()
        for gg, (l, c) in enumerate(zip(gcls, gtpcs)):
            l, c = int(l), int(c)
            if not (0 <= l < len(ginfo)) or not (0 <= c < n_light):
                continue
            a_vec = np.asarray(amps[gg], np.float64)
            ginfo[l]["amp"][c] = a_vec
            ginfo[l]["sup"][c] = np.flatnonzero(a_vec * tmpl64.max() > 1.0)
    else:
        # sum of the members' existing stage-1 amplitudes (img = amp x tmpl)
        image_maps = ns["imageMaps"]
        labs_g = np.asarray(ns["labels_global"])
        tmpl_max = float(tmpl64.max())
        by_label = {}
        for (lab, c), im in image_maps.items():
            by_label.setdefault(int(lab), []).append((int(c), im))
        for gi, g in enumerate(ginfo):
            for lab in np.unique(labs_g[g["mask"]]):
                for c, im in by_label.get(int(lab), []):
                    if not (0 <= c < n_light):
                        continue
                    a_vec = np.asarray(im, np.float64).max(axis=1) / tmpl_max
                    if c in g["amp"]:
                        g["amp"][c] += a_vec
                    else:
                        g["amp"][c] = a_vec
            for c, a_vec in g["amp"].items():
                g["sup"][c] = np.flatnonzero(a_vec * tmpl_max > 1.0)

    _t_render = _time.time() - _t0
    _t0 = _time.time()

    def _shift1(vec, sh):
        return timeinterpolation(vec[None, :], shift=float(sh), baseline=0)[0]

    tpc_groups = {}
    for gi, g in enumerate(ginfo):
        for c in g["amp"]:
            tpc_groups.setdefault(c, []).append(gi)

    def _field(c, rows, exclude_gi):
        out = np.zeros((rows.size, n_ticks))
        for gj in tpc_groups.get(c, []):
            if gj == exclude_gi:
                continue
            gj_g = ginfo[gj]
            out += gj_g["amp"][c][rows, None] * _shift1(tmpl64, gj_g["t"])[None, :]
        return out

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
        for c, a_vec in g["amp"].items():
            ch = g["sup"][c]
            if not ch.size:
                continue
            if ch.size > TOP_ROWS:
                ch = ch[np.argsort(a_vec[ch])[::-1][:TOP_ROWS]]
            any_sup = True
            base = _field(c, ch, gi)
            a_c, w_c = actual[c][ch], w_full[c][ch]
            amp_col = a_vec[ch, None]
            for k, dt in enumerate(offs):
                pred = np.minimum(base + amp_col * _shift1(tmpl64, g["t"] + dt)[None, :], CLIP)
                curve[k] += float((w_c * np.square(pred - a_c)).sum())
        g["t_new"] = g["t"]
        if not any_sup:
            continue
        xs, why, a_curv = _quad(curve)
        g["curve_min"] = float(curve.min())
        g["curve_contrast"] = float((curve.max() - curve.min()) / max(abs(curve.min()), 1.0))
        g["fit_status"] = {"ok": 0.0, "range": 1.0, "nonconvex": 2.0}[why]
        g["n_rows_fit"] = float(sum(min(len(g["sup"][c]), TOP_ROWS) for c in g["amp"]))
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
    st["t_load"] = round(_t_load, 2)
    st["t_vox"] = round(_t_vox, 2)
    st["t_model"] = round(_t_model, 2)
    st["t_render"] = round(_t_render, 2)
    st["t_grid"] = round(_time.time() - _t0, 2)

    # ---- stage F: per-channel fits -> per-hit inverse-distance weighting ----
    chdump = [] if dump_path is not None else None
    if hit_refine:
        raise NotImplementedError(
            "stage F (per-hit) is parked on the git branch "
            "intra-hit-development; the rank-1 stage-E rewrite on main "
            "does not materialize the per-group images it needs")
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

    # ---- confidence features (v0.9.2 study): waveform comparison at the
    # final time + quadratic-fit evidence, one row per group ----
    tmax_i = int(np.argmax(tmpl64))
    tmpl_pk = float(tmpl64.max())
    conf_feat, conf_meta = [], []
    for gi, g in enumerate(ginfo):
        if not g["amp"]:
            continue
        k0 = int(round(g["t_new"])) + tmax_i
        zs, amps_v, resid_v = [], [], []
        for c, a_vec in g["amp"].items():
            ch = g["sup"][c]
            if not ch.size or not (0 <= k0 < n_ticks):
                continue
            if ch.size > TOP_ROWS:
                ch = ch[np.argsort(a_vec[ch])[::-1][:TOP_ROWS]]
            own_pk = a_vec[ch] * tmpl_pk
            base_pk = _field(c, ch, gi)[:, k0]
            a_pk = actual[c][ch, k0]
            model = np.minimum(base_pk + own_pk, CLIP)
            zs.append((model - a_pk) / np.sqrt(GAIN * np.maximum(a_pk, 100.0)))
            amps_v.append(own_pk)
            resid_v.append(np.clip(a_pk - base_pk, 0.0, None))
        if not zs:
            continue
        z = np.concatenate(zs)
        av = np.concatenate(amps_v)
        rv = np.concatenate(resid_v)
        na, nr = float(np.linalg.norm(av)), float(np.linalg.norm(rv))
        cosr = float(av @ rv / (na * nr)) if na > 1e-9 and nr > 1e-9 else 0.0
        mhits = g["mask"]
        conf_feat.append([
            np.log10(max(g["e"], 1e-3)),
            min(g.get("sig_t", np.inf), 10.0),
            g.get("fit_status", 2.0),
            g["curve_min"] / max(g["n_rows_fit"] * n_ticks, 1.0),
            g["curve_contrast"],
            float(np.mean(np.abs(z))),
            float(np.mean(np.abs(z) > 2.0)),
            cosr,
            float(len(g["amp"])),
            np.log10(max(int(mhits.sum()), 1)),
        ])
        ok_e = wrong_e = 0.0
        if truth_t0 is not None:
            tr = np.asarray(truth_t0, np.float64)[mhits]
            ee = e[mhits]
            fin = np.isfinite(tr)
            ok_e = float(ee[fin & (np.abs(ts[mhits] - tr) <= 10.0)].sum())
            wrong_e = float(ee[fin].sum()) - ok_e
        conf_meta.append([gi, g["e"], ok_e, wrong_e])

    # calibrated 0-1 confidence (v0.9.2): logistic on both evidence
    # families, trained/calibrated on file 0000001 LOO (AUC 0.881;
    # reliability monotone 22% -> 99.8%); see subtick_conf_study.py
    hit_conf = np.full(ts.size, np.nan, np.float32)
    cmdl = _get_conf_model()
    if cmdl is not None and conf_feat:
        F = np.asarray(conf_feat, np.float64)[:, cmdl["cols"]]
        Z = (F - np.asarray(cmdl["mu"])[cmdl["cols"]]) / np.asarray(cmdl["sd"])[cmdl["cols"]]
        wv = np.asarray(cmdl["w"])
        p = 1.0 / (1.0 + np.exp(-(Z @ wv + cmdl["b"])))
        for row, meta in enumerate(conf_meta):
            gi = int(meta[0])
            ginfo[gi]["conf"] = float(p[row])
            hit_conf[ginfo[gi]["mask"]] = np.float32(p[row])
        st["conf_med"] = round(float(np.median(p)), 3)
        st["conf_frac_lo"] = round(float(np.mean(p < 0.5)), 3)
    ns["hit_t0_conf"] = hit_conf          # 0=likely wrong, 1=confident; NaN=no estimate

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
            hit_conf=hit_conf,
            conf_feat=np.asarray(conf_feat, np.float32),
            conf_meta=np.asarray(conf_meta, np.float32),
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
