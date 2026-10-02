"""Integrated backbone re-run with the saving algorithm applied.

CANONICAL STAGE STRUCTURE (v0.4.x numbering; integer = assignment stage,
decimal = correction / sub-stage; per-event durations land in row["stage_t"]):

  STAGE 1 - clustering & ML prediction
    1.1 RANSAC track/shower clustering (+ absorption, length gate)
    1.2 vertex-track merge (--merge-vertex)
    1.3 perceiver light-amplitude prediction, GPU (>= --predict-min-energy MeV)
    1.4 support channels (90% / gap 2) + saturation veto (> --max-clip-ticks)
    [all inside run_first_stage_charge_light_matching; its built-in production
     track placement is superseded by stage 2/3.1 below when --backbone stack2]
  STAGE 2 - Phase 1 assignment
    2.1 showers at their scan optimum
    2.2 tracks crossing >= 3 TPCs
  STAGE 3 - Phase 2 assignment (multi-TPC or > --p2-threshold MeV)
    3.1 MERGED permutation pool (--merge-p2, canonical since v0.4.1):
        1-2-TPC backbone tracks + every P2-class cluster, energy-banded
        batches (s = 0.6, cap 5, cossoft pick, defer gate; includes
        single-TPC <= 30 MeV backbone tracks - the one known impurity)
    3.2 corrections on stages 2-3.1 (full-model trigger under the merge):
        overflow trigger -> Stage A / Stage B / N=1 saving -> ADOPT ->
        swap & move phase (covers merged placements too)
    3.3 large-cluster scan (V11 boundary patched by --p2-threshold, forced
        accept) + placement police; under the merge only 3.2-detached parts
        and stragglers remain here
    3.5 V2 spatial + light rescue + police            [correction]
    3.9 spatial family expansion (--phase3 expand|both; absorbs small
        clusters into placed interactions; regular clipped-Poisson loss)
  STAGE 4 - Phase 3 assignment (whatever is still unplaced, per TPC)
    4.1 v11 error-matrix association (single-TPC, predict-min..30 MeV)
    4.5 per-TPC swap fix (--swap-fix, ratio --swap-ratio-max)  [correction]
    4.9 force coverage (--force-coverage; coverage mandate 100%)
  output: per-TPC / per-cluster confidence (annotation, not assignment)

Per event: first stage -> BASELINE pass (Phase 2 + V2, unmodified) -> rescue
decisions (trigger -> groups -> Stage A anchored permutations -> Stage B
per-TPC part detachment on all-stay groups) -> ADOPT (move members, detach
parts as NEW cluster labels routed to Phase 2/3 by energy) -> RESCUED pass
(Phase 2 + V2 on the amended state).  Reports the energy-weighted hit-level
assignment accuracy (|reco-truth| <= 10 ticks among assigned, truth-valid
hits) for both passes.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

_HERE = os.path.abspath(os.path.dirname(__file__))
BASE_DIR = os.path.abspath(os.path.join(_HERE, "..", "CLMatching_AlphaRelease"))
for _p in (os.path.join(BASE_DIR, "M5p1"), os.path.join(BASE_DIR, "NewMLSection"), BASE_DIR):
    while _p in sys.path:
        sys.path.remove(_p)
sys.path.insert(0, BASE_DIR)
sys.path.insert(1, os.path.join(BASE_DIR, "NewMLSection"))
sys.path.insert(2, os.path.join(BASE_DIR, "M5p1"))
sys.path.insert(3, _HERE)

import numpy as np
from scipy.signal import fftconvolve

# v1.2.2: per-event, per-channel light saturation ceiling (int16 rail minus the dynamic
# baseline); registered once per event after the first stage, used by every clip below and
# by the testing-side stage modules.  CLIP stays only as the legacy fallback.
import sat_clip as _sat_clip
from sat_clip import clip_rows, clip_col

# checkpoint-key -> canonical stage id (see stage structure in the docstring)
STAGE_CHECKPOINTS = [
    ("first_stage", "1_clustering_prediction"),
    ("after_backbone", "2+3.1_phase1_backbone"),
    ("after_rescue_adopt", "3.2_trigger_rescue_swap"),
    ("after_early_expand", "3.25_early_expansion"),
    ("after_staged", "3.3+3.5_p2_v2_police"),
    ("after_expand", "3.9_expansion"),
    ("after_joint", "3.9_joint"),
    ("after_matrix", "4.1_matrix"),
    ("after_matrix_relax", "4.2_matrix_relax"),
    ("after_pass2", "3.95_pass2_trade"),
    ("after_pass3f", "3.97_pass3_family"),
    ("after_redecide", "3.98_redecide"),
    ("after_v200", "3.99_v200_blocks"),
    ("after_swap_fix", "4.5_swap_fix"),
    ("after_blip", "4.7_blip_rehome"),
    ("after_force", "4.9_force_coverage"),
    ("after_precision", "7_precision"),
]


def _stage_durations(tim: dict) -> dict:
    out, prev = {}, 0.0
    for key, stage in STAGE_CHECKPOINTS:
        if key in tim:
            out[stage] = round(float(tim[key]) - prev, 3)
            prev = float(tim[key])
    return out

PEAK = 105
GAIN = 200.0
CLIP = 60780.0
COS_CUT = 0.75
Z_CUT = 2.0
EXCESS_GATE = 500.0
GROUP_DT = 5.0
MAX_N = 3
SEARCH = 895
CUT = 10.0
VAR_CKPT = ("/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/NewMLSection/var_prediction/"
            "runs/var_run_perceiver_aligned_0001000_v2/best_model.pt")


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


def cos_curve(pred, resid, max_shift=SEARCH):
    """cos(own channel pattern at the pulse peak, unexplained residual) per candidate shift.

    pred:  (n_ch, n_ticks) template rows (the cluster's images, support channels).
    resid: SAME rows of (background - actual) as passed to weighted_scan; the
           unexplained light is clip(actual - background, 0) = clip(-resid, 0).
    A shift s puts the pulse peak at tick s+PEAK, where the cluster's channel
    pattern is pred[:, PEAK] (shift-invariant); no evidence at a shift -> cos 0.
    """
    own = pred[:, PEAK]
    R = np.clip(-resid, 0.0, None)[:, PEAK: PEAK + max_shift + 1]
    no = float(np.linalg.norm(own))
    nr = np.linalg.norm(R, axis=0)
    c = (own @ R) / np.maximum(no * nr, 1e-9)
    c[nr < 1e-6] = 0.0
    out = np.zeros(max_shift + 1)
    out[: c.size] = c
    return out if no > 1e-9 else np.zeros_like(out)


def gated_argmin(curve, pred, resid, cos_sel, cos_window=0.0):
    """Shift selection with a cosine matched-filter preference.

    Eligibility: shifts with cos(own, residual) >= cos_sel (all shifts if none
    pass or the gate is off).  With cos_window > 0, the loss ordering inside a
    (1 + cos_window) band of the minimum is treated as not trustworthy (the
    chi2 is perceiver-systematics dominated) and the HIGHEST-cos shift in the
    band wins; otherwise plain min-loss among eligible shifts."""
    if cos_sel <= 0.0:
        return float(np.argmin(curve))
    cv = cos_curve(pred, resid, max_shift=curve.size - 1)
    ok = cv >= cos_sel
    if not ok.any():
        # nothing looks like this cluster anywhere: no basis for a cos
        # preference, fall back to the plain scan
        return float(np.argmin(curve))
    idx = np.flatnonzero(ok)
    if cos_window > 0.0:
        lmin = float(curve[idx].min())
        band = idx[curve[idx] <= lmin * (1.0 + cos_window)]
        return float(band[np.argmax(cv[band])])
    return float(idx[np.argmin(curve[idx])])


def accuracy(hit_ts, truth_t0, e_pos, mask=None):
    r = np.asarray(hit_ts, dtype=np.float64)
    assigned = np.isfinite(r) & (r >= 0)
    if mask is not None:
        assigned = assigned & mask
    ok = assigned & np.isfinite(truth_t0)
    if not ok.any():
        return None, 0.0
    good = ok & (np.abs(r - truth_t0) <= CUT)
    acc = float(np.sum(e_pos[good]) / max(np.sum(e_pos[ok]), 1e-9))
    afrac = float(np.sum(e_pos[assigned]) / max(np.sum(e_pos), 1e-9))
    return acc, afrac


def make_police(actual, veto_mask, e_pos, support_cache, label_info, ti_fn,
                mode="locked", topk=18, z_cut=Z_CUT):
    """Placement police: after a stage, every label that the stage newly placed
    or moved is probed at its t0 (tick t0+105).  If the overflow gate fires,
    the placement is resolved IMMEDIATELY: keep / rewind (moved labels) /
    unanchored rescan — winner by Poisson union-support loss with pred clipped.
    Gate modes: "locked" = the phase-1 trigger (>=3 ch excess>=500 z_pois>z OR
    cos(own,resid)<0.75); "top18" = >=3 of the topk brightest unvetoed channels
    (by max(pred,actual) at the probe tick) with z_pois>z, no excess gate —
    bright channels are the ones the ML predicts well — plus the cos arm."""
    n_light, n_ch, n_ticks = actual.shape
    w_all = 1.0 / (GAIN * np.maximum(actual, 100.0))

    def label_sup(lab, tpcs, image_maps):
        lab_type = str(label_info.get(int(lab), {}).get("type", "track")).lower()
        sup = {}
        for c in tpcs:
            entry = support_cache.get((int(lab), int(c))) if lab_type == "track" else None
            if entry is not None:
                ch = np.asarray(entry["channel_indices"], dtype=int)
            else:
                ch = np.flatnonzero(np.asarray(image_maps[(int(lab), int(c))]).max(axis=1) > 1.0)
            ch = ch[~veto_mask[int(c), ch]]
            if ch.size:
                sup[int(c)] = np.sort(ch)
        return sup

    def fired(lab, t0, bi, image_maps, tpcs):
        tick = int(round(t0)) + PEAK
        if not (0 <= tick < n_ticks):
            return False
        cmin = 1.0
        for c in tpcs:
            fin = ~veto_mask[c]
            a = actual[c][:, tick]
            p = clip_col(np.asarray(bi[c], np.float64)[:, tick], c)
            ex = p - a
            z = ex / np.sqrt(GAIN * np.maximum(a, 100.0))
            if mode == "top18":
                amp = np.maximum(p, a)
                amp[~fin] = -1.0
                sel = np.argsort(amp)[::-1][:topk]
                if int(np.sum(z[sel] > z_cut)) >= 3:
                    return True
            else:
                zz = np.sort(z[fin & (ex >= EXCESS_GATE)])[::-1]
                if zz.size >= 3 and zz[2] > z_cut:
                    return True
            o = ti_fn(np.asarray(image_maps[(int(lab), int(c))], np.float64), t0)[:, tick]
            res = np.clip(np.clip(a, 0, None) - np.clip(p - o, 0, None), 0, None)
            no, nr = np.linalg.norm(o[fin]), np.linalg.norm(res[fin])
            if no > 1e-9 and nr > 1e-9:
                cmin = min(cmin, float(np.dot(o[fin], res[fin]) / (no * nr)))
        return cmin < COS_CUT

    def per_label_t0(ts, labels):
        out = {}
        assigned = np.isfinite(ts) & (ts >= 0)
        for lab in np.unique(labels[assigned & (labels >= 0)]):
            out[int(lab)] = float(np.median(ts[assigned & (labels == lab)]))
        return out

    def police(ns, ts_prev, tag):
        image_maps = ns["imageMaps"]
        labels = np.asarray(ns["labels_global"])
        ts = np.asarray(ns["hit_timestamps"], dtype=np.float64).copy()
        bi = np.asarray(ns["baseImage"], dtype=np.float32)
        cur = per_label_t0(ts, labels)
        prev = per_label_t0(np.asarray(ts_prev, dtype=np.float64), labels)
        todo = []
        for lab, t in cur.items():
            tp_ = prev.get(lab)
            if tp_ is None:
                todo.append((lab, t, None, "new"))
            elif abs(t - tp_) > CUT:
                todo.append((lab, t, tp_, "moved"))
        todo.sort(key=lambda x: -float(e_pos[labels == x[0]].sum()))
        st = {"n_checked": 0, "n_fired": 0, "n_rescan": 0, "n_revert": 0,
              "n_keep": 0, "fired_labels": []}
        for lab, t_cur, t_prev, kind in todo:
            tpcs = [c for c in range(n_light) if (int(lab), int(c)) in image_maps]
            if not tpcs:
                continue
            st["n_checked"] += 1
            if not fired(lab, t_cur, bi, image_maps, tpcs):
                continue
            st["n_fired"] += 1
            sup = label_sup(lab, tpcs, image_maps)
            if not sup:
                continue
            for c in tpcs:
                bi[c] -= ti_fn(np.asarray(image_maps[(int(lab), int(c))], np.float64),
                               t_cur).astype(np.float32)
            prs = np.concatenate([np.asarray(image_maps[(int(lab), int(c))], np.float64)[ch]
                                  for c, ch in sup.items()])
            rrs = np.concatenate([np.asarray(bi[c], np.float64)[ch] - actual[c][ch]
                                  for c, ch in sup.items()])
            wrs = np.concatenate([w_all[c][ch] for c, ch in sup.items()])
            t_scan = float(np.argmin(weighted_scan(prs, rrs, wrs)))
            cands = [("keep", t_cur), ("rescan", t_scan)]
            if t_prev is not None:
                cands.append(("revert", t_prev))

            def uloss(t):
                tot = 0.0
                for c, ch in sup.items():
                    mdl = (np.asarray(bi[c], np.float64)[ch]
                           + ti_fn(np.asarray(image_maps[(int(lab), int(c))], np.float64), t)[ch])
                    tot += float(np.sum(w_all[c][ch]
                                        * (clip_rows(mdl, c, ch) - actual[c][ch]) ** 2))
                return tot

            action, t_new = min(cands, key=lambda kv: uloss(kv[1]))
            st[f"n_{action}"] += 1
            st["fired_labels"].append({"label": int(lab), "kind": kind,
                                       "t_cur": float(t_cur),
                                       "t_prev": (float(t_prev) if t_prev is not None else None),
                                       "t_new": float(t_new), "action": action})
            for c in tpcs:
                bi[c] += ti_fn(np.asarray(image_maps[(int(lab), int(c))], np.float64),
                               t_new).astype(np.float32)
            if abs(t_new - t_cur) > 1e-9:
                mask = (labels == lab) & np.isfinite(ts) & (ts >= 0)
                ts[mask] = t_new
        np.clip(bi, 0.0, None, out=bi)
        ns["baseImage"] = bi
        ns["hit_timestamps"] = ts.astype(np.float32)
        return st

    return police


def staged_run(ns, v2mod, v2cfg, run_phase2, snapshot_fn, verify_fn, truth_t0, e_pos, tag,
               police=None, skip_v2=False):
    """Phase 2 then V2 with cohort decomposition. Returns metrics dict."""
    ts1 = np.asarray(ns["hit_timestamps"], dtype=np.float64).copy()
    A1 = np.isfinite(ts1) & (ts1 >= 0)
    m = {}
    m["p1_acc"], m["p1_afrac"] = accuracy(ts1, truth_t0, e_pos)
    snap = snapshot_fn(hit_timestamps=ns["hit_timestamps"], labels_global=ns["labels_global"],
                       v8_absorbed_hit_parent=ns.get("v8_absorbed_hit_parent"),
                       track_shower_labels=ns["track_shower_labels"])
    run_phase2(ns, verbose=False)
    verify_fn(snap, hit_timestamps=ns["hit_timestamps"], stage_name=f"P2({tag})")
    if police is not None:
        m["police_p2"] = police(ns, ts1, f"p2({tag})")
    ts2 = np.asarray(ns["hit_timestamps"], dtype=np.float64).copy()
    A2 = np.isfinite(ts2) & (ts2 >= 0)
    m["p2_acc"], m["p2_afrac"] = accuracy(ts2, truth_t0, e_pos)
    m["p2_acc_A1"], _ = accuracy(ts2, truth_t0, e_pos, mask=A1)
    m["p2_acc_new"], _ = accuracy(ts2, truth_t0, e_pos, mask=A2 & ~A1)
    if not skip_v2:
        v2mod.run_trial2_v2_light_rescue_from_namespace(ns, config=v2cfg, commit=True)
        if police is not None:
            m["police_v2"] = police(ns, ts2, f"v2({tag})")
    ts3 = np.asarray(ns["hit_timestamps"], dtype=np.float64).copy()
    m["v2_acc"], m["v2_afrac"] = accuracy(ts3, truth_t0, e_pos)
    m["v2_acc_A1"], _ = accuracy(ts3, truth_t0, e_pos, mask=A1)
    m["v2_acc_new"], _ = accuracy(ts3, truth_t0, e_pos, mask=A2 & ~A1)
    moved = A1 & np.isfinite(ts3) & (np.abs(ts3 - ts1) > CUT)
    m["v2_moved_A1_efrac"] = float(np.sum(e_pos[moved]) / max(np.sum(e_pos[A1]), 1e-9))
    m["v2_moved_A1_acc_before"], _ = accuracy(ts1, truth_t0, e_pos, mask=moved)
    m["v2_moved_A1_acc_after"], _ = accuracy(ts3, truth_t0, e_pos, mask=moved)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--event-stride", type=int, default=1)
    ap.add_argument("--event-offset", type=int, default=0)
    ap.add_argument("--sigma-mode", choices=["poisson", "model"], default="poisson")
    ap.add_argument("--z-cut", type=float, default=None,
                    help="trigger z threshold (default: module Z_CUT; model-sigma needs ~8)")
    ap.add_argument("--stay-margin", type=float, default=0.0,
                    help="Stage A moves only if best-perm loss < stay loss * (1 - margin)")
    ap.add_argument("--cos-sel", type=float, default=0.0,
                    help="rescue scans pick min-loss among shifts with cos(own, residual) >= "
                         "this (0 = off, plain argmin)")
    ap.add_argument("--cos-window", type=float, default=0.0,
                    help="treat losses within (1+w)*min as comparable and pick the "
                         "highest-cos shift in that band (needs --cos-sel > 0)")
    ap.add_argument("--swap-phase", action="store_true",
                    help="post-adoption lightweight pass: pairwise t0 swaps between "
                         "co-TPC tracks + cos-gated solo relocations, greedy by loss gain")
    ap.add_argument("--swap-sweeps", type=int, default=2)
    ap.add_argument("--backbone", choices=["prod", "v02", "stack2"], default="prod",
                    help="replace the phase-1 pool placement before baseline+rescue: "
                         "v02 = s=0.60 permutation batches; stack2 = v02 + cossoft "
                         "pick + defer gate + swap + trigger-rescue passes")
    ap.add_argument("--v2-tune", default="",
                    help="comma-separated V2 config variants (or 'all'): the rescued "
                         "pass runs Phase 2 once, then each V2 variant against a "
                         "restored post-P2 snapshot")
    ap.add_argument("--skip-baseline", action="store_true",
                    help="skip the baseline Phase2+V2 pass (saves ~90 s/event)")
    ap.add_argument("--dump-dir", default="",
                    help="write per-event final-state npz (positions, labels, final ts, "
                         "backbone base image, own-images of the worst wrong labels)")
    ap.add_argument("--evo", action="store_true",
                    help="record per-cluster median t at every stage checkpoint "
                         "(row['t_evo'] + dump key 't_evo'); read-only snapshots")
    ap.add_argument("--shower-eat", action="store_true",
                    help="UNVERSIONED candidate: eager shower absorption — "
                         "placed showers (explicit or emergent) immediately "
                         "eat surrounding clusters to 60%% light fulfillment "
                         "(preference-margin ranked; contact/overflow guards)")
    ap.add_argument("--subtick", action="store_true",
                    help="stage 6: continuous t0s - interaction-level "
                         "re-prediction + 0.25-tick grid + analytic "
                         "quadratic refinement (2x2 stage E port); "
                         "membership never touched")
    ap.add_argument("--subtick-hit", action="store_true",
                    help="EXPERIMENTAL (parked on git branch "
                         "intra-hit-development; NOT part of v1): stage F "
                         "per-HIT t0s from per-channel fits (implies --subtick)")
    ap.add_argument("--conf-cluster", action="store_true",
                    help="v0.9.3: calibrated cluster-level confidence "
                         "(3 evidence families, HGB+isotonic); needs --subtick")
    ap.add_argument("--conf-model", default="",
                    help="override path to subtick_conf_model3.joblib")
    ap.add_argument("--subtick-merge", type=float, default=0.5,
                    help="stage-E family merge tolerance in ticks (default 0.5)")
    ap.add_argument("--subtick-study", action="store_true",
                    help="dump study arms: grid-argmin + per-cluster fits")
    ap.add_argument("--subtick-dump", default="",
                    help="dir for per-hit (ts_before, ts_after, truth, e) "
                         "npz dumps from the subtick stage")
    ap.add_argument("--pool-mode", choices=["energy", "light", "light-order"], default="energy",
                    help="v1.1: batch the merged pool by light-coupling COMPONENTS "
                         "(cosine of amplitude patterns on a shared TPC >= --pool-cos) "
                         "and place each component by exact COMBINATION search over "
                         "the members' candidate times (v110_components.py); "
                         "'energy' = v1.0 energy-banded ordering search")
    ap.add_argument("--pool-cos", type=float, default=0.80,
                    help="v1.1 light-coupling edge threshold (pattern cosine)")
    ap.add_argument("--pool-k", type=int, default=6,
                    help="v1.1 candidate times per member (isolated-curve minima)")
    ap.add_argument("--pool-nmax", type=int, default=8,
                    help="v1.1 exact-enumeration chunk size (k^n table)")
    ap.add_argument("--predict-min-energy-shower", type=float, default=-1.0,
                    help="v1.1: prediction floor [MeV] for blob clusters touching a "
                         "shower TPC (> 1.4 GeV charge); < 0 = same as "
                         "--predict-min-energy")
    ap.add_argument("--pool-commit-min", type=float, default=0.0,
                    help="v1.1: pool members below this energy [MeV] take part in the "
                         "joint component decision as light hypotheses but are NOT "
                         "committed by it (the tail solver places them with its spatial "
                         "prior); 20 = the light-information boundary")
    ap.add_argument("--pool-joint-min", type=int, default=1,
                    help="v1.1: components smaller than this use v1.0's ordering rule; "
                         "the exact combination search runs only from this size up "
                         "(audit: joint pays only in crowded components)")
    ap.add_argument("--pool-gain", type=float, default=0.0,
                    help="v1.1 amplitude profiling band b: every phase-1/pool commit "
                         "fits its per-TPC amplitude in [1-b,1+b] against the residual "
                         "without itself (template rescaled in place); 0 = off")
    ap.add_argument("--precision", action="store_true",
                    help="v1.2 stage P: per-hit precision timing (per-family support "
                         "channels, delay-aware per-channel fits on the pile-up residual, "
                         "weighted-median family apex time, apex-clock per-hit deposition "
                         "times, in-situ k_idx/per-type slew calibration); after stage E")
    ap.add_argument("--precision-calib", default=os.path.join(_HERE, "v120_calib_dev13.json"),
                    help="v1.2 calibration JSON (k_idx, beta_t, type_of) from v120_calibrate.py")
    ap.add_argument("--precision-rounds", type=int, default=3,
                    help="v1.2 pile-up model iterations")
    ap.add_argument("--precision-pieces", action="store_true",
                    help="v1.2.1: split each family into contact-chained pieces; the core keeps "
                         "the apex clock, every displaced piece >= 5 MeV gets its OWN free time "
                         "(scan +-10 ticks on the residual) -> per-piece delay, distance, beta")
    ap.add_argument("--pool-spatial", type=float, default=0.0,
                    help="v1.1 charge-contact prior in the component search: connected "
                         "members (hit clouds within 3 cm) get a same-time bonus of this "
                         "fraction of the chunk's median unary spread (v0.7 energy-unit "
                         "convention); applies to the pool and to --redecide; 0 = off")
    ap.add_argument("--redecide", action="store_true",
                    help="v1.1 stage 3.98: after pass2, re-decide light-coupled "
                         "COMPONENTS of placed imaged clusters jointly on the "
                         "leave-component-out base (exact combination search; "
                         "winner must beat the incumbent's exact loss; singletons "
                         "never touched; shower TPCs excluded)")
    ap.add_argument("--redecide-emin", type=float, default=5.0,
                    help="v1.1 re-decision participant floor [MeV]")
    ap.add_argument("--v200-min-e", type=float, default=None,
                    help="v2.0 test: stage 3.99 exact-block LNS re-solve of clusters >= E [MeV] (testing/v200)")
    ap.add_argument("--v200-sweeps", type=int, default=3)
    ap.add_argument("--fast-scan", action="store_true",
                    help="rank-1 scan fast path: collapse multi-channel FFT "
                         "scans to one 1-D template correlation (exact - "
                         "rank-1 structure verified per call, else fallback)")
    ap.add_argument("--edge-penalty", action="store_true",
                    help="2x2-ported boundary-hiding guard: charge predicted "
                         "light shifted past the window end against the "
                         "channel quiet-tail weight in every scan")
    ap.add_argument("--le-retime", action="store_true",
                    help="stage 5: re-time LE blobs (noise + sub-1.5 MeV "
                         "clusters) by ranked candidate families on the "
                         "placed backbone (le_ranker.json)")
    ap.add_argument("--gate-excess", type=float, default=500.0,
                    help="overflow gate: min per-channel excess ADC")
    ap.add_argument("--gate-zcut", type=float, default=2.0,
                    help="overflow gate: min poisson z of counted excess")
    ap.add_argument("--gate-nch", type=int, default=3,
                    help="overflow gate: channels over threshold to fire")
    ap.add_argument("--gate-cos", type=float, default=0.75,
                    help="overflow gate: cos(own,resid) firing threshold")
    ap.add_argument("--gate-support", action="store_true",
                    help="v0.8.7: restrict the overflow/defer+rescue trigger "
                         "to the subject's support channels (kills the "
                         "summation-floor false defers at big families)")
    ap.add_argument("--skip-v2", action="store_true",
                    help="speed: skip the V2 rescue stage (~38-50 s/ev) — "
                         "candidate superseded by joint+PASS2")
    ap.add_argument("--fast-track-stage", action="store_true",
                    help="speed: disable superseded track-stage refinements "
                         "(second-pass rescan, overlap swap, fine correction)"
                         " — stack2 re-places the backbone anyway")
    ap.add_argument("--light-ckpt", default="",
                    help="override the amplitude-perceiver checkpoint (e.g. "
                         "the 4x-smaller ndfull_small4x model; arch read "
                         "from the checkpoint's 'arch' dict)")
    ap.add_argument("--decline-r", type=float, default=0.0,
                    help="v0.8.4: after force, audit small imaged clusters; "
                         "if best-case light delta-loss / template-norm > r, "
                         "try support-channel permutation repair, else "
                         "unassign (-10000). 0 = off")
    ap.add_argument("--prune-frac", type=float, default=0.0,
                    help="v0.8.3: confidence-prune this fraction of assigned "
                         "hits at end of chain (charge+light badness model; "
                         "stats in row['prune']; headline metrics unchanged)")
    ap.add_argument("--prune-model", default=os.path.join(
        _HERE, "v083_prune_cuts.json"),
                    help=".json = 4-threshold cuts rule (default, data-"
                         "auditable); .joblib = the 40-feature HGB bundle")
    ap.add_argument("--max-clip-ticks", type=int, default=None,
                    help="override the vAlpha saturation veto (default: config value 6)")
    ap.add_argument("--police", choices=["", "locked", "top18"], default="",
                    help="overflow-police Phase 2 and V2 placements: newly placed/"
                         "moved labels that fire the gate are immediately resolved "
                         "(keep/rewind/rescan by union loss)")
    ap.add_argument("--police-topk", type=int, default=18)
    ap.add_argument("--merge-vertex", action="store_true",
                    help="enable the v0.2.5 vertex-track merge in the clustering phase")
    ap.add_argument("--cheat", choices=["none", "type1", "type12"], default="none",
                    help="three-stage cheating study: truth-lock type-1 (showers/"
                         ">=3-TPC tracks) or types 1+2 (all big clusters) at MC truth")
    ap.add_argument("--phase3", choices=["none", "matrix", "expand", "both", "joint"], default="none",
                    help="small-cluster stage after V2: matrix = v11 error-matrix "
                         "(old way), expand = v0.3.0 family expansion, both = "
                         "expansion then matrix tail")
    ap.add_argument("--predict-min-energy", type=float, default=3.0,
                    help="prediction threshold [MeV] when --phase3 != none "
                         "(<=0 -> production None = predict everything)")
    ap.add_argument("--early-expand", action="store_true",
                    help="v0.4.5 EXPERIMENT (stage 3.25): run a restricted family expansion "
                         "right after the stage-2/3.1 permutation placements — placed clusters "
                         "absorb only SMALL (< --early-expand-emax MeV) unassigned clusters "
                         "(no noise attach, no new-t0 opening), so their predicted light looks "
                         "full before P2/V2 instead of other clusters filling the space")
    ap.add_argument("--early-expand-emax", type=float, default=30.0)
    ap.add_argument("--merge-p2", action="store_true",
                    help="merge stages 3.1+3.3: every P2-class cluster (multi-TPC "
                         "or > --p2-threshold MeV, imaged) joins the backbone "
                         "permutation pool and is placed by energy-banded "
                         "all-orderings batches; the Phase-2 scan then only "
                         "handles parts detached by the rescue stage, and the "
                         "3.2 corrections run after ALL big placements")
    ap.add_argument("--frontend-repo", default=None,
                    help="path to an external front-end clustering repo "
                         "(clmatching-frontend): its recluster.run_clustering "
                         "replaces the built-in v0.5.5 clustering")
    ap.add_argument("--frontend-params", default=None,
                    help="params json for the external front end "
                         "(default <repo>/params_full.json)")
    ap.add_argument("--pass2-trade", action="store_true",
                    help="v0.8 PASS 2: spatial cluster trading between reco "
                         "families (v4 score + light-curve veto), after the "
                         "light stages, before swap-fix")
    ap.add_argument("--pass2-model", default=os.path.join(_HERE, "spatial_score_v4.joblib"))
    ap.add_argument("--pass2-margin", type=float, default=0.3)
    ap.add_argument("--pass2-pmin", type=float, default=0.6)
    ap.add_argument("--pass2-veto-frac", type=float, default=0.25)
    ap.add_argument("--pass2-light-nom", action="store_true",
                    help="v0.8.1: light-nominated trade candidates — add "
                         "families whose t0 the cluster's own loss curve "
                         "finds near-optimal to the 8-nearest spatial set")
    ap.add_argument("--pass2-light-prop", action="store_true",
                    help="v0.8.2: symmetric dual acceptance — light PROPOSES "
                         "(strong curve preference) with spatial as a weak "
                         "veto floor, alongside the spatial-margin path")
    ap.add_argument("--pass2-iterations", type=int, default=1,
                    help="run PASS 2 trading this many times (Q2 re-asked on "
                         "the improved geometry)")
    ap.add_argument("--loop-p23", type=int, default=0,
                    help="v0.8.2: alternate PASS2 (Q2 trade) and PASS3 (Q1 "
                         "family move/close) this many rounds — the full "
                         "light<->spatial iteration; overrides the separate "
                         "pass2-iterations/pass3-family paths")
    ap.add_argument("--pass3-family", action="store_true",
                    help="v0.8 PASS 3 (Q1): rigid family MOVE rescans + "
                         "ridge-flash CLOSE dissolution, after PASS 2")
    ap.add_argument("--joint-tpc-emax", type=float, default=1400.0,
                    help="--phase3 joint: clusters in TPCs above this charge "
                         "energy [MeV] are left for the matrix (shower TPCs); "
                         "<=0 disables the split")
    ap.add_argument("--matrix-relax", action="store_true",
                    help="v0.7 experiment: after the first matrix pass, KEEP the "
                         "matrix propagating over the still-unassigned imaged tail "
                         "with a relaxed worsen tolerance (light-loss increases "
                         "allowed), so force-coverage only gets the true dregs")
    ap.add_argument("--matrix-relax-tol", type=float, default=1.0,
                    help="relaxed matrix_worsen_tolerance_norm for the second pass "
                         "(first pass uses 0.15)")
    ap.add_argument("--matrix-relax-eps", type=float, default=0.0,
                    help="relaxed full_scan_assign_eps for the second pass "
                         "(first pass uses -20000)")
    ap.add_argument("--frontend-shower-min-tracks", type=int, default=0,
                    help="re-type external vertex groups of >= N merged tracks "
                         "as showers (builtin convention; 0 = off)")
    ap.add_argument("--loss", choices=["chi2", "cubic", "signed2"], default="chi2",
                    help="v0.5.0 EXPERIMENTAL placement loss for the pool (3.1), "
                         "its swap/rescue passes and the expansion (3.9): "
                         "cubic = sum w*r^3, signed2 = sum w*sign(r)*r^2, both "
                         "masked where pred AND actual < 200 ADC, decisions "
                         "driven toward |L|=0. chi2 (default) = exact v0.4.1. "
                         "Trigger/police/V2/matrix/force always stay chi2.")
    ap.add_argument("--expand-score", choices=["special", "plain"], default="plain",
                    help="family-expansion score: plain = raw clipped-Poisson "
                         "delta-chi2 only (regular pipeline loss; default since "
                         "2026-07-15 A/B tie), special = per-MeV delta-chi2 + "
                         "geodesic term (legacy v0.3.0)")
    ap.add_argument("--tf32", action="store_true",
                    help="allow TF32 tensor-core matmul in the perceiver "
                         "(faster stage 1; slightly different amplitudes -> "
                         "requires A/B before adoption)")
    ap.add_argument("--force-simple", action="store_true",
                    help="v0.10.6: KD-tree force coverage (nearest assigned "
                         "hit; no geodesics/ML) instead of force_full_coverage")
    ap.add_argument("--isolate-cm", type=float, default=0.0,
                    help="hits with no other hit within this distance get NO "
                         "t0 (NaN) at any stage (0 = off)")
    ap.add_argument("--force-coverage", action="store_true",
                    help="final stage: force-assign EVERYTHING left (imaged leftovers "
                         "at best of family-t0 vs own scan minimum even if loss "
                         "increases; sub-threshold clusters + noise by geometry)")
    ap.add_argument("--p2-threshold", type=float, default=50.0,
                    help="P2-vs-P3 boundary [MeV] (V11_LARGE_CLUSTER_ENERGY_MEV)")
    ap.add_argument("--light-fraction", type=float, default=0.90,
                    help="support = smallest channel set holding this light fraction")
    ap.add_argument("--support-max-gap", type=int, default=2)
    ap.add_argument("--swap-ratio-max", type=float, default=0.9,
                    help="swap-fix acceptance: chi2_swap < ratio * chi2_cur")
    ap.add_argument("--shower-absorb", action="store_true",
                    help="showers admit label -1 hits via their PCA ellipsoid")
    ap.add_argument("--track-absorb-scale", type=float, default=1.5,
                    help="toolbox track noise-absorption radius scale (default 1.5)")
    ap.add_argument("--shower-tpc", action="store_true",
                    help="dedicated shower-TPC pass: redo TPCs > --shower-thresh from "
                         "scratch (anchor max cluster to max flash, 70%% geodesic fill, "
                         "spatial-dislike scans, prune & grow)")
    ap.add_argument("--shower-thresh", type=float, default=1500.0)
    ap.add_argument("--shower-fill", type=float, default=0.70)
    ap.add_argument("--swap-fix", action="store_true",
                    help="after phase3, correct pairwise t0 SWAPS: reassign co-located "
                         "clusters to the flash their saturation-masked spatial light best "
                         "fits (Hungarian-style pairwise local search)")
    ap.add_argument("--swap-min-energy", type=float, default=25.0)
    ap.add_argument("--blip-rehome", action="store_true",
                    help="stage 4.7 (blip_rehome.py): 0.5-20 MeV clusters, after swap-fix and before "
                         "force: (1) contact certificate - touching placed >=20 MeV clusters that share "
                         "one time -> that time; (2) isolated re-homing - quiet-in-own-TPC times of "
                         "nearby big clusters in other TPCs, local-channel group cosine, host-distance "
                         "and glare gates")
    ap.add_argument("--blip-no-contact", action="store_true")
    ap.add_argument("--blip-no-iso", action="store_true")
    ap.add_argument("--blip-contact-cm", type=float, default=2.0)
    ap.add_argument("--blip-near", action="store_true",
                    help="stage 4.7 operator 1b: near-contact (nearest placed big cluster within 15 cm, no "
                         "co-timed big within 20 cm, light does not prefer the current slot, buried in glare)")
    ap.add_argument("--blip-reattach", action="store_true",
                    help="stage 4.7 operator 3: re-attach strays with no co-timed host to the nearby big "
                         "cluster minimising own-light/glare x distance (<= 1.25 cm)")
    ap.add_argument("--resid-fill", action="store_true",
                    help="stage 4.8 (testing/resid_fill.py): fill unexplained residual flashes with the templated small "
                         "cluster of the TPC whose predicted pattern matches the residual best (reverse construction, "
                         "user 2026-10-01); runs after 4.7 and before 4.9")
    ap.add_argument("--rf-small-max", type=float, default=100.0); ap.add_argument("--rf-cos-min", type=float, default=0.9)
    ap.add_argument("--rf-dcos", type=float, default=0.1); ap.add_argument("--rf-ratio-hi", type=float, default=3.0)
    ap.add_argument("--rf-tail", type=float, default=0.5); ap.add_argument("--rf-host-min", type=float, default=20.0)
    ap.add_argument("--rf-mode", default="sup"); ap.add_argument("--rf-kprom", type=float, default=10.0)
    ap.add_argument("--michel-nd", action="store_true",
                    help="stage 4.85 (testing/michel_nd.py): piece-first Michel / delayed-daughter re-timing - Bragg-attached end "
                         "pieces and small whole clusters get their own later flash when the perceiver light of the piece matches "
                         "the residual there and the prompt shows the deficit (user 2026-10-02); after 4.8, before 4.9")
    ap.add_argument("--mnd-host-extent-cm", type=float, default=0.0); ap.add_argument("--mnd-bragg-end-cm", type=float, default=1e9)
    ap.add_argument("--mnd-piece-max-mev", type=float, default=1e9); ap.add_argument("--mnd-delay-max", type=float, default=300.0)
    ap.add_argument("--mnd-host-dedx-max", type=float, default=1e9)   # host cluster E / skeleton extent (MeV/cm); proton-like hosts carry no Michel
    ap.add_argument("--mnd-tip-face-cm", type=float, default=0.0)      # free tip must be >= this far from the outer detector box (exiting tracks)
    ap.add_argument("--mnd-set", nargs="*", default=[])              # any MichelNDConfig field, k=v (applied after the named flags)
    ap.add_argument("--mnd-batch", type=int, default=16)   # 64 x dense voxel grids = 11 GiB: two workers per GPU ran out of memory
    ap.add_argument("--force-light-cm", type=float, default=None,
                    help="stage 4.9 variant (force_light_vote.py): cluster-density vote for noise hits beyond 3 cm, "
                         "multiplied by the positive light residual on the 6 nearest channels for hits farther than "
                         "this distance [cm] from any assigned hit (user 2026-09-30); overrides --noise-vote")
    ap.add_argument("--noise-vote", action="store_true",
                    help="4.9 force (with --force-simple): noise hits beyond 3 cm of any assigned hit take "
                         "the time of a distance-weighted vote of nearby assigned clusters "
                         "(force_noise_vote.py) instead of the nearest assigned hit")
    ap.add_argument("--swap-min-gain", type=float, default=0.02)
    ap.add_argument("--reassign", action="store_true",
                    help="after phase3, residual-driven per-TPC MOVE: fill a light-deficit flash "
                         "with the cluster-part whose amplitude pattern matches the missing light "
                         "(safe alternative to --swap-fix; won't move correctly-placed clusters)")
    ap.add_argument("--new-flash-rescue", action="store_true",
                    help="after force-coverage, residual-peak NEW-FLASH rescue: quick peak-find on "
                         "the channel-summed residual to discover unexplained flashes, spatially "
                         "match a nearby compact BLOB (tracks excluded) via the light-detector y-map, "
                         "and MOVE it to the new t0 if that lowers this-TPC light loss")
    args = ap.parse_args()
    if args.tf32:
        import torch
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if args.edge_penalty:
        global EDGE_PENALTY
        EDGE_PENALTY = True
        if _HERE not in sys.path:
            sys.path.insert(0, _HERE)
        import v02_ordering_pipeline as _vop
        import v030_expansion as _v30
        _vop.EDGE_PENALTY = True
        _v30.EDGE_PENALTY = True
    if args.fast_scan:
        global SCAN_TEMPLATE
        if _HERE not in sys.path:
            sys.path.insert(0, _HERE)
        import v02_ordering_pipeline as _vop2
        import v030_expansion as _v302
        _tmplfs = np.load(os.path.join(BASE_DIR, "assets", "avg_pulse.npy")).astype(np.float64) / 999.0
        _tc = np.argsort(_tmplfs)[::-1][[0, 40, 150]]   # peak + shoulder + tail
        SCAN_TEMPLATE = _tmplfs
        _vop2.SCAN_TEMPLATE = _tmplfs
        _v302.SCAN_TEMPLATE = _tmplfs
        global SCAN_TCOLS
        SCAN_TCOLS = _tc
        _vop2.SCAN_TCOLS = _tc
        _v302.SCAN_TCOLS = _tc
    if args.merge_p2 and args.backbone == "prod":
        ap.error("--merge-p2 requires --backbone stack2 (or v02); the merge "
                 "lives in the backbone swap-in block")
    if args.frontend_repo:
        if _HERE not in sys.path:
            sys.path.insert(0, _HERE)
        from frontend_adapter import make_external_clustering
        import M5p1.first_stage_matching.pipeline as _fsp_mod
        _fsp_mod.run_global_track_clustering = make_external_clustering(
            args.frontend_repo, args.frontend_params,
            shower_min_tracks=int(args.frontend_shower_min_tracks))
        print(f"FRONT-END clustering -> "
              f"{_fsp_mod.run_global_track_clustering.frontend_tag}", flush=True)
    if args.loss != "chi2":
        import v02_refine_pipeline as _vrp
        _vrp.configure_loss(args.loss)
        print(f"v0.5.0 loss -> {args.loss} (pool + refine passes + expansion; "
              f"trigger/police/V2/matrix/force stay chi2)", flush=True)
    if args.dump_dir:
        os.makedirs(args.dump_dir, exist_ok=True)

    import h5py
    from M5p1.first_stage_matching import load_first_stage_models, run_first_stage_charge_light_matching
    from M5p1.first_stage_matching.track_stage import fast_stack_images, append_candidate_t0
    from M5p1.phase25_trial2_valpha_batch import (
        _build_namespace, _run_phase2, build_default_first_stage_config,
        build_default_v2_config, _list_events_in_file)
    from M5p1 import phase25_trial2_v2_light_rescue as v2mod
    from v11_phased_matching import snapshot_backbone_hits_v11, verify_backbone_hits_unchanged_v11
    from M5p1 import truth_plotting as tp
    from pulse_shapes import timeinterpolation

    import M5p1.phase25_trial2_valpha_batch as _batchmod
    if abs(args.p2_threshold - 50.0) > 1e-9:
        _batchmod.V11_LARGE_CLUSTER_ENERGY_MEV = float(args.p2_threshold)
        print(f"P2/P3 boundary -> {args.p2_threshold} MeV", flush=True)

    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    from v02_refine_pipeline import configure_gate as _cfg_gate
    _cfg_gate(excess=args.gate_excess, zcut=args.gate_zcut,
              nch=args.gate_nch, cos=args.gate_cos)
    if (args.gate_excess, args.gate_zcut, args.gate_nch, args.gate_cos) != (500.0, 2.0, 3, 0.75):
        print(f"gate thresholds -> excess {args.gate_excess} z {args.gate_zcut} "
              f"nch {args.gate_nch} cos {args.gate_cos}", flush=True)
    cfg = build_default_first_stage_config()
    if os.environ.get("CLM_IMAGE_BATCH"):        # shrink render footprint for a memory-constrained GPU
        cfg.prediction.image_batch_size = int(os.environ["CLM_IMAGE_BATCH"])
        print(f"[mem] image_batch_size={cfg.prediction.image_batch_size}", flush=True)
    if args.light_ckpt:
        # v0.6.2 / v0.8.5: alternate (e.g. 4x-smaller) amplitude perceiver;
        # the loader reads the checkpoint's "arch" dict when present
        cfg.model.light_checkpoint = args.light_ckpt
        print(f"light checkpoint override -> {args.light_ckpt}", flush=True)
    if args.fast_track_stage:
        cfg.track_stage.enable_second_pass_rescan = False
        cfg.track_stage.enable_overlap_swap = False
        cfg.track_stage.enable_fine_correction = False
        print("fast track stage: rescan/overlap-swap/fine-correction OFF", flush=True)
    cfg.prediction.min_cluster_energy_mev = 50.0
    if args.phase3 != "none":
        cfg.prediction.min_cluster_energy_mev = (
            None if args.predict_min_energy <= 0 else float(args.predict_min_energy))
        if args.predict_min_energy_shower >= 0:
            cfg.prediction.min_cluster_energy_shower_tpc_mev = float(args.predict_min_energy_shower)
            print(f"prediction floor in shower TPCs -> {args.predict_min_energy_shower} MeV", flush=True)
        print(f"phase3={args.phase3}: prediction threshold -> "
              f"{cfg.prediction.min_cluster_energy_mev} MeV", flush=True)
    if args.merge_vertex:
        cfg.clustering.vertex_merge_enable = True
        print("v0.2.5 vertex-track merge ENABLED", flush=True)
    cfg.support.light_fraction = float(args.light_fraction)
    cfg.support.max_gap = int(args.support_max_gap)
    if args.shower_absorb:
        cfg.clustering.shower_absorb_enable = True
        print("shower noise absorption ENABLED", flush=True)
    cfg.clustering.track_noise_absorb_radius_scale = float(args.track_absorb_scale)
    if args.max_clip_ticks is not None:
        cfg.support.saturation_max_clip_ticks = int(args.max_clip_ticks)
        print(f"saturation veto tightened: max_clip_ticks = {args.max_clip_ticks}", flush=True)
    v2cfg = build_default_v2_config(n_outer_passes=1, light_max_total_moves=24,
                                    light_max_moves_per_tpc=3, device_policy="auto", verbose=False)
    v2_variants = {}
    if args.v2_tune:
        def _mk(**kw):
            c = build_default_v2_config(n_outer_passes=1, light_max_total_moves=24,
                                        light_max_moves_per_tpc=3, device_policy="auto",
                                        verbose=False)
            for k, v in kw.items():
                if not hasattr(c, k):
                    raise AttributeError(f"V2 config has no field {k}")
                setattr(c, k, v)
            return c
        _all = {
            "def": _mk(),
            "strict": _mk(phys_min_dchi2_improvement=2.0e3, phys_min_dchi2_per_mev=5.0e1,
                          phys_min_source_ofch_reduction=12),
            "loose": _mk(phys_min_dchi2_improvement=2.5e2, phys_min_dchi2_per_mev=5.0,
                         phys_min_source_ofch_reduction=5),
            "2pass": _mk(n_outer_passes=2),
            "moves48": _mk(light_max_total_moves=48, light_max_moves_per_tpc=6),
            "branches": _mk(light_use_rescue_branches=True),
            "sigma2": _mk(light_overflow_sigma=2.0, light_min_overflow_channels=4),
            "sigma45": _mk(light_overflow_sigma=4.5, light_min_overflow_channels=8),
        }
        names = [n.strip() for n in args.v2_tune.split(",") if n.strip()]
        if names == ["all"]:
            names = list(_all)
        v2_variants = {n: _all[n] for n in names}
        print(f"V2 tune variants: {list(v2_variants)}", flush=True)

    print("loading models ...", flush=True)
    models = load_first_stage_models(cfg)

    if args.backbone != "prod":
        if _HERE not in sys.path:
            sys.path.insert(0, _HERE)
        from v02_ordering_pipeline import place_pool_v02, add_label as v02_add_label
        from v02_refine_pipeline import (make_cos_pick, make_defer_hook,
                                         swap_pass, rescue_pass)

    Z = float(args.z_cut) if args.z_cut is not None else Z_CUT
    var_model = None
    if args.sigma_mode == "model":
        import torch
        from var_prediction.inference_ndfl_v2 import load_model as load_var_model, predict as predict_var
        _vdev = "cuda" if torch.cuda.is_available() else "cpu"
        var_model, _vmeta = load_var_model(VAR_CKPT, device=_vdev, waveform_len=1000, num_channels=120)
        print(f"variance model loaded on {_vdev} | trigger z-cut {Z}", flush=True)

    work = []
    for fp in args.files:
        try:
            ev_ids = _list_events_in_file(fp)
        except Exception as exc:
            print(f"SKIP {fp}: {exc}", flush=True)
            continue
        work.extend((fp, int(e)) for e in ev_ids)
    work = work[int(args.event_offset):: max(1, int(args.event_stride))]
    print(f"{len(work)} events for this worker", flush=True)

    fout = open(args.out, "w")
    h5_cache = {}
    for file_path, ev_id in work:
        t_ev = time.time()
        try:
            h5 = h5_cache.get(file_path)
            if h5 is None:
                h5 = h5py.File(file_path, "r")
                h5_cache[file_path] = h5
            # ========== STAGE 1: clustering & ML prediction ==========
            fs = run_first_stage_charge_light_matching(h5=h5, event_id=int(ev_id), config=cfg,
                                                       models=models, verbose=False)
            tim = {"first_stage": time.time() - t_ev}
            stage1_t = {k: round(float(v), 2)
                        for k, v in (getattr(fs, "timings", None) or {}).items()}
            labels_global = np.asarray(fs.clustering.labels_global)
            split_index = int(fs.clustering.split_index)
            label_info = fs.clustering.label_info
            image_maps = fs.prediction.image_maps
            cluster_to_tpcs = fs.prediction.cluster_to_tpcs
            tpc_to_clusters = fs.prediction.tpc_to_clusters
            support_cache = fs.prediction.cluster_channel_support_cache
            veto_mask = np.asarray(fs.prediction.saturated_channel_cache["veto_mask"], dtype=bool)
            # v1.2.2: per-channel saturation ceiling = int16 rail (2**15-1) minus the dynamic
            # baseline, built with the waveform in the first stage; every clip below and in the
            # testing-side stages (through sat_clip) uses it instead of the constant CLIP.
            sat_ceil = fs.prediction.saturated_channel_cache.get("ceiling")
            sat_ceil = (np.full(veto_mask.shape, CLIP, dtype=np.float32) if sat_ceil is None
                        else np.asarray(sat_ceil, dtype=np.float32))
            _sat_clip.set_ceiling(sat_ceil)
            actual = np.asarray(fs.event.full_light_waveform, dtype=np.float64)
            n_light, n_ch, n_ticks = actual.shape
            hit_t0 = np.asarray(fs.track_stage.hit_t0)
            hit_tpc = np.asarray(fs.event.hit_tpc_id)
            e_pos = np.clip(np.asarray(fs.event.energy, dtype=np.float64), 0.0, None)

            # --evo: per-cluster median-t snapshot at each stage checkpoint
            # (read-only — never touches the arrays it reads)
            t_evo = {}
            def _evo(stage, ts_arr):
                if not args.evo:
                    return
                ts_a = np.asarray(ts_arr, dtype=np.float64)
                okm = np.isfinite(ts_a) & (ts_a >= 0) & (labels_global >= 0)
                t_evo[stage] = {int(l): round(float(np.median(ts_a[(labels_global == l) & okm])), 2)
                                for l in np.unique(labels_global[okm])}

            truth = tp.extract_event_hit_energy_and_truth_t0(
                h5, h5["charge/events/ref/charge/calib_prompt_hits/ref"], eventid=int(ev_id),
                hits_full=h5["charge/calib_prompt_hits/data"], convert_to_matching_ticks=True)
            truth_t0 = np.asarray(truth["truth_t0"], dtype=np.float64)

            def _label_truth(lab):
                m = labels_global == lab
                tt = truth_t0[m]
                ww = e_pos[m]
                ok = np.isfinite(tt)
                if not ok.any():
                    return None
                o = np.argsort(tt[ok])
                cw = np.cumsum(ww[ok][o])
                return float(tt[ok][o][np.searchsorted(cw, 0.5 * cw[-1])])

            # ---- v0.2.5 vertex-merge truth audit (decision quality) ----
            vm_row = None
            if args.merge_vertex:
                vm = fs.clustering.debug.get("vertex_merge", {})
                pre = np.asarray(vm.get("labels_premerge", np.full_like(labels_global, -1)))
                vm_groups = []
                for g in vm.get("groups", []):
                    truths, energies = [], []
                    for lab in g:
                        mm = (pre == lab) & np.isfinite(truth_t0)
                        energies.append(float(e_pos[pre == lab].sum()))
                        if not mm.any():
                            truths.append(None)
                            continue
                        tv, wv = truth_t0[mm], e_pos[mm]
                        o = np.argsort(tv)
                        cw = np.cumsum(wv[o])
                        truths.append(float(tv[o][np.searchsorted(cw, 0.5 * cw[-1])]))
                    ts_ = [t for t in truths if t is not None]
                    dt = float(max(ts_) - min(ts_)) if len(ts_) >= 2 else None
                    vm_groups.append({"labels": [int(l) for l in g],
                                      "energies": [round(v, 1) for v in energies],
                                      "dt_truth": dt,
                                      "good": bool(dt is not None and dt <= 3.0)})
                vm_row = {"groups": vm_groups, "n_groups": len(vm_groups),
                          "n_good": sum(1 for r in vm_groups if r["good"]),
                          "pairs": vm.get("pairs", [])}

            # per-channel-per-tick variance: ML prediction or the Poisson g=200 form
            if var_model is not None:
                var_all = predict_var(var_model, np.asarray(actual, dtype=np.float32),
                                      batch_size=8, input_scale=1e-3, target_scale=1e-3,
                                      device=_vdev, return_variance=True,
                                      min_sigma=1.0).astype(np.float64)
                var_all = np.maximum(var_all, 1.0)
            else:
                var_all = GAIN * np.maximum(actual, 100.0)

            # ---------- three-stage cheating setup ----------
            cheat_t0, cheat_is_backbone, ht_map, Scheat = {}, {}, None, None
            if args.cheat != "none":
                from cheat_stages import (classify_clusters, split_and_predict,
                                          make_repin, hit_types, cheat_metrics)
                from M5p1.first_stage_matching.prediction import (
                    predict_first_stage_images_and_std as _predict_fn)
                types_map = classify_clusters(labels_global, split_index, label_info,
                                              cluster_to_tpcs, e_pos, n_light)
                want = {1} if args.cheat == "type1" else {1, 2}
                cheat_set = {l for l, ty in types_map.items() if ty in want}
                Scheat = dict(fs=fs, labels_global=labels_global, hit_tpc=hit_tpc,
                              truth_t0=truth_t0, e_pos=e_pos, n_light=n_light,
                              cluster_to_tpcs=cluster_to_tpcs, label_info=label_info,
                              image_maps=image_maps, support_cache=support_cache,
                              actual=actual)
                cheat_t0 = split_and_predict(Scheat, cheat_set, _predict_fn, models, cfg)
                types_map = classify_clusters(labels_global, split_index, label_info,
                                              cluster_to_tpcs, e_pos, n_light)
                for _lab in cheat_t0:
                    _par = label_info.get(int(_lab), {}).get("truth_split_from", _lab)
                    cheat_is_backbone[int(_lab)] = int(_par) < split_index
                ht_map = hit_types(Scheat, types_map)
                _cheat_ti = lambda a, t: timeinterpolation(a, shift=float(t), baseline=0)
                cheat_repin = make_repin(Scheat, cheat_t0, _cheat_ti)

                def _repin_ns(nsx):
                    tsx = np.asarray(nsx["hit_timestamps"], dtype=np.float64).copy()
                    nfx = cheat_repin(np.asarray(nsx["baseImage"]), tsx)
                    nsx["hit_timestamps"] = tsx.astype(np.float32)
                    return nfx
                print(f"cheat={args.cheat}: {len(cheat_t0)} truth-locked clusters "
                      f"({sum(cheat_is_backbone.values())} backbone-class)", flush=True)

            # ========== STAGE 2 (Phase 1: showers + >=3-TPC tracks) and
            # ========== STAGE 3.1 (1-2-TPC backbone tracks, permutation batches)
            # ---------- optional v0.2 backbone swap-in ----------
            v02_moved = {}
            merged_set, merged_placed, merge_stats = set(), {}, None
            pool_stats, rescue_audit = None, []
            if args.backbone != "prod":
                def ti_f(arr, t):
                    return timeinterpolation(arr, shift=float(t), baseline=0).astype(np.float64)
                w_full = 1.0 / (GAIN * np.maximum(actual, 100.0))
                info_b = {}
                for lab in range(split_index):
                    tpcs = sorted(int(t) for t in cluster_to_tpcs.get(int(lab), [])
                                  if int(t) < n_light and (int(lab), int(t)) in image_maps)
                    if not tpcs:
                        continue
                    members = np.flatnonzero(labels_global == lab)
                    if members.size == 0:
                        continue
                    t0v = hit_t0[members]
                    t0v = t0v[np.isfinite(t0v) & (t0v >= 0)]
                    if t0v.size == 0:
                        continue
                    lab_type = str(label_info.get(int(lab), {}).get("type", "track")).lower()
                    sup = {}
                    for c in tpcs:
                        entry = (support_cache.get((int(lab), int(c)))
                                 if lab_type == "track" else None)
                        ch = (np.asarray(entry["channel_indices"], dtype=int) if entry is not None
                              else np.arange(n_ch))
                        ch = ch[~veto_mask[int(c), ch]]
                        if ch.size:
                            sup[int(c)] = np.sort(ch)
                    if not sup:
                        continue
                    info_b[lab] = {"t0": float(np.median(t0v)), "tpcs": tpcs, "sup": sup,
                                   "type": lab_type, "energy": float(e_pos[members].sum())}
                for _lab, _tc in cheat_t0.items():
                    if _lab in info_b:
                        info_b[_lab]["t0"] = float(_tc)
                    elif cheat_is_backbone.get(int(_lab), False):
                        _tpcs = [int(t) for t in cluster_to_tpcs.get(int(_lab), [])
                                 if int(t) < n_light and (int(_lab), int(t)) in image_maps]
                        _sup = {}
                        for _c in _tpcs:
                            _img = np.asarray(image_maps[(int(_lab), int(_c))])
                            _ch = np.flatnonzero(_img.max(axis=1) > 1.0)
                            _ch = _ch[~veto_mask[int(_c), _ch]]
                            if _ch.size:
                                _sup[int(_c)] = np.sort(_ch)
                        if _sup:
                            _mm = labels_global == _lab
                            info_b[_lab] = {"t0": float(_tc), "tpcs": _tpcs, "sup": _sup,
                                            "type": "track",
                                            "energy": float(e_pos[_mm].sum())}
                if args.merge_p2:
                    # 3.1+3.3 merge: every P2-class cluster (multi-TPC or
                    # > p2-threshold, imaged) joins the permutation pool
                    for lab in map(int, fs.clustering.cluster_labels):
                        if lab in cheat_t0 or lab in info_b:
                            continue
                        tpcs = sorted(int(t) for t in cluster_to_tpcs.get(lab, [])
                                      if int(t) < n_light and (lab, int(t)) in image_maps)
                        if not tpcs:
                            continue
                        members = np.flatnonzero(labels_global == lab)
                        if members.size == 0:
                            continue
                        e_c = float(e_pos[members].sum())
                        if len(tpcs) < 2 and e_c <= float(args.p2_threshold):
                            continue
                        sup = {}
                        for c in tpcs:
                            entry = support_cache.get((lab, int(c)))
                            ch = (np.asarray(entry["channel_indices"], dtype=int)
                                  if entry is not None else np.arange(n_ch))
                            ch = ch[~veto_mask[int(c), ch]]
                            if ch.size:
                                sup[int(c)] = np.sort(ch)
                        if not sup:
                            continue
                        info_b[lab] = {"t0": float("nan"), "tpcs": tpcs, "sup": sup,
                                       "type": "cluster", "energy": e_c}
                        merged_set.add(lab)
                showers_b = [l for l in sorted(info_b) if info_b[l]["type"] == "shower"]
                multi_b = [l for l in sorted(info_b) if info_b[l]["type"] != "shower"
                           and len(info_b[l]["tpcs"]) >= 3 and l not in merged_set]
                pool_b = [l for l in sorted(info_b) if info_b[l]["type"] != "shower"
                          and (len(info_b[l]["tpcs"]) <= 2 or l in merged_set)
                          and l not in cheat_t0]
                fixed_extra = [l for l in sorted(info_b) if l in cheat_t0
                               and cheat_is_backbone.get(int(l), False)
                               and l not in showers_b and l not in multi_b]
                b_full = np.zeros_like(actual)
                p1_gain_log = []
                if args.pool_gain > 0:
                    if _HERE not in sys.path:
                        sys.path.insert(0, _HERE)
                    from v110_components import profile_gain as _v110_gain
                for lab in showers_b + multi_b + fixed_extra:
                    v02_add_label(b_full, lab, info_b[lab]["t0"], info_b, image_maps, ti_f)
                    if args.pool_gain > 0:
                        _g = _v110_gain(lab, info_b[lab]["t0"], b_full, info_b, image_maps,
                                        actual, w_full, ti_f, float(args.pool_gain))
                        if _g:
                            p1_gain_log.append({"label": int(lab), "e": float(info_b[lab]["energy"]),
                                                "gains": {int(c): float(v) for c, v in _g.items()}})
                    hit_t0[labels_global == lab] = np.float32(info_b[lab]["t0"])
                se_eatenA, se_stA = {}, {}
                if args.shower_eat and showers_b:
                    if _HERE not in sys.path:
                        sys.path.insert(0, _HERE)
                    from shower_eat import run_shower_eat
                    _xyz_se = np.stack([np.asarray(fs.event.x, np.float64),
                                        np.asarray(fs.event.y, np.float64),
                                        np.asarray(fs.event.z, np.float64)], 1)
                    _seedsA = [dict(t0=float(info_b[l]["t0"]), labels=[int(l)])
                               for l in showers_b]
                    se_eatenA, se_stA = run_shower_eat(
                        seeds=_seedsA, b_full=b_full, hit_t0=hit_t0,
                        image_maps=image_maps, actual=actual, w_full=w_full,
                        ti=ti_f, labels_global=labels_global, e_pos=e_pos,
                        xyz=_xyz_se, cluster_to_tpcs=cluster_to_tpcs,
                        label_info=label_info, n_light=n_light,
                        placed=set(info_b) | set(cheat_t0),
                        truth_fn=_label_truth)
                if args.backbone == "stack2":
                    stx = {}
                    # v0.8.7 candidate: restrict the overflow/defer gate to the
                    # subject's SUPPORT channels — the per-cluster-render sum
                    # smears +90-115% floors onto far channels at big families
                    # and the all-channel gate defers correct placements there
                    gate_sup = None
                    if args.gate_support:
                        def gate_sup(lab, c, _sc=support_cache, _im=image_maps,
                                     _li=label_info):
                            lt = str(_li.get(int(lab), {}).get("type", "track")).lower()
                            entry = _sc.get((int(lab), int(c))) if lt == "track" else None
                            if entry is not None:
                                return np.asarray(entry["channel_indices"], dtype=int)
                            img = _im.get((int(lab), int(c)))
                            if img is None:
                                return None
                            return np.flatnonzero(np.asarray(img).max(axis=1) > 1.0)
                    hook = make_defer_hook(info_b, image_maps, actual, veto_mask, ti_f, stx,
                                           sup_fn=gate_sup)
                    if args.pool_mode in ("light", "light-order"):
                        if _HERE not in sys.path:
                            sys.path.insert(0, _HERE)
                        from v110_components import place_pool_v110
                        pool_gain_log = []
                        tmap, pool_st = place_pool_v110(
                            pool_b, b_full, info_b, image_maps, actual, w_full, ti_f,
                            veto_mask, pick=make_cos_pick("soft"), commit_hook=hook,
                            cos_min=float(args.pool_cos), k=int(args.pool_k),
                            n_max=(min(int(args.pool_nmax), 5) if args.pool_mode == "light-order"
                                   else int(args.pool_nmax)),
                            gain_band=float(args.pool_gain),
                            gain_log=pool_gain_log,
                            spatial_frac=float(args.pool_spatial),
                            hits_of=(lambda _l: np.flatnonzero(labels_global == int(_l))),
                            xyz=(np.stack([np.asarray(fs.event.x, np.float64),
                                           np.asarray(fs.event.y, np.float64),
                                           np.asarray(fs.event.z, np.float64)], 1)
                                 if args.pool_spatial > 0 else None),
                            labels=labels_global,
                            preplaced={int(_l): float(info_b[_l]["t0"])
                                       for _l in showers_b + multi_b + fixed_extra},
                            truth_fn=_label_truth,
                            search=("ordering" if args.pool_mode == "light-order" else "combination"),
                            joint_min=int(args.pool_joint_min),
                            commit_min=float(args.pool_commit_min))
                    else:
                        tmap, pool_st = place_pool_v02(pool_b, 0.60, b_full, info_b, image_maps,
                                                       actual, w_full, ti_f,
                                                       pick=make_cos_pick("soft"), commit_hook=hook)
                    n_swp = swap_pass(tmap, b_full, info_b, image_maps, actual, w_full, ti_f)
                    nfp, nrp = rescue_pass(tmap, b_full, info_b, image_maps, actual, w_full,
                                           veto_mask, ti_f, sup_fn=gate_sup)
                else:
                    tmap, pool_st = place_pool_v02(pool_b, 0.60, b_full, info_b, image_maps,
                                                   actual, w_full, ti_f)
                    stx, n_swp, nfp, nrp = {}, 0, 0, 0
                for lab, t in tmap.items():
                    hit_t0[labels_global == lab] = np.float32(t)
                    if lab in merged_set:
                        merged_placed[int(lab)] = (float(t), info_b[lab]["tpcs"])
                    elif abs(t - info_b[lab]["t0"]) > CUT:
                        v02_moved[lab] = (float(t), info_b[lab]["tpcs"])
                if args.merge_p2:
                    merge_stats = {
                        "n_pool": len(merged_set),
                        "n_placed": len(merged_placed),
                        "e_placed": float(sum(info_b[l]["energy"]
                                              for l in merged_placed))}
                se_row = None
                if args.shower_eat:
                    if _HERE not in sys.path:
                        sys.path.insert(0, _HERE)
                    from shower_eat import run_shower_eat, emergent_groups
                    _xyz_se = np.stack([np.asarray(fs.event.x, np.float64),
                                        np.asarray(fs.event.y, np.float64),
                                        np.asarray(fs.event.z, np.float64)], 1)
                    _pm = {int(l): float(info_b[l]["t0"])
                           for l in showers_b + multi_b}
                    _pm.update({int(l): float(t) for l, t in tmap.items()})
                    _seedsB = emergent_groups(_pm, info_b, label_info,
                                              labels_global, _xyz_se, e_pos)
                    se_eatenB, se_stB = run_shower_eat(
                        seeds=_seedsB, b_full=b_full, hit_t0=hit_t0,
                        image_maps=image_maps, actual=actual, w_full=w_full,
                        ti=ti_f, labels_global=labels_global, e_pos=e_pos,
                        xyz=_xyz_se, cluster_to_tpcs=cluster_to_tpcs,
                        label_info=label_info, n_light=n_light,
                        placed=set(info_b) | set(cheat_t0) | set(se_eatenA),
                        truth_fn=_label_truth)
                    for _l, (_t0e, _tp) in {**se_eatenA, **se_eatenB}.items():
                        merged_placed[int(_l)] = (float(_t0e), list(_tp))
                    _audA = se_stA.pop("audit", []) if se_stA else []
                    _audB = se_stB.pop("audit", [])
                    se_row = {"explicit": se_stA, "emergent": se_stB,
                              "n_groups": len(_seedsB),
                              "eaten_ok": sum(1 for a in _audA + _audB if a.get("ok")),
                              "eaten_wrong": sum(1 for a in _audA + _audB
                                                 if a.get("ok") is False),
                              "audit": (_audA + _audB)[:150]}
                _bh = {}
                for _b in pool_st.get("batch_sizes", []):
                    _bh[str(int(_b))] = _bh.get(str(int(_b)), 0) + 1
                pool_stats = {
                    "n_pool": len(pool_b), "n_showers": len(showers_b),
                    "n_multi3": len(multi_b),
                    "n_batches": int(pool_st.get("n_batches", 0)),
                    "batch_size_hist": dict(sorted(_bh.items())),
                    "n_scans": int(pool_st.get("n_scans", 0)),
                    "n_perm_leaves": int(pool_st.get("n_leaves", 0)),
                    "perm_total": int(pool_st.get("perm_total", 0)),
                    "perm_better": int(pool_st.get("perm_better", 0)),
                    "n_deferred": int(stx.get("n_deferred", 0)),
                    "swap_pass_swaps": int(n_swp),
                    "rescue_pass_flagged": int(nfp),
                    "rescue_pass_moved": int(nrp),
                    "time_s": round(float(pool_st.get("time_s", 0.0)), 2)}
                if args.pool_mode in ("light", "light-order"):
                    for _k in ("n_components", "n_edges", "comp_size_hist", "n_chunked",
                               "n_uncommitted", "e_uncommitted",
                               "n_rejected", "n_gained", "gain_med", "gain_p16", "gain_p84",
                               "gain_frac_at_edge", "n_exact_checks", "t_search",
                               "n_multi_batches"):
                        if _k in pool_st:
                            pool_stats[_k] = pool_st[_k]
                    pool_stats["mode"] = args.pool_mode
                    pool_stats["chunk_audit"] = pool_st.get("chunk_audit", [])
                if args.pool_gain > 0:
                    _pg = [v for d in p1_gain_log for v in d["gains"].values()]
                    pool_stats["p1_gain"] = {"n": len(p1_gain_log),
                                             "med": float(np.median(_pg)) if _pg else None,
                                             "p16": float(np.percentile(_pg, 16)) if _pg else None,
                                             "p84": float(np.percentile(_pg, 84)) if _pg else None}
                np.asarray(fs.track_stage.base_image)[:] = b_full.astype(np.float32)

            police_fn = None
            if args.police:
                def _ti_f(arr, t):
                    return timeinterpolation(arr, shift=float(t), baseline=0).astype(np.float64)
                police_fn = make_police(actual, veto_mask, e_pos, support_cache, label_info,
                                        _ti_f, mode=args.police, topk=int(args.police_topk))

            tim["after_backbone"] = time.time() - t_ev
            _evo("2+3.1_backbone_pool", hit_t0)

            # ---------- BASELINE pass ----------
            if args.skip_baseline:
                m0, acc0, af0 = None, None, None
            else:
                ns0 = _build_namespace(fs, h5, cfg)
                for j, (t0n, tpcs_j) in v02_moved.items():
                    for c in tpcs_j:
                        append_candidate_t0(ns0["t0Candidates"][c], t0n, min_sep=2, max_t0=SEARCH)
                        while len(ns0["flash_cluster_received_by_tpc"][c]) < len(ns0["t0Candidates"][c]):
                            ns0["flash_cluster_received_by_tpc"][c].append(False)
                for j, (t0n, tpcs_j) in merged_placed.items():
                    for c in tpcs_j:
                        append_candidate_t0(ns0["t0Candidates"][c], t0n, min_sep=2, max_t0=SEARCH)
                        while len(ns0["flash_cluster_received_by_tpc"][c]) < len(ns0["t0Candidates"][c]):
                            ns0["flash_cluster_received_by_tpc"][c].append(False)
                        ns0["assignment_info"][(int(j), int(c))] = {
                            "clusterid": int(j), "tpcid": int(c), "t0": float(t0n),
                            "mode": "merged_pool", "stage": "merged_pool",
                            "assigned": True,
                            "energy": float(info_b[j]["energy"]) if j in info_b else 0.0}
                if merged_placed:
                    ns0["cluster_labels"] = [cc for cc in ns0["cluster_labels"]
                                             if int(cc) not in merged_placed]
                m0 = staged_run(ns0, v2mod, v2cfg, _run_phase2, snapshot_backbone_hits_v11,
                                verify_backbone_hits_unchanged_v11, truth_t0, e_pos, "base",
                                police=police_fn)
                acc0, af0 = m0["v2_acc"], m0["v2_afrac"]

            # ========== STAGE 3.2: trigger corrections (A/B/N=1, adopt, swap) ====
            # ---------- rescue decisions (identical logic to rescue A+B) ----------
            # (with --merge-p2 the pool-placed P2-class labels are included, so
            #  the trigger base carries their light and they are rescuable)
            info = {}
            for lab in itertools.chain(range(split_index), sorted(merged_placed)):
                tpcs = sorted(int(t) for t in cluster_to_tpcs.get(int(lab), [])
                              if int(t) < n_light and (int(lab), int(t)) in image_maps)
                if not tpcs:
                    continue
                members = np.flatnonzero(labels_global == lab)
                if members.size == 0:
                    continue
                t0v = hit_t0[members]
                t0v = t0v[np.isfinite(t0v) & (t0v >= 0)]
                if t0v.size == 0:
                    continue
                lab_type = str(label_info.get(int(lab), {}).get("type", "track")).lower()
                sup = {}
                for c in tpcs:
                    entry = (support_cache.get((int(lab), int(c)))
                             if lab_type in ("track", "cluster") else None)
                    ch = (np.asarray(entry["channel_indices"], dtype=int) if entry is not None
                          else np.arange(n_ch))
                    ch = ch[~veto_mask[int(c), ch]]
                    if ch.size:
                        sup[int(c)] = ch
                if not sup:
                    continue
                info[lab] = {"t0": float(np.median(t0v)), "tpcs": tpcs, "sup": sup,
                             "type": lab_type,
                             "energy": float(e_pos[members].sum())}

            for _lab, _tc in cheat_t0.items():
                if _lab in info:
                    continue
                _mm = labels_global == _lab
                _hv = hit_t0[_mm]
                if not (np.isfinite(_hv) & (_hv >= 0)).any():
                    continue
                _tpcs = [int(t) for t in cluster_to_tpcs.get(int(_lab), [])
                         if int(t) < n_light and (int(_lab), int(t)) in image_maps]
                _sup = {}
                for _c in _tpcs:
                    _img = np.asarray(image_maps[(int(_lab), int(_c))])
                    _ch = np.flatnonzero(_img.max(axis=1) > 1.0)
                    _ch = _ch[~veto_mask[int(_c), _ch]]
                    if _ch.size:
                        _sup[int(_c)] = np.sort(_ch)
                if _sup:
                    info[_lab] = {"t0": float(_tc), "tpcs": _tpcs, "sup": _sup,
                                  "type": "track", "energy": float(e_pos[_mm].sum())}

            base = np.zeros_like(actual)
            flagged = []
            trig_rows = []
            if merged_placed:
                # merge mode: stack ALL placed light first so every probe sees
                # the full model (incl. merged P2 clusters), then test below
                for lab in sorted(info):
                    L = info[lab]
                    tpcs_arr = np.asarray(L["tpcs"], dtype=np.int32)
                    sh = timeinterpolation(fast_stack_images(image_maps, int(lab), tpcs_arr),
                                           shift=L["t0"], baseline=0).astype(np.float64)
                    base[tpcs_arr] += sh.reshape(tpcs_arr.size, n_ch, -1)
            for lab in sorted(info):
                L = info[lab]
                tpcs_arr = np.asarray(L["tpcs"], dtype=np.int32)
                sh = timeinterpolation(fast_stack_images(image_maps, int(lab), tpcs_arr),
                                       shift=L["t0"], baseline=0).astype(np.float64)
                own_full = sh.reshape(tpcs_arr.size, n_ch, -1)
                if not merged_placed:
                    base[tpcs_arr] += own_full
                tick = int(round(L["t0"])) + PEAK
                if not (0 <= tick < n_ticks) or int(lab) in cheat_t0 or \
                        (L["type"] != "track" and int(lab) not in merged_placed):
                    continue
                fire = False
                cmin = 1.0
                z3m = -100.0
                for ti, c in enumerate(L["tpcs"]):
                    fin = ~veto_mask[c]
                    a = actual[c, :, tick]
                    p = clip_col(base[c, :, tick], c)
                    ex = p - a
                    z = ex / np.sqrt(var_all[c, :, tick])
                    zz = np.sort(z[fin & (ex >= EXCESS_GATE)])[::-1]
                    if zz.size >= 3:
                        z3m = max(z3m, float(zz[2]))
                        if zz[2] > Z:
                            fire = True
                    o = own_full[ti, :, tick]
                    res = np.clip(np.clip(a, 0, None) - np.clip(p - o, 0, None), 0, None)
                    no, nr = np.linalg.norm(o[fin]), np.linalg.norm(res[fin])
                    if no > 1e-9 and nr > 1e-9:
                        cmin = min(cmin, float(np.dot(o[fin], res[fin]) / (no * nr)))
                if cmin < COS_CUT:
                    fire = True
                trig_rows.append({"label": int(lab), "e": L["energy"], "t0": L["t0"],
                                  "z3": z3m, "cmin": cmin, "fired": bool(fire)})
                if fire:
                    flagged.append(lab)

            groups = []
            lone = []
            seen = set()
            for k in flagged:
                K = info[k]
                mem = [j for j, J in info.items()
                       if abs(J["t0"] - K["t0"]) <= GROUP_DT and set(J["tpcs"]) & set(K["tpcs"])]
                if len(mem) > MAX_N:
                    others = sorted((j for j in mem if j != k), key=lambda j: -info[j]["energy"])
                    mem = [k] + others[: MAX_N - 1]
                key = frozenset(mem)
                if key in seen:
                    continue
                seen.add(key)
                if len(mem) < 2:
                    lone.append(k)
                    continue
                groups.append((k, sorted(mem)))

            group_diags = []
            moves = {}          # label -> new t0 (Stage A adoptions)
            detaches = []       # (parent, tpc_f) Stage B adoptions
            for k, mem in groups:
                if any(j in moves for j in mem):
                    continue
                slot_t0 = info[k]["t0"]
                base_wo = base.copy()
                for j in mem:
                    J = info[j]
                    tpcs_arr = np.asarray(J["tpcs"], dtype=np.int32)
                    sh = timeinterpolation(fast_stack_images(image_maps, int(j), tpcs_arr),
                                           shift=J["t0"], baseline=0).astype(np.float64)
                    base_wo[tpcs_arr] -= sh.reshape(tpcs_arr.size, n_ch, -1)
                np.clip(base_wo, 0.0, None, out=base_wo)

                union = {}
                for j in mem:
                    for c, ch in info[j]["sup"].items():
                        union[c] = np.union1d(union.get(c, np.array([], dtype=int)), ch)
                w_union = {c: 1.0 / var_all[c][ch] for c, ch in union.items()}

                def combined_loss(placement):
                    model = {c: base_wo[c][ch].copy() for c, ch in union.items()}
                    for j, t0j in placement.items():
                        J = info[j]
                        sh = timeinterpolation(
                            fast_stack_images(image_maps, int(j), np.asarray(J["tpcs"], dtype=np.int32)),
                            shift=float(t0j), baseline=0).astype(np.float64).reshape(len(J["tpcs"]), n_ch, -1)
                        for ti, c in enumerate(J["tpcs"]):
                            if c in union:
                                model[c] += sh[ti][union[c]]
                    return float(sum(np.sum(w_union[c] * (clip_rows(model[c], c, union[c]) - actual[c][union[c]]) ** 2)
                                     for c in union))

                best = None
                for perm in itertools.permutations(mem):
                    placement = {perm[0]: slot_t0}
                    cur = {c: base_wo[c].copy() for c in set(c for j in mem for c in info[j]["tpcs"])}
                    J0 = info[perm[0]]
                    sh0 = timeinterpolation(
                        fast_stack_images(image_maps, int(perm[0]), np.asarray(J0["tpcs"], dtype=np.int32)),
                        shift=float(slot_t0), baseline=0).astype(np.float64).reshape(len(J0["tpcs"]), n_ch, -1)
                    for ti, c in enumerate(J0["tpcs"]):
                        cur[c] += sh0[ti]
                    for j in perm[1:]:
                        J = info[j]
                        pr, rr, wr = [], [], []
                        for c, ch in J["sup"].items():
                            pr.append(np.asarray(image_maps[(int(j), int(c))], dtype=np.float64)[ch])
                            rr.append(cur[c][ch] - actual[c][ch])
                            wr.append(1.0 / var_all[c][ch])
                        pr_j, rr_j = np.concatenate(pr), np.concatenate(rr)
                        curve = weighted_scan(pr_j, rr_j, np.concatenate(wr))
                        t0j = gated_argmin(curve, pr_j, rr_j, float(args.cos_sel))
                        placement[j] = t0j
                        shj = timeinterpolation(
                            fast_stack_images(image_maps, int(j), np.asarray(J["tpcs"], dtype=np.int32)),
                            shift=t0j, baseline=0).astype(np.float64).reshape(len(J["tpcs"]), n_ch, -1)
                        for ti, c in enumerate(J["tpcs"]):
                            cur[c] += shj[ti]
                    fl = combined_loss(placement)
                    if best is None or fl < best[0]:
                        best = (fl, dict(placement))
                # stay-gate (mirrors the N=1 branch): only move if the best permutation
                # actually beats leaving everyone where they are
                loss_stay_A = combined_loss({j: info[j]["t0"] for j in mem})
                group_diags.append({
                    "mem": [int(j) for j in mem], "loss_stay": float(loss_stay_A),
                    "loss_best": float(best[0]),
                    "would_move": {int(j): float(best[1][j]) for j in mem
                                   if abs(best[1][j] - info[j]["t0"]) > CUT},
                    "truths": {int(j): _label_truth(j) for j in mem},
                    "t0s": {int(j): float(info[j]["t0"]) for j in mem},
                })
                if best[0] < loss_stay_A * (1.0 - float(args.stay_margin)):
                    _, placement = best
                else:
                    placement = {j: info[j]["t0"] for j in mem}
                all_stay = all(abs(placement[j] - slot_t0) <= CUT for j in mem)

                if not all_stay:
                    for j in mem:
                        if j in cheat_t0:
                            continue
                        if abs(placement[j] - info[j]["t0"]) > CUT:
                            moves[j] = placement[j]
                    continue

                # Stage B on all-stay groups
                K = info[k]
                tick = int(round(K["t0"])) + PEAK
                best_sig = None
                for c in K["tpcs"]:
                    fin = ~veto_mask[c]
                    a = actual[c, :, tick]
                    p = clip_col(base[c, :, tick], c)
                    ex = p - a
                    z = ex / np.sqrt(var_all[c, :, tick])
                    zz = np.sort(z[fin & (ex >= EXCESS_GATE)])[::-1]
                    z3v = float(zz[2]) if zz.size >= 3 else -100.0
                    if best_sig is None or z3v > best_sig[0]:
                        best_sig = (z3v, int(c))
                tpc_f = best_sig[1]
                ch_u = np.array([], dtype=int)
                for j in mem:
                    if tpc_f in info[j]["sup"]:
                        ch_u = np.union1d(ch_u, info[j]["sup"][tpc_f])
                if not ch_u.size:
                    continue
                w_f = 1.0 / var_all[tpc_f][ch_u]
                movers = []
                for j in mem:
                    if j in cheat_t0:
                        continue
                    if tpc_f not in info[j]["sup"] or (int(j), tpc_f) not in image_maps:
                        continue
                    if len(info[j]["tpcs"]) < 2:
                        continue   # a single-TPC cluster has no cross-TPC stitch to undo
                    P = np.asarray(image_maps[(int(j), tpc_f)], dtype=np.float64)
                    sh_t0 = timeinterpolation(P, shift=info[j]["t0"], baseline=0).astype(np.float64)
                    b_wo = np.clip(base[tpc_f] - sh_t0, 0.0, None)
                    chp = info[j]["sup"][tpc_f]
                    rr_b = (b_wo - actual[tpc_f])[chp]
                    curve = weighted_scan(P[chp], rr_b, 1.0 / var_all[tpc_f][chp])
                    t1 = gated_argmin(curve, P[chp], rr_b, float(args.cos_sel))
                    if abs(t1 - info[j]["t0"]) <= CUT:
                        continue
                    sh_t1 = timeinterpolation(P, shift=t1, baseline=0).astype(np.float64)
                    lm = float(np.sum(w_f * (clip_rows((b_wo + sh_t1)[ch_u], tpc_f, ch_u)
                                             - actual[tpc_f][ch_u]) ** 2))
                    movers.append((lm, int(j)))
                if movers:
                    movers.sort()
                    detaches.append((movers[0][1], tpc_f))

            # ---------- N=1 branch: lone flagged clusters ----------
            # No anchor assumption: the lone cluster may sit on NO real flash.
            # Options: stay / unanchored full-range rescan of the whole cluster /
            # cut a per-TPC part off the merged track (NEW cluster id) and rescan
            # the remainder.  Winner by the union loss with every piece placed.
            grouped_members = set(j for _, mem in groups for j in mem)
            n1_rem_moves = {}
            n1_actions = []

            for k in lone:
                if k in grouped_members or k in moves:
                    continue
                K = info[k]
                tpcs = K["tpcs"]
                tpcs_arr = np.asarray(tpcs, dtype=np.int32)
                stk = fast_stack_images(image_maps, int(k), tpcs_arr)
                base_wo = base.copy()
                base_wo[tpcs_arr] -= timeinterpolation(stk, shift=K["t0"], baseline=0).astype(
                    np.float64).reshape(tpcs_arr.size, n_ch, -1)
                np.clip(base_wo, 0.0, None, out=base_wo)
                union = dict(K["sup"])
                w_u = {c: 1.0 / var_all[c][ch] for c, ch in union.items()}

                def u_loss(parts):
                    tot = 0.0
                    for c, ch in union.items():
                        m = base_wo[c][ch] + parts.get(c, 0.0)
                        tot += float(np.sum(w_u[c] * (clip_rows(m, c, ch) - actual[c][ch]) ** 2))
                    return tot

                def contrib(label_stk, tpc_list, t0v, into=None):
                    sh = timeinterpolation(label_stk, shift=float(t0v), baseline=0).astype(
                        np.float64).reshape(len(tpc_list), n_ch, -1)
                    parts = {} if into is None else into
                    for ti, c in enumerate(tpc_list):
                        if c in union:
                            parts[c] = parts.get(c, 0.0) + sh[ti][union[c]]
                    return parts

                loss_stay = u_loss(contrib(stk, tpcs, K["t0"]))
                pr = np.concatenate([np.asarray(image_maps[(int(k), int(c))], float)[ch]
                                     for c, ch in union.items()])
                rr = np.concatenate([(base_wo[c] - actual[c])[ch] for c, ch in union.items()])
                wr = np.concatenate([w_u[c] for c in union])
                t0_move = gated_argmin(weighted_scan(pr, rr, wr), pr, rr, float(args.cos_sel), float(args.cos_window))
                loss_move = u_loss(contrib(stk, tpcs, t0_move))
                cands = [("move", loss_move, t0_move, None)] if abs(t0_move - K["t0"]) > CUT else []

                if len(tpcs) >= 2:
                    for tpf in tpcs:
                        if tpf not in union or (int(k), int(tpf)) not in image_maps:
                            continue
                        P = np.asarray(image_maps[(int(k), int(tpf))], dtype=np.float64)
                        chp = union[tpf]
                        rr_p = (base_wo[tpf] - actual[tpf])[chp]
                        t1 = gated_argmin(weighted_scan(P[chp], rr_p, w_u[tpf]),
                                          P[chp], rr_p, float(args.cos_sel))
                        if abs(t1 - K["t0"]) <= CUT:
                            continue
                        rem = [c for c in tpcs if c != tpf]
                        rem_arr = np.asarray(rem, dtype=np.int32)
                        stk_r = fast_stack_images(image_maps, int(k), rem_arr)
                        rem_sup = [c for c in rem if c in union]
                        if rem_sup:
                            pr_r = np.concatenate([np.asarray(image_maps[(int(k), int(c))], float)[union[c]]
                                                   for c in rem_sup])
                            rr_r = np.concatenate([(base_wo[c] - actual[c])[union[c]] for c in rem_sup])
                            wr_r = np.concatenate([w_u[c] for c in rem_sup])
                            t0_rem = gated_argmin(weighted_scan(pr_r, rr_r, wr_r),
                                                  pr_r, rr_r, float(args.cos_sel), float(args.cos_window))
                        else:
                            t0_rem = K["t0"]
                        parts = contrib(P, [tpf], t1)
                        parts = contrib(stk_r, rem, t0_rem, into=parts)
                        cands.append(("detach", u_loss(parts), t0_rem, tpf))

                if not cands:
                    continue
                cands.sort(key=lambda c: c[1])
                kind, loss_best, t0_best, tp_best = cands[0]
                if loss_best >= loss_stay:
                    continue
                tr = _label_truth(k)
                if kind == "move":
                    moves[k] = t0_best
                    n1_actions.append({"label": int(k), "action": "move", "old_t0": K["t0"],
                                       "new_t0": t0_best, "truth": tr, "energy": K["energy"]})
                else:
                    detaches.append((k, tp_best))
                    if abs(t0_best - K["t0"]) > CUT:
                        n1_rem_moves[k] = (t0_best, [c for c in tpcs if c != tp_best])
                    n1_actions.append({"label": int(k), "action": "detach", "old_t0": K["t0"],
                                       "part_tpc": int(tp_best), "rem_t0": t0_best, "truth": tr,
                                       "energy": K["energy"]})

            # attach truth verdicts to the trigger diagnostics BEFORE labels mutate
            for r in trig_rows:
                trr = _label_truth(r["label"])
                r["truth"] = trr
                r["wrong"] = bool(trr is not None and abs(r["t0"] - trr) > CUT)

            # ---------- ADOPT ----------
            next_label = int(labels_global.max()) + 1 if (labels_global >= 0).any() else 0
            base_img = np.asarray(fs.track_stage.base_image)
            for j, t0n in moves.items():
                J = info[j]
                tpcs_arr = np.asarray(J["tpcs"], dtype=np.int32)
                stk = fast_stack_images(image_maps, int(j), tpcs_arr)
                sh_old = timeinterpolation(stk, shift=J["t0"], baseline=0).astype(np.float32)
                sh_new = timeinterpolation(stk, shift=float(t0n), baseline=0).astype(np.float32)
                base_img[tpcs_arr] += (sh_new - sh_old).reshape(tpcs_arr.size, n_ch, -1)
                hit_t0[labels_global == j] = np.float32(t0n)
                _tr = _label_truth(int(j))
                rescue_audit.append({
                    "kind": "move", "label": int(j), "old_t0": float(J["t0"]),
                    "new_t0": float(t0n), "truth": _tr, "e": float(J["energy"]),
                    "was_ok": (_tr is not None and abs(J["t0"] - _tr) <= CUT),
                    "now_ok": (_tr is not None and abs(float(t0n) - _tr) <= CUT)})
            done_detach = set()
            n_new = 0
            for j, tpc_f in detaches:
                if (j, tpc_f) in done_detach or (int(j), int(tpc_f)) not in image_maps or j in moves:
                    continue
                done_detach.add((j, tpc_f))
                rescue_audit.append({"kind": "detach", "label": int(j),
                                     "tpc": int(tpc_f),
                                     "truth": _label_truth(int(j)),
                                     "e": float(info[j]["energy"])})
                newlab = next_label
                next_label += 1
                n_new += 1
                mask = (labels_global == j) & (hit_tpc == tpc_f)
                labels_global[mask] = newlab
                np.asarray(fs.track_stage.labels_global_with_leftovers)[mask] = newlab
                hit_t0[mask] = np.nan
                P = np.asarray(image_maps.pop((int(j), int(tpc_f))))
                base_img[tpc_f] -= timeinterpolation(P.astype(np.float64), shift=info[j]["t0"],
                                                     baseline=0).astype(np.float32)
                image_maps[(int(newlab), int(tpc_f))] = P
                cluster_to_tpcs[int(j)] = [t for t in cluster_to_tpcs.get(int(j), []) if int(t) != tpc_f]
                cluster_to_tpcs[int(newlab)] = [int(tpc_f)]
                tpc_to_clusters.setdefault(int(tpc_f), []).append(int(newlab))
                tpc_to_clusters[int(tpc_f)].sort()
                entry = support_cache.pop((int(j), int(tpc_f)), None)
                if entry is not None:
                    entry = dict(entry)
                    entry["clusterid"] = int(newlab)
                    support_cache[(int(newlab), int(tpc_f))] = entry
                label_info[int(newlab)] = {"type": "cluster", "tpcs": [int(tpc_f)],
                                           "n_hits": int(mask.sum()), "n_tracks": 0, "track_ids": [],
                                           "cluster_id": None, "backbone": False,
                                           "segment_sources": [], "n_rescue_segments": 0}
                if int(j) in label_info:
                    label_info[int(j)]["tpcs"] = [t for t in label_info[int(j)].get("tpcs", [])
                                                  if int(t) != tpc_f]
                fs.clustering.cluster_labels.append(int(newlab))
            # N=1 detach remainders: move the surviving multi-TPC stub to its
            # rescanned t0 (part hits are already relabeled, so labels==k is
            # exactly the remainder now)
            for k, (t0r, rem_tpcs) in n1_rem_moves.items():
                if not rem_tpcs or k in moves:
                    continue
                rem_arr = np.asarray(rem_tpcs, dtype=np.int32)
                try:
                    stk_r = fast_stack_images(image_maps, int(k), rem_arr)
                except KeyError:
                    continue
                sh_old = timeinterpolation(stk_r, shift=info[k]["t0"], baseline=0).astype(np.float32)
                sh_new = timeinterpolation(stk_r, shift=float(t0r), baseline=0).astype(np.float32)
                base_img[rem_arr] += (sh_new - sh_old).reshape(rem_arr.size, n_ch, -1)
                hit_t0[labels_global == k] = np.float32(t0r)
            np.clip(base_img, 0.0, None, out=base_img)

            # ---------- SWAP & MOVE phase (lightweight, post-adoption) ----------
            # Pairwise: any two placed tracks sharing a TPC try exchanging t0s.
            # Solo: every placed track gets one cos-gated relocation scan.
            # Greedy adoption by loss gain; a cluster moves at most once per sweep.
            swap_actions, swap_new_t0s = [], {}
            if args.swap_phase:
                t_sw0 = time.time()
                bsw = np.asarray(base_img, dtype=np.float64).copy()
                for sweep in range(int(args.swap_sweeps)):
                    cur = {}
                    for lab, L in info.items():
                        if L["type"] != "track" and int(lab) not in merged_placed:
                            continue
                        tpcs_l = [c for c in L["tpcs"] if (int(lab), int(c)) in image_maps]
                        sup_l = {c: L["sup"][c] for c in tpcs_l if c in L["sup"]}
                        if not sup_l:
                            continue
                        tv = hit_t0[labels_global == lab]
                        tv = tv[np.isfinite(tv) & (tv >= 0)]
                        if tv.size == 0:
                            continue
                        cur[lab] = {"t0": float(np.median(tv)), "tpcs": tpcs_l, "sup": sup_l}

                    def rend(lab, t0v):
                        ta = np.asarray(cur[lab]["tpcs"], dtype=np.int32)
                        sh = timeinterpolation(fast_stack_images(image_maps, int(lab), ta),
                                               shift=float(t0v), baseline=0).astype(np.float64)
                        return ta, sh.reshape(ta.size, n_ch, -1)

                    rend_cache = {lab: rend(lab, cur[lab]["t0"]) for lab in cur}

                    def contrib_of(lab, t0v, union):
                        ta, sh = rend_cache[lab] if t0v == cur[lab]["t0"] else rend(lab, t0v)
                        out = {}
                        for k_i, c in enumerate(ta.tolist()):
                            if c in union:
                                out[c] = out.get(c, 0.0) + sh[k_i][union[c]]
                        return out

                    def base_without(union, labs_rm):
                        out = {c: bsw[c][ch].copy() for c, ch in union.items()}
                        for lab in labs_rm:
                            ta, sh = rend_cache[lab]
                            for k_i, c in enumerate(ta.tolist()):
                                if c in out:
                                    out[c] -= sh[k_i][union[c]]
                        for c in out:
                            np.clip(out[c], 0.0, None, out=out[c])
                        return out

                    def eval_loss(union, bwo, parts_list):
                        tot = 0.0
                        for c, ch in union.items():
                            mdl = bwo[c].copy()
                            for p in parts_list:
                                if c in p:
                                    mdl += p[c]
                            tot += float(np.sum((1.0 / (GAIN * np.maximum(actual[c][ch], 100.0)))
                                                * (clip_rows(mdl, c, ch) - actual[c][ch]) ** 2))
                        return tot

                    labs_sorted = sorted(cur)
                    by_tpc = {}
                    for lab in labs_sorted:
                        for c in cur[lab]["tpcs"]:
                            by_tpc.setdefault(c, []).append(lab)
                    pairs = set()
                    for c, ls in by_tpc.items():
                        for xx in range(len(ls)):
                            for yy in range(xx + 1, len(ls)):
                                pairs.add((ls[xx], ls[yy]))
                    cands_sw = []
                    for i, j in sorted(pairs):
                        if abs(cur[i]["t0"] - cur[j]["t0"]) <= CUT:
                            continue
                        union = {}
                        for lab in (i, j):
                            for c, ch in cur[lab]["sup"].items():
                                union[c] = np.union1d(union.get(c, np.array([], dtype=int)), ch)
                        bwo = base_without(union, (i, j))
                        L_cur = eval_loss(union, bwo, [contrib_of(i, cur[i]["t0"], union),
                                                       contrib_of(j, cur[j]["t0"], union)])
                        L_swp = eval_loss(union, bwo, [contrib_of(i, cur[j]["t0"], union),
                                                       contrib_of(j, cur[i]["t0"], union)])
                        if L_swp < L_cur:
                            cands_sw.append((L_cur - L_swp, "swap", i, j, cur[j]["t0"], cur[i]["t0"]))
                    for i in labs_sorted:
                        union = dict(cur[i]["sup"])
                        bwo = base_without(union, (i,))
                        pr = np.concatenate([np.asarray(image_maps[(int(i), int(c))], float)[ch]
                                             for c, ch in union.items()])
                        rr = np.concatenate([bwo[c] - actual[c][union[c]] for c in union])
                        wr = np.concatenate([1.0 / (GAIN * np.maximum(actual[c][ch], 100.0))
                                             for c, ch in union.items()])
                        t_new = gated_argmin(weighted_scan(pr, rr, wr), pr, rr,
                                             float(args.cos_sel), float(args.cos_window))
                        if abs(t_new - cur[i]["t0"]) > CUT:
                            L_cur = eval_loss(union, bwo, [contrib_of(i, cur[i]["t0"], union)])
                            L_new = eval_loss(union, bwo, [contrib_of(i, t_new, union)])
                            if L_new < L_cur:
                                cands_sw.append((L_cur - L_new, "move", i, None, float(t_new), None))
                    cands_sw.sort(key=lambda x: -x[0])
                    touched = set()
                    n_adopt = 0
                    for gain, kind, i, j, ti_new, tj_new in cands_sw:
                        if i in touched or (j is not None and j in touched):
                            continue
                        for lab, t_new in [(i, ti_new)] + ([(j, tj_new)] if kind == "swap" else []):
                            ta, sh_old = rend_cache[lab]
                            _, sh_new = rend(lab, t_new)
                            bsw[ta] += sh_new - sh_old
                            base_img[ta] += (sh_new - sh_old).astype(np.float32)
                            hit_t0[labels_global == lab] = np.float32(t_new)
                            swap_actions.append({"sweep": sweep, "kind": kind, "label": int(lab),
                                                 "old_t0": cur[lab]["t0"], "new_t0": float(t_new),
                                                 "partner": (int(j) if kind == "swap" else None),
                                                 "gain": float(gain), "truth": _label_truth(lab),
                                                 "energy": info[lab]["energy"]})
                            touched.add(lab)
                            swap_new_t0s[lab] = float(t_new)
                            n_adopt += 1
                    if n_adopt == 0:
                        break
                np.clip(base_img, 0.0, None, out=base_img)
                print(f"  swap phase: {len(swap_actions)} actions "
                      f"({sum(1 for a in swap_actions if a['kind']=='swap')} swap-halves, "
                      f"{sum(1 for a in swap_actions if a['kind']=='move')} moves) "
                      f"in {time.time()-t_sw0:.1f}s", flush=True)

            if args.cheat != "none":
                _ts_tmp = np.asarray(hit_t0, dtype=np.float64).copy()
                cheat_repin(base_img, _ts_tmp)
                hit_t0[:] = _ts_tmp.astype(hit_t0.dtype)

            tim["after_rescue_adopt"] = time.time() - t_ev
            _evo("3.2_rescue_swap", hit_t0)

            # ========== STAGE 3.3 (P2 large-cluster scan + police) and ==========
            # ========== STAGE 3.5 (V2 spatial+light rescue + police)  ==========
            # ---------- RESCUED pass ----------
            ns1 = _build_namespace(fs, h5, cfg)
            for j, (t0n, tpcs_j) in v02_moved.items():
                for c in tpcs_j:
                    append_candidate_t0(ns1["t0Candidates"][c], t0n, min_sep=2, max_t0=SEARCH)
                    while len(ns1["flash_cluster_received_by_tpc"][c]) < len(ns1["t0Candidates"][c]):
                        ns1["flash_cluster_received_by_tpc"][c].append(False)
            for j, t0n in moves.items():
                for c in info[j]["tpcs"]:
                    if append_candidate_t0(ns1["t0Candidates"][c], t0n, min_sep=2, max_t0=SEARCH):
                        pass
                    while len(ns1["flash_cluster_received_by_tpc"][c]) < len(ns1["t0Candidates"][c]):
                        ns1["flash_cluster_received_by_tpc"][c].append(False)
            for j, t0n in swap_new_t0s.items():
                for c in info[j]["tpcs"]:
                    append_candidate_t0(ns1["t0Candidates"][c], t0n, min_sep=2, max_t0=SEARCH)
                    while len(ns1["flash_cluster_received_by_tpc"][c]) < len(ns1["t0Candidates"][c]):
                        ns1["flash_cluster_received_by_tpc"][c].append(False)
            for j, (t0n, tpcs_j) in merged_placed.items():
                _tv = hit_t0[labels_global == j]
                _tv = _tv[np.isfinite(_tv) & (_tv >= 0)]
                t0c = float(np.median(_tv)) if _tv.size else float(t0n)
                for c in tpcs_j:
                    append_candidate_t0(ns1["t0Candidates"][c], t0c, min_sep=2, max_t0=SEARCH)
                    while len(ns1["flash_cluster_received_by_tpc"][c]) < len(ns1["t0Candidates"][c]):
                        ns1["flash_cluster_received_by_tpc"][c].append(False)
                    ns1["assignment_info"][(int(j), int(c))] = {
                        "clusterid": int(j), "tpcid": int(c), "t0": float(t0c),
                        "mode": "merged_pool", "stage": "merged_pool",
                        "assigned": True,
                        "energy": float(info.get(j, {}).get("energy", 0.0))}
            if merged_placed:
                # merged labels are already placed: keep them out of every
                # downstream worklist (built from cluster_labels inside
                # _run_phase2), exactly like the cheat pre-place purge
                ns1["cluster_labels"] = [cc for cc in ns1["cluster_labels"]
                                         if int(cc) not in merged_placed]
            if args.cheat != "none":
                _bi1 = np.asarray(ns1["baseImage"])
                _ts1 = np.asarray(ns1["hit_timestamps"], dtype=np.float64).copy()
                for _lab, _tc in cheat_t0.items():
                    if cheat_is_backbone.get(int(_lab), False):
                        continue          # backbone-class already placed in stage 1
                    _mm = labels_global == _lab
                    _ts1[_mm] = float(_tc)
                    for _c in cluster_to_tpcs.get(int(_lab), []):
                        _key = (int(_lab), int(_c))
                        if _key in image_maps and int(_c) < n_light:
                            _bi1[_c] += timeinterpolation(
                                np.asarray(image_maps[_key], np.float64),
                                shift=float(_tc), baseline=0).astype(_bi1.dtype)
                            append_candidate_t0(ns1["t0Candidates"][int(_c)], float(_tc),
                                                min_sep=2, max_t0=SEARCH)
                            while len(ns1["flash_cluster_received_by_tpc"][int(_c)]) <                                     len(ns1["t0Candidates"][int(_c)]):
                                ns1["flash_cluster_received_by_tpc"][int(_c)].append(False)
                ns1["hit_timestamps"] = _ts1.astype(np.float32)
                # keep cheated clusters out of EVERY downstream worklist: the v11
                # structures are built inside _run_phase2 from cluster_labels
                ns1["cluster_labels"] = [cc for cc in ns1["cluster_labels"]
                                         if int(cc) not in cheat_t0]

            # ========== STAGE 3.25 (v0.4.5 experiment): EARLY family expansion ======
            # Placed stage-2/3.1 clusters absorb small (<emax) unassigned neighbours NOW,
            # so their predicted waveform is full before the P2/V2 scans (instead of other
            # clusters "filling in the space").  Absorbed labels are purged from
            # cluster_labels, so every downstream worklist (built inside _run_phase2)
            # skips them — the same protection pattern as merged_placed / cheat.
            ee_row = None
            if args.early_expand:
                from v030_expansion import run_family_expansion as _rfe_early
                t_ee = time.time()
                ee_row = _rfe_early(ns1, verbose=False,
                                    params={"E_MAX_CAND": float(args.early_expand_emax),
                                            "ASSIGN_NOISE": False, "OPEN_NEW_T0": False,
                                            "PLAIN_SCORE": args.expand_score == "plain"})
                ee_row.pop("assigned", None)
                _taken_ee = set(ee_row.pop("member_labels", []))
                ns1["cluster_labels"] = [cc for cc in ns1["cluster_labels"]
                                         if int(cc) not in _taken_ee]
                ee_row["n_taken"] = len(_taken_ee)
                ee_row["time_s"] = round(time.time() - t_ee, 2)
                tim["after_early_expand"] = time.time() - t_ev
                _evo("3.25_early_expand", ns1["hit_timestamps"])

            if not v2_variants:
                m1 = staged_run(ns1, v2mod, v2cfg, _run_phase2, snapshot_backbone_hits_v11,
                                verify_backbone_hits_unchanged_v11, truth_t0, e_pos, "rescued",
                                police=police_fn, skip_v2=bool(args.skip_v2))
                acc1, af1 = m1["v2_acc"], m1["v2_afrac"]
                if args.cheat != "none":
                    _repin_ns(ns1)
            else:
                # Phase 2 once, then each V2 variant on a restored post-P2 snapshot.
                # V2 (incl. its inner trial2 spatial commit) replaces exactly these
                # namespace keys: hit_timestamps, baseImage, t0Candidates,
                # flash_cluster_received_by_tpc (verified against both modules).
                import copy as _cp
                ts1v = np.asarray(ns1["hit_timestamps"], dtype=np.float64).copy()
                A1 = np.isfinite(ts1v) & (ts1v >= 0)
                m1 = {}
                m1["p1_acc"], m1["p1_afrac"] = accuracy(ts1v, truth_t0, e_pos)
                snap = snapshot_backbone_hits_v11(
                    hit_timestamps=ns1["hit_timestamps"], labels_global=ns1["labels_global"],
                    v8_absorbed_hit_parent=ns1.get("v8_absorbed_hit_parent"),
                    track_shower_labels=ns1["track_shower_labels"])
                _run_phase2(ns1, verbose=False)
                verify_backbone_hits_unchanged_v11(snap, hit_timestamps=ns1["hit_timestamps"],
                                                   stage_name="P2(tune)")
                ts2v = np.asarray(ns1["hit_timestamps"], dtype=np.float64).copy()
                A2 = np.isfinite(ts2v) & (ts2v >= 0)
                m1["p2_acc"], m1["p2_afrac"] = accuracy(ts2v, truth_t0, e_pos)
                m1["p2_acc_A1"], _ = accuracy(ts2v, truth_t0, e_pos, mask=A1)
                m1["p2_acc_new"], _ = accuracy(ts2v, truth_t0, e_pos, mask=A2 & ~A1)
                post = {"hit_timestamps": np.asarray(ns1["hit_timestamps"]).copy(),
                        "baseImage": np.asarray(ns1["baseImage"]).copy(),
                        "t0Candidates": _cp.deepcopy(ns1["t0Candidates"]),
                        "flash_cluster_received_by_tpc":
                            _cp.deepcopy(ns1.get("flash_cluster_received_by_tpc"))}
                m1["v2"] = {}
                for vname, vcfg in v2_variants.items():
                    ns1["hit_timestamps"] = post["hit_timestamps"].copy()
                    ns1["baseImage"] = post["baseImage"].copy()
                    ns1["t0Candidates"] = _cp.deepcopy(post["t0Candidates"])
                    if post["flash_cluster_received_by_tpc"] is not None:
                        ns1["flash_cluster_received_by_tpc"] = _cp.deepcopy(
                            post["flash_cluster_received_by_tpc"])
                    ns1.pop("phase25_trial2_v2_result", None)
                    t_v = time.time()
                    v2mod.run_trial2_v2_light_rescue_from_namespace(ns1, config=vcfg, commit=True)
                    ts3v = np.asarray(ns1["hit_timestamps"], dtype=np.float64).copy()
                    e = {}
                    e["v2_acc"], e["v2_afrac"] = accuracy(ts3v, truth_t0, e_pos)
                    e["v2_acc_A1"], _ = accuracy(ts3v, truth_t0, e_pos, mask=A1)
                    e["v2_acc_new"], _ = accuracy(ts3v, truth_t0, e_pos, mask=A2 & ~A1)
                    moved = A1 & np.isfinite(ts3v) & (np.abs(ts3v - ts1v) > CUT)
                    e["moved_efrac"] = float(np.sum(e_pos[moved]) / max(np.sum(e_pos[A1]), 1e-9))
                    e["moved_acc_before"], _ = accuracy(ts1v, truth_t0, e_pos, mask=moved)
                    e["moved_acc_after"], _ = accuracy(ts3v, truth_t0, e_pos, mask=moved)
                    e["time_s"] = time.time() - t_v
                    m1["v2"][vname] = e
                _ref = m1["v2"].get("def") or next(iter(m1["v2"].values()))
                acc1, af1 = _ref["v2_acc"], _ref["v2_afrac"]

            tim["after_staged"] = time.time() - t_ev
            _evo("3.3+3.5_p2_v2", ns1["hit_timestamps"])

            # ========== STAGE 3.9 (expansion) + STAGE 4 (Phase 3) ==========
            # ---------- v0.3.0 small-cluster stage (where Phase 3 lives) ----------
            p3_row = None
            if args.phase3 != "none":
                t_p3 = time.time()
                ts_pre3 = np.asarray(ns1["hit_timestamps"], dtype=np.float64).copy()
                A_pre3 = np.isfinite(ts_pre3) & (ts_pre3 >= 0)
                p3_row = {"mode": args.phase3}
                if args.phase3 == "joint":
                    if _HERE not in sys.path:
                        sys.path.insert(0, _HERE)
                    from v070_joint import run_joint_assignment
                    st_j = run_joint_assignment(
                        ns1, verbose=False,
                        params={"TPC_E_MAX": (float(args.joint_tpc_emax)
                                              if args.joint_tpc_emax > 0 else None)})
                    for _aj in st_j.get("assigned", []):
                        _labj, _t0j = int(_aj["label"]), float(_aj["t0"])
                        for _cj in cluster_to_tpcs.get(_labj, []):
                            _cj = int(_cj)
                            if _cj >= n_light or (_labj, _cj) not in ns1["imageMaps"]:
                                continue
                            append_candidate_t0(ns1["t0Candidates"][_cj], _t0j,
                                                min_sep=2, max_t0=SEARCH)
                            while len(ns1["flash_cluster_received_by_tpc"][_cj]) < \
                                    len(ns1["t0Candidates"][_cj]):
                                ns1["flash_cluster_received_by_tpc"][_cj].append(False)
                            ns1["assignment_info"][(_labj, _cj)] = {
                                "clusterid": _labj, "tpcid": _cj, "t0": _t0j,
                                "mode": "joint", "stage": "joint",
                                "assigned": True, "energy": float(_aj["e"])}
                    st_j.pop("assigned", None)
                    ts_pj = np.asarray(ns1["hit_timestamps"], dtype=np.float64)
                    new_j = np.isfinite(ts_pj) & (ts_pj >= 0) & ~A_pre3
                    m1["p3e_acc"], m1["p3e_afrac"] = accuracy(ts_pj, truth_t0, e_pos)
                    m1["p3e_acc_new"], _ = accuracy(ts_pj, truth_t0, e_pos, mask=new_j)
                    m1["p3e_new_efrac"] = float(e_pos[new_j].sum() / max(e_pos.sum(), 1e-9))
                    _taken_j = set(st_j.pop("member_labels", []))
                    for _k in ("v11_iterative_single_tpc", "unassigned_by_tpc"):
                        for _t, _lst in list(ns1[_k].items()):
                            ns1[_k][_t] = [cc for cc in _lst if int(cc) not in _taken_j]
                    ns1["v11_pruned_iterative_clusters"] = [
                        cc for cc in ns1.get("v11_pruned_iterative_clusters", [])
                        if int(cc) not in _taken_j]
                    p3_row["joint"] = st_j
                    tim["after_joint"] = time.time() - t_ev
                    _evo("3.9_joint", ns1["hit_timestamps"])
                if args.phase3 in ("expand", "both"):
                    if _HERE not in sys.path:
                        sys.path.insert(0, _HERE)
                    from v030_expansion import run_family_expansion
                    st_e = run_family_expansion(
                        ns1, verbose=False,
                        params={"PLAIN_SCORE": args.expand_score == "plain",
                                "LOSS_MODE": args.loss})
                    st_e.pop("assigned", None)
                    # purge expansion-assigned clusters from the matrix worklists
                    # (else the matrix phase re-places them: double light + overwrites)
                    taken = set(st_e.pop("member_labels", []))
                    for _k in ("v11_iterative_single_tpc", "unassigned_by_tpc"):
                        for _t, _lst in list(ns1[_k].items()):
                            ns1[_k][_t] = [cc for cc in _lst if int(cc) not in taken]
                    # the pruned list too: the matrix marks every pruned cluster
                    # unassigned WITHOUT subtracting its light from the base, so
                    # an expansion-assigned sub-threshold cluster would be wiped
                    # and its light double-counted when force re-places it
                    ns1["v11_pruned_iterative_clusters"] = [
                        cc for cc in ns1.get("v11_pruned_iterative_clusters", [])
                        if int(cc) not in taken]
                    ts_pe = np.asarray(ns1["hit_timestamps"], dtype=np.float64)
                    new_e = np.isfinite(ts_pe) & (ts_pe >= 0) & ~A_pre3
                    m1["p3e_acc"], m1["p3e_afrac"] = accuracy(ts_pe, truth_t0, e_pos)
                    m1["p3e_acc_new"], _ = accuracy(ts_pe, truth_t0, e_pos, mask=new_e)
                    m1["p3e_new_efrac"] = float(e_pos[new_e].sum() / max(e_pos.sum(), 1e-9))
                    p3_row["expand"] = st_e
                    tim["after_expand"] = time.time() - t_ev
                    _evo("3.9_expansion", ns1["hit_timestamps"])
                if args.phase3 in ("matrix", "both", "joint"):
                    from v11_phased_matching import run_small_cluster_matrix_phase_v11
                    (p3_base, p3_ts, p3_cand, p3_ainfo, p3_unass, p3_log, _p3u,
                     p3_stats) = run_small_cluster_matrix_phase_v11(
                        active_cluster_tpcs=ns1["v11_active_cluster_tpcs"],
                        iterative_single_tpc=ns1["v11_iterative_single_tpc"],
                        pruned_iterative_clusters=ns1["v11_pruned_iterative_clusters"],
                        image_maps=ns1["imageMaps"],
                        base_image=ns1["baseImage"],
                        full_light_waveform=ns1["fullLightWaveform"],
                        full_light_std=ns1["fullLightStd"],
                        channel_support_cache=ns1["cluster_channel_support_cache"],
                        labels_global=ns1["labels_global"],
                        hit_timestamps=ns1["hit_timestamps"],
                        t0_candidates=ns1["t0Candidates"],
                        assignment_info=ns1["assignment_info"],
                        unassigned_by_tpc=ns1["unassigned_by_tpc"],
                        cluster_energies=ns1["cluster_energies"],
                        energy_band_fraction=0.20,
                        positive_row_margin=-1e-4,
                        matrix_worsen_tolerance_norm=0.15,
                        search_range=SEARCH,
                        adc_clip=sat_ceil,
                        collect_scan_losses=False,
                        full_scan_assign_eps=-20000,
                        backward_peak_align_ticks=5,
                        leftover_absorption_context=ns1.get("v11_1_leftover_absorption_context"),
                        saturated_channel_cache=ns1["saturated_channel_cache"],
                    )
                    ns1["baseImage"] = p3_base
                    ns1["hit_timestamps"] = p3_ts
                    ns1["t0Candidates"] = p3_cand
                    ns1["assignment_info"] = p3_ainfo
                    ns1["unassigned_by_tpc"] = p3_unass
                    p3_row["matrix"] = {
                        "n_proposed": int(len(p3_log)),
                        "n_assigned": int(sum(1 for r in p3_log if r.get("assigned")))}
                    tim["after_matrix"] = time.time() - t_ev
                    _evo("4.1_matrix", ns1["hit_timestamps"])
                    if args.matrix_relax:
                        # v0.7-a: keep the matrix propagating over the imaged
                        # tail (would-be force cohort) with light-loss increases
                        # allowed, instead of forcing by pure distance
                        _ts_r = np.asarray(ns1["hit_timestamps"], dtype=np.float64)
                        _asg_r = np.isfinite(_ts_r) & (_ts_r >= 0)
                        _acc_r0, _ = accuracy(_ts_r, truth_t0, e_pos)
                        _work2 = {}
                        for _lab in ns1["cluster_labels"]:
                            _lab = int(_lab)
                            _m2 = labels_global == _lab
                            if not _m2.any() or _asg_r[_m2].any():
                                continue
                            _tp2 = sorted(int(c) for c in cluster_to_tpcs.get(_lab, [])
                                          if int(c) < n_light
                                          and (_lab, int(c)) in ns1["imageMaps"])
                            if len(_tp2) != 1:
                                continue
                            _work2.setdefault(_tp2[0], []).append(_lab)
                        if _work2:
                            (p3_base, p3_ts, p3_cand, p3_ainfo, p3_unass, p3_log2,
                             _p3u2, _p3s2) = run_small_cluster_matrix_phase_v11(
                                active_cluster_tpcs=ns1["v11_active_cluster_tpcs"],
                                iterative_single_tpc=_work2,
                                pruned_iterative_clusters=[],
                                image_maps=ns1["imageMaps"],
                                base_image=ns1["baseImage"],
                                full_light_waveform=ns1["fullLightWaveform"],
                                full_light_std=ns1["fullLightStd"],
                                channel_support_cache=ns1["cluster_channel_support_cache"],
                                labels_global=ns1["labels_global"],
                                hit_timestamps=ns1["hit_timestamps"],
                                t0_candidates=ns1["t0Candidates"],
                                assignment_info=ns1["assignment_info"],
                                unassigned_by_tpc=ns1["unassigned_by_tpc"],
                                cluster_energies=ns1["cluster_energies"],
                                energy_band_fraction=0.20,
                                positive_row_margin=-1e-4,
                                matrix_worsen_tolerance_norm=float(args.matrix_relax_tol),
                                search_range=SEARCH,
                                adc_clip=sat_ceil,
                                collect_scan_losses=False,
                                full_scan_assign_eps=float(args.matrix_relax_eps),
                                backward_peak_align_ticks=5,
                                leftover_absorption_context=ns1.get("v11_1_leftover_absorption_context"),
                                saturated_channel_cache=ns1["saturated_channel_cache"],
                            )
                            ns1["baseImage"] = p3_base
                            ns1["hit_timestamps"] = p3_ts
                            ns1["t0Candidates"] = p3_cand
                            ns1["assignment_info"] = p3_ainfo
                            ns1["unassigned_by_tpc"] = p3_unass
                            _ts_r2 = np.asarray(ns1["hit_timestamps"], dtype=np.float64)
                            _new_r = (np.isfinite(_ts_r2) & (_ts_r2 >= 0) & ~_asg_r)
                            _acc_r2n, _ = accuracy(_ts_r2, truth_t0, e_pos, mask=_new_r)
                            p3_row["matrix_relax"] = {
                                "n_work": int(sum(len(v) for v in _work2.values())),
                                "n_proposed": int(len(p3_log2)),
                                "n_assigned": int(sum(1 for r in p3_log2 if r.get("assigned"))),
                                "new_efrac": float(e_pos[_new_r].sum() / max(e_pos.sum(), 1e-9)),
                                "acc_new": _acc_r2n}
                        else:
                            p3_row["matrix_relax"] = {"n_work": 0}
                        tim["after_matrix_relax"] = time.time() - t_ev
                        _evo("4.2_matrix_relax", ns1["hit_timestamps"])
                # ---------- STAGE 4.5: per-TPC swap fix ----------
                if args.loop_p23 > 0 and args.pass2_trade:
                    # ---- v0.8.2: the full Q1<->Q2 alternation ----
                    if _HERE not in sys.path:
                        sys.path.insert(0, _HERE)
                    from v080_pass2 import run_pass2_trading
                    from v080_pass3 import run_pass3_family_light
                    audit_p2, audit_p3f, loop_log = [], [], []
                    st_p2, st_p3f = {}, {}
                    for _rd in range(int(args.loop_p23)):
                        st_p2 = run_pass2_trading(
                            ns1, model_path=args.pass2_model,
                            margin=float(args.pass2_margin),
                            p_min=float(args.pass2_pmin),
                            veto_frac=float(args.pass2_veto_frac),
                            light_nominate=bool(args.pass2_light_nom),
                            light_propose=bool(args.pass2_light_prop),
                            truth_fn=_label_truth, verbose=False)
                        audit_p2 += st_p2.pop("audit", [])
                        st_p3f = run_pass3_family_light(
                            ns1, truth_fn=_label_truth, verbose=False)
                        audit_p3f += st_p3f.pop("audit", [])
                        loop_log.append({
                            "round": _rd + 1,
                            "p2_traded": int(st_p2.get("n_traded", 0)),
                            "p2_via_light": int(st_p2.get("n_traded_via_light", 0)),
                            "p3_moved": int(st_p3f.get("n_moved", 0)),
                            "p3_swapped": int(st_p3f.get("n_swapped", 0)),
                            "p3_closed": int(st_p3f.get("n_closed", 0))})
                        if (st_p2.get("n_traded", 0) == 0
                                and st_p3f.get("n_moved", 0) == 0
                                and st_p3f.get("n_swapped", 0) == 0
                                and st_p3f.get("n_closed", 0) == 0):
                            break
                    st_p2["iterations"] = len(loop_log)
                    st_p2["n_fixed"] = sum(1 for a in audit_p2
                                           if not a.get("was_ok") and a.get("now_ok"))
                    st_p2["n_broke"] = sum(1 for a in audit_p2
                                           if a.get("was_ok") and not a.get("now_ok"))
                    for _op in ("move", "swap", "close"):
                        st_p3f[f"{_op}_fixed"] = sum(
                            1 for a in audit_p3f if a["op"] == _op
                            and not a.get("was_ok") and a.get("now_ok"))
                        st_p3f[f"{_op}_broke"] = sum(
                            1 for a in audit_p3f if a["op"] == _op
                            and a.get("was_ok") and not a.get("now_ok"))
                    p3_row["loop"] = loop_log
                    p3_row["pass2"] = st_p2
                    p3_row["pass2_audit"] = audit_p2
                    p3_row["pass3_family"] = st_p3f
                    p3_row["pass3_audit"] = audit_p3f
                    tim["after_pass2"] = time.time() - t_ev
                    _evo("3.95_pass2_trade", ns1["hit_timestamps"])
                    tim["after_pass3f"] = time.time() - t_ev
                    _evo("3.97_pass3_family", ns1["hit_timestamps"])
                elif args.pass2_trade:
                    if _HERE not in sys.path:
                        sys.path.insert(0, _HERE)
                    from v080_pass2 import run_pass2_trading
                    audit_p2 = []
                    for _it in range(max(1, int(args.pass2_iterations))):
                        st_p2 = run_pass2_trading(
                            ns1, model_path=args.pass2_model,
                            margin=float(args.pass2_margin),
                            p_min=float(args.pass2_pmin),
                            veto_frac=float(args.pass2_veto_frac),
                            light_nominate=bool(args.pass2_light_nom),
                            light_propose=bool(args.pass2_light_prop),
                            truth_fn=_label_truth, verbose=False)
                        audit_p2 += st_p2.pop("audit", [])
                        if st_p2.get("n_traded", 0) == 0:
                            break
                    st_p2["iterations"] = _it + 1
                    st_p2["n_fixed"] = sum(1 for a in audit_p2
                                           if not a.get("was_ok") and a.get("now_ok"))
                    st_p2["n_broke"] = sum(1 for a in audit_p2
                                           if a.get("was_ok") and not a.get("now_ok"))
                    p3_row["pass2"] = st_p2
                    p3_row["pass2_audit"] = audit_p2
                    tim["after_pass2"] = time.time() - t_ev
                    _evo("3.95_pass2_trade", ns1["hit_timestamps"])
                if args.redecide:
                    # ---------- STAGE 3.98 (v1.1): component re-decision ----------
                    if _HERE not in sys.path:
                        sys.path.insert(0, _HERE)
                    from v110_redecide import run_redecide
                    from v02_refine_pipeline import make_cos_pick as _mcp_rd

                    def _ti_rd(arr, t):
                        return timeinterpolation(arr, shift=float(t), baseline=0).astype(np.float64)
                    st_rd = run_redecide(
                        ns1, veto_mask=veto_mask, support_cache=support_cache,
                        label_info=label_info, ti=_ti_rd, pick=_mcp_rd("soft"),
                        e_min=float(args.redecide_emin), cos_min=float(args.pool_cos),
                        k=int(args.pool_k), n_max=int(args.pool_nmax),
                        truth_fn=_label_truth, spatial_frac=float(args.pool_spatial))
                    p3_row["redecide_audit"] = st_rd.pop("audit", [])
                    p3_row["redecide"] = st_rd
                    tim["after_redecide"] = time.time() - t_ev
                    _evo("3.98_redecide", ns1["hit_timestamps"])
                if args.v200_min_e is not None:
                    # ---------- STAGE 3.99 (v2.0 test): exact-block LNS re-solve of clusters >= E ----------
                    # (V200_DESIGN.md) warm start from the current assignment; clusters below the floor
                    # stay frozen and only shape the residual; light energy on the TPC channel unions.
                    _v200_dir = os.path.join(_HERE, "v200")
                    if _v200_dir not in sys.path:
                        sys.path.insert(0, _v200_dir)
                    os.environ.setdefault("V200_DOM", "3")
                    from v200_stage import run_v200_stage

                    def _ti_v2(arr, t):
                        return timeinterpolation(arr, shift=float(t), baseline=0).astype(np.float64)
                    st_v2 = run_v200_stage(ns1, veto_mask=veto_mask, sat_ceil=sat_ceil, ti=_ti_v2,
                                           min_e=float(args.v200_min_e), sweeps=int(args.v200_sweeps),
                                           truth_t0=truth_t0)
                    p3_row["v200"] = st_v2
                    print(f"  [3.99 v200] re-solved {st_v2['n_resolved']} clusters >= {args.v200_min_e} MeV: "
                          f"{st_v2['n_changed']} moved ({st_v2['e_changed']:.0f} MeV), light E {st_v2['E_hist'][0]:.0f} -> "
                          f"{st_v2['E_light_after']:.0f}, audit fixed/broke {st_v2.get('audit_fixed')}/{st_v2.get('audit_broke')}, "
                          f"{st_v2['time_s']:.1f}s", flush=True)
                    tim["after_v200"] = time.time() - t_ev
                    _evo("3.99_v200_blocks", ns1["hit_timestamps"])
                if args.pass3_family and args.loop_p23 == 0:
                    if _HERE not in sys.path:
                        sys.path.insert(0, _HERE)
                    from v080_pass3 import run_pass3_family_light
                    st_p3f = run_pass3_family_light(ns1, truth_fn=_label_truth,
                                                    verbose=False)
                    audit_p3f = st_p3f.pop("audit", [])
                    for _op in ("move", "swap", "close"):
                        st_p3f[f"{_op}_fixed"] = sum(
                            1 for a in audit_p3f if a["op"] == _op
                            and not a.get("was_ok") and a.get("now_ok"))
                        st_p3f[f"{_op}_broke"] = sum(
                            1 for a in audit_p3f if a["op"] == _op
                            and a.get("was_ok") and not a.get("now_ok"))
                    p3_row["pass3_family"] = st_p3f
                    p3_row["pass3_audit"] = audit_p3f
                    tim["after_pass3f"] = time.time() - t_ev
                    _evo("3.97_pass3_family", ns1["hit_timestamps"])
                if args.swap_fix:
                    if _HERE not in sys.path:
                        sys.path.insert(0, _HERE)
                    from swap_fix import run_swap_correction
                    ts_bs = np.asarray(ns1["hit_timestamps"], dtype=np.float64)
                    acc_bs, _ = accuracy(ts_bs, truth_t0, e_pos)
                    st_sw = run_swap_correction(
                        ns1, hit_tpc=hit_tpc, cluster_to_tpcs=cluster_to_tpcs,
                        min_energy=args.swap_min_energy, min_cos_gain=args.swap_min_gain,
                        chi2_ratio_max=float(args.swap_ratio_max),
                        truth_fn=_label_truth, verbose=False)
                    ts_as = np.asarray(ns1["hit_timestamps"], dtype=np.float64)
                    st_sw["acc_before"], _ = accuracy(ts_bs, truth_t0, e_pos)
                    st_sw["acc_after"], _ = accuracy(ts_as, truth_t0, e_pos)
                    p3_row["swap"] = {k: v for k, v in st_sw.items() if k != "audit"}
                    p3_row["swap_audit"] = st_sw["audit"]
                    if args.cheat != "none":
                        _repin_ns(ns1)
                    tim["after_swap_fix"] = time.time() - t_ev
                    _evo("4.5_swap_fix", ns1["hit_timestamps"])
                if args.reassign:
                    if _HERE not in sys.path:
                        sys.path.insert(0, _HERE)
                    from residual_reassign import run_residual_reassign
                    ts_br = np.asarray(ns1["hit_timestamps"], dtype=np.float64)
                    st_ra = run_residual_reassign(
                        ns1, hit_tpc=hit_tpc, cluster_to_tpcs=cluster_to_tpcs,
                        min_energy=args.swap_min_energy, truth_fn=_label_truth, verbose=False)
                    ts_ar = np.asarray(ns1["hit_timestamps"], dtype=np.float64)
                    st_ra["acc_before"], _ = accuracy(ts_br, truth_t0, e_pos)
                    st_ra["acc_after"], _ = accuracy(ts_ar, truth_t0, e_pos)
                    # store under 'swap' key so score_swap.py works unchanged
                    p3_row["swap"] = {k: v for k, v in st_ra.items() if k != "audit"}
                    p3_row["swap_audit"] = st_ra["audit"]
                if args.blip_rehome:
                    # ---------- STAGE 4.7: blip re-homing (contact certificate + isolated re-homing) ----------
                    if _HERE not in sys.path:
                        sys.path.insert(0, _HERE)
                    from blip_rehome import run_blip_rehome
                    _bimg = ns1["imageMaps"]
                    _bbase = ns1["baseImage"]
                    _bamp = {(int(L), int(c)): np.asarray(im, np.float64).max(axis=1)
                             for (L, c), im in _bimg.items() if 0 <= int(c) < n_light}

                    def _brender(L, c, t_old, t_new):
                        im = np.asarray(_bimg[(L, c)], np.float64)
                        if t_old is not None:
                            _bbase[c] -= timeinterpolation(im, shift=float(t_old), baseline=0).astype(_bbase.dtype)
                        if t_new is not None:
                            _bbase[c] += timeinterpolation(im, shift=float(t_new), baseline=0).astype(_bbase.dtype)
                    _bts = np.asarray(ns1["hit_timestamps"], dtype=np.float64).copy()
                    st_bl = run_blip_rehome(
                        ts=_bts, labels=np.asarray(ns1["labels_global"]), energy=np.asarray(ns1["Eset"], np.float64),
                        xyz=np.stack([np.asarray(ns1["xset"], np.float64), np.asarray(ns1["yset"], np.float64),
                                      np.asarray(ns1["zset"], np.float64)], axis=1),
                        tpc=np.asarray(ns1["hitTPCid"]).astype(np.int64), amp=_bamp,
                        actual=np.asarray(ns1["fullLightWaveform"], np.float64), base=_bbase, render=_brender,
                        contact=not args.blip_no_contact, isolated=not args.blip_no_iso,
                        contact_cm=float(args.blip_contact_cm), truth_t0=truth_t0,
                        near=bool(args.blip_near), reattach=bool(args.blip_reattach))
                    ns1["hit_timestamps"] = _bts.astype(np.asarray(ns1["hit_timestamps"]).dtype)
                    p3_row["blip_audit"] = st_bl.pop("audit", [])
                    p3_row["blip"] = st_bl
                    print(f"  [4.7 blip] contact moved {st_bl['contact_moved']} ({st_bl['e_contact']:.0f} MeV, "
                          f"fixed/broke {st_bl.get('contact_fixed_e', 0):.0f}/{st_bl.get('contact_broke_e', 0):.0f} MeV), "
                          f"iso moved {st_bl['iso_moved']} ({st_bl['e_iso']:.0f} MeV, fixed/broke "
                          f"{st_bl.get('iso_fixed_e', 0):.0f}/{st_bl.get('iso_broke_e', 0):.0f} MeV), "
                          f"near {st_bl['near_moved']} ({st_bl.get('near_fixed_e', 0):.0f}/{st_bl.get('near_broke_e', 0):.0f}), "
                          f"reattach {st_bl['reattach_moved']} ({st_bl.get('reattach_fixed_e', 0):.0f}/"
                          f"{st_bl.get('reattach_broke_e', 0):.0f}), "
                          f"{st_bl['time_s']:.1f}s", flush=True)
                    tim["after_blip"] = time.time() - t_ev
                    _evo("4.7_blip_rehome", ns1["hit_timestamps"])
                # ---------- STAGE 4.8: residual-flash filling ----------
                if args.resid_fill:
                    if _HERE not in sys.path:
                        sys.path.insert(0, _HERE)
                    from resid_fill import run_resid_fill
                    from subtick_refine import build_sipm_positions
                    _rimg = ns1["imageMaps"]; _rbase = ns1["baseImage"]
                    _ramp = {(int(L), int(c)): np.asarray(im, np.float64).max(axis=1)
                             for (L, c), im in _rimg.items() if 0 <= int(c) < n_light}

                    def _rrender(L, c, t_old, t_new):
                        im = np.asarray(_rimg[(L, c)], np.float64)
                        if t_old is not None:
                            _rbase[c] -= timeinterpolation(im, shift=float(t_old), baseline=0).astype(_rbase.dtype)
                        if t_new is not None:
                            _rbase[c] += timeinterpolation(im, shift=float(t_new), baseline=0).astype(_rbase.dtype)
                    _rts = np.asarray(ns1["hit_timestamps"], dtype=np.float64).copy()
                    st_rf = run_resid_fill(
                        ts=_rts, labels=np.asarray(ns1["labels_global"]), energy=np.asarray(ns1["Eset"], np.float64),
                        xyz=np.stack([np.asarray(ns1["xset"], np.float64), np.asarray(ns1["yset"], np.float64),
                                      np.asarray(ns1["zset"], np.float64)], axis=1),
                        tpc=np.asarray(ns1["hitTPCid"]).astype(np.int64), amp=_ramp,
                        actual=np.asarray(ns1["fullLightWaveform"]), base=_rbase, veto=veto_mask,
                        chan_pos=build_sipm_positions(file_path), render_cb=_rrender,
                        pulse_path=os.path.join(BASE_DIR, "assets", "avg_pulse.npy"),
                        small_min=0.5, small_max=float(args.rf_small_max), mode=str(args.rf_mode), K=8,
                        cos_min=float(args.rf_cos_min), dcos_min=float(args.rf_dcos), ratio_lo=0.5,
                        ratio_hi=float(args.rf_ratio_hi), k_prom=float(args.rf_kprom), fam_tol=5.0,
                        tail_ratio=float(args.rf_tail), min_adc=600.0, max_per_flash=4, glare_min=None,
                        host_min_cm=float(args.rf_host_min), truth_t0=truth_t0)
                    ns1["hit_timestamps"] = _rts.astype(np.asarray(ns1["hit_timestamps"]).dtype)
                    st_rf.pop("audit", None); p3_row["resid_fill"] = st_rf
                    print(f"  [4.8 resid-fill] residual flashes {st_rf['n_flash']}, moved {st_rf['n_moved']} clusters "
                          f"({st_rf['e_moved']:.0f} MeV), fixed/broke {st_rf['fixed_e']:.0f}/{st_rf['broke_e']:.0f} MeV, "
                          f"{st_rf['time_s']:.2f}s", flush=True)
                    tim["after_resid_fill"] = time.time() - t_ev
                    _evo("4.8_resid_fill", ns1["hit_timestamps"])
                # ---------- STAGE 4.85: Michel / delayed-daughter re-timing (piece-first) ----------
                if args.michel_nd:
                    if _HERE not in sys.path:
                        sys.path.insert(0, _HERE)
                    from michel_nd import run_michel_nd, MichelNDConfig
                    from M5p1.first_stage_matching.streaming_prediction import process_clusters_to_imageMaps_streaming as _mnd_pred
                    _mimg = ns1["imageMaps"]
                    _mamp = {(int(L), int(c)): np.asarray(im, np.float64).max(axis=1) for (L, c), im in _mimg.items() if 0 <= int(c) < n_light}
                    _mx = np.asarray(ns1["xset"], np.float32); _my = np.asarray(ns1["yset"], np.float32); _mz = np.asarray(ns1["zset"], np.float32)
                    _me = np.asarray(ns1["Eset"], np.float32); _mtpc = np.asarray(ns1["hitTPCid"]).astype(np.int64); _mlab = np.asarray(ns1["labels_global"]).astype(np.int64)
                    _mtmpl = np.asarray(models.waveform_template, np.float64) if hasattr(models, "waveform_template") else None

                    def _mnd_predict(items):
                        """items: [(key, hit indices)] -> {key: amp pattern}; hits of one spill may overlap between items -> passes"""
                        out = {}; passes = []
                        for j, (k, h) in enumerate(items):
                            for pm in passes:
                                if not pm["used"][h].any():
                                    pm["lab"][h] = 1_000_000 + j; pm["used"][h] = True; pm["want"].add(1_000_000 + j); pm["keys"].append((j, k)); break
                            else:
                                pm = {"lab": np.full(_mlab.size, -1, np.int64), "used": np.zeros(_mlab.size, bool), "want": set(), "keys": []}
                                pm["lab"][h] = 1_000_000 + j; pm["used"][h] = True; pm["want"].add(1_000_000 + j); pm["keys"].append((j, k)); passes.append(pm)
                        for pm in passes:
                            _, meta = _mnd_pred(_mx, _my, _mz, _me, _mtpc, pm["lab"], model=models.light_model, template=_mtmpl, batch_size=int(args.mnd_batch),
                                                restrict_clusters=pm["want"], device_policy="auto", amplitudes_only=True, min_prediction_threshold=0.0)
                            amps = {(int(c_), int(t_)): meta["amplitudes"][i] for i, (c_, t_) in enumerate(zip(meta["group_cls"], meta["group_tpcs"]))}
                            for j, k in pm["keys"]:
                                v = amps.get((1_000_000 + j, k[1])); out[k] = np.asarray(v, np.float64) if v is not None else np.zeros(n_light_ch)
                        return out
                    _mnd_T = np.load(os.path.join(BASE_DIR, "assets", "avg_pulse.npy")).astype(np.float64).ravel(); _mnd_T = _mnd_T / _mnd_T.max()
                    _mcfg = MichelNDConfig(); _mcfg.host_extent_min_cm = float(args.mnd_host_extent_cm); _mcfg.bragg_end_cm = float(args.mnd_bragg_end_cm); _mcfg.host_dedx_max = float(args.mnd_host_dedx_max); _mcfg.tip_face_min_cm = float(args.mnd_tip_face_cm)
                    for _kv in args.mnd_set:
                        _k, _v = _kv.split("=", 1); _cur = getattr(_mcfg, _k)
                        setattr(_mcfg, _k, float(_v) if _cur is None else (type(_cur)(float(_v)) if isinstance(_cur, (int, float)) and not isinstance(_cur, bool) else (_v.lower() in ("1", "true") if isinstance(_cur, bool) else _v)))
                    _mcfg.piece_max_mev = float(args.mnd_piece_max_mev); _mcfg.delay_max = float(args.mnd_delay_max)
                    _mts = np.asarray(ns1["hit_timestamps"], dtype=np.float64).copy(); _mbase = ns1["baseImage"]
                    n_light_ch = int(np.asarray(_mbase).shape[1])
                    st_m = run_michel_nd(xyz=np.stack([_mx, _my, _mz], 1).astype(np.float64), e=_me, tpc=_mtpc, labels=_mlab, ts=_mts, amp=_mamp,
                                         actual=np.asarray(ns1["fullLightWaveform"], np.float32), base=_mbase, veto=veto_mask, ceil=sat_ceil,
                                         template=_mnd_T, predict_fn=_mnd_predict, cfg=_mcfg, truth_t0=truth_t0, apply=True)
                    ns1["hit_timestamps"] = _mts.astype(np.asarray(ns1["hit_timestamps"]).dtype)
                    mv = st_m.pop("moves"); p3_row["michel_nd"] = st_m
                    p3_row["michel_nd_moves"] = [{k: v for k, v in m.items() if k != "hits"} for m in mv]
                    print(f"  [4.85 michel] pieces {st_m['n_cand']} wholes {st_m['n_whole']} predictions {st_m['n_pred']} moves {st_m['n_moves']} "
                          f"({st_m['e_moved']:.0f} MeV) fixed/broke {st_m.get('fixed_e', 0):.0f}/{st_m.get('broke_e', 0):.0f} MeV, {st_m['time_s']:.1f}s {st_m['tim']}", flush=True)
                    tim["after_michel_nd"] = time.time() - t_ev
                    _evo("4.85_michel_nd", ns1["hit_timestamps"])
                # ---------- STAGE 4.9: force coverage ----------
                if args.force_coverage:
                    if _HERE not in sys.path:
                        sys.path.insert(0, _HERE)
                    from v030_expansion import force_full_coverage
                    from force_simple import force_simple_coverage
                    ts_pf = np.asarray(ns1["hit_timestamps"], dtype=np.float64)
                    A_pref = np.isfinite(ts_pf) & (ts_pf >= 0)
                    if args.force_simple and args.force_light_cm is not None:
                        from force_light_vote import force_light_vote_coverage
                        from subtick_refine import build_sipm_positions
                        st_f = force_light_vote_coverage(
                            ns1, sipm=build_sipm_positions(file_path), isolate_cm=float(args.isolate_cm),
                            energy=np.asarray(ns1["Eset"], np.float64), light_cm=float(args.force_light_cm),
                            actual=np.asarray(ns1["fullLightWaveform"]), base=np.asarray(ns1["baseImage"]),
                            veto=veto_mask)
                    elif args.force_simple and args.noise_vote:
                        from force_noise_vote import force_noise_vote_coverage
                        st_f = force_noise_vote_coverage(
                            ns1, isolate_cm=float(args.isolate_cm),
                            energy=np.asarray(ns1["Eset"], np.float64))
                    elif args.force_simple:
                        st_f = force_simple_coverage(
                            ns1, isolate_cm=float(args.isolate_cm))
                    else:
                        st_f = force_full_coverage(ns1, verbose=False)
                    ts_af = np.asarray(ns1["hit_timestamps"], dtype=np.float64)
                    newf = np.isfinite(ts_af) & (ts_af >= 0) & ~A_pref
                    st_f["force_acc_new"], _ = accuracy(ts_af, truth_t0, e_pos, mask=newf)
                    st_f["force_new_efrac"] = float(e_pos[newf].sum() / max(e_pos.sum(), 1e-9))
                    # composition of what the force stage had to absorb
                    st_f["resid_noise_efrac"] = float(
                        e_pos[newf & (labels_global < 0)].sum() / max(e_pos[newf].sum(), 1e-9))
                    p3_row["force"] = st_f
                    if args.cheat != "none":
                        _repin_ns(ns1)
                    tim["after_force"] = time.time() - t_ev
                    _evo("4.9_force", ns1["hit_timestamps"])
                if args.decline_r > 0:
                    # ---- v0.8.4: decline-or-repair for the small tail ----
                    if _HERE not in sys.path:
                        sys.path.insert(0, _HERE)
                    from v084_decline import run_decline
                    t_dc = time.time()
                    st_dc = run_decline(ns1, r_max=float(args.decline_r),
                                        truth_fn=_label_truth, verbose=False)
                    aud_dc = st_dc.pop("audit", [])
                    st_dc["declined_wrong"] = sum(
                        1 for a in aud_dc if a.get("declined") and not a["was_ok"])
                    st_dc["declined_right"] = sum(
                        1 for a in aud_dc if a.get("declined") and a["was_ok"])
                    st_dc["mit_fixed"] = sum(
                        1 for a in aud_dc if not a.get("declined")
                        and not a["was_ok"] and a.get("now_ok"))
                    st_dc["mit_broke"] = sum(
                        1 for a in aud_dc if not a.get("declined")
                        and a["was_ok"] and not a.get("now_ok"))
                    st_dc["time_s"] = round(time.time() - t_dc, 2)
                    p3_row["decline"] = st_dc
                    p3_row["decline_audit"] = aud_dc
                if args.le_retime:
                    if _HERE not in sys.path:
                        sys.path.insert(0, _HERE)
                    from le_retime import run_le_retime
                    t_le = time.time()
                    st_le = run_le_retime(
                        ns1, truth_fn_hit=lambda blob: truth_t0[blob])
                    st_le["time_s"] = round(time.time() - t_le, 2)
                    p3_row["le_retime"] = st_le
                if args.new_flash_rescue:
                    if _HERE not in sys.path:
                        sys.path.insert(0, _HERE)
                    from new_flash_rescue import run_new_flash_rescue
                    ts_bnf = np.asarray(ns1["hit_timestamps"], dtype=np.float64).copy()
                    st_nf = run_new_flash_rescue(
                        ns1, hit_tpc=hit_tpc, x=fs.event.x, y=fs.event.y, z=fs.event.z,
                        truth_fn=_label_truth, verbose=False)
                    ts_anf = np.asarray(ns1["hit_timestamps"], dtype=np.float64)
                    chg = np.isfinite(ts_bnf) & np.isfinite(ts_anf) & (ts_bnf != ts_anf)
                    st_nf["moved_e_frac"] = float(e_pos[chg].sum() / max(e_pos.sum(), 1e-9))
                    p3_row["new_flash"] = st_nf
                if args.subtick or args.subtick_hit:
                    if _HERE not in sys.path:
                        sys.path.insert(0, _HERE)
                    from subtick_refine import run_subtick_refine
                    t_st6 = time.time()
                    _dp6 = None
                    if args.subtick_dump:
                        os.makedirs(args.subtick_dump, exist_ok=True)
                        _ftag6 = os.path.basename(file_path).split(".")[-3]
                        _dp6 = os.path.join(args.subtick_dump,
                                            f"{_ftag6}_ev{ev_id}.npz")
                    st6 = run_subtick_refine(
                        ns1, ckpt=args.light_ckpt, truth_t0=truth_t0,
                        group_merge=float(args.subtick_merge),
                        study=bool(args.subtick_study),
                        min_group_mev=5.0,   # sub-5 MeV = light-blind (E_tiny): no refinable information
                        hit_refine=("channel6" if args.subtick_hit else None),
                        flow_path=file_path, dump_path=_dp6)
                    st6["time_s"] = round(time.time() - t_st6, 2)
                    p3_row["subtick"] = st6
                    if args.conf_cluster:
                        # v0.9.3 cluster-level calibrated confidence
                        t_c93 = time.time()
                        try:
                            from conf_v093 import score_event as _c93
                            _lab_c, _hit_c93, st93 = _c93(
                                x=fs.event.x, y=fs.event.y, z=fs.event.z,
                                e=fs.event.energy, tpc=hit_tpc,
                                labels=labels_global,
                                ts_final=np.asarray(ns1["hit_timestamps"],
                                                    np.float64),
                                actual=actual, base_image=ns1["baseImage"],
                                veto_mask=veto_mask, image_maps=image_maps,
                                ns=ns1, n_light=n_light,
                                model_path=(args.conf_model or None))
                            ns1["hit_t0_conf_group"] = ns1.get("hit_t0_conf")
                            ns1["hit_t0_conf"] = _hit_c93
                            st93["time_s"] = round(time.time() - t_c93, 2)
                            p3_row["conf93"] = st93
                        except Exception as _c93e:
                            p3_row["conf93"] = {"error": repr(_c93e)}
                if args.precision:
                    # ---------- STAGE P (v1.2): per-hit precision timing ----------
                    if _HERE not in sys.path:
                        sys.path.insert(0, _HERE)
                    from v120_precision import run_precision
                    t_sp = time.time()
                    try:
                        stp = run_precision(ns1, flow_path=file_path,
                                            calib_path=(args.precision_calib or None),
                                            veto_mask=veto_mask,
                                            rounds=int(args.precision_rounds),
                                            pieces=bool(args.precision_pieces),
                                            collect=bool(args.precision_pieces))
                        stp["time_s"] = round(time.time() - t_sp, 2)
                        _frows = stp.pop("rows", [])
                        _pcs = stp.pop("pieces", [])
                        if args.precision_pieces:
                            p3_row["pieces"] = _pcs[:400]        # per-piece delay / distance / beta / sigma
                            # compact per-family table for time-of-flight pairing: apex time, sigma, apex xyz, E
                            p3_row["families"] = [dict(tpc=int(fr["tpc"]), t_chain=round(float(fr["t_chain"]), 3),
                                                       t0=round(float(fr["t0"]), 4), sig_ns=round(float(fr["sig_ns"]), 3),
                                                       E=round(float(fr["E"]), 2), n_ch=int(fr["n_ch"]),
                                                       apex=[round(float(v), 1) for v in fr["apex"]])
                                                  for fr in _frows][:1500]
                        p3_row["precision"] = stp
                    except Exception as _spe:
                        p3_row["precision"] = {"error": repr(_spe)}
                    tim["after_precision"] = time.time() - t_ev
                if args.shower_tpc:
                    if _HERE not in sys.path:
                        sys.path.insert(0, _HERE)
                    from shower_tpc import run_shower_tpc_pass

                    def _part_truth(lab, tpc_q):
                        m = (labels_global == lab) & (hit_tpc == tpc_q) & np.isfinite(truth_t0)
                        if not m.any():
                            return None
                        tv, wv = truth_t0[m], e_pos[m]
                        o = np.argsort(tv)
                        cw = np.cumsum(wv[o])
                        return float(tv[o][np.searchsorted(cw, 0.5 * cw[-1])])

                    ts_bsh = np.asarray(ns1["hit_timestamps"], dtype=np.float64).copy()
                    _ts_pre_shower = ts_bsh.copy()
                    st_sh = run_shower_tpc_pass(
                        ns1, hit_tpc=hit_tpc, e_thresh_mev=float(args.shower_thresh),
                        fill_frac=float(args.shower_fill), truth_fn=_part_truth,
                        verbose=False)
                    ts_ash = np.asarray(ns1["hit_timestamps"], dtype=np.float64)
                    tpc_e_sh = np.zeros(int(hit_tpc.max()) + 2)
                    np.add.at(tpc_e_sh, hit_tpc, e_pos)
                    mask_sh = tpc_e_sh[hit_tpc] > float(args.shower_thresh)
                    for nm, arr in (("sh_eff_before", ts_bsh), ("sh_eff_after", ts_ash)):
                        okh = mask_sh & np.isfinite(truth_t0)
                        gd = okh & np.isfinite(arr) & (arr >= 0) & (np.abs(arr - truth_t0) <= CUT)
                        st_sh[nm] = float(e_pos[gd].sum() / max(e_pos[okh].sum(), 1e-9))
                    p3_row["shower_tpc"] = st_sh
                ts_p3 = np.asarray(ns1["hit_timestamps"], dtype=np.float64)
                m1["p3_acc"], m1["p3_afrac"] = accuracy(ts_p3, truth_t0, e_pos)
                new3 = np.isfinite(ts_p3) & (ts_p3 >= 0) & ~A_pre3
                m1["p3_acc_new"], _ = accuracy(ts_p3, truth_t0, e_pos, mask=new3)
                m1["p3_new_efrac"] = float(e_pos[new3].sum() / max(e_pos.sum(), 1e-9))
                p3_row["time_s"] = time.time() - t_p3
                acc1, af1 = m1["p3_acc"], m1["p3_afrac"]

            # ---- TARGET METRIC: non-shower cohort = hits in TPCs <= 1.4 GeV ----
            tpc_e = np.zeros(int(hit_tpc.max()) + 2, dtype=np.float64)
            np.add.at(tpc_e, hit_tpc, e_pos)
            mask_ns = tpc_e[hit_tpc] <= 1400.0
            ts_now = np.asarray(ns1["hit_timestamps"], dtype=np.float64)
            m1["ns_acc"], _ = accuracy(ts_now, truth_t0, e_pos, mask=mask_ns)
            ok_ns = mask_ns & np.isfinite(truth_t0)
            asg_ns = np.isfinite(ts_now) & (ts_now >= 0) & mask_ns
            good_ns = ok_ns & asg_ns & (np.abs(ts_now - truth_t0) <= CUT)
            m1["ns_total_eff"] = float(e_pos[good_ns].sum() / max(e_pos[ok_ns].sum(), 1e-9))
            m1["ns_coverage"] = float(e_pos[asg_ns].sum() / max(e_pos[mask_ns].sum(), 1e-9))
            m1["ns_assigned_acc"] = float(
                e_pos[good_ns].sum() / max(e_pos[ok_ns & asg_ns].sum(), 1e-9))
            m1["ns_assigned_efrac"] = float(
                e_pos[ok_ns & asg_ns].sum() / max(e_pos[ok_ns].sum(), 1e-9))
            _ce_le = {}
            for _l in np.unique(labels_global[labels_global >= 0]):
                _ce_le[_l] = e_pos[labels_global == _l].sum()
            _small_le = np.isin(labels_global,
                                [l for l, en in _ce_le.items() if en < 1.5])
            _le = ((labels_global < 0) | _small_le) & np.isfinite(truth_t0) \
                & np.isfinite(ts_now) & (ts_now >= 0)
            _le_good = _le & (np.abs(ts_now - truth_t0) <= CUT)
            m1["le_acc"] = float(e_pos[_le_good].sum() / max(e_pos[_le].sum(), 1e-9))
            m1["le_efrac"] = float(e_pos[_le].sum() / max(e_pos[np.isfinite(truth_t0)].sum(), 1e-9))
            m1["ns_efrac"] = float(e_pos[mask_ns].sum() / max(e_pos.sum(), 1e-9))

            ts_fin = np.asarray(ns1["hit_timestamps"], dtype=np.float64)

            # ---- v0.8.3 (optional): confidence pruning of assigned hits ----
            row_prune = None
            if args.prune_frac > 0:
                try:
                    if _HERE not in sys.path:
                        sys.path.insert(0, _HERE)
                    from v083_prune import prune_event
                    from confidence_cluster import cluster_scores as _pr_cs
                    _conf_pr = _pr_cs(labels_global, hit_tpc,
                                      np.asarray(fs.event.y, np.float64),
                                      e_pos, ts_fin, actual,
                                      np.asarray(ns1["baseImage"], np.float64))
                    row_prune = prune_event(
                        dict(x=np.asarray(fs.event.x), y=np.asarray(fs.event.y),
                             z=np.asarray(fs.event.z), e=e_pos, tpc=hit_tpc,
                             labels=labels_global, ts_final=ts_fin,
                             truth_t0=truth_t0),
                        _conf_pr, args.prune_model, float(args.prune_frac))
                except Exception as _pe:
                    row_prune = {"error": repr(_pe)}

            if args.dump_dir:
                lab_rows = []
                for lab in np.unique(labels_global[labels_global >= 0]):
                    mm = labels_global == lab
                    ok = mm & np.isfinite(truth_t0) & np.isfinite(ts_fin) & (ts_fin >= 0)
                    if not ok.any():
                        continue
                    e_ok = float(e_pos[ok].sum())
                    e_wrong = float(e_pos[ok & (np.abs(ts_fin - truth_t0) > CUT)].sum())
                    tv = ts_fin[ok]
                    lab_rows.append({"label": int(lab), "e_ok": e_ok, "e_wrong": e_wrong,
                                     "reco_t0": float(np.median(tv)),
                                     "truth": _label_truth(int(lab)),
                                     "type": str(label_info.get(int(lab), {}).get("type", "?")),
                                     "p1_t0": (float(info[int(lab)]["t0"])
                                               if int(lab) in info else None),
                                     "tpcs": sorted(int(t) for t in
                                                    cluster_to_tpcs.get(int(lab), []))})
                extra = {}
                if args.evo:
                    extra["t_evo"] = json.dumps(t_evo)
                    try:
                        extra["assign_stage"] = json.dumps(
                            {f"{int(k[0])}_{int(k[1])}":
                                {"t0": float(v.get("t0", -1.0)),
                                 "stage": str(v.get("stage", v.get("mode", "?")))}
                             for k, v in ns1["assignment_info"].items()
                             if isinstance(k, tuple) and v.get("assigned")})
                    except Exception as _ase:
                        extra["assign_stage"] = json.dumps({"error": repr(_ase)})
                worst = sorted((r for r in lab_rows if r["e_wrong"] > 100.0
                                and r["e_wrong"] > 0.5 * r["e_ok"]),
                               key=lambda r: -r["e_wrong"])[:8]
                for r in worst:
                    for c in r["tpcs"]:
                        if (r["label"], int(c)) in image_maps and int(c) < n_light:
                            extra[f"img_{r['label']}_{c}"] = np.asarray(
                                image_maps[(r["label"], int(c))], np.float32)
                # shower-TPC offline-iteration extras: ALL part templates + flash
                # table for TPCs above the shower threshold (tune offline, not in-chain)
                if args.shower_tpc:
                    try:
                        extra["ts_pre_shower"] = _ts_pre_shower
                    except NameError:
                        pass
                    tpc_e_d = np.zeros(int(hit_tpc.max()) + 2)
                    np.add.at(tpc_e_d, hit_tpc, e_pos)
                    for t_sh in np.flatnonzero(tpc_e_d[:n_light] > float(args.shower_thresh)):
                        t_sh = int(t_sh)
                        extra[f"sh_t0cand_{t_sh}"] = np.asarray(
                            [float(v) for v in ns1["t0Candidates"][t_sh]], dtype=np.float64)
                        for lab in np.unique(labels_global[(hit_tpc == t_sh)
                                                           & (labels_global >= 0)]):
                            if (int(lab), t_sh) in image_maps:
                                extra[f"shimg_{int(lab)}_{t_sh}"] = np.asarray(
                                    image_maps[(int(lab), t_sh)], np.float32)
                # v0.5.6 release extras: per-cluster confidence components +
                # everything needed to RE-RUN the front-end clustering offline
                try:
                    from confidence_cluster import cluster_scores as _v056_cs
                    _cconf = _v056_cs(labels_global, hit_tpc,
                                      np.asarray(fs.event.y, np.float64), e_pos,
                                      ts_fin, actual,
                                      np.asarray(ns1["baseImage"], np.float64))
                    extra["cluster_conf"] = json.dumps(
                        {str(int(k)): {kk: float(vv) for kk, vv in v.items()}
                         for k, v in _cconf.items()})
                except Exception as _cce:
                    extra["cluster_conf"] = json.dumps({"error": repr(_cce)})
                if args.conf_cluster and ns1.get("hit_t0_conf") is not None:
                    extra["hit_conf93"] = np.asarray(ns1["hit_t0_conf"],
                                                     np.float32)
                if ns1.get("hit_t0_sigma_p") is not None:
                    extra["hit_sigma_p"] = np.asarray(ns1["hit_t0_sigma_p"], np.float32)
                # v0.9.3 confidence dev: per-label per-channel PEAK vectors
                # (rank-1 identity: predicted image = amp x shared template, so
                # peak-per-channel fully describes each label's light) — lets
                # scan-curve / margin features be developed offline.
                try:
                    _al, _at, _ar = [], [], []
                    for (_lab, _c), _im in image_maps.items():
                        if not (0 <= int(_c) < n_light):
                            continue
                        _al.append(int(_lab))
                        _at.append(int(_c))
                        _ar.append(np.asarray(_im, np.float32).max(axis=1))
                    extra["amp_lab"] = np.asarray(_al, np.int32)
                    extra["amp_tpc"] = np.asarray(_at, np.int32)
                    extra["amp_row"] = (np.asarray(_ar, np.float32)
                                        if _ar else np.zeros((0, 120), np.float32))
                except Exception as _ame:
                    extra["amp_err"] = repr(_ame)
                ftag = os.path.basename(file_path).split(".")[-3]
                np.savez_compressed(
                    os.path.join(args.dump_dir, f"{ftag}_ev{ev_id}.npz"),
                    x=fs.event.x, y=fs.event.y, z=fs.event.z, e=fs.event.energy,
                    io_group=np.asarray(fs.event.io_group),
                    split_index=int(split_index),
                    tpc=hit_tpc, labels=labels_global, hit_refs=fs.event.hit_refs,
                    ts_final=ts_fin.astype(np.float64), ts_p1=hit_t0.astype(np.float64),
                    truth_t0=truth_t0, veto_mask=veto_mask,
                    sat_ceiling=sat_ceil.astype(np.float32),
                    # FINAL predicted light (ns1['baseImage']): the matrix stage reassigns
                    # ns1['baseImage']=p3_base, so fs.track_stage.base_image is the STALE
                    # pre-phase3 snapshot (missing every expansion/matrix/force cluster) — do
                    # NOT dump that; ns1['baseImage'] (always built at line ~1108, updated by
                    # every later stage) is what the matcher's decisions actually use.
                    base_image=np.asarray(ns1["baseImage"], dtype=np.float32),
                    full_light=actual.astype(np.float32),
                    lab_rows=json.dumps(lab_rows), src_file=file_path, event=int(ev_id),
                    **extra)

            # ---- v0.4: per-TPC charge-light matching CONFIDENCE (~0.006% overhead) ----
            conf_components = {}
            try:
                from confidence_v04 import components as _cl_components, t0_list as _cl_t0
                _bimg = np.asarray(ns1["baseImage"]); _nl = _bimg.shape[0]
                _e = np.asarray(fs.event.energy, float)
                for _c in np.unique(hit_tpc[hit_tpc < _nl]):
                    _c = int(_c); _m = hit_tpc == _c
                    if not _m.any():
                        continue
                    _placed = []
                    for _L in np.unique(labels_global[_m & (labels_global >= 0)]):
                        _mm = _m & (labels_global == _L) & np.isfinite(ts_fin)
                        if _mm.any() and float(_e[_mm].sum()) >= 10:
                            _placed.append(int(round(np.median(ts_fin[_mm]))) + 105)
                    conf_components[str(_c)] = _cl_components(
                        actual[_c], _bimg[_c], _cl_t0(actual[_c], _placed), tpc=int(_c))
            except Exception as _ce:
                conf_components = {"error": repr(_ce)}

            try:
                import v02_ordering_pipeline as _vfs1
                import v030_expansion as _vfs2
                _fs = (FS_STAT + _vfs1.FS_STAT + _vfs2.FS_STAT).tolist()
                FS_STAT[:] = 0
                _vfs1.FS_STAT[:] = 0
                _vfs2.FS_STAT[:] = 0
            except Exception:
                _fs = None
            fout.write(json.dumps({
                "file": os.path.basename(file_path), "event": int(ev_id),
                "fs_stat": _fs,
                "confidence_components": conf_components,
                "m_baseline": m0, "m_rescued": m1,
                "acc_baseline": acc0, "assigned_frac_baseline": af0,
                "acc_rescued": acc1, "assigned_frac_rescued": af1,
                "n_groups": len(groups), "n_moves": len(moves), "n_detaches": n_new,
                "n_lone_flagged": len(lone), "n1_actions": n1_actions,
                "swap_actions": swap_actions,
                "backbone": args.backbone, "n_v02_moved": len(v02_moved),
                "vertex_merge": vm_row, "phase3": p3_row, "merge_p2": merge_stats,
                "pool_stats": pool_stats, "rescue_audit": rescue_audit,
                "loss_mode": args.loss,
                "frontend": (args.frontend_repo or "builtin"),
                "early_expand": ee_row,
                "cheat": (None if args.cheat == "none" else
                          {"mode": args.cheat,
                           "n_locked": len(cheat_t0),
                           **cheat_metrics(Scheat, ht_map,
                                           np.asarray(ns1["hit_timestamps"], np.float64))}),
                "timing": tim, "stage_t": _stage_durations(tim),
                "t_evo": (t_evo if args.evo else None),
                "prune": row_prune,
                "shower_eat": se_row if args.shower_eat else None,
                "stage1_t": stage1_t,
                "t_event_s": time.time() - t_ev,
                "sigma_mode": args.sigma_mode, "z_cut": Z, "trig_rows": trig_rows,
                "stay_margin": float(args.stay_margin), "cos_sel": float(args.cos_sel), "cos_window": float(args.cos_window), "group_diags": group_diags,
                "total_energy": float(e_pos.sum()),
                "truth_energy": float(e_pos[np.isfinite(truth_t0)].sum()),
            }) + "\n")
            fout.flush()
            a0s = f"{acc0:.4f}" if acc0 is not None else "skip"
            a1s = f"{acc1:.4f}" if acc1 is not None else "none"
            print(f"done {os.path.basename(file_path)} ev{ev_id} in {time.time()-t_ev:.0f}s | "
                  f"acc {a0s} -> {a1s} | moves {len(moves)} detaches {n_new}", flush=True)
        except Exception as exc:
            import traceback
            print(f"ERROR {file_path} ev{ev_id}: {exc}", flush=True)
            traceback.print_exc()
    fout.close()
    print("worker complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
