"""v0.4.5 pipeline = v0.4.0 + two early-absorption upgrades (COPY of
overflow_rescue_pipeline.py; the v0.4 script is untouched so old records stand).

  1. --track-absorb-scale default 1.5 -> 2.0: during the CLUSTER phase each
     track absorbs left-over (no-cluster) hits within 2x its own transverse
     radius (toolbox _fast_absorb_noise_into_tracks; already vectorized).
  2. --stage2-expand: joint geodesic-Voronoi family expansion DURING the stage-2
     permutation placement (v045_stage2_expansion), so a decided cluster's small
     neighbours fill the predicted waveform BEFORE the remaining backbone
     clusters are scanned onto that light.  Never absorbs >30 MeV (Phase 2 owns
     those); fronts of different families meet and stop (no crossing).

Original header follows.
---
Integrated backbone re-run with the saving algorithm applied.

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

_V045 = os.path.abspath(os.path.dirname(__file__))            # .../testing/v045
_HERE = os.path.abspath(os.path.join(_V045, ".."))            # .../testing (v030_expansion, v02_*)
BASE_DIR = os.path.abspath(os.path.join(_HERE, "..", "CLMatching_AlphaRelease"))
for _p in (os.path.join(BASE_DIR, "M5p1"), os.path.join(BASE_DIR, "NewMLSection"), BASE_DIR):
    while _p in sys.path:
        sys.path.remove(_p)
sys.path.insert(0, BASE_DIR)
sys.path.insert(1, os.path.join(BASE_DIR, "NewMLSection"))
sys.path.insert(2, os.path.join(BASE_DIR, "M5p1"))
sys.path.insert(3, _HERE)
sys.path.insert(4, _V045)

import numpy as np
from scipy.signal import fftconvolve

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


def weighted_scan(pred, resid, w, max_shift=SEARCH):
    n = pred.shape[1]
    t1 = fftconvolve(w, (pred * pred)[:, ::-1], mode="full", axes=1)[:, n - 1: n - 1 + max_shift + 1].sum(axis=0)
    t2 = fftconvolve(w * resid, pred[:, ::-1], mode="full", axes=1)[:, n - 1: n - 1 + max_shift + 1].sum(axis=0)
    return t1 + 2.0 * t2 + float(np.sum(w * resid * resid))


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
            p = np.minimum(np.asarray(bi[c], np.float64)[:, tick], CLIP)
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
                                        * (np.minimum(mdl, CLIP) - actual[c][ch]) ** 2))
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
               police=None):
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
    ap.add_argument("--stage2-expand", action="store_true",
                    help="v0.4.5: joint family expansion DURING the stage-2 permutation "
                         "placement (fills the waveform before later clusters are scanned)")
    ap.add_argument("--stage2-every", type=int, default=4,
                    help="run a joint expansion pass every N accepted stage-2 commits")
    ap.add_argument("--stage2-large-mev", type=float, default=30.0,
                    help="never absorb a cluster this big during stage 2 (Phase 2 owns it)")
    ap.add_argument("--stage2-d-geo", type=float, default=20.0,
                    help="stage-2 expansion geodesic reach [cm]")
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
    ap.add_argument("--phase3", choices=["none", "matrix", "expand", "both"], default="none",
                    help="small-cluster stage after V2: matrix = v11 error-matrix "
                         "(old way), expand = v0.3.0 family expansion, both = "
                         "expansion then matrix tail")
    ap.add_argument("--predict-min-energy", type=float, default=3.0,
                    help="prediction threshold [MeV] when --phase3 != none "
                         "(<=0 -> production None = predict everything)")
    ap.add_argument("--expand-score", choices=["special", "plain"], default="special",
                    help="family-expansion score: special = per-MeV delta-chi2 + "
                         "geodesic term (v0.3.0), plain = raw clipped-Poisson "
                         "delta-chi2 only (regular pipeline loss)")
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
    ap.add_argument("--track-absorb-scale", type=float, default=2.0,   # v0.4.5: was 1.5
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

    cfg = build_default_first_stage_config()
    if os.environ.get("CLM_IMAGE_BATCH"):        # shrink render footprint for a memory-constrained GPU
        cfg.prediction.image_batch_size = int(os.environ["CLM_IMAGE_BATCH"])
        print(f"[mem] image_batch_size={cfg.prediction.image_batch_size}", flush=True)
    cfg.prediction.min_cluster_energy_mev = 50.0
    if args.phase3 != "none":
        cfg.prediction.min_cluster_energy_mev = (
            None if args.predict_min_energy <= 0 else float(args.predict_min_energy))
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
            fs = run_first_stage_charge_light_matching(h5=h5, event_id=int(ev_id), config=cfg,
                                                       models=models, verbose=False)
            tim = {"first_stage": time.time() - t_ev}
            labels_global = np.asarray(fs.clustering.labels_global)
            split_index = int(fs.clustering.split_index)
            label_info = fs.clustering.label_info
            image_maps = fs.prediction.image_maps
            cluster_to_tpcs = fs.prediction.cluster_to_tpcs
            tpc_to_clusters = fs.prediction.tpc_to_clusters
            support_cache = fs.prediction.cluster_channel_support_cache
            veto_mask = np.asarray(fs.prediction.saturated_channel_cache["veto_mask"], dtype=bool)
            actual = np.asarray(fs.event.full_light_waveform, dtype=np.float64)
            n_light, n_ch, n_ticks = actual.shape
            hit_t0 = np.asarray(fs.track_stage.hit_t0)
            hit_tpc = np.asarray(fs.event.hit_tpc_id)
            e_pos = np.clip(np.asarray(fs.event.energy, dtype=np.float64), 0.0, None)

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

            # ---------- optional v0.2 backbone swap-in ----------
            v02_moved = {}
            stage2_absorbed: dict[int, float] = {}
            s2_stats = {}
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
                showers_b = [l for l in sorted(info_b) if info_b[l]["type"] == "shower"]
                multi_b = [l for l in sorted(info_b) if info_b[l]["type"] != "shower"
                           and len(info_b[l]["tpcs"]) >= 3]
                pool_b = [l for l in sorted(info_b) if info_b[l]["type"] != "shower"
                          and len(info_b[l]["tpcs"]) <= 2 and l not in cheat_t0]
                fixed_extra = [l for l in sorted(info_b) if l in cheat_t0
                               and cheat_is_backbone.get(int(l), False)
                               and l not in showers_b and l not in multi_b]
                b_full = np.zeros_like(actual)
                for lab in showers_b + multi_b + fixed_extra:
                    v02_add_label(b_full, lab, info_b[lab]["t0"], info_b, image_maps, ti_f)
                    hit_t0[labels_global == lab] = np.float32(info_b[lab]["t0"])
                if args.backbone == "stack2":
                    stx = {}
                    hook = make_defer_hook(info_b, image_maps, actual, veto_mask, ti_f, stx)
                    s2exp = None
                    if args.stage2_expand:
                        # v0.4.5: absorb small neighbours as soon as t0s are decided,
                        # so their light fills the waveform for the NEXT scans.
                        if _V045 not in sys.path:
                            sys.path.insert(0, _V045)
                        from v045_stage2_expansion import (Stage2Expander,
                                                           make_expansion_hook)
                        s2exp = Stage2Expander(
                            xyz=np.stack([np.asarray(fs.event.x), np.asarray(fs.event.y),
                                          np.asarray(fs.event.z)], axis=1),
                            e_pos=e_pos, labels=labels_global, image_maps=image_maps,
                            cluster_to_tpcs=cluster_to_tpcs, actual=actual, base=b_full,
                            ti_fn=ti_f, backbone_labels=list(info_b), n_light=n_light,
                            n_ch=n_ch, n_ticks=n_ticks,
                            params={"LARGE_MEV": float(args.stage2_large_mev),
                                    "D_GEO_MAX": float(args.stage2_d_geo)})
                        decided_s2 = {int(l): float(info_b[l]["t0"])
                                      for l in (showers_b + multi_b + fixed_extra)}

                        def _on_absorb(got):
                            for _l, _t in got.items():
                                hit_t0[labels_global == int(_l)] = np.float32(_t)
                                stage2_absorbed[int(_l)] = float(_t)

                        hook = make_expansion_hook(hook, s2exp, decided_s2, hit_tpc,
                                                   on_absorb=_on_absorb,
                                                   every=int(args.stage2_every))
                    tmap, _ = place_pool_v02(pool_b, 0.60, b_full, info_b, image_maps,
                                             actual, w_full, ti_f,
                                             pick=make_cos_pick("soft"), commit_hook=hook)
                    swap_pass(tmap, b_full, info_b, image_maps, actual, w_full, ti_f)
                    rescue_pass(tmap, b_full, info_b, image_maps, actual, w_full,
                                veto_mask, ti_f)
                    if s2exp is not None:
                        # final catch-up pass on the SETTLED backbone t0s
                        decided_s2.update({int(l): float(t) for l, t in tmap.items()})
                        _on_absorb(s2exp.expand(decided_s2, hit_tpc))
                        s2_stats = dict(s2exp.stats)
                else:
                    tmap, _ = place_pool_v02(pool_b, 0.60, b_full, info_b, image_maps,
                                             actual, w_full, ti_f)
                for lab, t in tmap.items():
                    hit_t0[labels_global == lab] = np.float32(t)
                    if abs(t - info_b[lab]["t0"]) > CUT:
                        v02_moved[lab] = (float(t), info_b[lab]["tpcs"])
                np.asarray(fs.track_stage.base_image)[:] = b_full.astype(np.float32)

            police_fn = None
            if args.police:
                def _ti_f(arr, t):
                    return timeinterpolation(arr, shift=float(t), baseline=0).astype(np.float64)
                police_fn = make_police(actual, veto_mask, e_pos, support_cache, label_info,
                                        _ti_f, mode=args.police, topk=int(args.police_topk))

            tim["after_backbone"] = time.time() - t_ev

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
                m0 = staged_run(ns0, v2mod, v2cfg, _run_phase2, snapshot_backbone_hits_v11,
                                verify_backbone_hits_unchanged_v11, truth_t0, e_pos, "base",
                                police=police_fn)
                acc0, af0 = m0["v2_acc"], m0["v2_afrac"]

            # ---------- rescue decisions (identical logic to rescue A+B) ----------
            info = {}
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
                    entry = support_cache.get((int(lab), int(c))) if lab_type == "track" else None
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
            for lab in sorted(info):
                L = info[lab]
                tpcs_arr = np.asarray(L["tpcs"], dtype=np.int32)
                sh = timeinterpolation(fast_stack_images(image_maps, int(lab), tpcs_arr),
                                       shift=L["t0"], baseline=0).astype(np.float64)
                own_full = sh.reshape(tpcs_arr.size, n_ch, -1)
                base[tpcs_arr] += own_full
                tick = int(round(L["t0"])) + PEAK
                if not (0 <= tick < n_ticks) or L["type"] != "track" or int(lab) in cheat_t0:
                    continue
                fire = False
                cmin = 1.0
                z3m = -100.0
                for ti, c in enumerate(L["tpcs"]):
                    fin = ~veto_mask[c]
                    a = actual[c, :, tick]
                    p = np.minimum(base[c, :, tick], CLIP)
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
                    return float(sum(np.sum(w_union[c] * (np.minimum(model[c], CLIP) - actual[c][union[c]]) ** 2)
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
                    p = np.minimum(base[c, :, tick], CLIP)
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
                    lm = float(np.sum(w_f * (np.minimum((b_wo + sh_t1)[ch_u], CLIP)
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
                        tot += float(np.sum(w_u[c] * (np.minimum(m, CLIP) - actual[c][ch]) ** 2))
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
            done_detach = set()
            n_new = 0
            for j, tpc_f in detaches:
                if (j, tpc_f) in done_detach or (int(j), int(tpc_f)) not in image_maps or j in moves:
                    continue
                done_detach.add((j, tpc_f))
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
                        if L["type"] != "track":
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
                                                * (np.minimum(mdl, CLIP) - actual[c][ch]) ** 2))
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

            if not v2_variants:
                m1 = staged_run(ns1, v2mod, v2cfg, _run_phase2, snapshot_backbone_hits_v11,
                                verify_backbone_hits_unchanged_v11, truth_t0, e_pos, "rescued",
                                police=police_fn)
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

            # ---------- v0.3.0 small-cluster stage (where Phase 3 lives) ----------
            p3_row = None
            if args.phase3 != "none":
                t_p3 = time.time()
                # v0.4.5: clusters already absorbed in stage 2 must never be
                # re-placed downstream (double light + overwrite -- the v0.3 bug)
                if stage2_absorbed:
                    for _k in ("v11_iterative_single_tpc", "unassigned_by_tpc"):
                        for _t, _lst in list(ns1.get(_k, {}).items()):
                            ns1[_k][_t] = [cc for cc in _lst
                                           if int(cc) not in stage2_absorbed]
                ts_pre3 = np.asarray(ns1["hit_timestamps"], dtype=np.float64).copy()
                A_pre3 = np.isfinite(ts_pre3) & (ts_pre3 >= 0)
                p3_row = {"mode": args.phase3}
                if args.phase3 in ("expand", "both"):
                    if _HERE not in sys.path:
                        sys.path.insert(0, _HERE)
                    from v030_expansion import run_family_expansion
                    st_e = run_family_expansion(
                        ns1, verbose=False,
                        params={"PLAIN_SCORE": args.expand_score == "plain"})
                    st_e.pop("assigned", None)
                    # purge expansion-assigned clusters from the matrix worklists
                    # (else the matrix phase re-places them: double light + overwrites)
                    taken = set(st_e.pop("member_labels", []))
                    for _k in ("v11_iterative_single_tpc", "unassigned_by_tpc"):
                        for _t, _lst in list(ns1[_k].items()):
                            ns1[_k][_t] = [cc for cc in _lst if int(cc) not in taken]
                    ts_pe = np.asarray(ns1["hit_timestamps"], dtype=np.float64)
                    new_e = np.isfinite(ts_pe) & (ts_pe >= 0) & ~A_pre3
                    m1["p3e_acc"], m1["p3e_afrac"] = accuracy(ts_pe, truth_t0, e_pos)
                    m1["p3e_acc_new"], _ = accuracy(ts_pe, truth_t0, e_pos, mask=new_e)
                    m1["p3e_new_efrac"] = float(e_pos[new_e].sum() / max(e_pos.sum(), 1e-9))
                    p3_row["expand"] = st_e
                if args.phase3 in ("matrix", "both"):
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
                        adc_clip=CLIP,
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
                if args.force_coverage:
                    if _HERE not in sys.path:
                        sys.path.insert(0, _HERE)
                    from v030_expansion import force_full_coverage
                    ts_pf = np.asarray(ns1["hit_timestamps"], dtype=np.float64)
                    A_pref = np.isfinite(ts_pf) & (ts_pf >= 0)
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
            m1["ns_efrac"] = float(e_pos[mask_ns].sum() / max(e_pos.sum(), 1e-9))

            if args.dump_dir:
                ts_fin = np.asarray(ns1["hit_timestamps"], dtype=np.float64)
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
                ftag = os.path.basename(file_path).split(".")[-3]
                np.savez_compressed(
                    os.path.join(args.dump_dir, f"{ftag}_ev{ev_id}.npz"),
                    x=fs.event.x, y=fs.event.y, z=fs.event.z, e=fs.event.energy,
                    tpc=hit_tpc, labels=labels_global, hit_refs=fs.event.hit_refs,
                    ts_final=ts_fin.astype(np.float64), ts_p1=hit_t0.astype(np.float64),
                    truth_t0=truth_t0, veto_mask=veto_mask,
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
                        actual[_c], _bimg[_c], _cl_t0(actual[_c], _placed))
            except Exception as _ce:
                conf_components = {"error": repr(_ce)}

            fout.write(json.dumps({
                "file": os.path.basename(file_path), "event": int(ev_id),
                "confidence_components": conf_components,
                "m_baseline": m0, "m_rescued": m1,
                "acc_baseline": acc0, "assigned_frac_baseline": af0,
                "acc_rescued": acc1, "assigned_frac_rescued": af1,
                "n_groups": len(groups), "n_moves": len(moves), "n_detaches": n_new,
                "n_lone_flagged": len(lone), "n1_actions": n1_actions,
                "swap_actions": swap_actions,
                "backbone": args.backbone, "n_v02_moved": len(v02_moved),
                "stage2_expand": bool(args.stage2_expand),
                "stage2": {**s2_stats, "n_absorbed": len(stage2_absorbed),
                           "e_absorbed": float(sum(
                               e_pos[labels_global == l].sum() for l in stage2_absorbed))},
                "vertex_merge": vm_row, "phase3": p3_row,
                "cheat": (None if args.cheat == "none" else
                          {"mode": args.cheat,
                           "n_locked": len(cheat_t0),
                           **cheat_metrics(Scheat, ht_map,
                                           np.asarray(ns1["hit_timestamps"], np.float64))}),
                "timing": tim, "t_event_s": time.time() - t_ev,
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
