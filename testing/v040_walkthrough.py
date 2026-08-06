"""Step-by-step v0.4.0 walkthrough engine for the notebook
(v040_step_by_step.ipynb).

Runs the FULL v0.4.0 chain on one event (whole ND, algorithm unchanged) while
firing a display callback every time the watched TPC's predicted light (base
image) is updated.  The callback receives (stage, note, base_view) — the
notebook draws the usual 2D light+charge display and an inline 3D
reconstructed view at each checkpoint.

Stage map (canonical v0.4.1 flags: stack2 backbone + MERGED 3.1/3.3 pool
(--merge-p2), poisson trigger, veto-4, police locked, merge-vertex, phase3
both + force, plain expansion score, swap 0.85, boundary 30, predict >= 1.5):
  1        clustering & ML prediction (inside the first-stage call)
  2        Phase 1: showers + >=3-TPC tracks (per-placement checkpoints)
  3.1      MERGED permutation pool: 1-2-TPC backbone tracks + every P2-class
           cluster (multi-TPC or > 30 MeV), energy-banded all-orderings
           batches with the defer gate (per-commit checkpoints)
  3.2      trigger + saving on the full model (per-ADOPT checkpoints)
  3.3      Phase-2 scan for what 3.2 detached + police
  3.5      V2 spatial+light rescue + police
  3.9      family expansion, plain clipped-Poisson score (per-assignment)
  4.1      error matrix     (stage checkpoint)
  4.5      per-TPC swap fix (stage checkpoint)
  4.9      force coverage   (stage checkpoint)
"""
from __future__ import annotations

import itertools
import os
import sys
import time
from typing import Any, Callable

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


def _ensure_here():
    # first_stage_matching.paths.configure_paths() strips every sys.path entry
    # under the project root that is not inside the release -- including this
    # testing dir.  Re-arm before any testing-local deferred import.
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)


# v0.6: set to the clmatching-frontend repo path (and optionally a params json)
# BEFORE load_chain() to run the external front-end clustering; None = builtin.
FRONTEND_REPO = None
FRONTEND_PARAMS = None
FRONTEND_SHOWER_MIN_TRACKS = 0

PEAK = 105
GAIN = 200.0
CLIP = 60780.0
COS_CUT = 0.75
Z_CUT = 2.0
EXCESS_GATE = 500.0
GROUP_DT = 5.0
MAX_N = 3
SEARCH = 800
CUT = 10.0
P2_THRESHOLD = 30.0
PREDICT_MIN = 1.5
SWAP_RATIO = 0.85


def _tick(S, name, t0):
    S["timings"].append((name, time.perf_counter() - t0))


# ---------------------------------------------------------------------------
def load_chain():
    """Heavy imports + models, v0.4.0 configuration."""
    t0 = time.perf_counter()
    import h5py  # noqa
    from M5p1.first_stage_matching import (load_first_stage_models,
                                           run_first_stage_charge_light_matching)
    from M5p1.first_stage_matching.track_stage import fast_stack_images, append_candidate_t0
    import M5p1.phase25_trial2_valpha_batch as batchmod
    from M5p1.phase25_trial2_valpha_batch import (_build_namespace, _run_phase2,
                                                  build_default_first_stage_config,
                                                  build_default_v2_config)
    from M5p1 import phase25_trial2_v2_light_rescue as v2mod
    from v11_phased_matching import (snapshot_backbone_hits_v11,
                                     verify_backbone_hits_unchanged_v11,
                                     run_small_cluster_matrix_phase_v11)
    from M5p1 import truth_plotting as truthmod
    from pulse_shapes import timeinterpolation

    batchmod.V11_LARGE_CLUSTER_ENERGY_MEV = P2_THRESHOLD
    cfg = build_default_first_stage_config()
    cfg.prediction.min_cluster_energy_mev = PREDICT_MIN
    cfg.support.saturation_max_clip_ticks = 4
    cfg.clustering.vertex_merge_enable = True
    v2cfg = build_default_v2_config(n_outer_passes=1, light_max_total_moves=24,
                                    light_max_moves_per_tpc=3, device_policy="auto",
                                    verbose=False)
    if FRONTEND_REPO:
        _ensure_here()
        from frontend_adapter import make_external_clustering
        import M5p1.first_stage_matching.pipeline as _fsp_mod
        _fsp_mod.run_global_track_clustering = make_external_clustering(
            FRONTEND_REPO, FRONTEND_PARAMS,
            shower_min_tracks=int(FRONTEND_SHOWER_MIN_TRACKS))
        print(f"[load_chain] FRONT-END clustering -> "
              f"{_fsp_mod.run_global_track_clustering.frontend_tag}")
    models = load_first_stage_models(cfg)
    ctx = dict(h5py=h5py, run_fs=run_first_stage_charge_light_matching,
               fast_stack_images=fast_stack_images, append_candidate_t0=append_candidate_t0,
               build_ns=_build_namespace, run_phase2=_run_phase2, v2mod=v2mod,
               snapshot=snapshot_backbone_hits_v11, verify=verify_backbone_hits_unchanged_v11,
               run_matrix=run_small_cluster_matrix_phase_v11, truthmod=truthmod,
               ti=lambda a, t: timeinterpolation(a, shift=float(t), baseline=0).astype(np.float64),
               cfg=cfg, v2cfg=v2cfg, models=models)
    print(f"[load_chain] models + config ready in {time.perf_counter()-t0:.1f}s "
          f"(boundary {P2_THRESHOLD} MeV, predict >= {PREDICT_MIN} MeV, veto-4, vertex merge ON)")
    return ctx


def first_stage(ctx, file_path, ev_id, watch_tpc):
    t0 = time.perf_counter()
    h5 = ctx["h5py"].File(file_path, "r")
    fs = ctx["run_fs"](h5=h5, event_id=int(ev_id), config=ctx["cfg"],
                       models=ctx["models"], verbose=False)
    truth = ctx["truthmod"].extract_event_hit_energy_and_truth_t0(
        h5, h5["charge/events/ref/charge/calib_prompt_hits/ref"], eventid=int(ev_id),
        hits_full=h5["charge/calib_prompt_hits/data"], convert_to_matching_ticks=True)
    S = dict(ctx=ctx, h5=h5, fs=fs, ev=int(ev_id), file_path=file_path,
             watch=int(watch_tpc), timings=[], phase="backbone",
             labels_global=np.asarray(fs.clustering.labels_global),
             split_index=int(fs.clustering.split_index),
             label_info=fs.clustering.label_info,
             image_maps=fs.prediction.image_maps,
             cluster_to_tpcs=fs.prediction.cluster_to_tpcs,
             tpc_to_clusters=fs.prediction.tpc_to_clusters,
             support_cache=fs.prediction.cluster_channel_support_cache,
             veto_mask=np.asarray(fs.prediction.saturated_channel_cache["veto_mask"], bool),
             actual=np.asarray(fs.event.full_light_waveform, np.float64),
             hit_t0=np.asarray(fs.track_stage.hit_t0),
             hit_t0_prod=np.asarray(fs.track_stage.hit_t0).copy(),
             hit_tpc=np.asarray(fs.event.hit_tpc_id),
             e_pos=np.clip(np.asarray(fs.event.energy, np.float64), 0, None),
             truth_t0=np.asarray(truth["truth_t0"], np.float64),
             hits_full_ev=h5["charge/calib_prompt_hits/data"][...][fs.event.hit_refs])
    S["n_light"], S["n_ch"], S["n_ticks"] = S["actual"].shape
    S["w_full"] = 1.0 / (GAIN * np.maximum(S["actual"], 100.0))
    _tick(S, "1 clustering & ML prediction (+production placement, superseded)", t0)
    print(f"[first_stage] {time.perf_counter()-t0:.1f}s | clusters {S['split_index']} backbone "
          f"labels | watched TPC {watch_tpc}: "
          f"{int(np.sum((S['hit_tpc']==watch_tpc)))} hits, "
          f"{S['e_pos'][S['hit_tpc']==watch_tpc].sum():.0f} MeV")
    return S


def label_truth(S, lab):
    m = S["labels_global"] == lab
    tt, ww = S["truth_t0"][m], S["e_pos"][m]
    ok = np.isfinite(tt)
    if not ok.any():
        return None
    o = np.argsort(tt[ok])
    cw = np.cumsum(ww[ok][o])
    return float(tt[ok][o][np.searchsorted(cw, 0.5 * cw[-1])])


def cluster_table(S, tpc=None):
    """Cluster definition for the watched TPC."""
    tpc = S["watch"] if tpc is None else tpc
    rows = []
    for lab in np.unique(S["labels_global"][(S["hit_tpc"] == tpc) & (S["labels_global"] >= 0)]):
        m = (S["labels_global"] == lab)
        mt = m & (S["hit_tpc"] == tpc)
        li = S["label_info"].get(int(lab), {})
        rows.append(dict(label=int(lab),
                         type=str(li.get("type", "cluster" if lab >= S["split_index"] else "?")),
                         backbone=bool(lab < S["split_index"]),
                         n_hits_tpc=int(mt.sum()), e_tpc=round(float(S["e_pos"][mt].sum()), 1),
                         e_total=round(float(S["e_pos"][m].sum()), 1),
                         tpcs=sorted(int(t) for t in S["cluster_to_tpcs"].get(int(lab), [])),
                         has_template=(int(lab), int(tpc)) in S["image_maps"],
                         vertex_merged=bool(li.get("vertex_merged", False)),
                         truth_t0=label_truth(S, int(lab))))
    n_noise = int(np.sum((S["hit_tpc"] == tpc) & (S["labels_global"] < 0)))
    return rows, n_noise


def current_base(S):
    if S["phase"] == "backbone":
        return np.asarray(S["fs"].track_stage.base_image)
    return np.asarray(S["ns"]["baseImage"])


def current_ts(S):
    if S["phase"] == "backbone":
        return np.asarray(S["hit_t0"], np.float64)
    return np.asarray(S["ns"]["hit_timestamps"], np.float64)


# ---------------------------------------------------------------------------
# Stage 1: stack2 backbone (rebuilt with checkpoints; identical machinery)
def stack2_backbone(S, cb: Callable):
    _ensure_here()
    from v02_ordering_pipeline import place_pool_v02, add_label
    from v02_refine_pipeline import make_cos_pick, make_defer_hook, swap_pass, rescue_pass

    t0 = time.perf_counter()
    ti = S["ctx"]["ti"]
    n_ch = S["n_ch"]
    info = {}
    for lab in range(S["split_index"]):
        tpcs = sorted(int(t) for t in S["cluster_to_tpcs"].get(int(lab), [])
                      if int(t) < S["n_light"] and (int(lab), int(t)) in S["image_maps"])
        if not tpcs:
            continue
        members = np.flatnonzero(S["labels_global"] == lab)
        if members.size == 0:
            continue
        t0v = S["hit_t0_prod"][members]
        t0v = t0v[np.isfinite(t0v) & (t0v >= 0)]
        if t0v.size == 0:
            continue
        lab_type = str(S["label_info"].get(int(lab), {}).get("type", "track")).lower()
        sup = {}
        for c in tpcs:
            entry = S["support_cache"].get((int(lab), int(c))) if lab_type == "track" else None
            ch = (np.asarray(entry["channel_indices"], int) if entry is not None
                  else np.arange(n_ch))
            ch = ch[~S["veto_mask"][int(c), ch]]
            if ch.size:
                sup[int(c)] = np.sort(ch)
        if not sup:
            continue
        info[lab] = {"t0": float(np.median(t0v)), "tpcs": tpcs, "sup": sup,
                     "type": lab_type, "energy": float(S["e_pos"][members].sum())}
    # ---- 3.1+3.3 merge: every P2-class cluster joins the permutation pool
    merged = set()
    for lab in map(int, S["fs"].clustering.cluster_labels):
        if lab in info:
            continue
        tpcs = sorted(int(t) for t in S["cluster_to_tpcs"].get(lab, [])
                      if int(t) < S["n_light"] and (lab, int(t)) in S["image_maps"])
        if not tpcs:
            continue
        members = np.flatnonzero(S["labels_global"] == lab)
        if members.size == 0:
            continue
        e_c = float(S["e_pos"][members].sum())
        if len(tpcs) < 2 and e_c <= P2_THRESHOLD:
            continue
        sup = {}
        for c in tpcs:
            entry = S["support_cache"].get((lab, int(c)))
            ch = (np.asarray(entry["channel_indices"], int) if entry is not None
                  else np.arange(n_ch))
            ch = ch[~S["veto_mask"][int(c), ch]]
            if ch.size:
                sup[int(c)] = np.sort(ch)
        if not sup:
            continue
        info[lab] = {"t0": float("nan"), "tpcs": tpcs, "sup": sup,
                     "type": "cluster", "energy": e_c}
        merged.add(lab)
    S["info_b"] = info
    S["merged_set"] = merged
    showers = [l for l in sorted(info) if info[l]["type"] == "shower"]
    multi = [l for l in sorted(info) if info[l]["type"] != "shower"
             and len(info[l]["tpcs"]) >= 3 and l not in merged]
    pool = [l for l in sorted(info) if info[l]["type"] != "shower"
            and (len(info[l]["tpcs"]) <= 2 or l in merged)]

    b_full = np.zeros_like(S["actual"])
    for lab in showers + multi:
        add_label(b_full, lab, info[lab]["t0"], info, S["image_maps"], ti)
        S["hit_t0"][S["labels_global"] == lab] = np.float32(info[lab]["t0"])
        if S["watch"] in info[lab]["tpcs"]:
            np.asarray(S["fs"].track_stage.base_image)[:] = b_full.astype(np.float32)
            cb("2 phase1", f"{'SHOWER' if lab in showers else '>=3-TPC track'} label {lab} "
               f"placed at production t0={info[lab]['t0']:.1f} "
               f"(E={info[lab]['energy']:.0f} MeV, truth={label_truth(S, lab)})")

    stx = {}
    hook_defer = make_defer_hook(info, S["image_maps"], S["actual"], S["veto_mask"], ti, stx)

    def hook(lab, t_new, baseA):
        ok = hook_defer(lab, t_new, baseA)
        if ok:
            S["hit_t0"][S["labels_global"] == lab] = np.float32(t_new)
        if S["watch"] in info[lab]["tpcs"]:
            np.asarray(S["fs"].track_stage.base_image)[:] = baseA.astype(np.float32)
            kind = "P2-class cluster" if lab in merged else "backbone track"
            cb("3.1 merged pool", f"{kind} {lab} (E={info[lab]['energy']:.0f} MeV) "
               f"{'COMMITTED' if ok else 'DEFERRED by overflow gate'} at t0={t_new:.0f} "
               f"(truth={label_truth(S, lab)})")
        return ok

    tmap, _st = place_pool_v02(pool, 0.60, b_full, info, S["image_maps"], S["actual"],
                               S["w_full"], ti, pick=make_cos_pick("soft"), commit_hook=hook)
    n_sw = swap_pass(tmap, b_full, info, S["image_maps"], S["actual"], S["w_full"], ti)
    nf, nr = rescue_pass(tmap, b_full, info, S["image_maps"], S["actual"], S["w_full"],
                         S["veto_mask"], ti)
    for lab, tv in tmap.items():
        S["hit_t0"][S["labels_global"] == lab] = np.float32(tv)
    np.asarray(S["fs"].track_stage.base_image)[:] = b_full.astype(np.float32)
    S["merged_placed"] = {int(l): (float(tv), info[l]["tpcs"])
                          for l, tv in tmap.items() if l in merged}
    S["v02_moved"] = {lab: (float(tv), info[lab]["tpcs"]) for lab, tv in tmap.items()
                     if lab not in merged and abs(tv - info[lab]["t0"]) > CUT}
    cb("3.1 merged pool done", f"pool placed {len(tmap)} objects "
       f"({len(S['merged_placed'])} P2-class, {len(tmap)-len(S['merged_placed'])} backbone) "
       f"+ swap pass ({n_sw} swaps) + rescue pass ({nf} flagged, {nr} moved); "
       f"deferred once: {stx.get('n_deferred', 0)}", force=True)
    _tick(S, "2+3.1 phase1 + merged permutation pool", t0)


# ---------------------------------------------------------------------------
# Stage 2: overflow trigger + saving algorithm (faithful copy of the pipeline)
def rescue_stage(S, cb: Callable):
    _ensure_here()
    from v02_ordering_pipeline import add_label  # noqa
    fast_stack = S["ctx"]["fast_stack_images"]
    ti_fn = S["ctx"]["ti"]
    t_st = time.perf_counter()
    info = S["info_b"]
    actual, veto = S["actual"], S["veto_mask"]
    n_ch, n_ticks = S["n_ch"], S["n_ticks"]
    var_all = GAIN * np.maximum(actual, 100.0)
    from v030_expansion import weighted_scan

    mp = S.get("merged_placed", {})
    for lab in sorted(info):
        L = info[lab]
        # refresh t0 from current hit_t0 (post-pool)
        mv = S["hit_t0"][S["labels_global"] == lab]
        mv = mv[np.isfinite(mv) & (mv >= 0)]
        if mv.size:
            L["t0"] = float(np.median(mv))
    base = np.zeros_like(actual)
    flagged = []
    if mp:
        # merge mode: stack the FULL model first, then probe against it
        for lab in sorted(info):
            L = info[lab]
            if not np.isfinite(L["t0"]):
                continue
            arr = np.asarray(L["tpcs"], np.int32)
            base[arr] += ti_fn(fast_stack(S["image_maps"], int(lab), arr),
                               L["t0"]).reshape(arr.size, n_ch, -1)
    for lab in sorted(info):
        L = info[lab]
        if not np.isfinite(L["t0"]):
            continue
        arr = np.asarray(L["tpcs"], np.int32)
        sh = ti_fn(fast_stack(S["image_maps"], int(lab), arr), L["t0"]).reshape(arr.size, n_ch, -1)
        if not mp:
            base[arr] += sh
        tick = int(round(L["t0"])) + PEAK
        if not (0 <= tick < n_ticks) or (L["type"] != "track" and int(lab) not in mp):
            continue
        fire, cmin = False, 1.0
        for i_t, c in enumerate(L["tpcs"]):
            fin = ~veto[c]
            a = actual[c, :, tick]
            p = np.minimum(base[c, :, tick], CLIP)
            ex = p - a
            z = ex / np.sqrt(var_all[c, :, tick])
            zz = np.sort(z[fin & (ex >= EXCESS_GATE)])[::-1]
            if zz.size >= 3 and zz[2] > Z_CUT:
                fire = True
            o = sh[i_t, :, tick]
            res = np.clip(np.clip(a, 0, None) - np.clip(p - o, 0, None), 0, None)
            no, nr = np.linalg.norm(o[fin]), np.linalg.norm(res[fin])
            if no > 1e-9 and nr > 1e-9:
                cmin = min(cmin, float(np.dot(o[fin], res[fin]) / (no * nr)))
        if cmin < COS_CUT:
            fire = True
        if fire:
            flagged.append(lab)

    groups, lone, seen = [], [], set()
    for k in flagged:
        K = info[k]
        mem = [j for j, J in info.items()
               if abs(J["t0"] - K["t0"]) <= GROUP_DT and set(J["tpcs"]) & set(K["tpcs"])]
        if len(mem) > MAX_N:
            others = sorted((j for j in mem if j != k), key=lambda j: -info[j]["energy"])
            mem = [k] + others[:MAX_N - 1]
        key = frozenset(mem)
        if key in seen:
            continue
        seen.add(key)
        (lone if len(mem) < 2 else groups).append(k if len(mem) < 2 else (k, sorted(mem)))
    cb("3.2 trigger", f"trigger fired on {len(flagged)} placements -> {len(groups)} co-located "
       f"groups + {len(lone)} lone; running Stage A/B/N=1 decisions", force=True)

    moves, detaches, n1_rem = {}, [], {}
    for k, mem in groups:
        if any(j in moves for j in mem):
            continue
        slot = info[k]["t0"]
        base_wo = base.copy()
        for j in mem:
            arr = np.asarray(info[j]["tpcs"], np.int32)
            base_wo[arr] -= ti_fn(fast_stack(S["image_maps"], int(j), arr),
                                  info[j]["t0"]).reshape(arr.size, n_ch, -1)
        np.clip(base_wo, 0, None, out=base_wo)
        union = {}
        for j in mem:
            for c, ch in info[j]["sup"].items():
                union[c] = np.union1d(union.get(c, np.array([], int)), ch)
        w_u = {c: 1.0 / var_all[c][ch] for c, ch in union.items()}

        def closs(pl):
            model = {c: base_wo[c][ch].copy() for c, ch in union.items()}
            for j, tj in pl.items():
                arr = np.asarray(info[j]["tpcs"], np.int32)
                sh = ti_fn(fast_stack(S["image_maps"], int(j), arr), tj).reshape(arr.size, n_ch, -1)
                for i_t, c in enumerate(info[j]["tpcs"]):
                    if c in union:
                        model[c] += sh[i_t][union[c]]
            return float(sum(np.sum(w_u[c] * (np.minimum(model[c], CLIP) - actual[c][ch]) ** 2)
                             for c, ch in union.items()))

        best = None
        for perm in itertools.permutations(mem):
            pl = {perm[0]: slot}
            cur = {c: base_wo[c].copy() for c in set(c for j in mem for c in info[j]["tpcs"])}
            arr0 = np.asarray(info[perm[0]]["tpcs"], np.int32)
            sh0 = ti_fn(fast_stack(S["image_maps"], int(perm[0]), arr0), slot).reshape(arr0.size, n_ch, -1)
            for i_t, c in enumerate(info[perm[0]]["tpcs"]):
                cur[c] += sh0[i_t]
            for j in perm[1:]:
                pr, rr, wr = [], [], []
                for c, ch in info[j]["sup"].items():
                    pr.append(np.asarray(S["image_maps"][(int(j), int(c))], np.float64)[ch])
                    rr.append(cur[c][ch] - actual[c][ch])
                    wr.append(1.0 / var_all[c][ch])
                tj = float(np.argmin(weighted_scan(np.concatenate(pr), np.concatenate(rr),
                                                   np.concatenate(wr))))
                pl[j] = tj
                arrj = np.asarray(info[j]["tpcs"], np.int32)
                shj = ti_fn(fast_stack(S["image_maps"], int(j), arrj), tj).reshape(arrj.size, n_ch, -1)
                for i_t, c in enumerate(info[j]["tpcs"]):
                    cur[c] += shj[i_t]
            fl = closs(pl)
            if best is None or fl < best[0]:
                best = (fl, dict(pl))
        stay = closs({j: info[j]["t0"] for j in mem})
        pl = best[1] if best[0] < stay else {j: info[j]["t0"] for j in mem}
        if not all(abs(pl[j] - slot) <= CUT for j in mem):
            for j in mem:
                if abs(pl[j] - info[j]["t0"]) > CUT:
                    moves[j] = pl[j]
            continue
        # Stage B
        K = info[k]
        tick = int(round(K["t0"])) + PEAK
        best_sig = None
        for c in K["tpcs"]:
            fin = ~veto[c]
            a = actual[c, :, tick]
            p = np.minimum(base[c, :, tick], CLIP)
            ex = p - a
            z = ex / np.sqrt(var_all[c, :, tick])
            zz = np.sort(z[fin & (ex >= EXCESS_GATE)])[::-1]
            z3 = float(zz[2]) if zz.size >= 3 else -100.0
            if best_sig is None or z3 > best_sig[0]:
                best_sig = (z3, int(c))
        tpc_f = best_sig[1]
        ch_u = np.array([], int)
        for j in mem:
            if tpc_f in info[j]["sup"]:
                ch_u = np.union1d(ch_u, info[j]["sup"][tpc_f])
        if not ch_u.size:
            continue
        w_f = 1.0 / var_all[tpc_f][ch_u]
        movers = []
        for j in mem:
            if tpc_f not in info[j]["sup"] or (int(j), tpc_f) not in S["image_maps"]:
                continue
            if len(info[j]["tpcs"]) < 2:
                continue
            P = np.asarray(S["image_maps"][(int(j), tpc_f)], np.float64)
            b_wo = np.clip(base[tpc_f] - ti_fn(P, info[j]["t0"]), 0, None)
            chp = info[j]["sup"][tpc_f]
            t1 = float(np.argmin(weighted_scan(P[chp], (b_wo - actual[tpc_f])[chp],
                                               1.0 / var_all[tpc_f][chp])))
            if abs(t1 - info[j]["t0"]) <= CUT:
                continue
            lm = float(np.sum(w_f * (np.minimum((b_wo + ti_fn(P, t1))[ch_u], CLIP)
                                     - actual[tpc_f][ch_u]) ** 2))
            movers.append((lm, int(j)))
        if movers:
            movers.sort()
            detaches.append((movers[0][1], tpc_f))

    grouped = set(j for _, mem in groups for j in mem)
    for k in lone:
        if k in grouped or k in moves:
            continue
        K = info[k]
        arr = np.asarray(K["tpcs"], np.int32)
        stk = fast_stack(S["image_maps"], int(k), arr)
        base_wo = base.copy()
        base_wo[arr] -= ti_fn(stk, K["t0"]).reshape(arr.size, n_ch, -1)
        np.clip(base_wo, 0, None, out=base_wo)
        union = dict(K["sup"])
        w_u = {c: 1.0 / var_all[c][ch] for c, ch in union.items()}

        def uloss(parts):
            return float(sum(np.sum(w_u[c] * (np.minimum(base_wo[c][ch] + parts.get(c, 0.0),
                                                         CLIP) - actual[c][ch]) ** 2)
                             for c, ch in union.items()))

        def contrib(stk_, tl, tv, into=None):
            sh = ti_fn(stk_, tv).reshape(len(tl), n_ch, -1)
            parts = {} if into is None else into
            for i_t, c in enumerate(tl):
                if c in union:
                    parts[c] = parts.get(c, 0.0) + sh[i_t][union[c]]
            return parts

        stay = uloss(contrib(stk, K["tpcs"], K["t0"]))
        pr = np.concatenate([np.asarray(S["image_maps"][(int(k), int(c))], float)[ch]
                             for c, ch in union.items()])
        rr = np.concatenate([(base_wo[c] - actual[c])[ch] for c, ch in union.items()])
        wr = np.concatenate([w_u[c] for c in union])
        t_mv = float(np.argmin(weighted_scan(pr, rr, wr)))
        cands = ([("move", uloss(contrib(stk, K["tpcs"], t_mv)), t_mv, None)]
                 if abs(t_mv - K["t0"]) > CUT else [])
        if len(K["tpcs"]) >= 2:
            for tpf in K["tpcs"]:
                if tpf not in union or (int(k), int(tpf)) not in S["image_maps"]:
                    continue
                P = np.asarray(S["image_maps"][(int(k), int(tpf))], np.float64)
                chp = union[tpf]
                t1 = float(np.argmin(weighted_scan(P[chp], (base_wo[tpf] - actual[tpf])[chp],
                                                   w_u[tpf])))
                if abs(t1 - K["t0"]) <= CUT:
                    continue
                rem = [c for c in K["tpcs"] if c != tpf]
                rem_sup = [c for c in rem if c in union]
                if rem_sup:
                    prr = np.concatenate([np.asarray(S["image_maps"][(int(k), int(c))],
                                                     float)[union[c]] for c in rem_sup])
                    rrr = np.concatenate([(base_wo[c] - actual[c])[union[c]] for c in rem_sup])
                    wrr = np.concatenate([w_u[c] for c in rem_sup])
                    t_rem = float(np.argmin(weighted_scan(prr, rrr, wrr)))
                else:
                    t_rem = K["t0"]
                parts = contrib(P, [tpf], t1)
                parts = contrib(fast_stack(S["image_maps"], int(k),
                                           np.asarray(rem, np.int32)), rem, t_rem, into=parts)
                cands.append(("detach", uloss(parts), t_rem, tpf))
        if not cands:
            continue
        cands.sort(key=lambda c: c[1])
        kind, lb, tb, tpb = cands[0]
        if lb >= stay:
            continue
        if kind == "move":
            moves[k] = tb
        else:
            detaches.append((k, tpb))
            if abs(tb - K["t0"]) > CUT:
                n1_rem[k] = (tb, [c for c in K["tpcs"] if c != tpb])

    # ---------------- ADOPT with checkpoints ----------------
    next_label = int(S["labels_global"].max()) + 1
    base_img = np.asarray(S["fs"].track_stage.base_image)
    for j, t0n in moves.items():
        J = info[j]
        arr = np.asarray(J["tpcs"], np.int32)
        stk = fast_stack(S["image_maps"], int(j), arr)
        base_img[arr] += (ti_fn(stk, t0n) - ti_fn(stk, J["t0"])).reshape(arr.size, n_ch, -1).astype(np.float32)
        S["hit_t0"][S["labels_global"] == j] = np.float32(t0n)
        if S["watch"] in J["tpcs"]:
            cb("3.2 saving: MOVE", f"label {j} moved {J['t0']:.0f} -> {t0n:.0f} "
               f"(truth={label_truth(S, j)})")
    n_new = 0
    done = set()
    for j, tpc_f in detaches:
        if (j, tpc_f) in done or (int(j), int(tpc_f)) not in S["image_maps"] or j in moves:
            continue
        done.add((j, tpc_f))
        newlab = next_label
        next_label += 1
        n_new += 1
        mask = (S["labels_global"] == j) & (S["hit_tpc"] == tpc_f)
        S["labels_global"][mask] = newlab
        np.asarray(S["fs"].track_stage.labels_global_with_leftovers)[mask] = newlab
        S["hit_t0"][mask] = np.nan
        P = np.asarray(S["image_maps"].pop((int(j), int(tpc_f))))
        base_img[tpc_f] -= ti_fn(P.astype(np.float64), info[j]["t0"]).astype(np.float32)
        S["image_maps"][(int(newlab), int(tpc_f))] = P
        S["cluster_to_tpcs"][int(j)] = [t for t in S["cluster_to_tpcs"].get(int(j), [])
                                        if int(t) != tpc_f]
        S["cluster_to_tpcs"][int(newlab)] = [int(tpc_f)]
        S["tpc_to_clusters"].setdefault(int(tpc_f), []).append(int(newlab))
        entry = S["support_cache"].pop((int(j), int(tpc_f)), None)
        if entry is not None:
            entry = dict(entry)
            entry["clusterid"] = int(newlab)
            S["support_cache"][(int(newlab), int(tpc_f))] = entry
        S["label_info"][int(newlab)] = {"type": "cluster", "tpcs": [int(tpc_f)],
                                        "n_hits": int(mask.sum()), "n_tracks": 0,
                                        "track_ids": [], "cluster_id": None, "backbone": False,
                                        "segment_sources": [], "n_rescue_segments": 0}
        S["fs"].clustering.cluster_labels.append(int(newlab))
        if S["watch"] == tpc_f or S["watch"] in info[j]["tpcs"]:
            cb("3.2 saving: DETACH", f"label {j}'s part in TPC {tpc_f} cut off as NEW cluster "
               f"{newlab} (goes to Phase 2/3); parent truth={label_truth(S, j)}")
    for k, (t0r, rem) in n1_rem.items():
        if not rem or k in moves:
            continue
        arr = np.asarray(rem, np.int32)
        try:
            stk = fast_stack(S["image_maps"], int(k), arr)
        except KeyError:
            continue
        base_img[arr] += (ti_fn(stk, t0r) - ti_fn(stk, info[k]["t0"])).reshape(arr.size, n_ch, -1).astype(np.float32)
        S["hit_t0"][S["labels_global"] == k] = np.float32(t0r)
        if S["watch"] in rem:
            cb("3.2 saving: N=1 remainder", f"label {k} remainder moved to {t0r:.0f}")
    np.clip(base_img, 0, None, out=base_img)
    S["rescue_moves"] = moves
    cb("3.2 saving done", f"{len(moves)} moves, {n_new} detached parts", force=True)
    _tick(S, "3.2 trigger + saving algorithm", t_st)


# ---------------------------------------------------------------------------
def staged_stage(S, cb: Callable):
    """Phase 2 + police, then V2 + police."""
    _ensure_here()
    from v02_refine_pipeline import trigger_fired  # noqa
    ctx = S["ctx"]
    t0 = time.perf_counter()
    ns = ctx["build_ns"](S["fs"], S["h5"], ctx["cfg"])
    S["ns"] = ns
    S["phase"] = "ns"
    for j, (t0n, tpcs_j) in S.get("v02_moved", {}).items():
        for c in tpcs_j:
            ctx["append_candidate_t0"](ns["t0Candidates"][c], t0n, min_sep=2, max_t0=SEARCH)
            while len(ns["flash_cluster_received_by_tpc"][c]) < len(ns["t0Candidates"][c]):
                ns["flash_cluster_received_by_tpc"][c].append(False)
    for j, t0n in S.get("rescue_moves", {}).items():
        for c in S["info_b"][j]["tpcs"]:
            ctx["append_candidate_t0"](ns["t0Candidates"][c], t0n, min_sep=2, max_t0=SEARCH)
            while len(ns["flash_cluster_received_by_tpc"][c]) < len(ns["t0Candidates"][c]):
                ns["flash_cluster_received_by_tpc"][c].append(False)
    for j, (t0n, tpcs_j) in S.get("merged_placed", {}).items():
        tv = S["hit_t0"][S["labels_global"] == j]
        tv = tv[np.isfinite(tv) & (tv >= 0)]
        t0c = float(np.median(tv)) if tv.size else float(t0n)
        for c in tpcs_j:
            ctx["append_candidate_t0"](ns["t0Candidates"][c], t0c, min_sep=2, max_t0=SEARCH)
            while len(ns["flash_cluster_received_by_tpc"][c]) < len(ns["t0Candidates"][c]):
                ns["flash_cluster_received_by_tpc"][c].append(False)
            ns["assignment_info"][(int(j), int(c))] = {
                "clusterid": int(j), "tpcid": int(c), "t0": float(t0c),
                "mode": "merged_pool", "stage": "merged_pool", "assigned": True,
                "energy": float(S["info_b"][j]["energy"]) if j in S["info_b"] else 0.0}
    if S.get("merged_placed"):
        # merged labels are already placed: keep them out of the P2/matrix worklists
        ns["cluster_labels"] = [cc for cc in ns["cluster_labels"]
                                if int(cc) not in S["merged_placed"]]
    _tick(S, "3.3a namespace build", t0)

    ts1 = np.asarray(ns["hit_timestamps"], np.float64).copy()
    t0 = time.perf_counter()
    snap = ctx["snapshot"](hit_timestamps=ns["hit_timestamps"], labels_global=ns["labels_global"],
                           v8_absorbed_hit_parent=ns.get("v8_absorbed_hit_parent"),
                           track_shower_labels=ns["track_shower_labels"])
    ctx["run_phase2"](ns, verbose=False)
    ctx["verify"](snap, hit_timestamps=ns["hit_timestamps"], stage_name="P2(walkthrough)")
    _tick(S, "3.3 Phase 2 scan (3.2-detached parts + leftovers)", t0)
    ts2 = np.asarray(ns["hit_timestamps"], np.float64)
    new_here = _new_in_watch(S, ts1, ts2)
    cb("3.3 Phase 2", f"placed {len(new_here)} clusters in watched TPC: {new_here}", force=True)

    t0 = time.perf_counter()
    from overflow_rescue_pipeline import make_police
    police = make_police(S["actual"], S["veto_mask"], S["e_pos"], S["support_cache"],
                         S["label_info"], ctx["ti"], mode="locked")
    stp2 = police(ns, ts1, "p2")
    _tick(S, "3.3b police (post-P2)", t0)
    fixes = [f for f in stp2["fired_labels"]
             if S["watch"] in S["cluster_to_tpcs"].get(f["label"], [])]
    cb("3.3 police", f"police checked {stp2['n_checked']}, fired {stp2['n_fired']}; "
       f"fixes touching this TPC: {fixes}", force=bool(fixes))

    ts2b = np.asarray(ns["hit_timestamps"], np.float64).copy()
    t0 = time.perf_counter()
    ctx["v2mod"].run_trial2_v2_light_rescue_from_namespace(ns, config=ctx["v2cfg"], commit=True)
    _tick(S, "3.5 V2 spatial+light rescue", t0)
    ts3 = np.asarray(ns["hit_timestamps"], np.float64)
    moved_here = _moved_in_watch(S, ts2b, ts3)
    cb("3.5 V2", f"V2 moved/assigned in watched TPC: {moved_here}", force=True)

    t0 = time.perf_counter()
    stv2 = police(ns, ts2b, "v2")
    _tick(S, "3.5b police (post-V2)", t0)
    fixes = [f for f in stv2["fired_labels"]
             if S["watch"] in S["cluster_to_tpcs"].get(f["label"], [])]
    cb("3.5 police", f"police checked {stv2['n_checked']}, fired {stv2['n_fired']}; "
       f"fixes touching this TPC: {fixes}", force=bool(fixes))


def _new_in_watch(S, ts_a, ts_b):
    new = np.isfinite(ts_b) & (ts_b >= 0) & ~(np.isfinite(ts_a) & (ts_a >= 0))
    labs = np.unique(S["labels_global"][new & (S["hit_tpc"] == S["watch"])])
    return [(int(l), round(float(np.median(ts_b[(S["labels_global"] == l) & new])), 1))
            for l in labs if l >= 0]


def _moved_in_watch(S, ts_a, ts_b):
    ok = np.isfinite(ts_b) & (ts_b >= 0)
    chg = ok & (~(np.isfinite(ts_a) & (ts_a >= 0)) | (np.abs(ts_b - np.where(
        np.isfinite(ts_a), ts_a, -1e9)) > CUT))
    labs = np.unique(S["labels_global"][chg & (S["hit_tpc"] == S["watch"])])
    return [(int(l), round(float(np.median(ts_b[(S["labels_global"] == l) & chg])), 1))
            for l in labs if l >= 0]


# ---------------------------------------------------------------------------
def smallcluster_stage(S, cb: Callable):
    """Expansion (per-assignment replay) -> matrix -> swap fix -> force."""
    _ensure_here()
    from v030_expansion import run_family_expansion, force_full_coverage
    from swap_fix import run_swap_correction
    ns, ctx = S["ns"], S["ctx"]

    t0 = time.perf_counter()
    base_before = np.asarray(ns["baseImage"], np.float32)[S["watch"]].copy()
    st_e = run_family_expansion(ns, verbose=False, params={"PLAIN_SCORE": True})
    _tick(S, "3.9 family expansion (plain score)", t0)
    frames = [a for a in st_e.get("assigned", [])
              if S["watch"] in S["cluster_to_tpcs"].get(a["label"], [])]
    view = base_before.copy()
    for a in frames:
        key = (int(a["label"]), S["watch"])
        if key in S["image_maps"]:
            img = np.asarray(S["image_maps"][key], np.float64)
            sft = np.zeros_like(img)
            s0 = int(round(a["t0"]))
            if 0 <= s0 < img.shape[1]:
                sft[:, s0:] = img[:, :img.shape[1] - s0]
            view = view + sft.astype(np.float32)
        cb("3.9 expansion", f"family {a['fam']} absorbed cluster {a['label']} "
           f"(E={a['e']:.1f} MeV) at t0={a['t0']:.0f} | score {a['score']:.2f}, "
           f"d_geo {a['d_geo']:.1f} cm (truth={label_truth(S, a['label'])})",
           base_view=view.copy())
    taken = set(st_e.get("member_labels", []))
    for k in ("v11_iterative_single_tpc", "unassigned_by_tpc"):
        for tk, lst in list(ns[k].items()):
            ns[k][tk] = [cc for cc in lst if int(cc) not in taken]
    ns["v11_pruned_iterative_clusters"] = [
        cc for cc in ns.get("v11_pruned_iterative_clusters", [])
        if int(cc) not in taken]
    cb("3.9 expansion done", f"assigned {st_e['n_assigned']} clusters "
       f"({st_e['e_assigned']:.0f} MeV) event-wide; dissents {st_e['dissents']}, "
       f"ejections {st_e['ejections']}, new t0s {st_e['new_t0']}", force=True)

    t0 = time.perf_counter()
    (p3b, p3t, p3c, p3a, p3u, p3l, _u, _s) = ctx["run_matrix"](
        active_cluster_tpcs=ns["v11_active_cluster_tpcs"],
        iterative_single_tpc=ns["v11_iterative_single_tpc"],
        pruned_iterative_clusters=ns["v11_pruned_iterative_clusters"],
        image_maps=ns["imageMaps"], base_image=ns["baseImage"],
        full_light_waveform=ns["fullLightWaveform"], full_light_std=ns["fullLightStd"],
        channel_support_cache=ns["cluster_channel_support_cache"],
        labels_global=ns["labels_global"], hit_timestamps=ns["hit_timestamps"],
        t0_candidates=ns["t0Candidates"], assignment_info=ns["assignment_info"],
        unassigned_by_tpc=ns["unassigned_by_tpc"], cluster_energies=ns["cluster_energies"],
        energy_band_fraction=0.20, positive_row_margin=-1e-4,
        matrix_worsen_tolerance_norm=0.15, search_range=SEARCH, adc_clip=CLIP,
        collect_scan_losses=False, full_scan_assign_eps=-20000,
        backward_peak_align_ticks=5,
        leftover_absorption_context=ns.get("v11_1_leftover_absorption_context"),
        saturated_channel_cache=ns["saturated_channel_cache"])
    ns["baseImage"], ns["hit_timestamps"] = p3b, p3t
    ns["t0Candidates"], ns["assignment_info"], ns["unassigned_by_tpc"] = p3c, p3a, p3u
    _tick(S, "4.1 error-matrix Phase 3", t0)
    cb("4.1 error matrix", f"matrix proposed {len(p3l)}, assigned "
       f"{sum(1 for r in p3l if r.get('assigned'))} event-wide", force=True)

    t0 = time.perf_counter()
    st_sw = run_swap_correction(ns, hit_tpc=S["hit_tpc"], cluster_to_tpcs=S["cluster_to_tpcs"],
                                min_energy=25.0, chi2_ratio_max=SWAP_RATIO,
                                truth_fn=lambda l: label_truth(S, l), verbose=False)
    _tick(S, "4.5 per-TPC swap fix", t0)
    cb("4.5 swap fix", f"{st_sw.get('n_swaps', 0)} swaps event-wide "
       f"({st_sw.get('e_swapped', 0):.0f} MeV)", force=True)

    t0 = time.perf_counter()
    st_f = force_full_coverage(ns, verbose=False)
    _tick(S, "4.9 force coverage", t0)
    cb("4.9 force coverage", f"imaged {st_f['n_img']} | no-image {st_f['n_noimg']} | "
       f"noise hits {st_f['n_noise']} -> coverage 100%", force=True)


def tpc_summary(S):
    ts = current_ts(S)
    m = (S["hit_tpc"] == S["watch"]) & np.isfinite(S["truth_t0"])
    ok = m & np.isfinite(ts) & (ts >= 0)
    good = ok & (np.abs(ts - S["truth_t0"]) <= CUT)
    return dict(e_truth=float(S["e_pos"][m].sum()),
                coverage=float(S["e_pos"][ok].sum() / max(S["e_pos"][m].sum(), 1e-9)),
                eff=float(S["e_pos"][good].sum() / max(S["e_pos"][ok].sum(), 1e-9)))
