"""v0.2 refinement tests on the s=0.60 backbone (phase-1 only).

Variants per event (same first stage, same fixed showers+multi-TPC cohort):
  prod      production t0 for the pool (reference)
  v02       plain s=0.60 energy-banded permutation placement
  swap      v02 + pairwise t0-exchange pass (shared-TPC pairs, adopt if the
            pair's union-support loss improves; sweeps to convergence)
  relax     v02 + iterative remove-and-rescan sweeps (energy order)
  cosgate   v02 with the scan argmin restricted to shifts whose
            cos(own-peak-pattern, unexplained-light pattern) >= 0.6
            (fallback: plain argmin if no shift qualifies)
  cossoft   v02 with score = loss * (1 + (1 - cos))  (soft cosine penalty)
  defer     v02 with the LOCKED overflow trigger (cos<0.75 OR 3ch excess>=500
            z_pois>2) as a commit gate: a firing commit is rejected once and
            the cluster returns to the pool (placed later with more backbone)
  rescue    v02 + final pass: trigger-flagged placed clusters get an
            unanchored rescan, adopted if own-support loss improves
  stack     cosgate placement + defer gate + swap + rescue
"""
from __future__ import annotations

import argparse
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

from v02_ordering_pipeline import (PEAK, GAIN, SEARCH, CUT, weighted_scan, acc_sums,
                                   scan_label, add_label, place_pool_v02)
from sat_clip import clip_rows, clip_col

# v0.5.0 switchable loss (kept in lockstep with v02_ordering_pipeline)
LOSS_MODE = "chi2"


def configure_loss(mode: str) -> None:
    global LOSS_MODE
    LOSS_MODE = str(mode)
    import v02_ordering_pipeline as _vop
    _vop.configure_loss(mode)

S_BAND = 0.60
COS_CUT = 0.75
Z_CUT = 2.0
EXCESS_GATE = 500.0
N_CH_GATE = 3


def configure_gate(excess=None, zcut=None, nch=None, cos=None):
    """Re-tune the overflow/defer trigger thresholds (support-gate era)."""
    global EXCESS_GATE, Z_CUT, N_CH_GATE, COS_CUT
    if excess is not None:
        EXCESS_GATE = float(excess)
    if zcut is not None:
        Z_CUT = float(zcut)
    if nch is not None:
        N_CH_GATE = int(nch)
    if cos is not None:
        COS_CUT = float(cos)


def make_cos_pick(mode, cos_min=0.6, lam=1.0):
    def pick(curve, P, B, A, W):
        own = P[:, PEAK]
        no = float(np.linalg.norm(own))
        if no <= 1e-9:
            return float(np.argmin(curve))
        R = np.clip(np.clip(A, 0, None) - np.clip(B, 0, None), 0, None)[:, PEAK: PEAK + curve.size]
        dn = np.linalg.norm(R, axis=0)
        cos = np.zeros(curve.size)
        m = dn > 1e-9
        cos[: dn.size][m] = (own @ R[:, m]) / (no * dn[m])
        if mode == "gate":
            ok = cos >= cos_min
            if ok.any():
                return float(np.argmin(np.where(ok, curve, np.inf)))
            return float(np.argmin(curve))
        c = np.clip(cos, 0.0, 1.0)
        return float(np.argmin(curve * (1.0 + lam * (1.0 - c))))
    return pick


def trigger_fired(lab, t0, baseA, info, image_maps, actual, veto_mask, ti,
                  sup_fn=None):
    """The locked v0.1 trigger: cos(own,resid)<0.75 OR >=3 gated channels z_pois>2.

    sup_fn (optional): (lab, tpc) -> support channel indices.  When given,
    BOTH branches are restricted to the subject's support channels — the
    per-cluster-render summation smears a floor onto far channels (measured
    +90-115% on non-support channels at big families), and counting excess
    there defers correctly-placed clusters."""
    n_ticks = actual.shape[2]
    tick = int(round(t0)) + PEAK
    if not (0 <= tick < n_ticks):
        return False
    cmin = 1.0
    for c in info[lab]["tpcs"]:
        fin = ~veto_mask[c]
        if sup_fn is not None:
            ch_sup = sup_fn(int(lab), int(c))
            if ch_sup is not None and len(ch_sup):
                msup = np.zeros(fin.shape[0], dtype=bool)
                msup[np.asarray(ch_sup, dtype=int)] = True
                fin = fin & msup
        a = actual[c][:, tick]
        p = clip_col(baseA[c][:, tick], c)
        ex = p - a
        z = ex / np.sqrt(GAIN * np.maximum(a, 100.0))
        zz = np.sort(z[fin & (ex >= EXCESS_GATE)])[::-1]
        if zz.size >= N_CH_GATE and zz[N_CH_GATE - 1] > Z_CUT:
            return True
        o = ti(np.asarray(image_maps[(int(lab), int(c))], np.float64), t0)[:, tick]
        res = np.clip(np.clip(a, 0, None) - np.clip(p - o, 0, None), 0, None)
        no, nr = np.linalg.norm(o[fin]), np.linalg.norm(res[fin])
        if no > 1e-9 and nr > 1e-9:
            cmin = min(cmin, float(np.dot(o[fin], res[fin]) / (no * nr)))
    return cmin < COS_CUT


def make_defer_hook(info, image_maps, actual, veto_mask, ti, stats, sup_fn=None):
    deferred = set()

    def hook(lab, t0, baseA):
        if lab in deferred:
            return True
        if trigger_fired(lab, t0, baseA, info, image_maps, actual, veto_mask, ti,
                         sup_fn=sup_fn):
            deferred.add(lab)
            stats["n_deferred"] = stats.get("n_deferred", 0) + 1
            return False
        return True
    return hook


def pair_union(info, i, j):
    uch = {}
    for lab in (i, j):
        for c, ch in info[lab]["sup"].items():
            uch[c] = np.union1d(uch.get(c, np.array([], dtype=int)), ch)
    return uch


def swap_pass(tmap, baseA, info, image_maps, actual, w_full, ti, max_sweeps=3):
    placed = sorted(tmap)
    tpc_map = {}
    for lab in placed:
        for c in info[lab]["tpcs"]:
            tpc_map.setdefault(c, []).append(lab)
    pairs = sorted({(i, j) for labs in tpc_map.values()
                    for ai, i in enumerate(labs) for j in labs[ai + 1:]})
    n_sw = 0
    for _ in range(max_sweeps):
        changed = False
        for i, j in pairs:
            t_i, t_j = tmap[i], tmap[j]
            if abs(t_i - t_j) <= CUT:
                continue
            uch = pair_union(info, i, j)
            rows = {c: baseA[c][ch].copy() for c, ch in uch.items()}
            P_r = {}
            for lab, tt in ((i, t_i), (j, t_j)):
                for c in info[lab]["tpcs"]:
                    if c in uch:
                        P_r[(lab, c)] = np.asarray(image_maps[(int(lab), int(c))],
                                                   np.float64)[uch[c]]
                        rows[c] -= ti(P_r[(lab, c)], tt)

            def pl(ta, tb):
                tot = 0.0
                use_signed = LOSS_MODE != "chi2"
                if use_signed:
                    import v050_loss as _v050
                for c, ch in uch.items():
                    m = rows[c].copy()
                    if (i, c) in P_r:
                        m += ti(P_r[(i, c)], ta)
                    if (j, c) in P_r:
                        m += ti(P_r[(j, c)], tb)
                    if use_signed:
                        tot += _v050.point_loss(m, actual[c][ch], w_full[c][ch],
                                                LOSS_MODE)
                    else:
                        tot += float(np.sum(w_full[c][ch]
                                            * (clip_rows(m, c, ch) - actual[c][ch]) ** 2))
                return abs(tot) if use_signed else tot

            if pl(t_j, t_i) < pl(t_i, t_j) - 1e-9:
                for lab, told, tnew in ((i, t_i, t_j), (j, t_j, t_i)):
                    add_label(baseA, lab, told, info, image_maps, ti, sign=-1.0)
                    add_label(baseA, lab, tnew, info, image_maps, ti)
                    tmap[lab] = tnew
                n_sw += 1
                changed = True
        if not changed:
            break
    return n_sw


def relax_pass(tmap, baseA, info, image_maps, actual, w_full, ti, pick=None, max_sweeps=3):
    n_moved = 0
    for _ in range(max_sweeps):
        changed = 0
        for lab in sorted(tmap, key=lambda l: -info[l]["energy"]):
            t_old = tmap[lab]
            add_label(baseA, lab, t_old, info, image_maps, ti, sign=-1.0)
            t_new = scan_label(lab, baseA, info, image_maps, actual, w_full, pick=pick)
            if abs(t_new - t_old) > CUT:
                changed += 1
                n_moved += 1
            tmap[lab] = t_new
            add_label(baseA, lab, t_new, info, image_maps, ti)
        if changed == 0:
            break
    return n_moved


def rescue_pass(tmap, baseA, info, image_maps, actual, w_full, veto_mask, ti,
                sup_fn=None):
    n_flag = n_moved = 0
    for lab in sorted(tmap, key=lambda l: -info[l]["energy"]):
        t_old = tmap[lab]
        if not trigger_fired(lab, t_old, baseA, info, image_maps, actual, veto_mask, ti,
                             sup_fn=sup_fn):
            continue
        n_flag += 1
        add_label(baseA, lab, t_old, info, image_maps, ti, sign=-1.0)
        prs, bss, acs, wrs = [], [], [], []
        for c, ch in info[lab]["sup"].items():
            prs.append(np.asarray(image_maps[(int(lab), int(c))], np.float64)[ch])
            bss.append(baseA[c][ch])
            acs.append(actual[c][ch])
            wrs.append(w_full[c][ch])
        if LOSS_MODE != "chi2":
            import v050_loss as _v050
            curve = np.abs(_v050.scan_curve(np.concatenate(prs),
                                            np.concatenate(bss) - np.concatenate(acs),
                                            np.concatenate(wrs), mode=LOSS_MODE))
        else:
            curve = weighted_scan(np.concatenate(prs),
                                  np.concatenate(bss) - np.concatenate(acs),
                                  np.concatenate(wrs))
        t_new = float(np.argmin(curve))
        if abs(t_new - t_old) > CUT and curve[int(t_new)] < curve[int(t_old)] - 1e-9:
            tmap[lab] = t_new
            n_moved += 1
            add_label(baseA, lab, t_new, info, image_maps, ti)
        else:
            add_label(baseA, lab, t_old, info, image_maps, ti)
    return n_flag, n_moved


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--event-stride", type=int, default=1)
    ap.add_argument("--event-offset", type=int, default=0)
    ap.add_argument("--max-clip-ticks", type=int, default=None,
                    help="override the vAlpha saturation veto (default keeps the "
                         "config value, 6): channels clipped for MORE than this "
                         "many ticks are removed everywhere")
    args = ap.parse_args()

    import h5py
    from M5p1.first_stage_matching import load_first_stage_models, run_first_stage_charge_light_matching
    from M5p1.phase25_trial2_valpha_batch import build_default_first_stage_config, _list_events_in_file
    from M5p1 import truth_plotting as truthmod
    from pulse_shapes import timeinterpolation

    def ti(arr, t):
        return timeinterpolation(arr, shift=float(t), baseline=0).astype(np.float64)

    cfg = build_default_first_stage_config()
    cfg.prediction.min_cluster_energy_mev = 50.0
    if args.max_clip_ticks is not None:
        cfg.support.saturation_max_clip_ticks = int(args.max_clip_ticks)
        print(f"saturation veto tightened: max_clip_ticks = {args.max_clip_ticks}", flush=True)
    print("loading models ...", flush=True)
    models = load_first_stage_models(cfg)

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
            labels_global = np.asarray(fs.clustering.labels_global)
            split_index = int(fs.clustering.split_index)
            label_info = fs.clustering.label_info
            image_maps = fs.prediction.image_maps
            cluster_to_tpcs = fs.prediction.cluster_to_tpcs
            support_cache = fs.prediction.cluster_channel_support_cache
            veto_mask = np.asarray(fs.prediction.saturated_channel_cache["veto_mask"], dtype=bool)
            actual = np.asarray(fs.event.full_light_waveform, dtype=np.float64)
            n_light, n_ch, n_ticks = actual.shape
            hit_t0 = np.asarray(fs.track_stage.hit_t0)
            e_pos = np.clip(np.asarray(fs.event.energy, dtype=np.float64), 0.0, None)
            w_full = 1.0 / (GAIN * np.maximum(actual, 100.0))

            truth = truthmod.extract_event_hit_energy_and_truth_t0(
                h5, h5["charge/events/ref/charge/calib_prompt_hits/ref"], eventid=int(ev_id),
                hits_full=h5["charge/calib_prompt_hits/data"], convert_to_matching_ticks=True)
            truth_t0 = np.asarray(truth["truth_t0"], dtype=np.float64)

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
                        sup[int(c)] = np.sort(ch)
                if not sup:
                    continue
                info[lab] = {"t0": float(np.median(t0v)), "tpcs": tpcs, "sup": sup,
                             "type": lab_type, "energy": float(e_pos[members].sum())}

            showers = [l for l in sorted(info) if info[l]["type"] == "shower"]
            multi = [l for l in sorted(info) if info[l]["type"] != "shower"
                     and len(info[l]["tpcs"]) >= 3]
            pool = [l for l in sorted(info) if info[l]["type"] != "shower"
                    and len(info[l]["tpcs"]) <= 2]

            def label_truth(lab):
                m = labels_global == lab
                tt, ww = truth_t0[m], e_pos[m]
                okm = np.isfinite(tt)
                if not okm.any():
                    return None
                o = np.argsort(tt[okm])
                cw = np.cumsum(ww[okm][o])
                return float(tt[okm][o][np.searchsorted(cw, 0.5 * cw[-1])])

            base_fixed = np.zeros_like(actual)
            for lab in showers + multi:
                add_label(base_fixed, lab, info[lab]["t0"], info, image_maps, ti)

            A = (info, image_maps, actual, w_full, ti)
            variants, v_stats = {}, {}
            variants["prod"] = {l: info[l]["t0"] for l in pool}

            def run(name, pick=None, use_defer=False, use_swap=False,
                    use_relax=False, use_rescue=False):
                t0c = time.time()
                st = {}
                b = base_fixed.copy()
                hook = (make_defer_hook(info, image_maps, actual, veto_mask, ti, st)
                        if use_defer else None)
                tmap, _ = place_pool_v02(pool, S_BAND, b, info, image_maps, actual,
                                         w_full, ti, pick=pick, commit_hook=hook)
                if use_swap:
                    st["n_swaps"] = swap_pass(tmap, b, *A)
                if use_relax:
                    st["n_relaxed"] = relax_pass(tmap, b, *A, pick=pick)
                if use_rescue:
                    st["n_flagged"], st["n_rescued"] = rescue_pass(
                        tmap, b, info, image_maps, actual, w_full, veto_mask, ti)
                st["time_s"] = time.time() - t0c
                variants[name] = tmap
                v_stats[name] = st

            run("v02")
            run("stack2", pick=make_cos_pick("soft"), use_defer=True,
                use_swap=True, use_rescue=True)
            run("stack2r", pick=make_cos_pick("soft"), use_defer=True,
                use_swap=True, use_relax=True, use_rescue=True)

            pool_mask = np.isin(labels_global, np.asarray(pool, dtype=labels_global.dtype))
            ts_fixed = np.full(labels_global.shape, np.nan)
            for lab in showers + multi:
                ts_fixed[labels_global == lab] = info[lab]["t0"]
            fg, fo = acc_sums(ts_fixed, truth_t0, e_pos)
            row_variants = {}
            for name, tmap in variants.items():
                ts = ts_fixed.copy()
                for lab, t in tmap.items():
                    ts[labels_global == lab] = t
                g, o = acc_sums(ts, truth_t0, e_pos)
                gp, op = acc_sums(ts, truth_t0, e_pos, mask=pool_mask)
                row_variants[name] = {"good": g, "ok": o, "good_pool": gp, "ok_pool": op,
                                      "acc": (g / o if o > 0 else None),
                                      "acc_pool": (gp / op if op > 0 else None)}

            pool_rows = []
            for lab in pool:
                pool_rows.append({"label": int(lab), "energy": info[lab]["energy"],
                                  "tpcs": info[lab]["tpcs"], "truth": label_truth(lab),
                                  **{f"t_{name}": variants[name].get(lab)
                                     for name in variants}})

            fout.write(json.dumps({
                "file": os.path.basename(file_path), "event": int(ev_id),
                "n_shower": len(showers), "n_multi": len(multi), "n_pool": len(pool),
                "fixed_cohort": {"good": fg, "ok": fo},
                "variants": row_variants, "v_stats": v_stats, "pool_labels": pool_rows,
                "total_energy": float(e_pos.sum()),
                "truth_energy": float(e_pos[np.isfinite(truth_t0)].sum()),
                "t_event_s": time.time() - t_ev,
            }) + "\n")
            fout.flush()
            accs = " ".join(f"{n}={v['acc']:.4f}" for n, v in row_variants.items()
                            if v["acc"] is not None)
            print(f"done {os.path.basename(file_path)} ev{ev_id} in {time.time()-t_ev:.0f}s | "
                  f"{accs}", flush=True)
        except Exception as exc:
            import traceback
            print(f"ERROR {file_path} ev{ev_id}: {exc}", flush=True)
            traceback.print_exc()
    fout.close()
    print("worker complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
