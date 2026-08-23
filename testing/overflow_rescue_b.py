"""Stage A + Stage B of the saving algorithm (user spec, 2026-07-06).

Post-phase-1 rescue: at every t0 slot flagged by the pinpointing trigger
(cos(own, residual) < 0.75 OR [>=3 ch, excess >= 500 ADC, z_pois > 2]),
take the co-located clusters c1..cN (|dt0| <= 5 ticks, shared TPC, N <= 3)
and try all anchored permutations:
  - anchor the first cluster of the permutation at the slot t0,
  - scan each subsequent cluster over the FULL t0 range (Poisson-weighted
    loss on its own support channels, given everything placed so far),
  - after all N are placed, score the arrangement by the summed Poisson loss
    over the UNION of involved TPCs (support channels only, full waveform,
    model clipped at ADC_CLIP) -- TPCs untouched by a move contribute a
    constant, keeping comparisons apples-to-apples.
Adopt the minimum-loss permutation.  Groups where every cluster still lands
at the slot t0 are Stage-B candidates ("stop believing cross-TPC matching").

Writes one JSON line per processed group with truth bookkeeping.
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


def weighted_scan(pred, resid, w, max_shift=SEARCH):
    n = pred.shape[1]
    t1 = fftconvolve(w, (pred * pred)[:, ::-1], mode="full", axes=1)[:, n - 1: n - 1 + max_shift + 1].sum(axis=0)
    t2 = fftconvolve(w * resid, pred[:, ::-1], mode="full", axes=1)[:, n - 1: n - 1 + max_shift + 1].sum(axis=0)
    return t1 + 2.0 * t2 + float(np.sum(w * resid * resid))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--event-stride", type=int, default=1)
    ap.add_argument("--event-offset", type=int, default=0)
    args = ap.parse_args()

    import h5py
    from M5p1.first_stage_matching import load_first_stage_models, run_first_stage_charge_light_matching
    from M5p1.first_stage_matching.track_stage import fast_stack_images
    from M5p1.phase25_trial2_valpha_batch import build_default_first_stage_config, _list_events_in_file
    from M5p1 import truth_plotting as tp
    from pulse_shapes import timeinterpolation

    cfg = build_default_first_stage_config()
    cfg.prediction.min_cluster_energy_mev = 50.0
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
            hit_t0 = np.asarray(fs.track_stage.hit_t0, dtype=np.float64)
            e_pos = np.clip(np.asarray(fs.event.energy, dtype=np.float64), 0.0, None)

            truth = tp.extract_event_hit_energy_and_truth_t0(
                h5, h5["charge/events/ref/charge/calib_prompt_hits/ref"], eventid=int(ev_id),
                hits_full=h5["charge/calib_prompt_hits/data"], convert_to_matching_ticks=True)
            truth_t0 = np.asarray(truth["truth_t0"], dtype=np.float64)

            # per-label state
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
                tt = truth_t0[members]
                ww = e_pos[members]
                ok = np.isfinite(tt)
                tmed = None
                if ok.any():
                    order = np.argsort(tt[ok])
                    cw = np.cumsum(ww[ok][order])
                    tmed = float(tt[ok][order][np.searchsorted(cw, 0.5 * cw[-1])])
                info[lab] = {
                    "t0": float(np.median(t0v)), "tpcs": tpcs, "sup": sup,
                    "truth": tmed, "energy": float(ww.sum()), "type": lab_type,
                }

            # trigger at placement time (sequential replay)
            base = np.zeros_like(actual)
            flagged = []
            for lab in sorted(info):
                L = info[lab]
                tpcs_arr = np.asarray(L["tpcs"], dtype=np.int32)
                stacked = fast_stack_images(image_maps, int(lab), tpcs_arr)
                shifted = timeinterpolation(stacked, shift=L["t0"], baseline=0).astype(np.float64)
                own_full = shifted.reshape(tpcs_arr.size, n_ch, -1)
                base[tpcs_arr] += own_full
                tick = int(round(L["t0"])) + PEAK
                if not (0 <= tick < n_ticks) or L["type"] != "track":
                    continue
                fire = False
                cmin = 1.0
                for ti, c in enumerate(L["tpcs"]):
                    fin = ~veto_mask[c]
                    a = actual[c, :, tick]
                    p = np.minimum(base[c, :, tick], CLIP)
                    ex = p - a
                    z = ex / np.sqrt(GAIN * np.maximum(a, 100.0))
                    zz = np.sort(z[fin & (ex >= EXCESS_GATE)])[::-1]
                    if zz.size >= 3 and zz[2] > Z_CUT:
                        fire = True
                    o = own_full[ti, :, tick]
                    res = np.clip(np.clip(a, 0, None) - np.clip(p - o, 0, None), 0, None)
                    no, nr = np.linalg.norm(o[fin]), np.linalg.norm(res[fin])
                    if no > 1e-9 and nr > 1e-9:
                        cmin = min(cmin, float(np.dot(o[fin], res[fin]) / (no * nr)))
                if cmin < COS_CUT:
                    fire = True
                if fire:
                    flagged.append(lab)

            # groups
            groups = []
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
                groups.append((k, sorted(mem)))

            for k, mem in groups:
                if len(mem) < 2:
                    fout.write(json.dumps({"file": os.path.basename(file_path), "event": int(ev_id),
                                           "flag": int(k), "members": mem, "skipped": "N==1"}) + "\n")
                    continue
                slot_t0 = info[k]["t0"]
                # base without the group
                base_wo = base.copy()
                shifted_orig = {}
                for j in mem:
                    J = info[j]
                    tpcs_arr = np.asarray(J["tpcs"], dtype=np.int32)
                    sh = timeinterpolation(fast_stack_images(image_maps, int(j), tpcs_arr),
                                           shift=J["t0"], baseline=0).astype(np.float64)
                    shifted_orig[j] = sh.reshape(tpcs_arr.size, n_ch, -1)
                    base_wo[tpcs_arr] -= shifted_orig[j]
                np.clip(base_wo, 0.0, None, out=base_wo)

                # union loss channels + weights (constant across permutations)
                union = {}
                for j in mem:
                    for c, ch in info[j]["sup"].items():
                        union[c] = np.union1d(union.get(c, np.array([], dtype=int)), ch)
                w_union = {c: 1.0 / (GAIN * np.maximum(actual[c][ch], 100.0)) for c, ch in union.items()}

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

                loss_orig = combined_loss({j: info[j]["t0"] for j in mem})

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
                        pred_rows, res_rows, w_rows = [], [], []
                        for c, ch in J["sup"].items():
                            pred_rows.append(np.asarray(image_maps[(int(j), int(c))], dtype=np.float64)[ch])
                            res_rows.append(cur[c][ch] - actual[c][ch])
                            w_rows.append(1.0 / (GAIN * np.maximum(actual[c][ch], 100.0)))
                        curve = weighted_scan(np.concatenate(pred_rows), np.concatenate(res_rows),
                                              np.concatenate(w_rows))
                        t0j = float(np.argmin(curve))
                        placement[j] = t0j
                        shj = timeinterpolation(
                            fast_stack_images(image_maps, int(j), np.asarray(J["tpcs"], dtype=np.int32)),
                            shift=t0j, baseline=0).astype(np.float64).reshape(len(J["tpcs"]), n_ch, -1)
                        for ti, c in enumerate(J["tpcs"]):
                            cur[c] += shj[ti]
                    fl = combined_loss(placement)
                    if best is None or fl < best[0]:
                        best = (fl, dict(placement), [int(v) for v in perm])

                loss_new, placement, order = best
                all_stay = all(abs(placement[j] - slot_t0) <= 10.0 for j in mem)

                stage_b = None
                if all_stay:
                    # firing TPC: strongest trigger signature among flagged member's TPCs
                    K = info[k]
                    tick = int(round(K["t0"])) + PEAK
                    best_sig = None
                    for c in K["tpcs"]:
                        fin = ~veto_mask[c]
                        a = actual[c, :, tick]
                        p = np.minimum(base[c, :, tick], CLIP)
                        ex = p - a
                        z = ex / np.sqrt(GAIN * np.maximum(a, 100.0))
                        zz = np.sort(z[fin & (ex >= EXCESS_GATE)])[::-1]
                        z3v = float(zz[2]) if zz.size >= 3 else -100.0
                        if best_sig is None or z3v > best_sig[0]:
                            best_sig = (z3v, int(c))
                    tpc_f = best_sig[1]

                    # union of group support channels in the firing TPC (for the TPC-only loss)
                    ch_u = np.array([], dtype=int)
                    for j in mem:
                        if tpc_f in info[j]["sup"]:
                            ch_u = np.union1d(ch_u, info[j]["sup"][tpc_f])
                    if ch_u.size:
                        w_f = 1.0 / (GAIN * np.maximum(actual[tpc_f][ch_u], 100.0))

                        def tpc_loss(model_tpc):
                            return float(np.sum(w_f * (np.minimum(model_tpc[ch_u], CLIP)
                                                       - actual[tpc_f][ch_u]) ** 2))

                        loss_stay = tpc_loss(base[tpc_f])
                        tries = []
                        for j in mem:
                            if tpc_f not in info[j]["sup"] or (int(j), tpc_f) not in image_maps:
                                continue
                            P = np.asarray(image_maps[(int(j), tpc_f)], dtype=np.float64)
                            sh_t0 = timeinterpolation(P, shift=info[j]["t0"], baseline=0).astype(np.float64)
                            b_wo = np.clip(base[tpc_f] - sh_t0, 0.0, None)
                            chp = info[j]["sup"][tpc_f]
                            curve = weighted_scan(
                                P[chp], (b_wo - actual[tpc_f])[chp],
                                1.0 / (GAIN * np.maximum(actual[tpc_f][chp], 100.0)))
                            t1 = float(np.argmin(curve))
                            # part hits + truth
                            pm = (labels_global == j) & (np.asarray(fs.event.hit_tpc_id) == tpc_f)
                            e_part = float(e_pos[pm].sum())
                            tt = truth_t0[pm]; wwp = e_pos[pm]
                            okp = np.isfinite(tt)
                            tmed_p = None
                            if okp.any():
                                op = np.argsort(tt[okp]); cwp = np.cumsum(wwp[okp][op])
                                tmed_p = float(tt[okp][op][np.searchsorted(cwp, 0.5 * cwp[-1])])
                            if abs(t1 - info[j]["t0"]) > 10.0:
                                sh_t1 = timeinterpolation(P, shift=t1, baseline=0).astype(np.float64)
                                l_move = tpc_loss(b_wo + sh_t1)
                            else:
                                l_move = None
                            tries.append({"member": int(j), "t1": t1, "loss_move": l_move,
                                          "e_part": e_part, "truth_part": tmed_p})
                        movers = [t for t in tries if t["loss_move"] is not None]
                        pick = min(movers, key=lambda t: t["loss_move"]) if movers else None
                        stage_b = {"tpc_f": int(tpc_f), "loss_stay": loss_stay,
                                   "tries": tries,
                                   "detach": None if pick is None else pick["member"],
                                   "detach_t1": None if pick is None else pick["t1"],
                                   "detach_e": None if pick is None else pick["e_part"],
                                   "detach_truth": None if pick is None else pick["truth_part"],
                                   "detach_loss": None if pick is None else pick["loss_move"]}

                fout.write(json.dumps({
                    "file": os.path.basename(file_path), "event": int(ev_id),
                    "flag": int(k), "members": [int(j) for j in mem],
                    "slot_t0": slot_t0, "winner_order": order,
                    "old_t0": {str(j): info[j]["t0"] for j in mem},
                    "new_t0": {str(j): placement[j] for j in mem},
                    "truth": {str(j): info[j]["truth"] for j in mem},
                    "energy": {str(j): info[j]["energy"] for j in mem},
                    "loss_orig": loss_orig, "loss_new": loss_new,
                    "all_stay": bool(all_stay), "stage_b": stage_b,
                }) + "\n")
            fout.flush()
            print(f"done {os.path.basename(file_path)} ev{ev_id} in {time.time()-t_ev:.0f}s "
                  f"({len(flagged)} flagged, {len(groups)} groups)", flush=True)
        except Exception as exc:
            import traceback
            print(f"ERROR {file_path} ev{ev_id}: {exc}", flush=True)
            traceback.print_exc()
    fout.close()
    print("worker complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
