"""Census of placements STILL WRONG after the locked phase-1 rescue chain
(trigger cos<0.75 OR 3ch/excess500/z>2; Stage A anchored perms + stay-gate;
Stage B part detach; N=1 with COS_SEL=0.7 gate + COS_WINDOW=0.15 on relocations).

Per event: phase-1 -> rescue -> adopt, then for every backbone label score the
hit-level wrong energy at the rescued state, categorize (never-flagged /
flagged-no-action / action-still-wrong / broken / shower), and record
SAVABILITY diagnostics: whole-cluster union loss+cos at the rescued slot vs at
the TRUTH slot in the post-rescue base.  Labels with wrongE >= DUMP_MEV also get
a channel/waveform npz dump for the gallery.
"""
import argparse
import itertools
import json
import os
import sys
import time

_HERE = os.path.abspath(os.path.dirname(__file__))
BASE_DIR = os.path.abspath(os.path.join(_HERE, "..", "CLMatching_AlphaRelease"))
sys.path.insert(0, BASE_DIR)
sys.path.insert(1, os.path.join(BASE_DIR, "NewMLSection"))
sys.path.insert(2, os.path.join(BASE_DIR, "M5p1"))

import numpy as np
import h5py
from scipy.signal import fftconvolve

PEAK, GAIN, CLIP = 105, 200.0, 60780.0
COS_CUT, Z_CUT, EXCESS_GATE = 0.75, 2.0, 500.0
GROUP_DT, MAX_N, SEARCH, CUT = 5.0, 3, 800, 10.0
COS_SEL, COS_WINDOW = 0.7, 0.15
DUMP_MEV = 20.0

from M5p1.first_stage_matching import load_first_stage_models, run_first_stage_charge_light_matching
from M5p1.first_stage_matching.track_stage import fast_stack_images
from M5p1.phase25_trial2_valpha_batch import build_default_first_stage_config, _list_events_in_file
from M5p1 import truth_plotting as tp
from pulse_shapes import timeinterpolation


def weighted_scan(pred, resid, w, max_shift=SEARCH):
    n = pred.shape[1]
    t1 = fftconvolve(w, (pred * pred)[:, ::-1], mode="full", axes=1)[:, n - 1: n - 1 + max_shift + 1].sum(axis=0)
    t2 = fftconvolve(w * resid, pred[:, ::-1], mode="full", axes=1)[:, n - 1: n - 1 + max_shift + 1].sum(axis=0)
    return t1 + 2.0 * t2 + float(np.sum(w * resid * resid))


def cos_curve(pred, resid, max_shift=SEARCH):
    own = pred[:, PEAK]
    R = np.clip(-resid, 0.0, None)[:, PEAK: PEAK + max_shift + 1]
    no = float(np.linalg.norm(own))
    nr = np.linalg.norm(R, axis=0)
    c = (own @ R) / np.maximum(no * nr, 1e-9)
    c[nr < 1e-6] = 0.0
    return c if no > 1e-9 else np.zeros_like(c)


def gated_argmin(curve, pred, resid, cos_sel, cos_window=0.0):
    if cos_sel <= 0.0:
        return float(np.argmin(curve))
    cv = cos_curve(pred, resid, max_shift=curve.size - 1)
    ok = cv >= cos_sel
    if not ok.any():
        return float(np.argmin(curve))
    idx = np.flatnonzero(ok)
    if cos_window > 0.0:
        lmin = float(curve[idx].min())
        band = idx[curve[idx] <= lmin * (1.0 + cos_window)]
        return float(band[np.argmax(cv[band])])
    return float(idx[np.argmin(curve[idx])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--event-stride", type=int, default=1)
    ap.add_argument("--event-offset", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.dump_dir, exist_ok=True)

    cfg = build_default_first_stage_config()
    cfg.prediction.min_cluster_energy_mev = 50.0
    print("loading models ...", flush=True)
    models = load_first_stage_models(cfg)

    work = []
    for fp in args.files:
        work.extend((fp, int(e)) for e in _list_events_in_file(fp))
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
            labels_lo = np.asarray(fs.track_stage.labels_global_with_leftovers)
            split_index = int(fs.clustering.split_index)
            label_info = fs.clustering.label_info
            image_maps = fs.prediction.image_maps
            cluster_to_tpcs = fs.prediction.cluster_to_tpcs
            support_cache = fs.prediction.cluster_channel_support_cache
            veto_mask = np.asarray(fs.prediction.saturated_channel_cache["veto_mask"], dtype=bool)
            actual = np.asarray(fs.event.full_light_waveform, dtype=np.float64)
            n_light, n_ch, n_ticks = actual.shape
            hit_t0 = np.asarray(fs.track_stage.hit_t0)
            base_img = np.asarray(fs.track_stage.base_image)
            hit_tpc = np.asarray(fs.event.hit_tpc_id)
            e_pos = np.clip(np.asarray(fs.event.energy, dtype=np.float64), 0.0, None)
            labels_phase1 = labels_global.copy()

            truth = tp.extract_event_hit_energy_and_truth_t0(
                h5, h5["charge/events/ref/charge/calib_prompt_hits/ref"], eventid=int(ev_id),
                hits_full=h5["charge/calib_prompt_hits/data"], convert_to_matching_ticks=True)
            truth_t0 = np.asarray(truth["truth_t0"], dtype=np.float64)

            def masked_truth(m):
                tt, ww = truth_t0[m], e_pos[m]
                ok = np.isfinite(tt)
                if not ok.any():
                    return None
                o = np.argsort(tt[ok])
                cw = np.cumsum(ww[ok][o])
                return float(tt[ok][o][np.searchsorted(cw, 0.5 * cw[-1])])

            # ---- info + replay trigger (identical to the locked chain) ----
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
                             "type": lab_type, "energy": float(e_pos[members].sum())}

            base = np.zeros_like(actual)
            flagged, trig_diag = [], {}
            for lab in sorted(info):
                L = info[lab]
                ta = np.asarray(L["tpcs"], dtype=np.int32)
                sh = timeinterpolation(fast_stack_images(image_maps, int(lab), ta),
                                       shift=L["t0"], baseline=0).astype(np.float64)
                own_full = sh.reshape(ta.size, n_ch, -1)
                base[ta] += own_full
                tick = int(round(L["t0"])) + PEAK
                if not (0 <= tick < n_ticks) or L["type"] != "track":
                    continue
                fire = False
                cmin, z3m = 1.0, -100.0
                for ti, c in enumerate(L["tpcs"]):
                    fin = ~veto_mask[c]
                    a = actual[c, :, tick]
                    p = np.minimum(base[c, :, tick], CLIP)
                    ex = p - a
                    z = ex / np.sqrt(GAIN * np.maximum(a, 100.0))
                    zz = np.sort(z[fin & (ex >= EXCESS_GATE)])[::-1]
                    if zz.size >= 3:
                        z3m = max(z3m, float(zz[2]))
                        if zz[2] > Z_CUT:
                            fire = True
                    o = own_full[ti, :, tick]
                    res = np.clip(np.clip(a, 0, None) - np.clip(p - o, 0, None), 0, None)
                    no, nr = np.linalg.norm(o[fin]), np.linalg.norm(res[fin])
                    if no > 1e-9 and nr > 1e-9:
                        cmin = min(cmin, float(np.dot(o[fin], res[fin]) / (no * nr)))
                if cmin < COS_CUT:
                    fire = True
                trig_diag[lab] = {"cmin": cmin, "z3": z3m}
                if fire:
                    flagged.append(lab)

            # ---- groups / Stage A / Stage B / N=1 (locked knobs) ----
            groups, lone, seen = [], [], set()
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
                (lone if len(mem) < 2 else groups).append(k if len(mem) < 2 else (k, sorted(mem)))

            moves, detaches, all_stay_groups = {}, [], []
            for k, mem in groups:
                if any(j in moves for j in mem):
                    continue
                slot_t0 = info[k]["t0"]
                base_wo = base.copy()
                for j in mem:
                    J = info[j]
                    ta = np.asarray(J["tpcs"], dtype=np.int32)
                    base_wo[ta] -= timeinterpolation(fast_stack_images(image_maps, int(j), ta),
                                                     shift=J["t0"], baseline=0).astype(np.float64).reshape(ta.size, n_ch, -1)
                np.clip(base_wo, 0.0, None, out=base_wo)
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
                            wr.append(1.0 / (GAIN * np.maximum(actual[c][ch], 100.0)))
                        pr_j, rr_j = np.concatenate(pr), np.concatenate(rr)
                        curve = weighted_scan(pr_j, rr_j, np.concatenate(wr))
                        t0j = gated_argmin(curve, pr_j, rr_j, COS_SEL)
                        placement[j] = t0j
                        shj = timeinterpolation(
                            fast_stack_images(image_maps, int(j), np.asarray(J["tpcs"], dtype=np.int32)),
                            shift=t0j, baseline=0).astype(np.float64).reshape(len(J["tpcs"]), n_ch, -1)
                        for ti, c in enumerate(J["tpcs"]):
                            cur[c] += shj[ti]
                    fl = combined_loss(placement)
                    if best is None or fl < best[0]:
                        best = (fl, dict(placement))
                loss_stay_A = combined_loss({j: info[j]["t0"] for j in mem})
                placement = best[1] if best[0] < loss_stay_A else {j: info[j]["t0"] for j in mem}
                all_stay = all(abs(placement[j] - slot_t0) <= CUT for j in mem)
                if all_stay:
                    all_stay_groups.append((k, mem))
                    continue
                for j in mem:
                    if abs(placement[j] - info[j]["t0"]) > CUT:
                        moves[j] = placement[j]

            for k, mem in all_stay_groups:
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
                ch_u = np.array([], dtype=int)
                for j in mem:
                    if tpc_f in info[j]["sup"]:
                        ch_u = np.union1d(ch_u, info[j]["sup"][tpc_f])
                if not ch_u.size:
                    continue
                w_f = 1.0 / (GAIN * np.maximum(actual[tpc_f][ch_u], 100.0))
                movers = []
                for j in mem:
                    if tpc_f not in info[j]["sup"] or (int(j), tpc_f) not in image_maps:
                        continue
                    if len(info[j]["tpcs"]) < 2:
                        continue
                    P = np.asarray(image_maps[(int(j), tpc_f)], dtype=np.float64)
                    sh_t0 = timeinterpolation(P, shift=info[j]["t0"], baseline=0).astype(np.float64)
                    b_wo = np.clip(base[tpc_f] - sh_t0, 0.0, None)
                    chp = info[j]["sup"][tpc_f]
                    rr_b = (b_wo - actual[tpc_f])[chp]
                    t1 = gated_argmin(weighted_scan(P[chp], rr_b, 1.0 / (GAIN * np.maximum(actual[tpc_f][chp], 100.0))),
                                      P[chp], rr_b, COS_SEL)
                    if abs(t1 - info[j]["t0"]) <= CUT:
                        continue
                    sh_t1 = timeinterpolation(P, shift=t1, baseline=0).astype(np.float64)
                    lm = float(np.sum(w_f * (np.minimum((b_wo + sh_t1)[ch_u], CLIP) - actual[tpc_f][ch_u]) ** 2))
                    movers.append((lm, int(j)))
                if movers:
                    movers.sort()
                    detaches.append((movers[0][1], tpc_f))

            grouped_members = set(j for _, mem in groups for j in mem)
            n1_rem_moves = {}
            for k in lone:
                if k in grouped_members or k in moves:
                    continue
                K = info[k]
                tpcs = K["tpcs"]
                ta = np.asarray(tpcs, dtype=np.int32)
                stk = fast_stack_images(image_maps, int(k), ta)
                base_wo = base.copy()
                base_wo[ta] -= timeinterpolation(stk, shift=K["t0"], baseline=0).astype(np.float64).reshape(ta.size, n_ch, -1)
                np.clip(base_wo, 0.0, None, out=base_wo)
                union = dict(K["sup"])
                w_u = {c: 1.0 / (GAIN * np.maximum(actual[c][ch], 100.0)) for c, ch in union.items()}

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
                t0_move = gated_argmin(weighted_scan(pr, rr, wr), pr, rr, COS_SEL, COS_WINDOW)
                loss_move = u_loss(contrib(stk, tpcs, t0_move))
                cands = [("move", loss_move, t0_move, None)] if abs(t0_move - K["t0"]) > CUT else []
                if len(tpcs) >= 2:
                    for tpf in tpcs:
                        if tpf not in union or (int(k), int(tpf)) not in image_maps:
                            continue
                        P = np.asarray(image_maps[(int(k), int(tpf))], dtype=np.float64)
                        chp = union[tpf]
                        rr_p = (base_wo[tpf] - actual[tpf])[chp]
                        t1 = gated_argmin(weighted_scan(P[chp], rr_p, w_u[tpf]), P[chp], rr_p, COS_SEL)
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
                            t0_rem = gated_argmin(weighted_scan(pr_r, rr_r, wr_r), pr_r, rr_r,
                                                  COS_SEL, COS_WINDOW)
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
                if kind == "move":
                    moves[k] = t0_best
                else:
                    detaches.append((k, tp_best))
                    if abs(t0_best - K["t0"]) > CUT:
                        n1_rem_moves[k] = (t0_best, [c for c in tpcs if c != tp_best])

            # ---- ADOPT ----
            next_label = int(labels_global.max()) + 1 if (labels_global >= 0).any() else 0
            action = {}
            for j, t0n in moves.items():
                J = info[j]
                ta = np.asarray(J["tpcs"], dtype=np.int32)
                stk = fast_stack_images(image_maps, int(j), ta)
                sh_old = timeinterpolation(stk, shift=J["t0"], baseline=0).astype(np.float32)
                sh_new = timeinterpolation(stk, shift=float(t0n), baseline=0).astype(np.float32)
                base_img[ta] += (sh_new - sh_old).reshape(ta.size, n_ch, -1)
                hit_t0[labels_global == j] = np.float32(t0n)
                action[j] = f"move->{t0n:.0f}"
            done_detach = set()
            for j, tpc_f in detaches:
                if (j, tpc_f) in done_detach or (int(j), int(tpc_f)) not in image_maps or j in moves:
                    continue
                done_detach.add((j, tpc_f))
                newlab = next_label
                next_label += 1
                mask = (labels_global == j) & (hit_tpc == tpc_f)
                labels_global[mask] = newlab
                labels_lo[mask] = newlab
                hit_t0[mask] = np.nan
                P = np.asarray(image_maps.pop((int(j), int(tpc_f))))
                base_img[tpc_f] -= timeinterpolation(P.astype(np.float64), shift=info[j]["t0"],
                                                     baseline=0).astype(np.float32)
                image_maps[(int(newlab), int(tpc_f))] = P
                action[j] = (action.get(j, "") + f" detachT{tpc_f}").strip()
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
                action[k] = (action.get(k, "") + f" rem->{t0r:.0f}").strip()
            np.clip(base_img, 0.0, None, out=base_img)

            # ---- census of STILL-WRONG labels at the rescued state ----
            rts = hit_t0.astype(np.float64)
            base_res = np.asarray(base_img, dtype=np.float64)
            ftag = os.path.basename(file_path).split(".")[-3]
            ev_rows, dump_labels = [], []
            for lab in sorted(info):
                m = labels_phase1 == lab
                tv = m & np.isfinite(truth_t0)
                a_now = tv & np.isfinite(rts) & (rts >= 0)
                wrong_e = float(e_pos[a_now & (np.abs(rts - truth_t0) > CUT)].sum())
                if wrong_e < 5.0:
                    continue
                tr = masked_truth(m)
                fin_now = np.isfinite(rts[m]) & (rts[m] >= 0)
                t0_now = float(np.median(rts[m][fin_now])) if fin_now.any() else float("nan")
                d_p1 = abs(info[lab]["t0"] - tr) if tr is not None else float("nan")
                d_now = abs(t0_now - tr) if (tr is not None and np.isfinite(t0_now)) else float("nan")
                if not (np.isfinite(d_now) and d_now > CUT):
                    continue    # only labels whose surviving placement is still wrong
                # category
                if info[lab]["type"] != "track":
                    cat = "shower-untriggered"
                elif lab not in flagged:
                    cat = "never-flagged"
                elif lab in action:
                    cat = "action-still-wrong" if d_p1 > CUT else "BROKEN-by-rescue"
                else:
                    cat = "flagged-no-action" if d_p1 > CUT else "BROKEN?"
                # savability: post-rescue whole-cluster union scan (base without itself)
                sav = {}
                if tr is not None and all((int(lab), int(c)) in image_maps for c in info[lab]["sup"]):
                    tpcs = [c for c in info[lab]["tpcs"] if (int(lab), int(c)) in image_maps]
                    if tpcs and np.isfinite(t0_now):
                        ta = np.asarray(tpcs, dtype=np.int32)
                        stk = fast_stack_images(image_maps, int(lab), ta)
                        bw = base_res.copy()
                        bw[ta] -= timeinterpolation(stk, shift=t0_now, baseline=0).astype(np.float64).reshape(ta.size, n_ch, -1)
                        np.clip(bw, 0.0, None, out=bw)
                        sup = {c: ch for c, ch in info[lab]["sup"].items() if c in tpcs}
                        pr = np.concatenate([np.asarray(image_maps[(int(lab), int(c))], float)[ch]
                                             for c, ch in sup.items()])
                        rr = np.concatenate([(bw[c] - actual[c])[ch] for c, ch in sup.items()])
                        wr = np.concatenate([1.0 / (GAIN * np.maximum(actual[c][ch], 100.0)) for c, ch in sup.items()])
                        curve = weighted_scan(pr, rr, wr)
                        cv = cos_curve(pr, rr)
                        sT, sN = int(round(tr)), int(round(t0_now))
                        if 0 <= sT <= SEARCH and 0 <= sN <= SEARCH:
                            sav = {"cos_truth": float(cv[sT]), "cos_now": float(cv[sN]),
                                   "loss_truth": float(curve[sT]), "loss_now": float(curve[sN]),
                                   "loss_min": float(curve.min()), "argmin": int(np.argmin(curve))}
                row = {"file": ftag, "event": int(ev_id), "label": int(lab),
                       "type": info[lab]["type"], "e": info[lab]["energy"], "wrong_e": wrong_e,
                       "t0_p1": info[lab]["t0"], "t0_now": t0_now, "truth": tr,
                       "d_now": d_now, "tpcs": info[lab]["tpcs"],
                       "flagged": bool(lab in flagged), "action": action.get(lab, "-"),
                       "cmin": trig_diag.get(lab, {}).get("cmin"),
                       "z3": trig_diag.get(lab, {}).get("z3"),
                       "cat": cat, "sav": sav}
                ev_rows.append(row)
                if wrong_e >= DUMP_MEV:
                    dump_labels.append(lab)

            if dump_labels:
                dd = {"x": np.asarray(fs.event.x, np.float32), "y": np.asarray(fs.event.y, np.float32),
                      "z": np.asarray(fs.event.z, np.float32), "e": e_pos.astype(np.float32),
                      "tpc": hit_tpc.astype(np.int16), "labels_p1": labels_phase1.astype(np.int32),
                      "hit_t0_rescued": rts.astype(np.float32), "truth_t0": truth_t0.astype(np.float32)}
                for lab in dump_labels:
                    tpcs = [c for c in info[lab]["tpcs"] if (int(lab), int(c)) in image_maps]
                    for c in info[lab]["sup"]:
                        if c not in tpcs:
                            continue
                        dd[f"L{lab}_T{c}_own"] = np.asarray(image_maps[(int(lab), int(c))], np.float32)
                        dd[f"L{lab}_T{c}_sup"] = np.asarray(info[lab]["sup"][c], np.int32)
                    for c in set(info[lab]["tpcs"]):
                        dd[f"T{c}_actual"] = actual[c].astype(np.float32)
                        dd[f"T{c}_base"] = base_res[c].astype(np.float32)
                np.savez_compressed(os.path.join(args.dump_dir, f"{ftag}_ev{ev_id}.npz"), **dd)

            fout.write(json.dumps({"file": ftag, "event": int(ev_id), "rows": ev_rows,
                                   "n_flagged": len(flagged), "n_moves": len(moves),
                                   "n_detaches": len(done_detach),
                                   "truth_energy": float(e_pos[np.isfinite(truth_t0)].sum())}) + "\n")
            fout.flush()
            print(f"done {ftag} ev{ev_id} in {time.time()-t_ev:.0f}s | still-wrong labels {len(ev_rows)} "
                  f"({sum(r['wrong_e'] for r in ev_rows):.0f} MeV)", flush=True)
        except Exception as exc:
            import traceback
            print(f"ERROR {file_path} ev{ev_id}: {exc}", flush=True)
            traceback.print_exc()
    fout.close()
    print("worker complete", flush=True)


if __name__ == "__main__":
    main()
