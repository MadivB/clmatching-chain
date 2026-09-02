"""v0.2 ordering-test backbone (phase-1 only; no Phase 2 / 2.5 / V2).

Per event:
  showers                   -> keep production t0   ("do showering as usual")
  tracks spanning >=3 TPCs  -> keep production t0   ("current method")
  pool = remaining predicted clusters (1-2 TPCs; all >=50 MeV by config),
  re-placed by the v0.2 energy-banded permutation batches:
    seed c1 = most energetic unplaced; band = pool members sharing a TPC with
    c1 and energy > (1-s)*E1, capped at MAX_BATCH total by energy; try ALL
    orderings, each ordering places greedily at the Poisson-weighted FFT-scan
    argmin given base-so-far; the ordering with the smallest union-support
    loss (pred clipped at ADC clip) wins; commit its first N-1 placements,
    return the LAST-placed to the pool; iterate until the pool is empty.

Controls on the same pool with the same scan machinery:
  ctrlL = production label order, ctrlG = plain energy-descending greedy
  (ctrlG == v0.2 with permutations disabled), plus prod = production t0s.

Sweeps s (default 0.20,0.30,0.40,0.50); reports energy-weighted hit-level
accuracy (|reco-truth| <= 10 ticks) overall and on the pool cohort, and the
compute cost (batches / permutation leaves / scans / wall time) per s.
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

import numpy as np
from scipy.signal import fftconvolve

PEAK = 105
GAIN = 200.0
CLIP = 60780.0
SEARCH = 895
CUT = 10.0
MAX_BATCH = 5

# v0.5.0: switchable placement loss ("chi2" = v0.4.1; "cubic"/"signed2" =
# signed residual loss, decisions driven toward |L| = 0; see v050_loss.py)
LOSS_MODE = "chi2"


def configure_loss(mode: str) -> None:
    global LOSS_MODE
    LOSS_MODE = str(mode)


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


def _rank1_amp(pred, tmpl, tmax):
    amp = pred.max(axis=1) / tmax
    ok = np.allclose(pred, amp[:, None] * tmpl[None, :], rtol=1e-3, atol=0.5)
    return (amp if ok else None)


def weighted_scan(pred, resid, w, max_shift=SEARCH):
    n = pred.shape[1]
    if SCAN_TEMPLATE is not None and pred.shape[1] == SCAN_TEMPLATE.size:
        tmpl = SCAN_TEMPLATE
        amp = _rank1_amp(np.asarray(pred, np.float64), tmpl, float(tmpl.max()))
        if amp is not None:
            w2 = (w * np.square(amp)[:, None]).sum(axis=0)[None, :]
            wr = (w * resid * amp[:, None]).sum(axis=0)[None, :]
            t1 = fftconvolve(w2, np.square(tmpl)[None, ::-1], mode="full", axes=1)[0, n - 1: n - 1 + max_shift + 1]
            t2 = fftconvolve(wr, tmpl[None, ::-1], mode="full", axes=1)[0, n - 1: n - 1 + max_shift + 1]
            out = t1 + 2.0 * t2 + float(np.sum(w * resid * resid))
            if EDGE_PENALTY:
                wt = w[:, -EDGE_TAIL_TICKS:].mean(axis=1)
                suf = np.cumsum(np.square(tmpl)[::-1])[::-1]
                A = float((wt * np.square(amp)).sum()) * suf
                pen = np.zeros(max_shift + 1)
                sh = np.arange(1, max_shift + 1)
                idx = n - sh
                m = (idx >= 0) & (idx < n)
                pen[1:][m] = A[idx[m]]
                out = out + pen
            return out
    t1 = fftconvolve(w, (pred * pred)[:, ::-1], mode="full", axes=1)[:, n - 1: n - 1 + max_shift + 1].sum(axis=0)
    t2 = fftconvolve(w * resid, pred[:, ::-1], mode="full", axes=1)[:, n - 1: n - 1 + max_shift + 1].sum(axis=0)
    out = t1 + 2.0 * t2 + float(np.sum(w * resid * resid))
    if EDGE_PENALTY:
        out = out + _edge_penalty_vec(pred, w, max_shift)
    return out


def acc_sums(hit_ts, truth_t0, e_pos, mask=None):
    """(sum E correct, sum E truth-valid assigned) for pooling across events."""
    r = np.asarray(hit_ts, dtype=np.float64)
    assigned = np.isfinite(r) & (r >= 0)
    if mask is not None:
        assigned = assigned & mask
    ok = assigned & np.isfinite(truth_t0)
    good = ok & (np.abs(r - truth_t0) <= CUT)
    return float(e_pos[good].sum()), float(e_pos[ok].sum())


def scan_label(lab, baseA, info, image_maps, actual, w_full, pick=None):
    """Optimal shift for `lab` against the current base (support channels).

    `pick(curve, P, B, A, W)` may override plain argmin (e.g. cosine-aware)."""
    prs, bss, acs, wrs = [], [], [], []
    for c, ch in info[lab]["sup"].items():
        prs.append(np.asarray(image_maps[(int(lab), int(c))], np.float64)[ch])
        bss.append(baseA[c][ch])
        acs.append(actual[c][ch])
        wrs.append(w_full[c][ch])
    P = np.concatenate(prs)
    B = np.concatenate(bss)
    A = np.concatenate(acs)
    W = np.concatenate(wrs)
    if LOSS_MODE != "chi2":
        import v050_loss as _v050
        curve = _v050.scan_curve(P, B - A, W, mode=LOSS_MODE)
        t, _ = _v050.refine_argmin(P, B, A, W, curve, top_k=32, mode=LOSS_MODE)
        return float(t)
    curve = weighted_scan(P, B - A, W)
    if pick is None:
        return float(np.argmin(curve))
    return pick(curve, P, B, A, W)


def add_label(baseA, lab, t0, info, image_maps, ti, sign=1.0):
    for c in info[lab]["tpcs"]:
        baseA[c] += sign * ti(np.asarray(image_maps[(int(lab), int(c))], np.float64), t0)


def place_pool_seq(pool_order, baseA, info, image_maps, actual, w_full, ti):
    tmap = {}
    for lab in pool_order:
        t = scan_label(lab, baseA, info, image_maps, actual, w_full)
        tmap[lab] = t
        add_label(baseA, lab, t, info, image_maps, ti)
    return tmap


def place_pool_v02(pool, s, baseA, info, image_maps, actual, w_full, ti,
                   max_batch=MAX_BATCH, pick=None, commit_hook=None):
    """Energy-banded permutation placement. Mutates baseA; returns (tmap, stats).

    `pick` overrides plain argmin in every scan (see scan_label).
    `commit_hook(lab, t0, baseA)` runs AFTER the label's light is added to
    baseA; return False to reject the commit (light is removed, label returns
    to the pool).  The hook owns its own retry policy (e.g. defer-once)."""
    t_start = time.time()
    pool_set = set(pool)
    committed = {}
    deferred_q = []
    stats = {"n_batches": 0, "n_multi_batches": 0, "n_leaves": 0, "n_scans": 0,
             "batch_sizes": [], "perm_total": 0, "perm_better": 0, "perm_rel_gain_sum": 0.0}

    def try_commit(lab, t0):
        add_label(baseA, lab, t0, info, image_maps, ti)
        if commit_hook is None or commit_hook(lab, t0, baseA):
            committed[lab] = t0
            pool_set.discard(lab)
            return True
        add_label(baseA, lab, t0, info, image_maps, ti, sign=-1.0)
        return False

    while pool_set:
        c1 = max(pool_set, key=lambda l: info[l]["energy"])
        E1 = info[c1]["energy"]
        mates = [l for l in pool_set
                 if l != c1 and info[l]["energy"] > (1.0 - s) * E1
                 and set(info[l]["tpcs"]) & set(info[c1]["tpcs"])]
        mates.sort(key=lambda l: -info[l]["energy"])
        B = [c1] + mates[: max_batch - 1]
        stats["n_batches"] += 1
        stats["batch_sizes"].append(len(B))

        if len(B) == 1:
            t = scan_label(c1, baseA, info, image_maps, actual, w_full, pick=pick)
            stats["n_scans"] += 1
            if not try_commit(c1, t):
                # true deferral: set aside, re-place after the pool drains
                pool_set.discard(c1)
                deferred_q.append(c1)
            continue

        # restricted working arrays over the union of support channels
        uch = {}
        for l in B:
            for c, ch in info[l]["sup"].items():
                uch[c] = np.union1d(uch.get(c, np.array([], dtype=int)), ch)
        act_u = {c: actual[c][ch] for c, ch in uch.items()}
        w_u = {c: w_full[c][ch] for c, ch in uch.items()}
        sb0 = {c: baseA[c][ch].copy() for c, ch in uch.items()}
        P_u, idx = {}, {}
        for l in B:
            for c in info[l]["tpcs"]:
                if c in uch:
                    P_u[(l, c)] = np.asarray(image_maps[(int(l), int(c))], np.float64)[uch[c]]
            for c, ch in info[l]["sup"].items():
                idx[(l, c)] = np.searchsorted(uch[c], ch)

        energy_order = tuple(B)
        best = [None]
        eloss = [None]
        counters = [0, 0]     # leaves, scans

        def uloss(sb):
            if LOSS_MODE != "chi2":
                import v050_loss as _v050
                tot = sum(_v050.point_loss(sb[c], act_u[c], w_u[c], LOSS_MODE)
                          for c in uch)
                return abs(float(tot))      # decision key: drive |L| -> 0
            return float(sum(np.sum(w_u[c] * (np.minimum(sb[c], CLIP) - act_u[c]) ** 2)
                             for c in uch))

        def dfs(remaining, sb, order, ts):
            if not remaining:
                lv = uloss(sb)
                counters[0] += 1
                if best[0] is None or lv < best[0][0]:
                    best[0] = (lv, list(order), dict(ts))
                if tuple(order) == energy_order:
                    eloss[0] = lv
                return
            for nxt in remaining:
                prs, bss, acs, wrs = [], [], [], []
                for c in info[nxt]["sup"]:
                    ii = idx[(nxt, c)]
                    prs.append(P_u[(nxt, c)][ii])
                    bss.append(sb[c][ii])
                    acs.append(act_u[c][ii])
                    wrs.append(w_u[c][ii])
                Pn = np.concatenate(prs)
                Bn = np.concatenate(bss)
                An = np.concatenate(acs)
                Wn = np.concatenate(wrs)
                if LOSS_MODE != "chi2":
                    import v050_loss as _v050
                    curve = _v050.scan_curve(Pn, Bn - An, Wn, mode=LOSS_MODE)
                    t0n = float(np.argmin(np.abs(curve)))
                else:
                    curve = weighted_scan(Pn, Bn - An, Wn)
                    t0n = float(np.argmin(curve)) if pick is None else pick(curve, Pn, Bn, An, Wn)
                counters[1] += 1
                sb2 = {c: a.copy() for c, a in sb.items()}
                for c in info[nxt]["tpcs"]:
                    if (nxt, c) in P_u:
                        sb2[c] += ti(P_u[(nxt, c)], t0n)
                order.append(nxt)
                ts[nxt] = t0n
                dfs([r for r in remaining if r != nxt], sb2, order, ts)
                order.pop()
                del ts[nxt]

        dfs(list(B), sb0, [], {})
        loss_b, order_b, ts_b = best[0]
        stats["n_multi_batches"] += 1
        stats["n_leaves"] += counters[0]
        stats["n_scans"] += counters[1]
        if eloss[0] is not None:
            stats["perm_total"] += 1
            if loss_b < eloss[0] - 1e-9:
                stats["perm_better"] += 1
                stats["perm_rel_gain_sum"] += (eloss[0] - loss_b) / max(eloss[0], 1e-12)
        for l in order_b[:-1]:
            if not try_commit(l, ts_b[l]):
                # rejected: the later ts_b assumed this label's light was in
                # the base -- leave the rest in the pool to be re-batched
                break
        # order_b[-1] returns to the pool for the next round

    for lab in deferred_q:
        t = scan_label(lab, baseA, info, image_maps, actual, w_full, pick=pick)
        stats["n_scans"] += 1
        try_commit(lab, t)

    stats["time_s"] = time.time() - t_start
    return committed, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--event-stride", type=int, default=1)
    ap.add_argument("--event-offset", type=int, default=0)
    ap.add_argument("--s-values", default="0.20,0.30,0.40,0.50")
    args = ap.parse_args()
    s_values = [float(x) for x in args.s_values.split(",") if x.strip()]

    import h5py
    from M5p1.first_stage_matching import load_first_stage_models, run_first_stage_charge_light_matching
    from M5p1.phase25_trial2_valpha_batch import build_default_first_stage_config, _list_events_in_file
    from M5p1 import truth_plotting as truthmod
    from pulse_shapes import timeinterpolation

    def ti(arr, t):
        return timeinterpolation(arr, shift=float(t), baseline=0).astype(np.float64)

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
            hit_t0 = np.asarray(fs.track_stage.hit_t0)
            e_pos = np.clip(np.asarray(fs.event.energy, dtype=np.float64), 0.0, None)
            w_full = 1.0 / (GAIN * np.maximum(actual, 100.0))

            truth = truthmod.extract_event_hit_energy_and_truth_t0(
                h5, h5["charge/events/ref/charge/calib_prompt_hits/ref"], eventid=int(ev_id),
                hits_full=h5["charge/calib_prompt_hits/data"], convert_to_matching_ticks=True)
            truth_t0 = np.asarray(truth["truth_t0"], dtype=np.float64)

            # ---------- per-label info (same construction as the rescue pipeline) ----------
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

            # ---------- variants ----------
            variants = {}
            variants["prod"] = {l: info[l]["t0"] for l in pool}
            bL = base_fixed.copy()
            variants["ctrlL"] = place_pool_seq(sorted(pool), bL, info, image_maps,
                                               actual, w_full, ti)
            bG = base_fixed.copy()
            variants["ctrlG"] = place_pool_seq(sorted(pool, key=lambda l: -info[l]["energy"]),
                                               bG, info, image_maps, actual, w_full, ti)
            s_stats = {}
            for s in s_values:
                key = f"s{int(round(s * 100))}"
                bV = base_fixed.copy()
                tmap, st = place_pool_v02(pool, s, bV, info, image_maps, actual, w_full, ti)
                variants[key] = tmap
                sizes = st.pop("batch_sizes")
                st["batch_size_hist"] = {str(k): sizes.count(k) for k in sorted(set(sizes))}
                s_stats[key] = st

            # ---------- scoring ----------
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
                row_variants[name] = {"good": g, "ok": o,
                                      "good_pool": gp, "ok_pool": op,
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
                "variants": row_variants, "s_stats": s_stats, "pool_labels": pool_rows,
                "total_energy": float(e_pos.sum()),
                "truth_energy": float(e_pos[np.isfinite(truth_t0)].sum()),
                "t_event_s": time.time() - t_ev,
            }) + "\n")
            fout.flush()
            accs = " ".join(f"{n}={v['acc']:.4f}" for n, v in row_variants.items()
                            if v["acc"] is not None)
            print(f"done {os.path.basename(file_path)} ev{ev_id} in {time.time()-t_ev:.0f}s | "
                  f"pool {len(pool)} | {accs}", flush=True)
        except Exception as exc:
            import traceback
            print(f"ERROR {file_path} ev{ev_id}: {exc}", flush=True)
            traceback.print_exc()
    fout.close()
    print("worker complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
