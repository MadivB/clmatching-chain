"""Census of WRONG backbone placements across phase 1 AND phase 2 (incl. the
iterative band/greedy placement), with decision-time loss landscapes.

Why a replay?  Phase 2 places clusters one at a time: every placement changes the
base image seen by the next one.  The algorithm only logs a loss curve for its
full-scan stages -- the ITERATIVE (greedy_discrete) stage logs none.  So we
re-play the recorded assignment order from the post-phase-1 base and, right
before each cluster is added, evaluate the matcher's OWN loss
(_scan_best_shift_multi on the support channels, weighted by full_light_std)
over all t0.  That is exactly the landscape the algorithm faced at that moment.

For each wrong placement we store both:
  * decision-time curve  -- base as it was when the cluster was placed
  * hindsight curve      -- final base minus the cluster's own contribution
Their difference separates "was misled by missing context" (iterative ordering)
from "the light genuinely prefers the wrong t0" (intrinsic).
We also store the discrete t0-candidate grid: iterative clusters may ONLY be
placed on it, so a truth outside the grid is unwinnable by construction.

Outputs: rows_worker*.jsonl (every wrong placement) + dumps/{tag}_ev{id}.npz.
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
import h5py

CUT = 10.0
SEARCH = 800
CLIP = 60780.0
DUMP_MIN_MEV = 20.0     # dump display state for wrong placements above this energy


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--event-stride", type=int, default=1)
    ap.add_argument("--event-offset", type=int, default=0)
    ap.add_argument("--with-iterative", action="store_true",
                    help="also run phase 3 (run_small_cluster_matrix_phase_v11) = the "
                         "ITERATIVE band-matrix small-cluster placement, which the "
                         "backbone-only driver skips entirely")
    ap.add_argument("--predict-all", action="store_true",
                    help="predict images for ALL clusters (min_cluster_energy_mev=None). "
                         "REQUIRED to exercise phase-2's ITERATIVE single-TPC path: in "
                         "backbone-only mode sub-50 MeV clusters get no image, so the "
                         "iterative band/greedy stage has nothing to place.")
    args = ap.parse_args()
    os.makedirs(args.dump_dir, exist_ok=True)

    from M5p1.first_stage_matching import load_first_stage_models, run_first_stage_charge_light_matching
    from M5p1.phase25_trial2_valpha_batch import (
        _build_namespace, build_default_first_stage_config, _list_events_in_file,
        V11_LARGE_CLUSTER_ENERGY_MEV, V11_MINIMUM_ITERATIVE_ENERGY_MEV,
        V11_PRIMARY_ASSIGNMENT_IMPROVEMENT_EPS, ADC_CLIP, T0_RESOLUTION)
    from M5p1.v11_phased_matching import (run_large_cluster_scan_phase_v11,
                                          run_small_cluster_matrix_phase_v11)
    from M5p1.v3_2_global_matching import (
        _stack_cluster_images, _scan_best_shift_multi, _shift_block)
    from M5p1 import truth_plotting as tp
    from M5p1.v12_saturation_mask import materialize_support_entry_v12

    cfg = build_default_first_stage_config()
    cfg.prediction.min_cluster_energy_mev = None if args.predict_all else 50.0
    print(f"min_cluster_energy_mev = {cfg.prediction.min_cluster_energy_mev} "
          f"(iterative path {'ACTIVE' if args.predict_all else 'inactive'})", flush=True)
    print("loading models ...", flush=True)
    models = load_first_stage_models(cfg)

    work = []
    for fp in args.files:
        work.extend((fp, int(e)) for e in _list_events_in_file(fp))
    work = work[int(args.event_offset):: max(1, int(args.event_stride))]
    print(f"{len(work)} events for this worker", flush=True)

    fout = open(args.out, "w")
    h5_cache: dict = {}
    for file_path, ev_id in work:
        t_ev = time.time()
        try:
            h5 = h5_cache.get(file_path)
            if h5 is None:
                h5 = h5py.File(file_path, "r")
                h5_cache[file_path] = h5
            fs = run_first_stage_charge_light_matching(h5=h5, event_id=int(ev_id), config=cfg,
                                                       models=models, verbose=False)
            ftag = os.path.basename(file_path).split(".")[-3]
            ns = _build_namespace(fs, h5, cfg)

            labels_global = np.asarray(ns["labels_global"])
            split_index = int(fs.clustering.split_index)
            e_pos = np.clip(np.asarray(ns["Eset"], dtype=np.float64), 0.0, None)
            hit_tpc = np.asarray(ns["hitTPCid"])
            image_maps = ns["imageMaps"]
            support_cache = ns["cluster_channel_support_cache"]
            sat_cache = ns["saturated_channel_cache"]
            actual = np.asarray(ns["fullLightWaveform"], dtype=np.float32)
            lstd = np.asarray(ns["fullLightStd"], dtype=np.float32)

            truth = tp.extract_event_hit_energy_and_truth_t0(
                h5, h5["charge/events/ref/charge/calib_prompt_hits/ref"], eventid=int(ev_id),
                hits_full=h5["charge/calib_prompt_hits/data"], convert_to_matching_ticks=True)
            truth_t0 = np.asarray(truth["truth_t0"], dtype=np.float64)

            def lab_truth(mask):
                tt, ww = truth_t0[mask], e_pos[mask]
                ok = np.isfinite(tt)
                if not ok.any():
                    return None
                o = np.argsort(tt[ok])
                cw = np.cumsum(ww[ok][o])
                return float(tt[ok][o][np.searchsorted(cw, 0.5 * cw[-1])])

            # ---------- phase-1 snapshot ----------
            ts1 = np.asarray(ns["hit_timestamps"], dtype=np.float64).copy()
            base_p1 = np.asarray(ns["baseImage"], dtype=np.float32).copy()
            cand_p1 = [list(map(int, c)) for c in ns["t0Candidates"]]

            # ---------- phase 2 (capture the log + scan losses) ----------
            cluster_energies = {int(c): float(e_pos[labels_global == c].sum())
                                for c in ns["cluster_labels"]}
            ns["cluster_energies"] = cluster_energies
            (baseImage, hit_timestamps, t0Candidates, assignment_info, unassigned_by_tpc,
             assign_log, scan_updates, p2stats, active_tpcs, iter_single_tpc,
             pruned_iter) = run_large_cluster_scan_phase_v11(
                cluster_labels=ns["cluster_labels"], cluster_to_tpcs=ns["cluster_to_tpcs"],
                image_maps=image_maps, base_image=ns["baseImage"],
                full_light_waveform=ns["fullLightWaveform"], full_light_std=ns["fullLightStd"],
                channel_support_cache=support_cache, labels_global=ns["labels_global"],
                hit_timestamps=ns["hit_timestamps"], t0_candidates=ns["t0Candidates"],
                assignment_info=ns["assignment_info"], unassigned_by_tpc=ns["unassigned_by_tpc"],
                cluster_energies=cluster_energies,
                large_cluster_energy_mev=V11_LARGE_CLUSTER_ENERGY_MEV,
                minimum_iterative_energy_mev=V11_MINIMUM_ITERATIVE_ENERGY_MEV,
                search_range=SEARCH, adc_clip=ADC_CLIP, collect_scan_losses=True,
                assignment_improvement_eps=V11_PRIMARY_ASSIGNMENT_IMPROVEMENT_EPS,
                flash_seed_t0s_by_tpc=ns["v8_1_flash_seed_t0s_by_tpc"],
                flash_seed_t0_resolution=int(T0_RESOLUTION),
                flash_cluster_received_by_tpc=ns["flash_cluster_received_by_tpc"],
                leftover_absorption_context=ns.get("v11_1_leftover_absorption_context"),
                saturated_channel_cache=ns["saturated_channel_cache"])
            ts2 = np.asarray(hit_timestamps, dtype=np.float64).copy()
            # cluster ids handled by the ITERATIVE single-TPC band/greedy path
            iter_cids = {int(c) for lst in iter_single_tpc.values() for c in lst}

            # ---------- phase 3: the ITERATIVE band-matrix small-cluster pass ----------
            # NOTE: run_large_cluster_scan_phase_v11 (phase 2) only does the primary
            # full scan -- it computes iterative_single_tpc but never places them.  The
            # backbone-only driver skips this phase entirely, so we call it explicitly.
            iter_log: list = []
            if args.with_iterative and iter_cids:
                (baseImage, hit_timestamps, t0Candidates, assignment_info, unassigned_by_tpc,
                 iter_log, iter_scan_updates, p3stats) = run_small_cluster_matrix_phase_v11(
                    active_cluster_tpcs=active_tpcs, iterative_single_tpc=iter_single_tpc,
                    pruned_iterative_clusters=pruned_iter, image_maps=image_maps,
                    base_image=baseImage, full_light_waveform=ns["fullLightWaveform"],
                    full_light_std=ns["fullLightStd"], channel_support_cache=support_cache,
                    labels_global=ns["labels_global"], hit_timestamps=hit_timestamps,
                    t0_candidates=t0Candidates, assignment_info=assignment_info,
                    unassigned_by_tpc=unassigned_by_tpc, cluster_energies=cluster_energies,
                    search_range=SEARCH, adc_clip=ADC_CLIP, collect_scan_losses=True,
                    leftover_absorption_context=ns.get("v11_1_leftover_absorption_context"),
                    saturated_channel_cache=ns["saturated_channel_cache"])
                scan_updates.update(iter_scan_updates)
            ts3 = np.asarray(hit_timestamps, dtype=np.float64).copy()
            base_final = np.asarray(baseImage, dtype=np.float32).copy()

            # ---------- faithful decision-time scan (matcher's own loss) ----------
            def scan_curve(clusterid, tpcs, base_img):
                tpcs = np.asarray(tpcs, dtype=int)
                try:
                    cb = _stack_cluster_images(image_maps, int(clusterid), tpcs)
                except KeyError:
                    return None, None
                fc, fb, fa, fe = [], [], [], []
                for bi, tpcid in enumerate(tpcs.tolist()):
                    se = materialize_support_entry_v12(
                        waveform=np.asarray(cb[bi], dtype=np.float32), tpcid=int(tpcid),
                        base_entry=None if support_cache is None else
                        support_cache.get((int(clusterid), int(tpcid))),
                        saturated_channel_cache=sat_cache)
                    ci = np.asarray(se["channel_indices"], dtype=np.int32)
                    if ci.size == 0:
                        continue
                    fc.append(np.asarray(cb[bi, ci, :], dtype=np.float32))
                    fb.append(np.asarray(base_img[int(tpcid), ci, :], dtype=np.float32))
                    fa.append(np.asarray(actual[int(tpcid), ci, :], dtype=np.float32))
                    fe.append(np.asarray(lstd[int(tpcid), ci, :], dtype=np.float32))
                if not fc:
                    return None, None
                bt, _, curve = _scan_best_shift_multi(
                    np.concatenate(fc, 0), np.concatenate(fb, 0), np.concatenate(fa, 0),
                    np.concatenate(fe, 0), search_range=SEARCH, adc_clip=CLIP,
                    return_curve=True)
                return curve, int(bt)

            def add_to_base(base_img, clusterid, tpcs, t0):
                tpcs = np.asarray(tpcs, dtype=int)
                try:
                    cb = _stack_cluster_images(image_maps, int(clusterid), tpcs)
                except KeyError:
                    return
                for bi, tpcid in enumerate(tpcs.tolist()):
                    sh = _shift_block(np.asarray(cb[bi], dtype=np.float32), int(t0))
                    base_img[int(tpcid)] = np.clip(base_img[int(tpcid)] + sh, None, CLIP)

            # which phase-2 placements are wrong?
            rows, dumps = [], {}
            base_replay = base_p1.copy()
            n_replay_ok, n_replay_tot = 0, 0
            # replay phase 2 then phase 3 in the exact order they were placed
            for entry in list(assign_log) + list(iter_log):
                cid = int(entry["clusterid"])
                if not entry.get("assigned"):
                    continue
                t0 = float(entry.get("t0", -1))
                tpcs = [int(t) for t in entry.get("tpcs", [])]
                m = labels_global == cid
                if not m.any() or t0 < 0:
                    continue
                tr = lab_truth(m)
                stage = str(entry.get("label", entry.get("mode", "?")))
                is_iter = (cid in iter_cids) or stage in ("greedy_discrete", "band_matrix")
                # decision-time curve BEFORE adding this cluster
                curve, bt = scan_curve(cid, tpcs, base_replay)
                # validate the replay against the algorithm's own logged scan minimum
                # (stage-agnostic: whenever it recorded one)
                su = scan_updates.get(cid, {})
                ref = su.get("best_t0_scan")
                if curve is not None and ref is not None:
                    n_replay_tot += 1
                    if abs(int(np.argmin(curve)) - int(ref)) <= 2:
                        n_replay_ok += 1
                wrong = (tr is not None) and (abs(t0 - tr) > CUT)
                e_lab = float(e_pos[m].sum())
                ph = "phase3_iterative" if is_iter else "phase2"
                ts_now = ts3 if is_iter else ts2
                if wrong:
                    rows.append({"file": ftag, "event": int(ev_id), "phase": ph,
                                 "label": cid, "tpcs": tpcs, "stage": stage,
                                 "mode": str(entry.get("mode", "")), "iterative": is_iter,
                                 "energy": e_lab, "t0": t0, "truth": tr,
                                 "n_hits": int(m.sum()),
                                 "wrong_e": float(e_pos[m & np.isfinite(truth_t0) &
                                                        (np.abs(ts_now - truth_t0) > CUT)].sum())})
                    # dump for figures: big clusters, and ANY iterative one (they are
                    # small by construction, <=50 MeV, but are the case we want to see)
                    if (e_lab >= DUMP_MIN_MEV or (is_iter and e_lab >= 2.0)) and curve is not None:
                        hind, _ = None, None
                        bh = base_final.copy()
                        for tpcid in tpcs:      # remove own contribution -> hindsight
                            try:
                                cb = _stack_cluster_images(image_maps, cid,
                                                           np.asarray([tpcid], dtype=int))
                                sh = _shift_block(np.asarray(cb[0], dtype=np.float32), int(round(t0)))
                                bh[int(tpcid)] = np.clip(bh[int(tpcid)] - sh, 0.0, None)
                            except KeyError:
                                pass
                        hind, _ = scan_curve(cid, tpcs, bh)
                        dumps[f"L{cid}_curve_dec"] = np.asarray(curve, np.float32)
                        if hind is not None:
                            dumps[f"L{cid}_curve_hind"] = np.asarray(hind, np.float32)
                        # candidates ENTERING phase 2 (what existed when placed) and the
                        # FINAL grid (phase 2 can expand it via _try_t0_expansion)
                        grid = sorted({int(g) for t in tpcs for g in cand_p1[int(t)]})
                        gridf = sorted({int(g) for t in tpcs for g in t0Candidates[int(t)]})
                        dumps[f"L{cid}_grid"] = np.asarray(grid, np.int32)
                        dumps[f"L{cid}_gridfinal"] = np.asarray(gridf, np.int32)
                        dumps[f"L{cid}_meta"] = np.asarray(
                            [cid, t0, tr if tr is not None else -1, e_lab,
                             1 if is_iter else 0, tpcs[0] if tpcs else -1], np.float64)
                add_to_base(base_replay, cid, tpcs, int(round(t0)))

            # ---------- phase-1 backbone wrong placements ----------
            for lab in range(split_index):
                m = labels_global == lab
                if not m.any():
                    continue
                tv = ts1[m]
                fin = np.isfinite(tv) & (tv >= 0)
                if not fin.any():
                    continue
                t0 = float(np.median(tv[fin]))
                tr = lab_truth(m)
                if tr is None or abs(t0 - tr) <= CUT:
                    continue
                tpcs = sorted({int(t) for t in hit_tpc[m]})
                rows.append({"file": ftag, "event": int(ev_id), "phase": "phase1",
                             "label": int(lab), "tpcs": tpcs, "stage": "backbone_placement",
                             "mode": "", "iterative": False,
                             "energy": float(e_pos[m].sum()), "t0": t0, "truth": tr,
                             "n_hits": int(m.sum()),
                             "wrong_e": float(e_pos[m & np.isfinite(truth_t0) &
                                                    (np.abs(ts1 - truth_t0) > CUT)].sum())})

            if dumps:
                dumps.update({
                    "x": np.asarray(ns["xset"], np.float32), "y": np.asarray(ns["yset"], np.float32),
                    "z": np.asarray(ns["zset"], np.float32), "e": e_pos.astype(np.float32),
                    "tpc": hit_tpc.astype(np.int16), "labels": labels_global.astype(np.int32),
                    "ts1": ts1.astype(np.float32), "ts2": ts2.astype(np.float32),
                    "ts3": ts3.astype(np.float32),
                    "truth_t0": truth_t0.astype(np.float32),
                })
                np.savez_compressed(os.path.join(args.dump_dir, f"{ftag}_ev{ev_id}.npz"), **dumps)

            for r in rows:
                fout.write(json.dumps(r) + "\n")
            fout.flush()
            n1 = sum(1 for r in rows if r["phase"] == "phase1")
            n2 = sum(1 for r in rows if r["phase"] == "phase2")
            ni = sum(1 for r in rows if r["phase"] == "phase3_iterative")
            rep = f"{n_replay_ok}/{n_replay_tot}" if n_replay_tot else "n/a"
            print(f"done {ftag} ev{ev_id} in {time.time()-t_ev:.0f}s | wrong: "
                  f"p1={n1} p2={n2} (iterative {ni}) | replay-validate {rep}", flush=True)
        except Exception as exc:
            import traceback
            print(f"ERROR {file_path} ev{ev_id}: {exc}", flush=True)
            traceback.print_exc()
    fout.close()
    print("worker complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
