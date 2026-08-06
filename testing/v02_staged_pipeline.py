"""v0.2 + re-attached Phase 2 + V2 (stage-2.5 light rescue) test.

Per event, THREE staged passes (Phase 2 then V2, cohort-tracked), all from the
same first stage:
  base   production v0.1 backbone (in-job baseline — clustering fix made the
         older 39-event references stale)
  v02    plain s=0.60 energy-banded permutation backbone
  stack  v02 with cos-gated scans + trigger defer gate + swap pass + final
         trigger rescue pass (the refine-job composition)
The v0.2 placements are written into fs (hit_t0 + base_image) and their t0s
appended to the per-TPC candidate lists before building each namespace;
hit_t0/base_image are restored from pristine copies between passes.
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

from v02_ordering_pipeline import (PEAK, GAIN, CLIP, SEARCH, CUT, acc_sums,
                                   add_label, place_pool_v02)
from v02_refine_pipeline import (make_cos_pick, make_defer_hook, swap_pass, rescue_pass)
from overflow_rescue_pipeline import staged_run

S_BAND = 0.60


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--event-stride", type=int, default=1)
    ap.add_argument("--event-offset", type=int, default=0)
    ap.add_argument("--variants", default="v02,stack")
    args = ap.parse_args()
    want = [v.strip() for v in args.variants.split(",") if v.strip()]

    import h5py
    from M5p1.first_stage_matching import load_first_stage_models, run_first_stage_charge_light_matching
    from M5p1.first_stage_matching.track_stage import append_candidate_t0
    from M5p1.phase25_trial2_valpha_batch import (
        _build_namespace, _run_phase2, build_default_first_stage_config,
        build_default_v2_config, _list_events_in_file)
    from M5p1 import phase25_trial2_v2_light_rescue as v2mod
    from v11_phased_matching import snapshot_backbone_hits_v11, verify_backbone_hits_unchanged_v11
    from M5p1 import truth_plotting as truthmod
    from pulse_shapes import timeinterpolation

    def ti(arr, t):
        return timeinterpolation(arr, shift=float(t), baseline=0).astype(np.float64)

    cfg = build_default_first_stage_config()
    cfg.prediction.min_cluster_energy_mev = 50.0
    v2cfg = build_default_v2_config(n_outer_passes=1, light_max_total_moves=24,
                                    light_max_moves_per_tpc=3, device_policy="auto", verbose=False)
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
            base_img = np.asarray(fs.track_stage.base_image)
            hit_t0_orig = hit_t0.copy()
            base_img_orig = base_img.copy()
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

            base_fixed = np.zeros_like(actual)
            for lab in showers + multi:
                add_label(base_fixed, lab, info[lab]["t0"], info, image_maps, ti)

            def build_backbone(name):
                b = base_fixed.copy()
                if name == "v02":
                    tmap, _ = place_pool_v02(pool, S_BAND, b, info, image_maps, actual,
                                             w_full, ti)
                elif name in ("stack", "stack2"):
                    st = {}
                    hook = make_defer_hook(info, image_maps, actual, veto_mask, ti, st)
                    pick = make_cos_pick("gate" if name == "stack" else "soft")
                    tmap, _ = place_pool_v02(pool, S_BAND, b, info, image_maps, actual,
                                             w_full, ti, pick=pick, commit_hook=hook)
                    swap_pass(tmap, b, info, image_maps, actual, w_full, ti)
                    rescue_pass(tmap, b, info, image_maps, actual, w_full, veto_mask, ti)
                elif name == "swap":
                    tmap, _ = place_pool_v02(pool, S_BAND, b, info, image_maps, actual,
                                             w_full, ti)
                    swap_pass(tmap, b, info, image_maps, actual, w_full, ti)
                else:
                    raise ValueError(name)
                return tmap, b

            # ---------- staged baseline on the pristine production state ----------
            ns0 = _build_namespace(fs, h5, cfg)
            metrics = {"base": staged_run(ns0, v2mod, v2cfg, _run_phase2,
                                          snapshot_backbone_hits_v11,
                                          verify_backbone_hits_unchanged_v11,
                                          truth_t0, e_pos, "base")}

            for name in want:
                tmap, b = build_backbone(name)
                hit_t0[:] = hit_t0_orig
                base_img[:] = base_img_orig
                for lab, t in tmap.items():
                    hit_t0[labels_global == lab] = np.float32(t)
                base_img[:] = b.astype(base_img.dtype)
                ns = _build_namespace(fs, h5, cfg)
                for lab, t in tmap.items():
                    if abs(t - info[lab]["t0"]) <= CUT:
                        continue
                    for c in info[lab]["tpcs"]:
                        append_candidate_t0(ns["t0Candidates"][c], t, min_sep=2, max_t0=SEARCH)
                        while len(ns["flash_cluster_received_by_tpc"][c]) < len(ns["t0Candidates"][c]):
                            ns["flash_cluster_received_by_tpc"][c].append(False)
                metrics[name] = staged_run(ns, v2mod, v2cfg, _run_phase2,
                                           snapshot_backbone_hits_v11,
                                           verify_backbone_hits_unchanged_v11,
                                           truth_t0, e_pos, name)
                hit_t0[:] = hit_t0_orig
                base_img[:] = base_img_orig

            fout.write(json.dumps({
                "file": os.path.basename(file_path), "event": int(ev_id),
                "n_shower": len(showers), "n_multi": len(multi), "n_pool": len(pool),
                "metrics": metrics,
                "total_energy": float(e_pos.sum()),
                "truth_energy": float(e_pos[np.isfinite(truth_t0)].sum()),
                "t_event_s": time.time() - t_ev,
            }) + "\n")
            fout.flush()
            line = " | ".join(f"{n}: p1={m['p1_acc']:.4f} v2={m['v2_acc']:.4f}"
                              for n, m in metrics.items() if m["p1_acc"] is not None)
            print(f"done {os.path.basename(file_path)} ev{ev_id} in {time.time()-t_ev:.0f}s | "
                  f"{line}", flush=True)
        except Exception as exc:
            import traceback
            print(f"ERROR {file_path} ev{ev_id}: {exc}", flush=True)
            traceback.print_exc()
    fout.close()
    print("worker complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
