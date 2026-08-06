"""Overflow-cut study: can a single-tick support-channel overflow threshold
(e.g. 20 sigma) flag wrongly-placed phase-1 backbone clusters?

For every phase-1 backbone label, replays the sequential placement (base =
every label placed before it, each at its final t0) exactly as a deployed
"check right after placement" would see it, then records

    max over the label's fit channels (support channels for tracks,
    all unsaturated channels for shower/vertex labels) of
    (pred - actual) / sigma   at tick round(t0) + 105  (the pulse peak)

with sigma^2 = max(actual, 100) (Poisson proxy; no variance model installed),
plus the MC-truth verdict for the label (wrong = |t0 - truth| > cut ticks).

Writes one JSON line per label so partial results survive crashes.
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

import numpy as np  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--out", required=True, help="output .jsonl path")
    ap.add_argument("--event-stride", type=int, default=1)
    ap.add_argument("--event-offset", type=int, default=0)
    ap.add_argument("--max-events-per-file", type=int, default=0)
    ap.add_argument("--residual-cut-ticks", type=float, default=10.0)
    ap.add_argument("--peak-offset", type=int, default=105)
    ap.add_argument("--sigma-floor", type=float, default=100.0)
    ap.add_argument("--predict-min-energy-mev", type=float, default=50.0)
    args = ap.parse_args()

    import h5py
    from M5p1.first_stage_matching import load_first_stage_models, run_first_stage_charge_light_matching
    from M5p1.first_stage_matching.track_stage import fast_stack_images
    from M5p1.phase25_trial2_valpha_batch import build_default_first_stage_config, _list_events_in_file
    from M5p1 import truth_plotting as tp
    from pulse_shapes import timeinterpolation

    cfg = build_default_first_stage_config()
    # prune Phase-3-only clusters from prediction (backbone always predicted);
    # we stop after the track stage, so Phase 2/3 image needs don't apply.
    cfg.prediction.min_cluster_energy_mev = float(args.predict_min_energy_mev)

    print("loading models ...", flush=True)
    models = load_first_stage_models(cfg)

    work = []
    for fp in args.files:
        try:
            ev_ids = _list_events_in_file(fp)
        except Exception as exc:
            print(f"SKIP unreadable {fp}: {exc}", flush=True)
            continue
        if int(args.max_events_per_file) > 0:
            ev_ids = ev_ids[: int(args.max_events_per_file)]
        work.extend((fp, int(e)) for e in ev_ids)
    work = work[int(args.event_offset):: max(1, int(args.event_stride))]
    print(f"{len(work)} events for this worker", flush=True)

    cut = float(args.residual_cut_ticks)
    n_rows = 0
    with open(args.out, "w") as fout:
        h5_cache: dict[str, h5py.File] = {}
        for file_path, ev_id in work:
            t_ev = time.time()
            try:
                h5 = h5_cache.get(file_path)
                if h5 is None:
                    h5 = h5py.File(file_path, "r")
                    h5_cache[file_path] = h5

                fs = run_first_stage_charge_light_matching(
                    h5=h5, event_id=int(ev_id), config=cfg, models=models, verbose=False)

                labels_global = np.asarray(fs.clustering.labels_global)
                split_index = int(fs.clustering.split_index)
                label_info = fs.clustering.label_info
                image_maps = fs.prediction.image_maps
                cluster_to_tpcs = fs.prediction.cluster_to_tpcs
                support_cache = fs.prediction.cluster_channel_support_cache
                veto_mask = np.asarray(fs.prediction.saturated_channel_cache["veto_mask"], dtype=bool)
                actual = np.asarray(fs.event.full_light_waveform, dtype=np.float64)
                n_light_tpc, n_ch, n_ticks = actual.shape
                hit_t0 = np.asarray(fs.track_stage.hit_t0, dtype=np.float64)
                e_pos = np.clip(np.asarray(fs.event.energy, dtype=np.float64), 0.0, None)

                truth = tp.extract_event_hit_energy_and_truth_t0(
                    h5,
                    h5["charge/events/ref/charge/calib_prompt_hits/ref"],
                    eventid=int(ev_id),
                    hits_full=h5["charge/calib_prompt_hits/data"],
                    convert_to_matching_ticks=True,
                )
                truth_t0 = np.asarray(truth["truth_t0"], dtype=np.float64)

                base = np.zeros_like(actual)
                for lab in range(split_index):          # exact placement order
                    tpcs = sorted(
                        int(t) for t in cluster_to_tpcs.get(int(lab), [])
                        if int(t) < n_light_tpc and (int(lab), int(t)) in image_maps
                    )
                    if not tpcs:
                        continue
                    members = np.flatnonzero(labels_global == lab)
                    if members.size == 0:
                        continue
                    t0v = hit_t0[members]
                    t0v = t0v[np.isfinite(t0v) & (t0v >= 0)]
                    if t0v.size == 0:
                        continue
                    t0 = float(np.median(t0v))

                    tpcs_arr = np.asarray(tpcs, dtype=np.int32)
                    stacked = fast_stack_images(image_maps, int(lab), tpcs_arr)
                    shifted = timeinterpolation(stacked, shift=float(t0), baseline=0).astype(np.float64)
                    base[tpcs_arr] += shifted.reshape(tpcs_arr.size, n_ch, -1)

                    tick = int(round(t0)) + int(args.peak_offset)
                    if not (0 <= tick < n_ticks):
                        continue

                    lab_type = str(label_info.get(int(lab), {}).get("type", "track")).lower()
                    own = shifted.reshape(tpcs_arr.size, n_ch, -1)[:, :, tick] if tick < shifted.shape[-1] else None
                    max_ov = None
                    max_ov_all = None   # same probe over ALL unsaturated channels of the TPC
                    n_fit_ch = 0
                    tpc_dump = []  # per probed TPC: full 120-ch [own, base, actual] + masks
                    for ti, tpc in enumerate(tpcs):
                        entry = support_cache.get((int(lab), int(tpc))) if lab_type == "track" else None
                        if entry is not None:
                            ch = np.asarray(entry["channel_indices"], dtype=int)
                        else:
                            ch = np.arange(n_ch)
                        ch = ch[~veto_mask[int(tpc), ch]]      # saturation veto
                        if ch.size == 0:
                            continue
                        a = actual[int(tpc), ch, tick]
                        p = base[int(tpc), ch, tick]           # includes this cluster (just added)
                        ov = (p - a) / np.sqrt(np.maximum(a, float(args.sigma_floor)))
                        m = float(np.max(ov))
                        max_ov = m if max_ov is None else max(max_ov, m)
                        n_fit_ch += int(ch.size)
                        # full-channel dump for multi-channel trigger studies
                        a_full = actual[int(tpc), :, tick]
                        p_full = base[int(tpc), :, tick]
                        o_full = own[ti] if own is not None else np.zeros(n_ch)
                        tpc_dump.append({
                            "tpc": int(tpc),
                            "own": [round(float(v), 1) for v in o_full],
                            "base": [round(float(v), 1) for v in (p_full - o_full)],
                            "actual": [round(float(v), 1) for v in a_full],
                            "support": [int(v) for v in ch],
                            "veto": [int(v) for v in np.flatnonzero(veto_mask[int(tpc)])],
                        })
                        ch_all = np.flatnonzero(~veto_mask[int(tpc)])
                        if ch_all.size:
                            ov_all = (p_full[ch_all] - a_full[ch_all]) / np.sqrt(
                                np.maximum(a_full[ch_all], float(args.sigma_floor)))
                            m_all = float(np.max(ov_all))
                            max_ov_all = m_all if max_ov_all is None else max(max_ov_all, m_all)
                    if max_ov is None:
                        continue

                    tt = truth_t0[members]
                    ww = e_pos[members]
                    ok = np.isfinite(tt)
                    if not ok.any():
                        truth_med, coherence, wrong = None, 0.0, None
                    else:
                        order = np.argsort(tt[ok])
                        cw = np.cumsum(ww[ok][order])
                        truth_med = float(tt[ok][order][np.searchsorted(cw, 0.5 * cw[-1])])
                        coherence = float(
                            np.sum(ww[ok][np.abs(tt[ok] - truth_med) <= cut]) / max(np.sum(ww[ok]), 1e-9))
                        wrong = bool(abs(t0 - truth_med) > cut)

                    fout.write(json.dumps({
                        "file": os.path.basename(file_path), "event": int(ev_id),
                        "label": int(lab), "type": lab_type,
                        "n_tpcs": len(tpcs), "tpcs": [int(t) for t in tpcs],
                        "n_fit_ch": int(n_fit_ch),
                        "n_hits": int(members.size), "energy_mev": float(e_pos[members].sum()),
                        "t0": float(t0), "truth_t0": truth_med,
                        "coherence": coherence, "wrong": wrong,
                        "max_overflow_sigma": float(max_ov),
                        "max_overflow_allch_sigma": float(max_ov_all) if max_ov_all is not None else None,
                        "tpc_channels": tpc_dump,
                    }) + "\n")
                    n_rows += 1
                fout.flush()
                # per-hit dump for hit-level ground truth (labels are placement-ordered)
                hits_dir = os.path.join(os.path.dirname(args.out), "hits")
                os.makedirs(hits_dir, exist_ok=True)
                ftag = os.path.basename(file_path).split(".")[-3]
                np.savez_compressed(
                    os.path.join(hits_dir, f"{ftag}_ev{ev_id}.npz"),
                    labels=labels_global.astype(np.int32),
                    tpc=np.asarray(fs.event.hit_tpc_id, dtype=np.int16),
                    e=np.asarray(fs.event.energy, dtype=np.float32),
                    reco_t0=hit_t0.astype(np.float32),
                    truth_t0=truth_t0.astype(np.float32),
                    split_index=np.int32(split_index),
                )
                print(f"done {os.path.basename(file_path)} ev{ev_id} in {time.time() - t_ev:.0f}s "
                      f"(rows so far {n_rows})", flush=True)
            except Exception as exc:
                import traceback
                print(f"ERROR {file_path} ev{ev_id}: {exc}", flush=True)
                traceback.print_exc()

    print(f"worker complete: {n_rows} rows -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
