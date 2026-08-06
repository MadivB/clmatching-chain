"""Dump per-event display state (post-phase-1) for the overflow example TPCs."""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.abspath(os.path.dirname(__file__))
BASE_DIR = os.path.abspath(os.path.join(_HERE, "..", "CLMatching_AlphaRelease"))
sys.path.insert(0, BASE_DIR)
sys.path.insert(1, os.path.join(BASE_DIR, "NewMLSection"))
sys.path.insert(2, os.path.join(BASE_DIR, "M5p1"))

import numpy as np
import h5py

from M5p1.first_stage_matching import load_first_stage_models, run_first_stage_charge_light_matching
from M5p1.phase25_trial2_valpha_batch import build_default_first_stage_config
from M5p1 import truth_plotting as tp

DATA = ("/global/cfs/cdirs/dunepro/people/abooth/nd-production/output/MiniProdN5/"
        "run-ndlar-flow/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift/FLOW/0000000")
EXAMPLES = [
    {"file": "MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.0000002.FLOW.hdf5", "event": 6,  "labels": [141, 24]},
    {"file": "MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.0000001.FLOW.hdf5", "event": 12, "labels": [210, 201]},
    {"file": "MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.0000002.FLOW.hdf5", "event": 7,  "labels": [50, 95]},
]
OUT = os.path.join(_HERE, "display_dump")
os.makedirs(OUT, exist_ok=True)

cfg = build_default_first_stage_config()
cfg.prediction.min_cluster_energy_mev = 50.0
models = load_first_stage_models(cfg)

for ex in EXAMPLES:
    fp = os.path.join(DATA, ex["file"])
    ev = int(ex["event"])
    with h5py.File(fp, "r") as h5:
        fs = run_first_stage_charge_light_matching(h5=h5, event_id=ev, config=cfg, models=models, verbose=False)
        truth = tp.extract_event_hit_energy_and_truth_t0(
            h5, h5["charge/events/ref/charge/calib_prompt_hits/ref"], eventid=ev,
            hits_full=h5["charge/calib_prompt_hits/data"], convert_to_matching_ticks=True)
        label_tpcs = {int(l): sorted(int(t) for t in fs.prediction.cluster_to_tpcs.get(int(l), []))
                      for l in ex["labels"]}
        tag = f"{ex['file'].split('.')[-3]}_ev{ev}"
        np.savez_compressed(
            os.path.join(OUT, f"{tag}.npz"),
            x=fs.event.x, y=fs.event.y, z=fs.event.z, e=fs.event.energy,
            tpc=fs.event.hit_tpc_id, hit_refs=fs.event.hit_refs,
            hit_t0=fs.track_stage.hit_t0, labels=np.asarray(fs.clustering.labels_global),
            truth_t0=np.asarray(truth["truth_t0"], dtype=np.float64),
            base_image=np.asarray(fs.track_stage.base_image, dtype=np.float32),
            full_light=np.asarray(fs.event.full_light_waveform, dtype=np.float32),
            label_tpcs=json.dumps(label_tpcs), src_file=fp, event=ev,
        )
        print(f"dumped {tag}: labels->tpcs {label_tpcs}", flush=True)
print("all dumps done", flush=True)
