"""Profile the stage-1 light prediction on dump events: all clusters >= E_min per light TPC, with/without image-map materialisation, batch sizes."""
import os, sys, glob, time, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.abspath(os.path.join(HERE, "..", "CLMatching_AlphaRelease"))
for p in (BASE, os.path.join(BASE, "M5p1"), os.path.join(BASE, "NewMLSection"), HERE): sys.path.insert(0, p)
from M5p1.first_stage_matching import load_first_stage_models
from M5p1.first_stage_matching.config import FirstStageConfig
from M5p1.first_stage_matching.streaming_prediction import process_clusters_to_imageMaps_streaming
SMALL = "/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/NewMLSection/runs/ndfull_small4x/best_model_arch.pt"
fcfg = FirstStageConfig(); fcfg.model.light_checkpoint = SMALL; models = load_first_stage_models(fcfg); tmpl = np.asarray(models.waveform_template, np.float64)
files = sorted(glob.glob("/pscratch/sd/y/yuxuan/v11arms/V14M3/full_dumps/*.npz"))[:3]
for f in files:
    z = np.load(f, allow_pickle=True); labs = z["labels"].astype(np.int64); e = np.clip(z["e"], 0, None).astype(np.float32); tpc = z["tpc"].astype(np.int64); x, y, zz = (z[k].astype(np.float32) for k in ("x", "y", "z"))
    Ecl = {int(L): float(e[labs == L].sum()) for L in np.unique(labs[labs >= 0])}
    for emin in (0.5, 1.5):
        want = {L for L, E in Ecl.items() if E >= emin}
        for amp_only in (False, True):
            for bs in (8, 32):
                t0 = time.time(); _, meta = process_clusters_to_imageMaps_streaming(x, y, zz, e, tpc, labs, model=models.light_model, template=tmpl, batch_size=bs, restrict_clusters=want, device_policy="auto", amplitudes_only=amp_only, min_prediction_threshold=0.0)
                dt = time.time() - t0; tm = meta.get("timings", {})
                print(f"{os.path.basename(f)[:-4]} emin {emin} amp_only {str(amp_only)[0]} bs {bs:2d}: groups {len(meta['group_cls'])} total {dt:5.1f}s  " + " ".join(f"{k} {v:.1f}" for k, v in tm.items() if isinstance(v, (int, float))), flush=True)
