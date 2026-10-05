"""bf16 autocast + cudnn.benchmark vs fp32 for the perceiver: time per group and amplitude agreement, on one dump event."""
import os, sys, glob, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.abspath(os.path.join(HERE, "..", "CLMatching_AlphaRelease"))
for p in (BASE, os.path.join(BASE, "M5p1"), os.path.join(BASE, "NewMLSection"), HERE): sys.path.insert(0, p)
from M5p1.first_stage_matching import load_first_stage_models
from M5p1.first_stage_matching.config import FirstStageConfig
from M5p1.first_stage_matching.streaming_prediction import process_clusters_to_imageMaps_streaming
SMALL = "/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/NewMLSection/runs/ndfull_small4x/best_model_arch.pt"
fcfg = FirstStageConfig(); fcfg.model.light_checkpoint = SMALL; models = load_first_stage_models(fcfg); tmpl = np.asarray(models.waveform_template, np.float64)
z = np.load(sorted(glob.glob("/pscratch/sd/y/yuxuan/v11arms/V14M3/full_dumps/*.npz"))[1], allow_pickle=True)
labs = z["labels"].astype(np.int64); e = np.clip(z["e"], 0, None).astype(np.float32); tpc = z["tpc"].astype(np.int64); x, y, zz = (z[k].astype(np.float32) for k in ("x", "y", "z"))
want = {int(L) for L in np.unique(labs[labs >= 0]) if e[labs == L].sum() >= 0.5}
def run(label, **kw):
    torch.cuda.synchronize(); t0 = time.time()
    _, meta = process_clusters_to_imageMaps_streaming(x, y, zz, e, tpc, labs, model=models.light_model, template=tmpl, restrict_clusters=want, device_policy="auto", amplitudes_only=True, min_prediction_threshold=0.0, **kw)
    torch.cuda.synchronize(); dt = time.time() - t0
    A = {(int(c), int(t)): np.asarray(meta["amplitudes"][i], np.float64) for i, (c, t) in enumerate(zip(meta["group_cls"], meta["group_tpcs"]))}
    print(f"{label:40s} groups {len(A)} {dt:5.1f}s  model_s {meta['timings'].get('model_s', 0):.1f}", flush=True); return A
ref = run("fp32 bs 8 (reference)", batch_size=8)
run("fp32 bs 8 again (warm)", batch_size=8)
run("fp32 bs 32", batch_size=32)
torch.backends.cudnn.benchmark = True
run("fp32 bs 8 cudnn.benchmark", batch_size=8); run("fp32 bs 32 cudnn.benchmark", batch_size=32)
for dt_ in ("bf16", "fp16"):
    try:
        A = run(f"{dt_} autocast bs 32 cudnn.benchmark", batch_size=32, use_mixed_precision=True, amp_dtype=dt_)
        rel = []; mx = []
        for k, v in ref.items():
            if k in A and v.max() > 100:
                rel.append(np.abs(A[k] - v).sum() / max(v.sum(), 1e-9)); mx.append(np.abs(A[k] - v).max() / max(v.max(), 1e-9))
        print(f"   {dt_} vs fp32: relative L1 of the pattern p50/90/99 {np.percentile(rel, [50, 90, 99]).round(4)}; max-channel rel diff p50/90/99 {np.percentile(mx, [50, 90, 99]).round(4)} (groups > 100 ADC: {len(rel)})", flush=True)
    except Exception as ex:
        print(f"   {dt_} failed: {ex}", flush=True)
torch.backends.cudnn.benchmark = False
try:
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    run("fp32 + TF32 bs 32", batch_size=32)
except Exception as ex:
    print("tf32 failed", ex)
