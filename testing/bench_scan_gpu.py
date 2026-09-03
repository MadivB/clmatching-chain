import numpy as np, sys, time, glob, os
sys.path.insert(0, '/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing')
os.chdir('/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing')
from scan_margin import event_scan_feats, event_scan_feats_ref, SCAN_NAMES
import torch
print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
files = sorted(glob.glob('v126MG_out/full_dumps/*.npz'), key=os.path.getsize)[::-1]
tot = {"ref": 0.0, "np": 0.0, "gpu": 0.0}
first = True
for ff in files[:5]:
    tag = os.path.basename(ff)
    sd = np.load(f'v126MG_out/dumps/{tag}'); fd = np.load(ff, allow_pickle=True)
    labs = fd['labels']; ts_b = sd['ts_before'].astype(float)
    base, act = fd['base_image'].astype(np.float64), fd['full_light'].astype(np.float64)
    al, at, ar = fd['amp_lab'], fd['amp_tpc'], fd['amp_row']
    lab_t0 = {}
    for L in np.unique(al):
        m = (labs == L) & np.isfinite(ts_b) & (ts_b >= 0)
        if m.any():
            v, c = np.unique(np.round(ts_b[m]).astype(int), return_counts=True)
            lab_t0[int(L)] = int(v[np.argmax(c)])
    args = (base, act, fd['veto_mask'], al, at, ar, lab_t0)
    if first:  # warm up cuda kernels/fft plans off the clock
        event_scan_feats(*args, use_gpu=True)
        first = False
    t0 = time.time(); r1 = event_scan_feats_ref(*args); tot["ref"] += time.time()-t0
    t0 = time.time(); r2 = event_scan_feats(*args); tot["np"] += time.time()-t0
    t0 = time.time(); r3 = event_scan_feats(*args, use_gpu=True); tot["gpu"] += time.time()-t0
    # gpu-vs-numpy parity on all 28 features
    bad = 0
    for L in r2:
        d = np.abs(r2[L][0] - r3[L][0]) / np.maximum(np.abs(r2[L][0]), 1e-3)
        if np.nanmax(np.where(np.isfinite(d), d, 0)) > 0.05: bad += 1
    print(f"{tag}: labels {len(r2)}, gpu-vs-np clusters >5% diff: {bad}")
print(f"\n5-event totals: ref {tot['ref']:.2f} s | numpy-f32 {tot['np']:.2f} s | GPU {tot['gpu']:.2f} s")
print(f"per event: ref {tot['ref']/5:.2f} | numpy {tot['np']/5:.2f} | GPU {tot['gpu']/5:.3f}")
