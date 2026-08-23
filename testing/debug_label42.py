"""Dump N=1 internals for the 'shower eats track' case: file 0000001 ev1, the
~350 MeV track placed at ~609.5 (truth ~389). Prints loss/cos landscapes for the
stay / move-whole / detach candidates so the adoption rule can be designed from data.
"""
import os
import sys

_HERE = os.path.abspath(os.path.dirname(__file__))
BASE_DIR = os.path.abspath(os.path.join(_HERE, "..", "CLMatching_AlphaRelease"))
sys.path.insert(0, BASE_DIR)
sys.path.insert(1, os.path.join(BASE_DIR, "NewMLSection"))
sys.path.insert(2, os.path.join(BASE_DIR, "M5p1"))

import numpy as np
import h5py
from scipy.signal import fftconvolve

PEAK, GAIN, CLIP, SEARCH, CUT = 105, 200.0, 60780.0, 895, 10.0

from M5p1.first_stage_matching import load_first_stage_models, run_first_stage_charge_light_matching
from M5p1.first_stage_matching.track_stage import fast_stack_images
from M5p1.phase25_trial2_valpha_batch import build_default_first_stage_config
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


DATA = ("/global/cfs/cdirs/dunepro/people/abooth/nd-production/output/MiniProdN5/"
        "run-ndlar-flow/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift/FLOW/0000000/"
        "MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.0000001.FLOW.hdf5")
EV = 1

cfg = build_default_first_stage_config()
cfg.prediction.min_cluster_energy_mev = 50.0
print("loading models ...", flush=True)
models = load_first_stage_models(cfg)
h5 = h5py.File(DATA, "r")
fs = run_first_stage_charge_light_matching(h5=h5, event_id=EV, config=cfg, models=models, verbose=False)

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

truth = tp.extract_event_hit_energy_and_truth_t0(
    h5, h5["charge/events/ref/charge/calib_prompt_hits/ref"], eventid=EV,
    hits_full=h5["charge/calib_prompt_hits/data"], convert_to_matching_ticks=True)
truth_t0 = np.asarray(truth["truth_t0"], dtype=np.float64)

# info + base (same as pipeline)
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
        ch = (np.asarray(entry["channel_indices"], dtype=int) if entry is not None else np.arange(n_ch))
        ch = ch[~veto_mask[int(c), ch]]
        if ch.size:
            sup[int(c)] = ch
    if not sup:
        continue
    info[lab] = {"t0": float(np.median(t0v)), "tpcs": tpcs, "sup": sup, "type": lab_type,
                 "energy": float(e_pos[members].sum())}

base = np.zeros_like(actual)
for lab in sorted(info):
    L = info[lab]
    ta = np.asarray(L["tpcs"], dtype=np.int32)
    base[ta] += timeinterpolation(fast_stack_images(image_maps, int(lab), ta),
                                  shift=L["t0"], baseline=0).astype(np.float64).reshape(ta.size, n_ch, -1)

# find the case: track, E>300 MeV, t0 in [600, 620]
cands = [(l, K) for l, K in info.items()
         if K["type"] == "track" and K["energy"] > 300 and 600 <= K["t0"] <= 620]
print("candidate labels:", [(l, round(K['energy']), K['t0'], K['tpcs']) for l, K in cands])
lab42, K = next(((l, Ki) for l, Ki in cands if Ki["tpcs"] == [17, 19]), cands[0])


def lab_truth(lab):
    m = labels_global == lab
    tt, ww = truth_t0[m], e_pos[m]
    ok = np.isfinite(tt)
    o = np.argsort(tt[ok]); cw = np.cumsum(ww[ok][o])
    return float(tt[ok][o][np.searchsorted(cw, 0.5 * cw[-1])])


TR = lab_truth(lab42)
print(f"\n=== label {lab42}: E={K['energy']:.0f} MeV t0={K['t0']} truth={TR:.1f} tpcs={K['tpcs']} ===")
# per-TPC part truth
for c in K["tpcs"]:
    m = (labels_global == lab42) & (np.asarray(fs.event.hit_tpc_id) == c)
    tt, ww = truth_t0[m], e_pos[m]
    ok = np.isfinite(tt)
    if ok.any():
        o = np.argsort(tt[ok]); cw = np.cumsum(ww[ok][o])
        pt = float(tt[ok][o][np.searchsorted(cw, 0.5 * cw[-1])])
        print(f"  part TPC {c}: {int(m.sum())} hits {ww.sum():.0f} MeV part-truth {pt:.1f}")

tpcs = K["tpcs"]
tpcs_arr = np.asarray(tpcs, dtype=np.int32)
stk = fast_stack_images(image_maps, int(lab42), tpcs_arr)
base_wo = base.copy()
base_wo[tpcs_arr] -= timeinterpolation(stk, shift=K["t0"], baseline=0).astype(np.float64).reshape(tpcs_arr.size, n_ch, -1)
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
    sh = timeinterpolation(label_stk, shift=float(t0v), baseline=0).astype(np.float64).reshape(len(tpc_list), n_ch, -1)
    parts = {} if into is None else into
    for ti, c in enumerate(tpc_list):
        if c in union:
            parts[c] = parts.get(c, 0.0) + sh[ti][union[c]]
    return parts


loss_stay = u_loss(contrib(stk, tpcs, K["t0"]))
pr = np.concatenate([np.asarray(image_maps[(int(lab42), int(c))], float)[ch] for c, ch in union.items()])
rr = np.concatenate([(base_wo[c] - actual[c])[ch] for c, ch in union.items()])
wr = np.concatenate([w_u[c] for c in union])
curve = weighted_scan(pr, rr, wr)
cv = cos_curve(pr, rr)
sT, s0 = int(round(TR)), int(round(K["t0"]))
print(f"\nWHOLE-cluster scan (union): loss_stay={loss_stay:.0f}")
print(f"  argmin loss: s={int(np.argmin(curve))} loss={curve.min():.0f} cos={cv[int(np.argmin(curve))]:.3f}")
print(f"  at TRUTH {sT}: loss={curve[sT]:.0f} cos={cv[sT]:.3f}")
print(f"  at PLACED {s0}: loss={curve[s0]:.0f} cos={cv[s0]:.3f}")
top = np.argsort(-cv)[:6]
print("  top-cos shifts:", [(int(s), round(float(cv[s]), 3), int(curve[s])) for s in top])
for gate in (0.5, 0.7, 0.8):
    ok = cv >= gate
    if ok.any():
        idx = np.flatnonzero(ok)
        sb = int(idx[np.argmin(curve[idx])])
        print(f"  gated argmin (cos>={gate}): s={sb} loss={curve[sb]:.0f} -> move loss {u_loss(contrib(stk, tpcs, float(sb))):.0f}")

print(f"  move-to-truth union loss: {u_loss(contrib(stk, tpcs, float(sT))):.0f}")

for tpf in tpcs:
    if tpf not in union:
        continue
    P = np.asarray(image_maps[(int(lab42), int(tpf))], dtype=np.float64)
    chp = union[tpf]
    rr_p = (base_wo[tpf] - actual[tpf])[chp]
    curve_p = weighted_scan(P[chp], rr_p, w_u[tpf])
    cv_p = cos_curve(P[chp], rr_p)
    rem = [c for c in tpcs if c != tpf]
    print(f"\nDETACH part=TPC{tpf}: part scan argmin s={int(np.argmin(curve_p))} "
          f"(cos there {cv_p[int(np.argmin(curve_p))]:.3f}); part cos@truth {cv_p[sT]:.3f} @placed {cv_p[s0]:.3f}")
    rem_sup = [c for c in rem if c in union]
    if rem_sup:
        pr_r = np.concatenate([np.asarray(image_maps[(int(lab42), int(c))], float)[union[c]] for c in rem_sup])
        rr_r = np.concatenate([(base_wo[c] - actual[c])[union[c]] for c in rem_sup])
        wr_r = np.concatenate([w_u[c] for c in rem_sup])
        curve_r = weighted_scan(pr_r, rr_r, wr_r)
        cv_r = cos_curve(pr_r, rr_r)
        sr = int(np.argmin(curve_r))
        print(f"  REMAINDER scan ({rem_sup}): argmin s={sr} loss={curve_r[sr]:.0f} cos={cv_r[sr]:.3f}")
        print(f"    at TRUTH {sT}: loss={curve_r[sT]:.0f} cos={cv_r[sT]:.3f} | at PLACED {s0}: loss={curve_r[s0]:.0f} cos={cv_r[s0]:.3f}")
        topr = np.argsort(-cv_r)[:6]
        print("    top-cos shifts:", [(int(s), round(float(cv_r[s]), 3), int(curve_r[s])) for s in topr])
        # full detach-candidate losses with remainder at its argmin vs at truth
        t1 = float(np.argmin(curve_p))
        parts = contrib(P, [tpf], t1)
        l_arg = u_loss(contrib(np.concatenate([np.asarray(image_maps[(int(lab42), int(c))], float) for c in rem]).reshape(len(rem), n_ch, -1) if len(rem) > 1 else np.asarray(image_maps[(int(lab42), int(rem[0]))], float), rem, float(sr), into=dict(parts)))
        l_tr = u_loss(contrib(np.concatenate([np.asarray(image_maps[(int(lab42), int(c))], float) for c in rem]).reshape(len(rem), n_ch, -1) if len(rem) > 1 else np.asarray(image_maps[(int(lab42), int(rem[0]))], float), rem, float(sT), into=dict(parts)))
        print(f"    detach candidate loss (rem@argmin {sr}): {l_arg:.0f} | (rem@TRUTH {sT}): {l_tr:.0f}")
print("\ndone", flush=True)
