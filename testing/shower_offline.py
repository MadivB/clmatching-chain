"""Offline shower-TPC iteration harness — tune without re-running the chain.

Loads v036v36_out/dumps (chain run ONCE with --shower-tpc --dump-dir, which
saves all part templates + flash tables + pre-shower timestamps for TPCs
above the threshold), rebuilds a minimal namespace per event, and runs
run_shower_tpc_pass with arbitrary parameters.  One full 39-event sweep takes
seconds-to-minutes on a login node — no GPU, no chain.

Usage:
    python shower_offline.py                       # default params, report
    python shower_offline.py fill_frac=0.6 grow_reach_cm=15   # override knobs
"""
import glob
import json
import re
import sys

import numpy as np

sys.path.insert(0, "/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing")
from shower_tpc import run_shower_tpc_pass

DUMPS = "/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing/v036v36_out/dumps"
CUT = 10.0
THRESH = 1500.0

kw = {}
for a in sys.argv[1:]:
    k, v = a.split("=")
    try:
        kw[k] = float(v)
    except ValueError:
        kw[k] = v


def run_event(fp, **params):
    d = np.load(fp, allow_pickle=True)
    if "ts_pre_shower" not in d:
        return None
    full_light = np.asarray(d["full_light"], np.float64)
    n_light = full_light.shape[0]
    labels = np.asarray(d["labels"], int)
    tpc = np.asarray(d["tpc"], int)
    e = np.clip(np.asarray(d["e"], float), 0, None)
    tt = np.asarray(d["truth_t0"], float)

    image_maps = {}
    t0cand = [[] for _ in range(n_light)]
    for key in d.files:
        m = re.match(r"shimg_(\d+)_(\d+)$", key)
        if m:
            image_maps[(int(m.group(1)), int(m.group(2)))] = np.asarray(d[key], np.float64)
        m = re.match(r"sh_t0cand_(\d+)$", key)
        if m:
            t0cand[int(m.group(1))] = [float(v) for v in d[key]]

    ns = {"imageMaps": image_maps,
          "baseImage": np.zeros_like(full_light, dtype=np.float32),
          "fullLightWaveform": full_light.astype(np.float32),
          "labels_global": labels,
          "hit_timestamps": np.asarray(d["ts_pre_shower"], np.float64).copy(),
          "xset": np.asarray(d["x"], float), "yset": np.asarray(d["y"], float),
          "zset": np.asarray(d["z"], float), "Eset": e,
          "hitTPCid": tpc, "t0Candidates": t0cand}

    def truth_fn(lab, tpc_q):
        m = (labels == lab) & (tpc == tpc_q) & np.isfinite(tt)
        if not m.any():
            return None
        tv, wv = tt[m], e[m]
        o = np.argsort(tv)
        cw = np.cumsum(wv[o])
        return float(tv[o][np.searchsorted(cw, 0.5 * cw[-1])])

    st = run_shower_tpc_pass(ns, hit_tpc=tpc, e_thresh_mev=THRESH,
                             truth_fn=truth_fn, **params)
    # cohort efficiency before/after
    tpc_e = np.zeros(int(tpc.max()) + 2)
    np.add.at(tpc_e, tpc, e)
    mask = tpc_e[tpc] > THRESH
    out = {}
    for nm, arr in (("before", np.asarray(d["ts_pre_shower"], float)),
                    ("after", np.asarray(ns["hit_timestamps"], float))):
        ok = mask & np.isfinite(tt)
        good = ok & np.isfinite(arr) & (arr >= 0) & (np.abs(arr - tt) <= CUT)
        out[nm] = (float(e[good].sum()), float(e[ok].sum()))
    out["tpc_recs"] = st["tpcs"]
    return out


if __name__ == "__main__":
    gb = ga = tot = 0.0
    recs = []
    for fp in sorted(glob.glob(f"{DUMPS}/*.npz")):
        r = run_event(fp, **kw)
        if r is None:
            continue
        gb += r["before"][0]
        ga += r["after"][0]
        tot += r["before"][1]
        recs += r["tpc_recs"]
    print(f"params: {kw or 'defaults'}")
    print(f"shower cohort (> {THRESH:.0f} MeV): {len(recs)} TPCs, {tot/1e3:.1f} GeV truth-valid")
    print(f"  eff BEFORE {gb/max(tot,1e-9):.4f} -> AFTER {ga/max(tot,1e-9):.4f} "
          f"({(ga-gb)/max(tot,1e-9)*100:+.2f}pp)")
    pe = [r["part_eff"] for r in recs if "part_eff" in r]
    if pe:
        print(f"  per-TPC part_eff: median {np.median(pe):.3f} min {min(pe):.3f}")
    worst = sorted(recs, key=lambda r: r.get("part_eff", 1.0))[:5]
    for r in worst:
        print(f"   worst: {r}")
