"""In-chain adapter for the v2.0 block solver (stage 3.99): re-solve clusters >= min_e MeV from
the chain's current assignment with exact block enumeration + LNS on the light energy, small
clusters frozen.  Writes the new times into ns["hit_timestamps"] and updates ns["baseImage"]."""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import v200_core as core  # noqa: E402


def run_v200_stage(ns: dict, *, veto_mask, sat_ceil, ti, min_e: float = 20.0, sweeps: int = 3,
                   truth_t0=None, verbose: bool = False, cold: bool = True) -> dict:
    """cold=True (offline arm C20, +0.19 pp on 39 events): the re-solved clusters are re-initialised by
    greedy self-placement on the residual of the frozen small clusters before the block sweeps."""
    t0 = time.time()
    image_maps = ns["imageMaps"]
    actual = np.asarray(ns["fullLightWaveform"], dtype=np.float64)
    n_light = actual.shape[0]
    labels = np.asarray(ns["labels_global"]).astype(int)
    ts = np.asarray(ns["hit_timestamps"], dtype=np.float64)
    amp = {}
    for (L, c), im in image_maps.items():
        if 0 <= int(c) < n_light:
            amp[(int(L), int(c))] = np.asarray(im, np.float64).max(axis=1)
    ev = core.Event.from_arrays(
        e=np.asarray(ns["Eset"], np.float64),
        xyz=np.stack([np.asarray(ns["xset"], np.float64), np.asarray(ns["yset"], np.float64),
                      np.asarray(ns["zset"], np.float64)], axis=1),
        tpc=np.asarray(ns["hitTPCid"]).astype(int), labels=labels, ts_chain=ts, A=actual,
        veto=np.asarray(veto_mask, bool), amp=amp,
        rail=None if sat_ceil is None else np.asarray(sat_ceil, np.float64),
        truth=None if truth_t0 is None else np.asarray(truth_t0, np.float64), tag="chain")
    before = ev.chain_assignment()
    res = core.solve_event(ev, n_sweeps=sweeps, verbose=verbose, warm_min_e=float(min_e), cold=bool(cold))
    after = res["assign"]
    changed = {L: (before.get(L), after[L]) for L in after if before.get(L) != after[L] and ev.lab_E[L] >= min_e}
    # write back: hit times and the base image (chain conventions: float32 image, timeinterpolation shift)
    base = ns["baseImage"]
    e_moved = 0.0
    for L, (t_old, t_new) in changed.items():
        idx = ev.lab_hits[L]
        ts[idx] = float(t_new)
        e_moved += float(ev.lab_E[L])
        for c in ev.lab_tpcs[L]:
            im = np.asarray(image_maps[(L, c)], np.float64)
            if t_old is not None:
                base[c] -= ti(im, t_old).astype(base.dtype)
            base[c] += ti(im, t_new).astype(base.dtype)
    ns["hit_timestamps"] = ts
    stats = {"n_resolved": int(sum(1 for L in ev.light_labels if ev.lab_E[L] >= min_e)),
             "n_changed": len(changed), "e_changed": e_moved, "E_light_before": None,
             "E_light_after": float(res["E_light"]), "E_hist": res["E_hist"], "n_decidable": res["n_decidable"],
             "time_s": time.time() - t0}
    if truth_t0 is not None:
        # audit only (never used in decisions): how many of the moved clusters got better / worse
        tr = np.asarray(truth_t0, np.float64)
        fixed = broke = 0
        for L, (t_old, t_new) in changed.items():
            idx = ev.lab_hits[L]; m = np.isfinite(tr[idx])
            if not m.any() or t_old is None:
                continue
            tt = float(np.median(tr[idx][m]))
            was = abs(t_old - tt) <= core.CUT; now = abs(t_new - tt) <= core.CUT
            fixed += int(now and not was); broke += int(was and not now)
        stats["audit_fixed"] = fixed; stats["audit_broke"] = broke
    return stats
