"""Offline emulation of stage 4.7 (blip_rehome.py) on chain dumps, with force coverage replayed.

The stage runs right before force coverage (4.9), so a v1.3 dump is its input once the force-placed
hits are cleared: noise hits (label < 0) and untemplated clusters (no amp row) are reset to NaN and
re-timed by force_simple in BOTH arms (baseline replay vs stage + replay), so noise hits dragged by
moved clusters are counted.  Hit-level, energy-weighted ledger; dev = files 1-3, holdout = 4-5.

usage: python blip_rehome_offline.py [--dumps DIR] [--no-contact] [--no-iso] [--contact-cm 2] ...
"""
from __future__ import annotations

import argparse
import glob
import multiprocessing as mp
import os
import sys

import numpy as np

_HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, _HERE)
from blip_rehome import PEAK, run_blip_rehome  # noqa: E402
from force_simple import force_simple_coverage  # noqa: E402
from force_noise_vote import force_noise_vote_coverage  # noqa: E402

PULSE = os.path.join(_HERE, "..", "CLMatching_AlphaRelease", "assets", "avg_pulse.npy")
CLASSES = [("noise", None, None), ("<1.5", 0, 1.5), ("1.5-5", 1.5, 5), ("5-20", 5, 20),
           ("20-100", 20, 100), (">=100", 100, 1e12)]
ARGS = None


def _pulse():
    p = np.load(PULSE).astype(np.float64).ravel()
    return p / p.max(), int(np.argmax(p))


def work(path):
    a = ARGS
    d = np.load(path, allow_pickle=True)
    x, y, z, e, tpc, lab, tsf, tt = [d[k] for k in ("x", "y", "z", "e", "tpc", "labels", "ts_final", "truth_t0")]
    lab = lab.astype(np.int64); tpc = tpc.astype(np.int64)
    xyz = np.c_[x, y, z].astype(np.float64)
    actual = d["full_light"].astype(np.float64)
    base0 = d["base_image"].astype(np.float64)
    amp = {(int(L), int(c)): np.asarray(r, np.float64) for L, c, r in zip(d["amp_lab"], d["amp_tpc"], d["amp_row"])}
    P, pk = _pulse()
    jj = np.arange(actual.shape[2], dtype=np.float64)

    def make_render(base):
        def render(L, c, t_old, t_new):
            s = amp[(L, c)][:, None]
            if t_old is not None:
                base[c] -= s * np.interp(jj - t_old - PEAK + pk, np.arange(P.size), P, left=0.0, right=0.0)[None, :]
            if t_new is not None:
                base[c] += s * np.interp(jj - t_new - PEAK + pk, np.arange(P.size), P, left=0.0, right=0.0)[None, :]
        return render

    templated = np.zeros(lab.size, bool)
    tl = np.unique(np.asarray(d["amp_lab"], np.int64))
    templated[np.isin(lab, tl)] = True
    ts0 = tsf.astype(np.float64).copy()
    ts0[(lab < 0) | ~templated] = np.nan            # force-placed in the chain -> re-timed by the replay

    def replay(ts, vote=False):
        ns = {"hit_timestamps": ts, "xset": x, "yset": y, "zset": z, "labels_global": lab}
        if vote:
            force_noise_vote_coverage(ns, isolate_cm=25.0, energy=e)
        else:
            force_simple_coverage(ns, isolate_cm=25.0)
        return np.asarray(ns["hit_timestamps"], np.float64)

    ts_base = replay(ts0.copy())
    ts_arm_in = ts0.copy(); base = base0.copy()
    if a.no_stage:
        a.no_contact = a.no_iso = True
    st = run_blip_rehome(ts=ts_arm_in, labels=lab, energy=e, xyz=xyz, tpc=tpc, amp=amp, actual=actual,
                         base=base, render=make_render(base), contact=not a.no_contact, isolated=not a.no_iso,
                         contact_cm=a.contact_cm, cos_min=a.cos_min, host_cm=a.host_cm, own_min=a.own_min,
                         cand_cm=a.cand_cm, truth_t0=tt, near=a.near, reattach=a.reattach)
    ts_arm = replay(ts_arm_in, vote=a.noise_vote)
    ok = np.isfinite(tt)
    ul, inv = np.unique(lab, return_inverse=True); cE = np.bincount(inv, weights=e)[inv]
    tpcE = np.bincount(tpc, minlength=70, weights=e)
    sh = tpcE[tpc] > 1400
    good_f = ok & (np.abs(tsf - tt) <= 10)
    good_b = ok & (np.abs(ts_base - tt) <= 10)
    good_a = ok & (np.abs(ts_arm - tt) <= 10)
    out = {"file": os.path.basename(path)[:7], "E_ok": float(e[ok].sum()), "E_ns": float(e[ok & ~sh].sum()),
           "good_f": float(e[good_f].sum()), "good_b": float(e[good_b].sum()), "good_a": float(e[good_a].sum()),
           "good_b_ns": float(e[good_b & ~sh].sum()), "good_a_ns": float(e[good_a & ~sh].sum()),
           "stats": {k: v for k, v in st.items() if k != "audit"}}
    for name, lo, hi in CLASSES:
        m = ok & ((lab < 0) if lo is None else ((lab >= 0) & (cE >= lo) & (cE < hi)))
        for shn, shm in (("ns", ~sh), ("sh", sh)):
            mm = m & shm
            out[f"fix_{name}_{shn}"] = float(e[mm & good_a & ~good_b].sum())
            out[f"brk_{name}_{shn}"] = float(e[mm & good_b & ~good_a].sum())
            out[f"tot_{name}_{shn}"] = float(e[mm].sum())
            out[f"wrong_b_{name}_{shn}"] = float(e[mm & ~good_b].sum())
    return out


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", default="/pscratch/sd/y/yuxuan/v11arms/V2A5/full_dumps")
    ap.add_argument("--no-contact", action="store_true")
    ap.add_argument("--no-iso", action="store_true")
    ap.add_argument("--contact-cm", type=float, default=2.0)
    ap.add_argument("--cos-min", type=float, default=0.9)
    ap.add_argument("--host-cm", type=float, default=20.0)
    ap.add_argument("--own-min", type=float, default=0.1)
    ap.add_argument("--cand-cm", type=float, default=150.0)
    ap.add_argument("--near", action="store_true")
    ap.add_argument("--reattach", action="store_true")
    ap.add_argument("--noise-vote", action="store_true")
    ap.add_argument("--no-stage", action="store_true", help="no cluster moves (e.g. noise vote alone)")
    ap.add_argument("--procs", type=int, default=8)
    ARGS = ap.parse_args()
    files = sorted(glob.glob(os.path.join(ARGS.dumps, "*.npz")))
    with mp.Pool(ARGS.procs) as pool:
        rows = pool.map(work, files)
    for split, sel in (("dev files 1-3", lambda r: r["file"] <= "0000003"), ("holdout files 4-5", lambda r: r["file"] > "0000003"), ("all", lambda r: True)):
        R = [r for r in rows if sel(r)]
        if not R:
            continue
        E = sum(r["E_ok"] for r in R); Ens = sum(r["E_ns"] for r in R)
        gb = sum(r["good_b"] for r in R); ga = sum(r["good_a"] for r in R); gf = sum(r["good_f"] for r in R)
        gbn = sum(r["good_b_ns"] for r in R); gan = sum(r["good_a_ns"] for r in R)
        d_ev = [r["good_a"] - r["good_b"] for r in R]
        print(f"=== {split}: {len(R)} events | all-TPC {gb/E*100:.3f} -> {ga/E*100:.3f} ({(ga-gb)/E*100:+.3f} pp; dump ts_final {gf/E*100:.3f}) "
              f"| NS {gbn/Ens*100:.3f} -> {gan/Ens*100:.3f} ({(gan-gbn)/Ens*100:+.3f}) | events better/worse {sum(v > 0 for v in d_ev)}/{sum(v < 0 for v in d_ev)}")
        for name, _, _ in CLASSES:
            parts = []
            for shn in ("ns", "sh"):
                fx = sum(r[f"fix_{name}_{shn}"] for r in R) / 1e3; bk = sum(r[f"brk_{name}_{shn}"] for r in R) / 1e3
                wb = sum(r[f"wrong_b_{name}_{shn}"] for r in R) / 1e3
                parts.append(f"{shn}: fixed {fx:5.2f} broke {bk:5.2f} (of {wb:5.2f} wrong)")
            print(f"   {name:>7}  " + " | ".join(parts))
        S = {}
        for r in R:
            for k, v in r["stats"].items():
                if isinstance(v, (int, float)):
                    S[k] = S.get(k, 0) + v
        print("   stage: " + ", ".join(f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}" for k, v in S.items() if k != "time_s")
              + f", time {S.get('time_s', 0)/len(R):.2f} s/ev")


if __name__ == "__main__":
    main()
