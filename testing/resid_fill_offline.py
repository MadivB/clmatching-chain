"""Offline emulation of stage 4.8 residual-flash filling (resid_fill.py) on chain dumps, force coverage replayed
with the v1.4 light-vote rule in both arms.  Hit-level energy-weighted ledger per file.
python resid_fill_offline.py <dump_dir> [--events N] [--mode near|top|sup] [--K 8] [--cos-min 0.9] [--dcos 0.1]
       [--ratio-lo 0.5] [--ratio-hi 20] [--kprom 10] [--tail 0.3] [--min-adc 600] [--glare-max X] [--small-max 20] [--procs 8]"""
from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import sys

import numpy as np

_HERE = os.path.abspath(os.path.dirname(__file__)); sys.path.insert(0, _HERE)
from resid_fill import run_resid_fill, render, _template  # noqa: E402
from force_light_vote import force_light_vote_coverage  # noqa: E402
from subtick_refine import build_sipm_positions  # noqa: E402

PULSE = os.path.join(_HERE, "..", "CLMatching_AlphaRelease", "assets", "avg_pulse.npy")
CUT = 10.0
ARGS = None


def ledger(ts, tru, e, labs, amp_labels):
    ok = np.isfinite(tru) & (tru >= 0); asg = np.isfinite(ts) & (ts >= 0)
    good = ok & asg & (np.abs(ts - tru) <= CUT)
    return float(e[ok & good].sum()), float(e[ok].sum())


def work(path):
    a = ARGS
    z = np.load(path, allow_pickle=True)
    e = np.clip(z["e"], 0, None).astype(float); ts = z["ts_final"].astype(float); tru = z["truth_t0"].astype(float)
    labs = z["labels"].astype(int); tpc = z["tpc"].astype(int); xyz = np.stack([z["x"], z["y"], z["z"]], 1).astype(float)
    full = z["full_light"]; base = z["base_image"].astype(np.float32).copy(); veto = z["veto_mask"]
    al = z["amp_lab"].astype(int); at = z["amp_tpc"].astype(int); ar = z["amp_row"].astype(float)
    amp = {(int(L), int(c)): ar[i] for i, (L, c) in enumerate(zip(al, at))}
    sipm = build_sipm_positions(str(z["src_file"]))
    templated = np.isin(labs, np.unique(al))
    ts0 = ts.copy(); ts0[(labs < 0) | ~templated] = np.nan        # force-placed -> replayed in both arms
    T = _template(PULSE)
    # the chain's base was rendered from the (continuous) cluster times; use lab_rows reco_t0 for the own render
    lr = {r["label"]: r.get("reco_t0") for r in json.loads(str(z["lab_rows"]))}
    ns_base = {"xset": xyz[:, 0], "yset": xyz[:, 1], "zset": xyz[:, 2], "labels_global": labs, "hitTPCid": tpc, "Eset": e}
    # arm A: baseline replay
    nsA = dict(ns_base); nsA["hit_timestamps"] = ts0.copy()
    force_light_vote_coverage(nsA, sipm=sipm, isolate_cm=25.0, energy=e, light_cm=20.0, actual=full, base=base, veto=veto)
    gA, tot = ledger(np.asarray(nsA["hit_timestamps"]), tru, e, labs, al)
    # arm B: stage + replay
    tsB = ts0.copy()
    for L in np.unique(labs[templated & (labs >= 0)]):
        v = ts[labs == L]; v = v[np.isfinite(v)]
        if v.size and lr.get(int(L)) is not None:
            tsB[labs == L] = float(np.median(v))

    def render_cb(L, c, t_old, t_new):
        aa = amp[(int(L), int(c))]
        if t_old is not None:
            base[c] -= render(aa, float(t_old), base.shape[2], T).astype(np.float32)
        if t_new is not None:
            base[c] += render(aa, float(t_new), base.shape[2], T).astype(np.float32)
    st = run_resid_fill(ts=tsB, labels=labs, energy=e, xyz=xyz, tpc=tpc, amp=amp, actual=full, base=base, veto=veto, chan_pos=sipm,
                        render_cb=render_cb, pulse_path=PULSE, small_min=a.small_min, small_max=a.small_max, mode=a.mode, K=a.K, cos_min=a.cos_min,
                        dcos_min=a.dcos, ratio_lo=a.ratio_lo, ratio_hi=a.ratio_hi, k_prom=a.kprom, fam_tol=5.0, tail_ratio=a.tail,
                        min_adc=a.min_adc, max_per_flash=4, glare_min=a.glare_min, host_min_cm=a.host_min, truth_t0=tru)
    nsB = dict(ns_base); nsB["hit_timestamps"] = tsB
    force_light_vote_coverage(nsB, sipm=sipm, isolate_cm=25.0, energy=e, light_cm=20.0, actual=full, base=base, veto=veto)
    gB, _ = ledger(np.asarray(nsB["hit_timestamps"]), tru, e, labs, al)
    # residual-flash purity (truth flash in TPC within 5 ticks) from the audit is not available; count real moves instead
    aud = st.pop("audit")
    return dict(file=os.path.basename(path)[:7], gA=gA, gB=gB, tot=tot, n_flash=st["n_flash"], n_moved=st["n_moved"], e_moved=st["e_moved"],
                fixed=st["fixed_e"], broke=st["broke_e"], t=st["time_s"], audit=aud)


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps"); ap.add_argument("--events", type=int, default=None); ap.add_argument("--mode", default="near"); ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--cos-min", type=float, default=0.9); ap.add_argument("--dcos", type=float, default=0.1); ap.add_argument("--ratio-lo", type=float, default=0.5)
    ap.add_argument("--ratio-hi", type=float, default=20.0); ap.add_argument("--kprom", type=float, default=10.0); ap.add_argument("--tail", type=float, default=0.3)
    ap.add_argument("--min-adc", type=float, default=600.0); ap.add_argument("--glare-min", type=float, default=None); ap.add_argument("--host-min", type=float, default=None); ap.add_argument("--small-min", type=float, default=0.5); ap.add_argument("--small-max", type=float, default=20.0)
    ap.add_argument("--procs", type=int, default=8); ap.add_argument("--tag", default=""); ap.add_argument("--files", default=None, help="comma-separated file tags, e.g. 0000004,0000005")
    ARGS = ap.parse_args()
    files = sorted(glob.glob(os.path.join(ARGS.dumps, "full_dumps", "*.npz")))
    if ARGS.files:
        tags = ARGS.files.split(","); files = [f for f in files if any(os.path.basename(f).startswith(t) for t in tags)]
    files = files[: ARGS.events]
    with mp.Pool(ARGS.procs) as pool:
        res = pool.map(work, files)
    by = {}
    for r in res:
        by.setdefault(r["file"], []).append(r)
    tot = sum(r["tot"] for r in res); dA = sum(r["gA"] for r in res); dB = sum(r["gB"] for r in res)
    print(f"[{ARGS.tag}] mode={ARGS.mode} K={ARGS.K} cos>={ARGS.cos_min} dcos>{ARGS.dcos} ratio[{ARGS.ratio_lo},{ARGS.ratio_hi}] kprom={ARGS.kprom} tail={ARGS.tail} min_adc={ARGS.min_adc} glare_min={ARGS.glare_min} host_min={ARGS.host_min} small=[{ARGS.small_min},{ARGS.small_max}]")
    print(f"  {len(res)} events: all-TPC {100*dA/tot:.3f} -> {100*dB/tot:.3f} ({100*(dB-dA)/tot:+.3f} pp); residual flashes/ev {np.mean([r['n_flash'] for r in res]):.1f}; moves/ev {np.mean([r['n_moved'] for r in res]):.1f} ({np.mean([r['e_moved'] for r in res]):.0f} MeV); fixed {sum(r['fixed'] for r in res)/1e3:.2f} GeV broke {sum(r['broke'] for r in res)/1e3:.2f} GeV; events better/worse {sum(r['gB']>r['gA']+1e-6 for r in res)}/{sum(r['gB']<r['gA']-1e-6 for r in res)}; stage {np.mean([r['t'] for r in res]):.2f} s/ev")
    for f, rs in sorted(by.items()):
        tt = sum(r["tot"] for r in rs); print(f"    {f}: {100*sum(r['gA'] for r in rs)/tt:.3f} -> {100*sum(r['gB'] for r in rs)/tt:.3f} ({100*sum(r['gB']-r['gA'] for r in rs)/tt:+.3f}) fixed {sum(r['fixed'] for r in rs)/1e3:.2f} broke {sum(r['broke'] for r in rs)/1e3:.2f}")
    aud = [x for r in res for x in r["audit"]]
    if aud:
        A = np.array(aud, float); right = A[:, 9] >= 0.5 * np.maximum(A[:, 8] + A[:, 9], 1e-9)
        was_right = A[:, 8] >= 0.5 * A[:, 2]
        print(f"  moves: {len(A)}; moved clusters that were wrong {np.mean(~was_right):.2f}; move outcome right {np.mean(A[:,9] > A[:,8]):.2f} / neutral {np.mean(np.isclose(A[:,9], A[:,8])):.2f} / wrong {np.mean(A[:,9] < A[:,8]):.2f}; cos p10/50/90 {np.percentile(A[:,5],[10,50,90]).round(2)}, ratio {np.percentile(A[:,7],[10,50,90]).round(2)}, E p50 {np.median(A[:,2]):.1f} MeV")


if __name__ == "__main__":
    main()
