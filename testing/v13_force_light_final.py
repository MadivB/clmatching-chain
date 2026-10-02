"""63-event evaluation of the production kernel force_light_vote on certification dumps, light_cm scan.
python v13_force_light_final.py <arm_dir> [n_events]"""
import glob
import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from subtick_refine import build_sipm_positions   # noqa: E402
from force_light_vote import force_light_vote_coverage   # noqa: E402
from force_simple import force_simple_coverage   # noqa: E402

CUT = 10.0


def main(arm, n_max=None):
    files = sorted(glob.glob(f"{arm}/full_dumps/*.npz"))[: int(n_max) if n_max else None]
    variants = [("nearest hit (current force_simple)", None), ("vote only (light off)", 1e9), ("vote x light beyond 20 cm", 20.0),
                ("vote x light beyond 15 cm", 15.0), ("vote x light beyond 10 cm", 10.0), ("vote x light beyond 3 cm (all voted)", 3.0)]
    D_BINS = [0, 3, 10, 15, 20, 1e9]
    T = {v[0]: np.zeros((len(D_BINS) - 1, 2)) for v in variants}; tt = {v[0]: 0.0 for v in variants}; sipm = None
    tot = np.zeros(2); allE = 0.0
    for ff in files:
        z = np.load(ff, allow_pickle=True)
        if sipm is None:
            sipm = build_sipm_positions(str(z["src_file"]))
        e = np.clip(z["e"], 0, None).astype(float); ts = z["ts_final"].astype(float); tru = z["truth_t0"].astype(float)
        labs = z["labels"].astype(int); tpc = z["tpc"].astype(int); xyz = np.stack([z["x"], z["y"], z["z"]], 1).astype(float)
        full = z["full_light"]; base = z["base_image"]; veto = z["veto_mask"]
        ok = np.isfinite(tru) & (tru >= 0) & (tru <= 885); allE += e[ok].sum()
        cl = (labs >= 0) & np.isfinite(ts) & (ts >= 0)
        ts0 = np.where(cl, ts, np.nan)      # pre-force state: clustered assignments only
        noise = (labs < 0) & ok
        d1, _ = cKDTree(xyz[cl]).query(xyz[noise], k=1)
        ns_base = {"xset": xyz[:, 0], "yset": xyz[:, 1], "zset": xyz[:, 2], "labels_global": labs, "hitTPCid": tpc, "Eset": e}
        for name, lc in variants:
            ns = dict(ns_base); ns["hit_timestamps"] = ts0.copy()
            t0 = time.time()
            if lc is None:
                force_simple_coverage(ns, isolate_cm=25.0)
            else:
                force_light_vote_coverage(ns, sipm=sipm, isolate_cm=25.0, energy=e, light_cm=lc, actual=full, base=base, veto=veto)
            tt[name] += time.time() - t0
            tn = np.asarray(ns["hit_timestamps"])[noise]
            good = np.isfinite(tn) & (np.abs(tn - tru[noise]) <= CUT)
            for b in range(len(D_BINS) - 1):
                m = (d1 >= D_BINS[b]) & (d1 < D_BINS[b + 1])
                T[name][b] += [e[noise][m].sum(), e[noise][m & good].sum()]
    print(f"{len(files)} events; noise energy {T[variants[0][0]][:,0].sum()/1e3:.2f} GeV of {allE/1e3:.0f} GeV")
    print(f"{'variant':40s} {'time/ev':>8s} " + " ".join(f"{D_BINS[b]:.0f}-{min(D_BINS[b+1],99):.0f}cm".rjust(9) for b in range(len(D_BINS) - 1)) + "   noise class   all-TPC delta (pp)")
    base_c = T[variants[0][0]][:, 1].sum()
    for name, _ in variants:
        A = T[name]
        print(f"{name:40s} {tt[name]/len(files):7.3f}s " + " ".join(f"{A[b,1]/max(A[b,0],1e-9):9.3f}" for b in range(len(D_BINS) - 1)) + f"   {A[:,1].sum()/A[:,0].sum():.3f}      {100*(A[:,1].sum()-base_c)/allE:+.3f}")
    print("energy per distance bin (GeV): " + " ".join(f"{T[variants[0][0]][b,0]/1e3:.2f}" for b in range(len(D_BINS) - 1)))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
