"""What would the chain gain if only the light-decidable population took the v2.0 block-solver
times?  Hybrid scores from a pass-1 result file:
  H_dec : v2.0 time for decidable clusters, chain time for the rest
  H_20  : v2.0 time for clusters >= 20 MeV, chain time for the rest
  H_dec20: v2.0 for (decidable AND >= 20 MeV), chain for the rest
python v200_hybrid.py results.json <dump_dir> [--workers 4]"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import v200_core as core  # noqa: E402

CLASSES = ["noise/blind", "<1.5", "1.5-5", "5-20", "20-100", ">=100"]


def one(args):
    r, dump_dir = args
    ev = core.Event(os.path.join(dump_dir, "full_dumps", r["tag"] + ".npz"))
    C = {c[0]: c for c in r["clusters"]}
    tv = {int(k): v for k, v in r["assign_v200"].items()}
    tc = {L: c[7] for L, c in C.items() if c[7] >= 0}
    dec = {L for L, c in C.items() if c[4] == 1}
    out = {"tag": r["tag"]}
    variants = {
        "chain": lambda L: tc.get(L),
        "v200": lambda L: tv.get(L),
        "H_dec": lambda L: tv.get(L) if L in dec else tc.get(L),
        "H_20": lambda L: tv.get(L) if ev.lab_E[L] >= 20 else tc.get(L),
        "H_dec20": lambda L: tv.get(L) if (L in dec and ev.lab_E[L] >= 20) else tc.get(L),
    }
    for name, pick in variants.items():
        ev.reset()
        for L in ev.light_labels:
            t = pick(L)
            if t is not None and t >= 0:
                ev.set_time(L, int(t))
        out[name] = ev.score(ev.hit_times())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results"); ap.add_argument("dump_dir"); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    res = json.load(open(a.results))
    with Pool(a.workers) as pool:
        rows = pool.map(one, [(r, a.dump_dir) for r in res], chunksize=1)
    names = ["chain", "v200", "H_dec", "H_20", "H_dec20"]
    print(f"{len(rows)} events")
    print(f"{'':>12} " + " ".join(f"{n:>9}" for n in names))
    eff = {n: 1 - sum(r[n]["E_wrong"] for r in rows) / sum(r[n]["E_ok"] for r in rows) for n in names}
    print(f"{'eff %':>12} " + " ".join(f"{100*eff[n]:9.3f}" for n in names))
    for k in CLASSES:
        line = []
        for n in names:
            wv = sum(r[n][k][0] for r in rows); Ek = sum(r[n][k][1] for r in rows)
            line.append(f"{100*wv/max(Ek,1e-9):9.2f}")
        print(f"{k:>12} " + " ".join(line))
    for n in names[1:]:
        b = sum(1 for r in rows if r[n]["E_wrong"] < r["chain"]["E_wrong"] - 1e-6)
        w = sum(1 for r in rows if r[n]["E_wrong"] > r["chain"]["E_wrong"] + 1e-6)
        print(f"{n}: events better/worse than chain {b}/{w}")


if __name__ == "__main__":
    main()
