"""Offline trigger sweep from rescue-pipeline trig_rows diagnostics.

Each row stores, per placed track: z3 (N_CH_FIRE-th highest gated z at the probe tick,
in whatever sigma mode the job ran), cmin (worst-TPC cos(own, residual)), truth and
wrongness. Sweep Z and COS cuts without re-running anything.
"""
import glob
import json
import sys

import numpy as np

out_dir = sys.argv[1] if len(sys.argv) > 1 else "rescue_pipe4_out"

tracks = []
mode = "?"
for fp in sorted(glob.glob(f"{out_dir}/groups_worker*.jsonl")):
    with open(fp) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            mode = r.get("sigma_mode", "?")
            for t in r.get("trig_rows", []):
                if t.get("truth") is not None:
                    tracks.append(t)

z3 = np.array([t["z3"] for t in tracks])
cm = np.array([t["cmin"] for t in tracks])
wrong = np.array([t["wrong"] for t in tracks], dtype=bool)
e = np.array([t["e"] for t in tracks])
print(f"{len(tracks)} truth-valid placed tracks | sigma mode: {mode} | "
      f"wrong {wrong.sum()} ({e[wrong].sum()/1e3:.1f} GeV)")


def stats(fire):
    catch = fire[wrong].mean() if wrong.any() else 0.0
    fp = fire[~wrong].mean() if (~wrong).any() else 0.0
    ecatch = e[fire & wrong].sum() / max(e[wrong].sum(), 1e-9)
    return catch, fp, ecatch


print(f"\n{'rule':<34} {'catch':>7} {'false':>7} {'E-catch':>8}")
for zc in (2, 4, 6, 8, 10, 12, 15, 20):
    c, f, ec = stats(z3 > zc)
    print(f"z3 > {zc:<28} {c:>7.1%} {f:>7.1%} {ec:>8.1%}")
for cc in (0.70, 0.75, 0.80, 0.85):
    c, f, ec = stats(cm < cc)
    print(f"cos < {cc:<27} {c:>7.1%} {f:>7.1%} {ec:>8.1%}")
print()
for cc in (0.75, 0.80):
    for zc in (4, 6, 8, 10, 12):
        c, f, ec = stats((cm < cc) | (z3 > zc))
        print(f"cos < {cc} OR z3 > {zc:<14} {c:>7.1%} {f:>7.1%} {ec:>8.1%}")
