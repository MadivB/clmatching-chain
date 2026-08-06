"""Report wrong backbone placements (phase 1 + phase 2) as event/TPC pairs.

Usage: python wrong_census_report.py [out_dir] [top_n]
  out_dir: p12wrong_out (backbone-only) or p12all_out (predict-all -> iterative active)

Classifies each dumped case by comparing the DECISION-TIME loss at the chosen t0
vs at truth:
  RECOVERABLE : loss(truth) < loss(chosen)  -> the light preferred truth but the
                algorithm did not take it (candidate grid / snapping / ordering)
  DEGENERATE  : loss(truth) > loss(chosen)  -> the loss itself prefers the wrong
                slot (real competing flash) -> needs a better discriminant
"""
import glob
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = sys.argv[1] if len(sys.argv) > 1 else "p12wrong_out"
TOP = int(sys.argv[2]) if len(sys.argv) > 2 else 30
CUT = 10.0

rows = []
for fp in sorted(glob.glob(os.path.join(HERE, OUTDIR, "rows_worker*.jsonl"))):
    for line in open(fp):
        if line.strip():
            rows.append(json.loads(line))
print(f"{len(rows)} wrong placements from {OUTDIR}")
print("by phase:", dict(Counter(r["phase"] for r in rows)))
print("phase-2 by stage:", dict(Counter(r["stage"] for r in rows if r["phase"] == "phase2")))
print("phase-2 iterative:", dict(Counter(r["iterative"] for r in rows if r["phase"] == "phase2")))
print(f"total wrong energy: {sum(r['energy'] for r in rows)/1e3:.1f} GeV "
      f"(phase1 {sum(r['energy'] for r in rows if r['phase']=='phase1')/1e3:.1f}, "
      f"phase2 {sum(r['energy'] for r in rows if r['phase']=='phase2')/1e3:.1f})")

# classify dumped phase-2 cases via their decision-time curves
cls = {}
dumpdir = os.path.join(HERE, OUTDIR, "dumps")
cache = {}
for r in rows:
    if r["phase"] != "phase2":
        continue
    tag = f"{r['file']}_ev{r['event']}"
    dp = os.path.join(dumpdir, f"{tag}.npz")
    if not os.path.exists(dp):
        continue
    if tag not in cache:
        cache[tag] = np.load(dp)
    d = cache[tag]
    k = f"L{r['label']}_curve_dec"
    if k not in d.files:
        continue
    dec = np.asarray(d[k], float)
    t0, tr = int(round(r["t0"])), int(round(r["truth"]))
    if not (0 <= t0 < dec.size and 0 <= tr < dec.size):
        continue
    grid = np.asarray(d.get(f"L{r['label']}_grid", np.array([], int)), int)
    reach = bool(grid.size and np.min(np.abs(grid - tr)) <= CUT)
    if abs(int(np.argmin(dec)) - tr) <= CUT:
        verdict = "RECOVERABLE"      # a free scan of this very landscape finds truth
    elif dec[tr] < dec[t0]:
        verdict = "PARTIAL"          # truth beats what we chose, but a 3rd slot wins
    else:
        verdict = "DEGENERATE"       # the loss genuinely prefers the wrong slot
    cls[(tag, r["label"])] = {
        "verdict": verdict, "reachable": reach,
        "dloss": float(dec[t0] - dec[tr]), "argmin": int(np.argmin(dec)),
        "argmin_is_truth": bool(abs(int(np.argmin(dec)) - tr) <= CUT)}

if cls:
    print(f"\n--- classification of {len(cls)} dumped phase-2 cases ---")
    print("verdict:", dict(Counter(v["verdict"] for v in cls.values())))
    print("truth reachable on the t0-candidate grid:",
          dict(Counter(v["reachable"] for v in cls.values())))
    print("free argmin would have been truth:",
          dict(Counter(v["argmin_is_truth"] for v in cls.values())))

# ---- the deliverable: event / TPC pairs ----
rows.sort(key=lambda r: -r["energy"])
print(f"\n{'='*118}\nTOP {TOP} WRONG PLACEMENTS  (event / TPC pairs)\n{'='*118}")
print(f"{'file':>8} {'ev':>3} {'TPCs':<12} {'label':>6} {'phase':>6} {'stage':<20} "
      f"{'E[MeV]':>8} {'t0':>6} {'truth':>6} {'off':>6}  verdict")
for r in rows[:TOP]:
    tag = f"{r['file']}_ev{r['event']}"
    c = cls.get((tag, r["label"]), {})
    v = c.get("verdict", "")
    if v and not c.get("reachable", True):
        v += " (truth off-grid)"
    it = "*ITER " if r.get("iterative") else ""
    print(f"{r['file']:>8} {r['event']:>3} {str(r['tpcs']):<12} {r['label']:>6} "
          f"{r['phase']:>6} {it+r['stage']:<20} {r['energy']:>8.0f} {r['t0']:>6.0f} "
          f"{r['truth']:>6.0f} {abs(r['t0']-r['truth']):>6.0f}  {v}")

# compact per-(event,TPC) pair list for the notebook
pairs = defaultdict(lambda: {"n": 0, "e": 0.0, "phases": set()})
for r in rows:
    for t in r["tpcs"]:
        k = (r["file"], r["event"], int(t))
        pairs[k]["n"] += 1
        pairs[k]["e"] += r["energy"]
        pairs[k]["phases"].add(r["phase"])
top_pairs = sorted(pairs.items(), key=lambda kv: -kv[1]["e"])[:TOP]
print(f"\n{'='*72}\nWORST (event, TPC) PAIRS  -> paste into the notebook\n{'='*72}")
print(f"{'file':>8} {'event':>5} {'TPC':>4} {'n_wrong':>8} {'wrongE[MeV]':>12}  phases")
for (f_, ev, t), v in top_pairs:
    print(f"{f_:>8} {ev:>5} {t:>4} {v['n']:>8} {v['e']:>12.0f}  {','.join(sorted(v['phases']))}")
print("\nPY_LIST = [")
for (f_, ev, t), v in top_pairs[:12]:
    print(f'    ("{f_}", {ev}, {t}),   # {v["n"]} wrong, {v["e"]:.0f} MeV')
print("]")
