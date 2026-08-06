"""Compare the v0.4.5 A/B arms on the events all arms completed.

  A = v0.4 baseline        (track-absorb 1.5, no stage-2 expansion)
  B = upgrade #1 only      (track-absorb 2.0)
  C = both                 (track-absorb 2.0 + stage-2 joint expansion)
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = sys.argv[1:] or ["A", "B", "C"]


def load(arm):
    rows = {}
    for fp in glob.glob(os.path.join(HERE, f"{arm}_out", "groups_worker*.jsonl")):
        for line in open(fp):
            if line.strip():
                r = json.loads(line)
                rows[(r["file"], r["event"])] = r
    return rows


data = {a: load(a) for a in ARMS}
for a in ARMS:
    print(f"{a}: {len(data[a])} events")
common = set.intersection(*[set(d) for d in data.values()]) if data else set()
print(f"common events: {len(common)}\n")
if not common:
    sys.exit("no common events yet")


def w_mean(arm, key, wkey="truth_energy"):
    num = den = 0.0
    for k in common:
        r = data[arm][k]
        v = r.get(key)
        if v is None:
            continue
        w = r.get(wkey, 1.0)
        num += v * w; den += w
    return num / max(den, 1e-9)


# discover which accuracy keys exist
sample = data[ARMS[0]][next(iter(common))]
keys = [k for k in ("acc_rescued", "ns_acc", "ns_total_eff", "ns_coverage") if k in sample]
m_keys = []
if "m_rescued" in sample:
    m_keys = [k for k in ("p1_acc", "p2_acc", "v2_acc", "p3_acc", "p3_acc_new", "p3_afrac")
              if k in (sample["m_rescued"] or {})]

print(f"{'metric':<22} " + " ".join(f"{a:>10}" for a in ARMS) + "   B-A     C-A")
for key in keys:
    vals = [w_mean(a, key) for a in ARMS]
    d = " ".join(f"{v:>10.4f}" for v in vals)
    print(f"{key:<22} {d}  {vals[1]-vals[0]:+.4f} {vals[2]-vals[0]:+.4f}"
          if len(vals) == 3 else f"{key:<22} {d}")

for key in m_keys:
    def wm(arm):
        num = den = 0.0
        for k in common:
            r = data[arm][k]; m = r.get("m_rescued") or {}
            v = m.get(key)
            if v is None:
                continue
            w = r.get("truth_energy", 1.0)
            num += v * w; den += w
        return num / max(den, 1e-9)
    vals = [wm(a) for a in ARMS]
    d = " ".join(f"{v:>10.4f}" for v in vals)
    print(f"{'m.'+key:<22} {d}" + (f"  {vals[1]-vals[0]:+.4f} {vals[2]-vals[0]:+.4f}"
                                   if len(vals) == 3 else ""))

# stage-2 expansion activity + cost
print()
for a in ARMS:
    s2 = [data[a][k].get("stage2") or {} for k in common]
    nab = sum(s.get("n_absorbed", 0) for s in s2)
    eab = sum(s.get("e_absorbed", 0.0) for s in s2)
    dij = sum(s.get("dijkstra_calls", 0) for s in s2)
    lg = sum(s.get("skipped_large", 0) for s in s2)
    if nab or dij:
        print(f"{a}: stage-2 absorbed {nab} clusters ({eab/1e3:.1f} GeV), "
              f"{dij} dijkstra calls, {lg} large skipped")

# per-event win/loss on the headline metric
head = "acc_rescued" if "acc_rescued" in sample else keys[0]
if len(ARMS) == 3:
    better = sum(1 for k in common
                 if (data["C"][k].get(head) or 0) > (data["A"][k].get(head) or 0) + 1e-6)
    worse = sum(1 for k in common
                if (data["C"][k].get(head) or 0) < (data["A"][k].get(head) or 0) - 1e-6)
    print(f"\nC vs A on {head}: better {better}, worse {worse}, flat {len(common)-better-worse}")
