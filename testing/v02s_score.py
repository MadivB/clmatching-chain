"""Aggregate v0.2 staged rows: pooled p1/p2/v2 accuracy per backbone variant."""
import glob
import json
import sys

out_dir = sys.argv[1] if len(sys.argv) > 1 else "v02s_out"

rows = []
for fp in sorted(glob.glob(f"{out_dir}/groups_worker*.jsonl")):
    with open(fp) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
print(f"{len(rows)} events from {out_dir}")
if not rows:
    sys.exit(0)

names = list(rows[0]["metrics"].keys())


def pooled(name, key):
    num = den = 0.0
    for r in rows:
        m = r["metrics"].get(name)
        if m is None:
            continue
        acc, af = m.get(f"{key}_acc"), m.get(f"{key}_afrac")
        if acc is None:
            continue
        w = r["truth_energy"] * af
        num += acc * w
        den += w
    return num / max(den, 1e-9)


print(f"\npooled energy-weighted accuracy (weight = truth_energy * assigned_frac):")
print(f"{'stage':<8}" + "".join(f"{n:>10}" for n in names))
for key in ("p1", "p2", "v2"):
    print(f"{key:<8}" + "".join(f"{pooled(n, key):>10.4f}" for n in names))

print("\ncohorts (weight = truth_energy):")
for key in ("p2_acc_A1", "p2_acc_new", "v2_acc_A1", "v2_acc_new"):
    vals = []
    for n in names:
        num = den = 0.0
        for r in rows:
            m = r["metrics"].get(n)
            if m and m.get(key) is not None:
                num += m[key] * r["truth_energy"]
                den += r["truth_energy"]
        vals.append(num / max(den, 1e-9))
    print(f"{key:<12}" + "".join(f"{v:>10.4f}" for v in vals))

print("\nassigned fraction after V2:")
for n in names:
    af = [r["metrics"][n]["v2_afrac"] for r in rows if r["metrics"].get(n)]
    print(f"  {n:<8} {sum(af)/len(af):.4f}")

# per-event v2 comparison vs base
import numpy as np
for n in names:
    if n == "base":
        continue
    d = np.array([r["metrics"][n]["v2_acc"] - r["metrics"]["base"]["v2_acc"]
                  for r in rows if r["metrics"].get(n) and r["metrics"]["base"]["v2_acc"] is not None])
    print(f"per-event v2 {n}-base: improved {np.sum(d>1e-6)} / flat {np.sum(np.abs(d)<=1e-6)} "
          f"/ worse {np.sum(d<-1e-6)} | mean {d.mean()*100:+.2f}pp worst {d.min()*100:+.2f}pp")
