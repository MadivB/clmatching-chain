"""Compare the v0.2.5 vertex-merge run against the identical no-merge reference,
plus the truth audit of every merge decision."""
import glob
import json
import sys

import numpy as np

out_dir = sys.argv[1] if len(sys.argv) > 1 else "v025merge_out"
REF = sys.argv[2] if len(sys.argv) > 2 else "v02police_l_out"


def load(d):
    rows = []
    for fp in sorted(glob.glob(f"{d}/groups_worker*.jsonl")):
        with open(fp) as f:
            rows.extend(json.loads(l) for l in f if l.strip())
    return {(r["file"], r["event"]): r for r in rows}


ref, mrg = load(REF), load(out_dir)
common = sorted(set(ref) & set(mrg))
print(f"{len(mrg)} merge-run events ({out_dir}), {len(common)} paired with {REF}")


def pooled(rows, which, key):
    num = den = 0.0
    for r in rows:
        m = r[which]
        if m is None or m.get(f"{key}_acc") is None:
            continue
        w = r["truth_energy"] * m[f"{key}_afrac"]
        num += m[f"{key}_acc"] * w
        den += w
    return num / max(den, 1e-9)


print(f"\n{'':22}{'no merge':>10}{'merged':>10}{'delta':>9}")
for which, label in (("m_baseline", "stack2+police"), ("m_rescued", "stack2+resc+police")):
    for key in ("p1", "p2", "v2"):
        a = pooled([ref[k] for k in common], which, key)
        b = pooled([mrg[k] for k in common], which, key)
        print(f"{label:>18} {key:<3}{a:>10.4f}{b:>10.4f}{(b-a)*100:>+8.2f}pp")

# per-event paired post-V2 (rescued)
d = np.array([mrg[k]["m_rescued"]["v2_acc"] - ref[k]["m_rescued"]["v2_acc"] for k in common])
print(f"\nper-event postV2 (rescued): better {np.sum(d>1e-6)} / flat {np.sum(np.abs(d)<=1e-6)} "
      f"/ worse {np.sum(d<-1e-6)} | mean {d.mean()*100:+.2f}pp worst {d.min()*100:+.2f}pp "
      f"best {d.max()*100:+.2f}pp")

# merge audit
groups = [g for r in mrg.values() for g in (r.get("vertex_merge") or {}).get("groups", [])]
scored = [g for g in groups if g["dt_truth"] is not None]
good = [g for g in scored if g["good"]]
bad = [g for g in scored if not g["good"]]
sizes = {}
for g in groups:
    sizes[len(g["labels"])] = sizes.get(len(g["labels"]), 0) + 1
print(f"\nvertex merges: {len(groups)} groups over {len(mrg)} events | sizes {sizes}")
if scored:
    eg = sum(sum(x["energies"]) for x in good) / 1e3
    eb = sum(sum(x["energies"]) for x in bad) / 1e3
    print(f"  truth-scored {len(scored)}: GOOD {len(good)} ({eg:.1f} GeV, "
          f"{100*len(good)/len(scored):.0f}%) | BAD {len(bad)} ({eb:.1f} GeV)")
    if bad:
        w = sorted(bad, key=lambda g: -sum(g["energies"]))[:6]
        print("  worst bad merges (labels, E, dt_truth):")
        for g in w:
            print(f"    {g['labels']} E={[round(e) for e in g['energies']]} "
                  f"dt={g['dt_truth']:.0f} ticks")
    dts = [g["dt_truth"] for g in scored]
    print(f"  dt_truth percentiles: 50%={np.percentile(dts,50):.1f} "
          f"90%={np.percentile(dts,90):.1f} max={max(dts):.0f} ticks")
