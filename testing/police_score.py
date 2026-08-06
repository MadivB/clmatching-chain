"""Compare a policed run against the unpoliced veto-4 reference (v02rescue4_out)."""
import glob
import json
import sys

REF = "v02rescue4_out"
out_dir = sys.argv[1] if len(sys.argv) > 1 else "v02police_t_out"


def load(d):
    rows = []
    for fp in sorted(glob.glob(f"{d}/groups_worker*.jsonl")):
        with open(fp) as f:
            rows.extend(json.loads(l) for l in f if l.strip())
    return {(r["file"], r["event"]): r for r in rows}


ref, pol = load(REF), load(out_dir)
common = sorted(set(ref) & set(pol))
print(f"{len(pol)} policed events ({out_dir}), {len(common)} paired with {REF}")


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


print(f"\n{'':16}{'unpoliced':>11}{'policed':>11}{'delta':>9}")
for which, label in (("m_baseline", "stack2"), ("m_rescued", "stack2+resc")):
    for key in ("p2", "v2"):
        a = pooled([ref[k] for k in common], which, key)
        b = pooled([pol[k] for k in common], which, key)
        print(f"{label:>12} {key:<3}{a:>11.4f}{b:>11.4f}{(b-a)*100:>+8.2f}pp")

for which in ("m_baseline", "m_rescued"):
    for key in ("p2_acc_new", "v2_acc_new", "v2_acc_A1"):
        na = nb = da = db = 0.0
        for k in common:
            for src, tgt in ((ref, "a"), (pol, "b")):
                m = src[k][which]
                if m and m.get(key) is not None:
                    w = src[k]["truth_energy"]
                    if tgt == "a":
                        na += m[key] * w
                        da += w
                    else:
                        nb += m[key] * w
                        db += w
        print(f"{which[2:]:>12} {key:<11} {na/max(da,1e-9):.4f} -> {nb/max(db,1e-9):.4f}")

# police action statistics
print("\npolice actions (summed):")
for which in ("m_baseline", "m_rescued"):
    for stage in ("police_p2", "police_v2"):
        agg = {}
        acts = []
        for r in pol.values():
            m = r[which]
            if not m or stage not in m:
                continue
            for kk, vv in m[stage].items():
                if kk == "fired_labels":
                    acts.extend(vv)
                else:
                    agg[kk] = agg.get(kk, 0) + vv
        print(f"  {which[2:]:>8} {stage}: {agg}")
        moved = [a for a in acts if a["action"] != "keep"]
        if moved:
            import numpy as np
            dt = [abs(a["t_new"] - a["t_cur"]) for a in moved]
            print(f"           moved {len(moved)} (median |dt| {np.median(dt):.0f} ticks; "
                  f"{sum(1 for a in moved if a['action']=='revert')} reverts, "
                  f"{sum(1 for a in moved if a['action']=='rescan')} rescans)")
