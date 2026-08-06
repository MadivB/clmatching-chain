"""Aggregate --v2-tune rows: pooled post-V2 accuracy per V2 config variant."""
import glob
import json
import sys

import numpy as np

out_dir = sys.argv[1] if len(sys.argv) > 1 else "v2tune_out"

rows = []
for fp in sorted(glob.glob(f"{out_dir}/groups_worker*.jsonl")):
    with open(fp) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
print(f"{len(rows)} events from {out_dir}")
if not rows:
    sys.exit(0)

names = list(rows[0]["m_rescued"]["v2"].keys())


def pooled(key_fn):
    num = den = 0.0
    for r in rows:
        acc, af = key_fn(r)
        if acc is None:
            continue
        w = r["truth_energy"] * af
        num += acc * w
        den += w
    return num / max(den, 1e-9)


p1 = pooled(lambda r: (r["m_rescued"]["p1_acc"], r["m_rescued"]["p1_afrac"]))
p2 = pooled(lambda r: (r["m_rescued"]["p2_acc"], r["m_rescued"]["p2_afrac"]))
print(f"\nbackbone (stack2+rescue) p1 {p1:.4f} | post-P2 {p2:.4f}")

print(f"\n{'variant':<10}{'post-V2':>9}{'A1':>9}{'new':>9}{'movedE%':>9}"
      f"{'mv-prec b->a':>14}{'time_s':>8}")
res = []
for n in names:
    acc = pooled(lambda r: (r["m_rescued"]["v2"][n]["v2_acc"],
                            r["m_rescued"]["v2"][n]["v2_afrac"]))

    def cohort(key):
        num = den = 0.0
        for r in rows:
            v = r["m_rescued"]["v2"][n].get(key)
            if v is not None:
                num += v * r["truth_energy"]
                den += r["truth_energy"]
        return num / max(den, 1e-9)

    a1, new = cohort("v2_acc_A1"), cohort("v2_acc_new")
    me = np.mean([r["m_rescued"]["v2"][n]["moved_efrac"] for r in rows])
    mb = cohort("moved_acc_before")
    ma = cohort("moved_acc_after")
    tt = sum(r["m_rescued"]["v2"][n]["time_s"] for r in rows)
    res.append((acc, n))
    print(f"{n:<10}{acc:>9.4f}{a1:>9.4f}{new:>9.4f}{me*100:>8.2f}%"
          f"{mb:>7.2f}->{ma:<5.2f}{tt:>8.0f}")

res.sort(reverse=True)
print(f"\nranking: {' > '.join(n for _, n in res)}")

# per-event paired comparison of each variant vs 'def'
if "def" in names:
    print("\npaired vs def (post-V2, per event):")
    for n in names:
        if n == "def":
            continue
        d = np.array([r["m_rescued"]["v2"][n]["v2_acc"] - r["m_rescued"]["v2"]["def"]["v2_acc"]
                      for r in rows])
        print(f"  {n:<10} better {np.sum(d>1e-6):>3} worse {np.sum(d<-1e-6):>3} "
              f"mean {d.mean()*100:+.3f}pp")
