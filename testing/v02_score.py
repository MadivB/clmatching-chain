"""Aggregate v0.2 ordering-test rows: pooled accuracies, pool-cohort, cost per s."""
import glob
import json
import sys

import numpy as np

CUT = 10.0
out_dir = sys.argv[1] if len(sys.argv) > 1 else "v02_out"

rows = []
for fp in sorted(glob.glob(f"{out_dir}/groups_worker*.jsonl")):
    with open(fp) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
print(f"{len(rows)} events from {out_dir}")
if not rows:
    sys.exit(0)

names = list(rows[0]["variants"].keys())

fg = sum(r["fixed_cohort"]["good"] for r in rows)
fo = sum(r["fixed_cohort"]["ok"] for r in rows)
print(f"\nfixed cohort (showers + >=3-TPC tracks, production t0): "
      f"{fg / max(fo, 1e-9):.4f} ({fo / 1e3:.1f} GeV truth-valid)")

print(f"\n{'variant':<8}{'overall':>10}{'pool-cohort':>13}   (energy-pooled)")
for n in names:
    g = sum(r["variants"][n]["good"] for r in rows)
    o = sum(r["variants"][n]["ok"] for r in rows)
    gp = sum(r["variants"][n]["good_pool"] for r in rows)
    op = sum(r["variants"][n]["ok_pool"] for r in rows)
    print(f"{n:<8}{g / max(o, 1e-9):>10.4f}{gp / max(op, 1e-9):>13.4f}")

# label-level wrongness on the pool
print(f"\npool label-level (|t - truth| > {CUT:.0f} ticks):")
for n in names:
    wrong, wE, tot, totE = 0, 0.0, 0, 0.0
    for r in rows:
        for pl in r["pool_labels"]:
            if pl["truth"] is None or pl.get(f"t_{n}") is None:
                continue
            tot += 1
            totE += pl["energy"]
            if abs(pl[f"t_{n}"] - pl["truth"]) > CUT:
                wrong += 1
                wE += pl["energy"]
    print(f"  {n:<8} wrong {wrong}/{tot} ({wE / 1e3:.1f} / {totE / 1e3:.1f} GeV)")

# compute cost per s
skeys = sorted({k for r in rows for k in r.get("s_stats", {})})
if skeys:
    print(f"\n{'s':<6}{'batches':>9}{'multi':>7}{'leaves':>8}{'scans':>8}"
          f"{'time_s':>8}{'perm>greedy':>13}{'rel gain':>10}{'maxB':>6}")
    for k in skeys:
        nb = sum(r["s_stats"][k]["n_batches"] for r in rows)
        nm = sum(r["s_stats"][k]["n_multi_batches"] for r in rows)
        nl = sum(r["s_stats"][k]["n_leaves"] for r in rows)
        ns = sum(r["s_stats"][k]["n_scans"] for r in rows)
        tt = sum(r["s_stats"][k]["time_s"] for r in rows)
        pb = sum(r["s_stats"][k]["perm_better"] for r in rows)
        pt = sum(r["s_stats"][k]["perm_total"] for r in rows)
        rg = sum(r["s_stats"][k]["perm_rel_gain_sum"] for r in rows)
        maxb = max(int(b) for r in rows for b in r["s_stats"][k]["batch_size_hist"])
        print(f"{k:<6}{nb:>9}{nm:>7}{nl:>8}{ns:>8}{tt:>8.0f}"
              f"{f'{pb}/{pt}':>13}{rg / max(pb, 1):>10.3f}{maxb:>6}")

# where do variants differ? pool labels whose placement differs from ctrlG
print("\nplacement differences vs ctrlG (pool labels, |dt| > 10):")
for n in names:
    if n == "ctrlG":
        continue
    diff, fixed, broke = 0, 0, 0
    for r in rows:
        for pl in r["pool_labels"]:
            t_n, t_g = pl.get(f"t_{n}"), pl.get("t_ctrlG")
            if t_n is None or t_g is None or abs(t_n - t_g) <= CUT:
                continue
            diff += 1
            if pl["truth"] is not None:
                ok_n = abs(t_n - pl["truth"]) <= CUT
                ok_g = abs(t_g - pl["truth"]) <= CUT
                fixed += (ok_n and not ok_g)
                broke += (ok_g and not ok_n)
    print(f"  {n:<8} differs {diff:>4} | better {fixed:>3} worse {broke:>3}")
