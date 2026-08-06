"""Sweep the Stage-A stay-margin offline from group_diags (loss_stay vs loss_best per group).

For each margin m: a group moves iff loss_best < loss_stay*(1-m). For moved groups, count
member-level fixes (was wrong by >CUT, lands within CUT of truth) and breaks (was right,
lands wrong) using the would-move t0s.
"""
import glob
import json
import sys

import numpy as np

CUT = 10.0
out_dir = sys.argv[1] if len(sys.argv) > 1 else "rescue_pipe6_out"

diags = []
for fp in sorted(glob.glob(f"{out_dir}/groups_worker*.jsonl")):
    with open(fp) as f:
        for line in f:
            if line.strip():
                diags.extend(json.loads(line).get("group_diags", []))
print(f"{len(diags)} Stage-A groups with loss diagnostics")

ratios = np.array([d["loss_best"] / max(d["loss_stay"], 1e-9) for d in diags])
print(f"loss_best/loss_stay: p10={np.percentile(ratios,10):.3f} p50={np.percentile(ratios,50):.3f} "
      f"p90={np.percentile(ratios,90):.3f} | groups with would-move: "
      f"{sum(1 for d in diags if d['would_move'])}")

print(f"\n{'margin':>7} {'moved-groups':>13} {'moves':>6} {'fixes':>6} {'breaks':>7} {'net':>5}")
for m in (0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80):
    ng = nm = fx = br = 0
    for d in diags:
        if not d["would_move"]:
            continue
        if d["loss_best"] >= d["loss_stay"] * (1.0 - m):
            continue
        ng += 1
        for j, t0n in d["would_move"].items():
            tr = d["truths"].get(j)
            if tr is None:
                continue
            nm += 1
            was_wrong = abs(d["t0s"][j] - tr) > CUT
            now_right = abs(t0n - tr) <= CUT
            if was_wrong and now_right:
                fx += 1
            elif not was_wrong and not now_right:
                br += 1
    print(f"{m:>7.2f} {ng:>13} {nm:>6} {fx:>6} {br:>7} {fx-br:>5}")
