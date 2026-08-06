"""Truth-score Stage A rescue results."""
import glob
import json
import os

import numpy as np

HERE = os.path.abspath(os.path.dirname(__file__))
rows = []
skipped1 = 0
for fp in sorted(glob.glob(os.path.join(HERE, "rescue_a_out", "groups_worker*.jsonl"))):
    with open(fp) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("skipped") == "N==1":
                skipped1 += 1
            else:
                rows.append(r)

n_events = len({(r["file"], r["event"]) for r in rows})
print(f"{len(rows)} groups (N>=2) over {n_events} events | N==1 flagged slots skipped: {skipped1}")
sizes = [len(r["members"]) for r in rows]
print(f"group sizes: N=2: {sizes.count(2)}, N=3: {sizes.count(3)}")

CUT = 10.0


def verdicts(r, t0map):
    out = []
    for j in r["members"]:
        tr = r["truth"][str(j)]
        if tr is None:
            out.append(None)
        else:
            out.append(abs(t0map[str(j)] - tr) <= CUT)
    return out


for adopt_name, adopt_fn in (
    ("always adopt winner", lambda r: True),
    ("adopt only if loss improves", lambda r: r["loss_new"] < r["loss_orig"]),
):
    fixed = broken = stay_right = stay_wrong = 0
    e_fixed = e_broken = 0.0
    moved = moved_right = 0
    for r in rows:
        use_new = adopt_fn(r)
        before = verdicts(r, r["old_t0"])
        after = verdicts(r, r["new_t0"] if use_new else r["old_t0"])
        for j, vb, va in zip(r["members"], before, after):
            if vb is None:
                continue
            e = r["energy"][str(j)]
            if not vb and va:
                fixed += 1
                e_fixed += e
            elif vb and not va:
                broken += 1
                e_broken += e
            elif vb and va:
                stay_right += 1
            else:
                stay_wrong += 1
            if use_new and abs(r["new_t0"][str(j)] - r["old_t0"][str(j)]) > CUT:
                moved += 1
                if va:
                    moved_right += 1
    tot_wrong_before = fixed + stay_wrong
    tot_right_before = broken + stay_right
    print(f"\n=== {adopt_name} ===")
    print(f"  wrong-before members : {tot_wrong_before} -> FIXED {fixed} "
          f"({fixed/max(tot_wrong_before,1):.0%}), still wrong {stay_wrong}")
    print(f"  right-before members : {tot_right_before} -> BROKEN {broken} "
          f"({broken/max(tot_right_before,1):.1%}), kept right {stay_right}")
    print(f"  energy: fixed {e_fixed:.0f} MeV | broken {e_broken:.0f} MeV | net {e_fixed-e_broken:+.0f} MeV")
    if adopt_name.startswith("always"):
        print(f"  members moved >10 ticks: {moved}, of which land truth-correct: {moved_right}")

n_stay = sum(1 for r in rows if r["all_stay"])
stay_with_wrong = sum(1 for r in rows if r["all_stay"] and any(
    v is False for v in verdicts(r, r["old_t0"])))
print(f"\nStage-B candidates (winner keeps ALL members at slot t0): {n_stay}/{len(rows)} groups")
print(f"  of those, groups that genuinely contain a wrong member: {stay_with_wrong} "
      f"(rest are trigger false alarms that Stage A leaves untouched)")
imp = sum(1 for r in rows if r["loss_new"] < r["loss_orig"])
print(f"groups where winner loss < original loss: {imp}/{len(rows)}")
