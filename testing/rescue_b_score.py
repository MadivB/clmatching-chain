"""Truth-score Stage B (per-TPC part detachment) decisions."""
import glob
import json
import os

import numpy as np

HERE = os.path.abspath(os.path.dirname(__file__))
rows = []
for fp in sorted(glob.glob(os.path.join(HERE, "rescue_b_out", "groups_worker*.jsonl"))):
    with open(fp) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if not r.get("skipped"):
                    rows.append(r)

CUT = 10.0
bgroups = [r for r in rows if r.get("stage_b")]
print(f"{len(rows)} N>=2 groups | Stage-A all-stay with Stage B evaluated: {len(bgroups)}")


def part_foreign(t, slot):
    return t is not None and abs(t - slot) > CUT


gen = []
for r in bgroups:
    sb = r["stage_b"]
    foreign_members = [t["member"] for t in sb["tries"] if part_foreign(t["truth_part"], r["slot_t0"])]
    gen.append(bool(foreign_members))
    r["_foreign"] = foreign_members
gen = np.array(gen)
print(f"genuine Stage-B groups (some member's part in the firing TPC is truly foreign): "
      f"{int(gen.sum())} | clean (trigger false alarms reaching B): {int((~gen).sum())}")

detach = np.array([r["stage_b"]["detach"] is not None for r in bgroups])
print(f"\nStage B outcome 1 (all parts stay, group accepted): {int((~detach).sum())}")
print(f"Stage B detaches: {int(detach.sum())}")

n_correct = n_wrong = 0
e_correct = e_wrong = 0.0
t1_good = 0
right_member = 0
routed_p2 = routed_p3 = 0
for r in bgroups:
    sb = r["stage_b"]
    if sb["detach"] is None:
        continue
    foreign = part_foreign(sb["detach_truth"], r["slot_t0"])
    if foreign:
        n_correct += 1
        e_correct += sb["detach_e"]
        if sb["detach_truth"] is not None and abs(sb["detach_t1"] - sb["detach_truth"]) <= CUT:
            t1_good += 1
        if sb["detach"] in r["_foreign"]:
            right_member += 1
    else:
        n_wrong += 1
        e_wrong += sb["detach_e"]
    if sb["detach_e"] >= 50.0:
        routed_p2 += 1
    else:
        routed_p3 += 1

print(f"  correct detaches (part truly foreign at slot): {n_correct} "
      f"({n_correct/max(int(detach.sum()),1):.0%} of detaches) | {e_correct:.0f} MeV freed")
print(f"  wrong detaches (healthy part split off):       {n_wrong} | {e_wrong:.0f} MeV "
      f"(mild cost: parts re-associate in phase 2/3)")
print(f"  scan's preferred t1 matches the part's truth:  {t1_good}/{n_correct}")
print(f"  routing of detached parts: {routed_p2} -> phase 2 (E>=50), {routed_p3} -> phase 3 (E<50)")

# coverage of the genuine population
gen_caught = sum(1 for r in bgroups if r["_foreign"] and r["stage_b"]["detach"] in r["_foreign"])
gen_missed = sum(1 for r in bgroups if r["_foreign"] and r["stage_b"]["detach"] is None)
gen_wrongpick = sum(1 for r in bgroups if r["_foreign"] and r["stage_b"]["detach"] is not None
                    and r["stage_b"]["detach"] not in r["_foreign"])
print(f"\nof {int(gen.sum())} genuine Stage-B groups: correct member detached {gen_caught}, "
      f"missed (outcome 1) {gen_missed}, wrong member picked {gen_wrongpick}")
clean_spur = sum(1 for r in bgroups if not r["_foreign"] and r["stage_b"]["detach"] is not None)
print(f"of {int((~gen).sum())} clean groups reaching B: spurious detaches {clean_spur} "
      f"({clean_spur/max(int((~gen).sum()),1):.1%})")
