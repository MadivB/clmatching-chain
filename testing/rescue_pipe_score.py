"""Aggregate rescue-pipeline rows: pooled staged accuracies + N=1 action truth stats."""
import glob
import json
import sys

import numpy as np

CUT = 10.0
out_dir = sys.argv[1] if len(sys.argv) > 1 else "rescue_pipe3_out"

rows = []
for fp in sorted(glob.glob(f"{out_dir}/groups_worker*.jsonl")):
    with open(fp) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
print(f"{len(rows)} events from {out_dir}")


def pooled(tag, key):
    num = den = 0.0
    for r in rows:
        m = r[f"m_{tag}"]
        acc, af = m.get(f"{key}_acc"), m.get(f"{key}_afrac")
        if acc is None:
            continue
        w = r["truth_energy"] * af          # ~ assigned truth-valid energy
        num += acc * w
        den += w
    return num / max(den, 1e-9)


print("\npooled energy-weighted accuracy (weight = truth_energy * assigned_frac):")
print(f"{'stage':<10}{'baseline':>10}{'rescued':>10}")
for key in ("p1", "p2", "v2"):
    print(f"{key:<10}{pooled('baseline', key):>10.4f}{pooled('rescued', key):>10.4f}")

for key in ("p2_acc_A1", "p2_acc_new", "v2_acc_A1", "v2_acc_new"):
    num = {"baseline": [0.0, 0.0], "rescued": [0.0, 0.0]}
    for r in rows:
        for tag in num:
            m = r[f"m_{tag}"]
            if m.get(key) is not None:
                w = r["truth_energy"]
                num[tag][0] += m[key] * w
                num[tag][1] += w
    b = num["baseline"][0] / max(num["baseline"][1], 1e-9)
    s = num["rescued"][0] / max(num["rescued"][1], 1e-9)
    print(f"{key:<14} baseline {b:.4f}  rescued {s:.4f}")

tot_groups = sum(r.get("n_groups", 0) for r in rows)
tot_moves = sum(r.get("n_moves", 0) for r in rows)
tot_det = sum(r.get("n_detaches", 0) for r in rows)
tot_lone = sum(r.get("n_lone_flagged", 0) for r in rows)
print(f"\ngroups {tot_groups} | moves {tot_moves} | detaches {tot_det} | lone flagged {tot_lone}")

acts = [a for r in rows for a in r.get("n1_actions", [])]
if acts:
    print(f"\nN=1 actions: {len(acts)} "
          f"(move {sum(a['action'] == 'move' for a in acts)}, "
          f"detach {sum(a['action'] == 'detach' for a in acts)})")
    for kind in ("move", "detach"):
        sel = [a for a in acts if a["action"] == kind and a["truth"] is not None]
        if not sel:
            continue
        e_tot = sum(a["energy"] for a in sel)
        was_wrong = [a for a in sel if abs(a["old_t0"] - a["truth"]) > CUT]
        newk = "new_t0" if kind == "move" else "rem_t0"
        fixed = [a for a in was_wrong if abs(a[newk] - a["truth"]) <= CUT]
        was_right = [a for a in sel if abs(a["old_t0"] - a["truth"]) <= CUT]
        broke = [a for a in was_right if abs(a[newk] - a["truth"]) > CUT]
        print(f"  {kind:<7} n={len(sel)} E={e_tot/1e3:.1f} GeV | was-wrong {len(was_wrong)} "
              f"-> fixed {len(fixed)} ({sum(a['energy'] for a in fixed)/1e3:.2f} GeV) | "
              f"was-right {len(was_right)} -> broke {len(broke)} "
              f"({sum(a['energy'] for a in broke)/1e3:.2f} GeV)")
        if kind == "detach":
            note = ("(rem_t0 vs parent truth is only a proxy; detached part gets "
                    "its own placement in Phase 2)")
            print(f"          {note}")

sw = [a for r in rows for a in r.get("swap_actions", [])]
if sw:
    print(f"\nSWAP phase: {len(sw)} actions "
          f"({sum(a['kind'] == 'swap' for a in sw)} swap-halves, "
          f"{sum(a['kind'] == 'move' for a in sw)} moves)")
    for kind in ("swap", "move"):
        sel = [a for a in sw if a["kind"] == kind and a["truth"] is not None]
        if not sel:
            continue
        ww = [a for a in sel if abs(a["old_t0"] - a["truth"]) > CUT]
        fx = [a for a in ww if abs(a["new_t0"] - a["truth"]) <= CUT]
        wr_ = [a for a in sel if abs(a["old_t0"] - a["truth"]) <= CUT]
        br = [a for a in wr_ if abs(a["new_t0"] - a["truth"]) > CUT]
        print(f"  {kind:<5} n={len(sel)} E={sum(a['energy'] for a in sel)/1e3:.1f} GeV | "
              f"was-wrong {len(ww)} -> fixed {len(fx)} ({sum(a['energy'] for a in fx)/1e3:.2f} GeV) | "
              f"was-right {len(wr_)} -> broke {len(br)} ({sum(a['energy'] for a in br)/1e3:.2f} GeV)")
