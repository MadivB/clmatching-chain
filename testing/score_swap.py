"""Compare the --swap-fix run (v030SW_out) against the D50 baseline (v030D50_out).

Reports pooled non-shower metrics and all-TPC accuracy for both, the swap-stage
audit (swaps applied, good vs bad by truth, energy recovered, acc before/after),
and per-event deltas.
"""
import glob, json, os, sys
import numpy as np
HERE = os.path.abspath(os.path.dirname(__file__))
SW_DIR = sys.argv[1] if len(sys.argv) > 1 else "v030SW_out"

def load(tag):
    rows = {}
    for fp in glob.glob(os.path.join(HERE, f"{tag}/groups_worker*.jsonl")):
        for l in open(fp):
            if l.strip():
                r = json.loads(l); rows[(r["file"], r["event"])] = r
    return rows

D = load("v030D50_out"); S = load(SW_DIR)
keys = sorted(set(D) & set(S))
print(f"D50 events={len(D)}  SW events={len(S)}  common={len(keys)}\n")

def wmean(rows, key, wkey="truth_energy"):
    num = den = 0.0
    for r in rows:
        m = r["m_rescued"]
        if m.get(key) is None:
            continue
        w = r.get(wkey, 1.0)
        num += m[key] * w; den += w
    return num / max(den, 1e-9)

for tag, rows in (("D50 (baseline)", [D[k] for k in keys]), ("SW (+swap-fix)", [S[k] for k in keys])):
    print(f"{tag:>18}: ns_total_eff={wmean(rows,'ns_total_eff'):.4f}  "
          f"ns_acc={wmean(rows,'ns_acc'):.4f}  "
          f"ns_cov={wmean(rows,'ns_coverage'):.4f}  "
          f"p3_acc(all)={wmean(rows,'p3_acc'):.4f}")

dse = wmean([S[k] for k in keys], 'ns_total_eff') - wmean([D[k] for k in keys], 'ns_total_eff')
print(f"\n  ==> ns_total_eff delta (SW - D50): {dse*100:+.2f} pp")

# swap-stage audit
tot = dict(n_swaps=0, n_good=0, n_bad=0, e_swapped=0.0)
ab = aa = wsum = 0.0
per_ev = []
for k in keys:
    sw = (S[k].get("phase3") or {}).get("swap") or {}
    for f in ("n_swaps", "n_good", "n_bad"):
        tot[f] += sw.get(f, 0)
    tot["e_swapped"] += sw.get("e_swapped", 0.0)
    if sw.get("acc_before") is not None:
        w = S[k].get("truth_energy", 1.0)
        ab += sw["acc_before"] * w; aa += sw["acc_after"] * w; wsum += w
    de = S[k]["m_rescued"]["ns_total_eff"] - D[k]["m_rescued"]["ns_total_eff"]
    per_ev.append((de, S[k]["file"], k[1], sw.get("n_swaps", 0), sw.get("n_good", 0), sw.get("n_bad", 0)))

print(f"\nSWAP STAGE: {tot['n_swaps']} per-TPC swaps over {len(keys)} events "
      f"({tot['n_good']} good / {tot['n_bad']} bad by truth"
      f" = {100*tot['n_good']/max(tot['n_swaps'],1):.0f}% correct), "
      f"{tot['e_swapped']/1e3:.1f} GeV moved")
print(f"pooled p3_acc(all-TPC) before swap {ab/max(wsum,1e-9):.4f} -> after {aa/max(wsum,1e-9):.4f} "
      f"({(aa-ab)/max(wsum,1e-9)*100:+.2f} pp)")

per_ev.sort()
def prow(t):
    de, f, ev, ns, ng, nb = t
    print(f"  {f[-13:-10]} ev{ev:<2} dEff={de*100:+6.2f}pp  swaps={ns} (good {ng}/bad {nb})")
print("\nMOST-IMPROVED events:")
for t in per_ev[-6:][::-1]:
    prow(t)
print("MOST-HURT events:")
for t in per_ev[:6]:
    prow(t)
print(f"\nevents improved: {sum(1 for t in per_ev if t[0]>1e-5)}  "
      f"unchanged: {sum(1 for t in per_ev if abs(t[0])<=1e-5)}  "
      f"hurt: {sum(1 for t in per_ev if t[0]<-1e-5)}")
