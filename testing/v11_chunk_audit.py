"""Analyse the v1.1 per-chunk decision audit:  python v11_chunk_audit.py <arm_dir> [...]

Pool chunks where the joint (combination) winner differs from the energy-order
greedy path: outcome (energy fixed/broke by the joint choice vs greedy) as a
function of the relative loss margin, and the net effect of accepting the
joint decision only above a margin floor.  Same for --redecide moves."""
import glob, json, sys
import numpy as np


def load(d):
    rows = []
    for f in glob.glob(f"{d}/groups_worker*.jsonl"):
        for line in open(f):
            rows.append(json.loads(line))
    return rows


def main(dirs):
    for d in dirs:
        rows = load(d)
        ch = [c for r in rows for c in ((r.get("pool_stats") or {}).get("chunk_audit") or [])]
        print(f"\n=== {d}: {len(rows)} events, {len(ch)} chunks where joint != greedy ===")
        if ch:
            dE, rg, nm = [], [], []
            for c in ch:
                if "greedy_ok" not in c:
                    continue
                de = 0.0
                for e_, g, k in zip(c["e"], c["greedy_ok"], c["chosen_ok"]):
                    if g is None or k is None:
                        continue
                    de += e_ * (float(k) - float(g))
                dE.append(de); rg.append(c["rel_gain"]); nm.append(len(c["labels"]))
            dE, rg, nm = np.array(dE), np.array(rg), np.array(nm)
            print(f"  joint vs greedy: net {dE.sum()/1e3:+.3f} GeV | chunks fixing {np.sum(dE>0.5)} (+{dE[dE>0.5].sum()/1e3:.3f}) | breaking {np.sum(dE<-0.5)} ({dE[dE<-0.5].sum()/1e3:.3f}) | neutral {np.sum(np.abs(dE)<=0.5)}")
            print(f"  rel_gain: median {np.median(rg):.4f} | fixing median {np.median(rg[dE>0.5]) if (dE>0.5).any() else float('nan'):.4f} | breaking median {np.median(rg[dE<-0.5]) if (dE<-0.5).any() else float('nan'):.4f}")
            print("  accept joint only if rel_gain > x:   x      net GeV   n_accepted  fixes  breaks")
            for x in (0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1):
                s = rg > x
                print(f"      {x:6.3f}  {dE[s].sum()/1e3:+8.3f}  {s.sum():6d}  {np.sum(s&(dE>0.5)):5d}  {np.sum(s&(dE<-0.5)):5d}")
            for lo, hi in ((2, 2), (3, 4), (5, 8)):
                s = (nm >= lo) & (nm <= hi)
                if s.any():
                    print(f"  chunk size {lo}-{hi}: n={s.sum()} net {dE[s].sum()/1e3:+.3f} GeV")
        rd = [a for r in rows for a in ((r.get("phase3") or {}).get("redecide_audit") or [])]
        st = [(r.get("phase3") or {}).get("redecide") or {} for r in rows]
        if any(st):
            print(f"  REDECIDE: participants/ev {np.mean([s.get('n_participants',0) for s in st]):.0f}, multi-components/ev {np.mean([s.get('n_multi',0) for s in st]):.0f}, chunks/ev {np.mean([s.get('n_chunks',0) for s in st]):.0f}, moved/ev {np.mean([s.get('n_moved',0) for s in st]):.1f}, time {np.median([s.get('time_s',0) for s in st]):.1f} s/ev")
            fx = sum(s.get("e_fixed", 0) for s in st); br = sum(s.get("e_broke", 0) for s in st)
            print(f"    moves: fixed {sum(s.get('n_fixed',0) for s in st)} (+{fx/1e3:.3f} GeV) broke {sum(s.get('n_broke',0) for s in st)} ({-br/1e3:.3f} GeV) net {(fx-br)/1e3:+.3f} GeV")
            if rd:
                rg = np.array([a.get("rel_gain", np.nan) for a in rd]); out = np.array([(1 if (not a.get("was_ok") and a.get("now_ok")) else -1 if (a.get("was_ok") and not a.get("now_ok")) else 0) for a in rd]); e = np.array([a["e"] for a in rd])
                cs = np.array([a.get("comp", 0) for a in rd])
                for lo, hi in ((2, 2), (3, 4), (5, 8)):
                    s_ = (cs >= lo) & (cs <= hi)
                    if s_.any():
                        print(f"    comp size {lo}-{hi}: moves {s_.sum()} net {(e[s_]*out[s_]).sum()/1e3:+.3f} GeV (fixes {np.sum(s_&(out>0))}, breaks {np.sum(s_&(out<0))})")
                for lo, hi in ((5, 20), (20, 100), (100, 1e9)):
                    s_ = (e >= lo) & (e < hi)
                    if s_.any():
                        print(f"    cluster E [{lo},{hi}): moves {s_.sum()} net {(e[s_]*out[s_]).sum()/1e3:+.3f} GeV (fixes {np.sum(s_&(out>0))}, breaks {np.sum(s_&(out<0))})")
                print("    accept move only if rel_gain > x:   x      net GeV   n   fixes  breaks")
                for x in (0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05):
                    s = rg > x
                    print(f"        {x:6.3f}  {(e[s]*out[s]).sum()/1e3:+8.3f}  {s.sum():4d}  {np.sum(s&(out>0)):4d}  {np.sum(s&(out<0)):4d}")


if __name__ == "__main__":
    main(sys.argv[1:])
