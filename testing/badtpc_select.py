"""Select a DIVERSE set of render targets from badtpc_ranked.json:
top swaps (headline) + worst LANDED_ON_FLASH + worst OFF_FLASH cases for
completeness.  Prints file:event:tpc tokens and the ranked indices used.
"""
import json, os, sys
HERE = os.path.abspath(os.path.dirname(__file__))
rows = json.load(open(os.path.join(HERE, "badtpc_ranked.json")))
ns = [r for r in rows if not r["shower"] and r["e_wrong_real"] > 0]
ns.sort(key=lambda r: -r["e_wrong_real"])

N_SWAP = int(sys.argv[1]) if len(sys.argv) > 1 else 6
N_LAND = int(sys.argv[2]) if len(sys.argv) > 2 else 2
N_OFF = int(sys.argv[3]) if len(sys.argv) > 3 else 2

picked, seen = [], set()
def take(pred, n, need_frac=0.5):
    got = 0
    for r in ns:
        if got >= n:
            break
        key = (r["file"], r["event"], r["tpc"])
        if key in seen:
            continue
        ce = r.get("catE", {}); tot = sum(ce.values()) or 1
        if pred(r, ce, tot):
            picked.append(r); seen.add(key); got += 1

# top swaps by wrong energy
take(lambda r, ce, tot: r.get("dom_cat") == "SWAP", N_SWAP)
# worst LANDED-dominated (its landed energy is the largest among its categories and substantial)
land = sorted([r for r in ns if r.get("dom_cat") == "LANDED"], key=lambda r: -r["catE"].get("LANDED", 0))
for r in land[:N_LAND]:
    key = (r["file"], r["event"], r["tpc"])
    if key not in seen:
        picked.append(r); seen.add(key)
off = sorted([r for r in ns if r.get("dom_cat") == "OFF"], key=lambda r: -r["catE"].get("OFF", 0))
for r in off[:N_OFF]:
    key = (r["file"], r["event"], r["tpc"])
    if key not in seen:
        picked.append(r); seen.add(key)

toks = ",".join(f"{r['file']}:{r['event']}:{r['tpc']}" for r in picked)
print(toks)
json.dump(picked, open(os.path.join(HERE, "bad_tpc_targets.json"), "w"), indent=1)
import sys as _s
for r in picked:
    print(f"  {r['file']} ev{r['event']} TPC{r['tpc']:<3} E={r['tpc_E']:.0f} eff={r['eff']:.2f} "
          f"wReal={r['e_wrong_real']:.0f} dom={r['dom_cat']} catE={r['catE']} nFlash={r.get('n_flashes')}",
          file=_s.stderr)
