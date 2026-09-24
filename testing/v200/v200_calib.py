"""In-situ prior calibration for v2.0_test (no truth anywhere).

From pass-1 results (light-only solve):
 * beta[contact feature]  = 2 ln P/(1-P), P = P(same time | contact pair, both clusters
   light-decidable)              -> pairwise Potts term when a contact pair disagrees
 * dprior[E bucket_d bucket] = -2 ln P/(1-P), P = P(a decidable cluster sits at family f |
   distance to f)                -> unary penalty for an undecidable cluster choosing f
 * pnew[E bucket]  = -2 ln P/(1-P), P = P(a decidable cluster sits at a time no other family
   in its TPCs has)              -> unary penalty for an undecidable cluster choosing its own time
python v200_calib.py results.json [...] --out prior.json [--min-count 20]"""
import argparse
import json
import math
from collections import defaultdict


def logit2(p: float) -> float:
    return 2.0 * math.log(p / (1.0 - p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-count", type=int, default=20)
    a = ap.parse_args()
    n = defaultdict(int); s = defaultdict(int)
    dn = defaultdict(int); ds = defaultdict(int)
    nn = defaultdict(int); ns = defaultdict(int)
    for f in a.results:
        for r in json.load(open(f)):
            for key, same, *_ in r.get("pair_obs", []):
                n[key] += 1; s[key] += int(same)
            for eb, db, same in r.get("dprior_obs", []):
                dn[f"{eb}_{db}"] += 1; ds[f"{eb}_{db}"] += int(same)
            for eb, is_new in r.get("new_obs", []):
                nn[eb] += 1; ns[eb] += int(is_new)
    beta, dprior, pnew = {}, {}, {}
    print("contact prior (pairwise):")
    print(f"{'feature':>8} {'pairs':>6} {'P(same)':>8} {'beta':>7}")
    for key in sorted(n):
        p = (s[key] + 0.5) / (n[key] + 1.0)
        b = logit2(p)
        flag = "" if n[key] >= a.min_count else "  (too few, 0)"
        print(f"{key:>8} {n[key]:6d} {p:8.3f} {b:7.2f}{flag}")
        beta[key] = b if n[key] >= a.min_count else 0.0
    print("\nfamily-distance prior (unary, penalty = -2 ln odds):")
    print(f"{'feature':>8} {'obs':>6} {'P(same)':>8} {'penalty':>8}")
    for key in sorted(dn):
        p = (ds[key] + 0.5) / (dn[key] + 1.0)
        pen = -logit2(p)
        flag = "" if dn[key] >= a.min_count else "  (too few, 0)"
        print(f"{key:>8} {dn[key]:6d} {p:8.3f} {pen:8.2f}{flag}")
        dprior[key] = pen if dn[key] >= a.min_count else 0.0
    print("\nnew-slot rate (unary penalty for an own time):")
    for key in sorted(nn):
        p = (ns[key] + 0.5) / (nn[key] + 1.0)
        pen = -logit2(p)
        print(f"{key:>8} {nn[key]:6d} {p:8.3f} {pen:8.2f}")
        pnew[key] = pen if nn[key] >= a.min_count else 0.0
    json.dump({"beta": beta, "dprior": dprior, "pnew": pnew}, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
