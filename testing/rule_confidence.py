"""Rule-based cluster confidence (user 2026-09-03: no MC-trained ML in the
data path — at most 4 thresholds tuned by looking at MC truth, combined
cleverly).

Rule form: k binary checks (k <= 4), each "feature (signed) >= threshold";
confidence LEVEL = number of passed checks (0..k). The delivered 0-1 value
is the level's measured E-weighted purity on file 0000001 (a k+1-entry
lookup table, fully inspectable - not a model).

Search: all 3- and 4-subsets of a physics candidate pool x coordinate-ascent
thresholds over quantile grids; objective = exact E-weighted midrank AUC of
the level score.

Usage: python rule_confidence.py v126MG_out
"""
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v083_prune as vp
from subtick_conf_study2 import load, auc_rank, GNAMES
from scan_margin import SCAN_NAMES as SCAN28

# candidate pool: (name, sign) - sign*value >= sign*threshold means PASS
POOL = [
    ("depth_n", +1),      # light visible at the chosen time
    ("margin2_n", +1),    # delta-chi2 to best competitor
    ("rel_margin", +1),   # margin / depth
    ("pcu_vote10", +1),   # channels agree on the location
    ("pcu_sig_med", +1),  # per-channel minima are CLEAR, not flat
    ("pcu_vote_med", -1), # median channel vote distance
    ("c_cosres", +1),     # residual pattern matches own light shape
    ("z_min", +1),        # anticorrelation swap partner (more neg = worse)
    ("dz", -1),           # z_own - z_min regret
    ("d_eu", -1),         # spatial isolation (works for light-blind)
]


def main():
    outdir = sys.argv[1]
    X, G, y, eg, ev, keys = load(outdir, with_keys=True)
    sm = np.load(os.path.join(outdir, "scan_margin_feats.npz"), allow_pickle=True)
    names_s = list(sm["names"])
    cols = [names_s.index(k) for k in SCAN28]
    lut = {(t, int(l)): f for t, l, f in zip(sm["tags"], sm["lab"], sm["F"][:, cols])}
    S = np.zeros((len(y), len(SCAN28) + 1))
    for i, k in enumerate(keys):
        f = lut.get(k)
        if f is not None:
            S[i, :-1] = np.nan_to_num(f, nan=0.0, posinf=1e9, neginf=-1e9)
            S[i, -1] = 1.0
    F = np.hstack([X, G, S])
    NAMES = list(vp.ALL_FEATS) + list(GNAMES) + list(SCAN28) + ["has_scan"]
    wrong = y == 0
    print(f"{len(y)} clusters, {wrong.sum()} wrong ({100*eg[wrong].sum()/eg.sum():.2f}% E)")

    feats = {}
    for nm, sg in POOL:
        v = sg * F[:, NAMES.index(nm)]
        feats[nm] = (v, sg)
    grids = {nm: np.unique(np.quantile(v, np.linspace(0.05, 0.95, 19)))
             for nm, (v, _) in feats.items()}

    def level_score(sel, ths):
        lv = np.zeros(len(y), np.int8)
        for nm, th in zip(sel, ths):
            lv += (feats[nm][0] >= th)
        return lv

    def objective(sel, ths):
        return auc_rank(y, level_score(sel, ths).astype(float), eg)

    results = []
    for k in (3, 4):
        for sel in itertools.combinations([nm for nm, _ in POOL], k):
            ths = [np.median(grids[nm]) for nm in sel]
            best = objective(sel, ths)
            for _sweep in range(3):
                improved = False
                for j, nm in enumerate(sel):
                    cur = ths[j]
                    for cand in grids[nm]:
                        ths[j] = cand
                        a = objective(sel, ths)
                        if a > best + 1e-6:
                            best, cur, improved = a, cand, True
                    ths[j] = cur
                if not improved:
                    break
            results.append((best, sel, list(ths)))
    results.sort(key=lambda r: -r[0])
    print("\ntop 8 rules (E-wt AUC of level score):")
    for a, sel, ths in results[:8]:
        lv = level_score(sel, ths).astype(float)
        print(f"  {a:.3f} (count {auc_rank(y, lv):.3f}): "
              + " & ".join(f"{nm}{'>=' if feats[nm][1]>0 else '<='}"
                           f"{feats[nm][1]*th:.3g}" for nm, th in zip(sel, ths)))

    a, sel, ths = results[0]
    lv = level_score(sel, ths)
    print(f"\nWINNER ({len(sel)} thresholds): E-wt AUC {a:.3f}, "
          f"count {auc_rank(y, lv.astype(float)):.3f}")
    for lo, hi, lab in [(5, 20, "5-20"), (20, 100, "20-100"), (100, 1e9, ">=100")]:
        m = (eg >= lo) & (eg < hi)
        if m.sum() > 10 and len(set(y[m])) == 2:
            print(f"  {lab}: {auc_rank(y[m], lv[m].astype(float)):.3f}")
    print("\nper-level purity table (the delivered confidence values):")
    table = {}
    for L in range(len(sel) + 1):
        m = lv == L
        if not m.any():
            continue
        pe = float(eg[m & (y == 1)].sum() / max(eg[m].sum(), 1e-9))
        pc = float((y[m] == 1).mean())
        table[int(L)] = round(pe, 4)
        print(f"  level {L}: n={m.sum():6d} ({100*m.mean():5.1f}%)  "
              f"E-share {100*eg[m].sum()/eg.sum():5.1f}%  "
              f"E-purity {100*pe:5.1f}%  count-purity {100*pc:5.1f}%")
    rule = {"checks": [{"feature": nm,
                        "op": ">=" if feats[nm][1] > 0 else "<=",
                        "threshold": float(feats[nm][1] * th)}
                       for nm, th in zip(sel, ths)],
            "level_purity_E": table,
            "tuned_on": "file 0000001 (13 events), 2026-09-03",
            "auc_Ewt": round(a, 4)}
    json.dump(rule, open("conf_rule_v0101.json", "w"), indent=1)
    print("\nrule -> conf_rule_v0101.json")


if __name__ == "__main__":
    main()
