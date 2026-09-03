"""User proposal: instead of weighting the two confidence families, take the
WORSE of the two (min of the calibrated scores) — a conservative AND.

Each family is scored by LOO-by-event logistic, then isotonic-calibrated
ALSO leave-one-event-out (no leakage), so both live on the same
"probability correct" scale before combining.

Combos: min / product / mean / learned 2-input stack, vs each family alone.
Reports count- & E-weighted AUC, banded AUC, and the catch table (wrong
clusters caught, and wrong ENERGY caught) at matched 10% / 20% flag rates.

Usage: python subtick_conf_min.py v125CD_out
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v083_prune as vp
from subtick_conf_study2 import load, loo, auc_rank

TRAD = [vp.ALL_FEATS.index(k) for k in ("z_min", "z_own", "dz", "chi2_loc", "n_tpc")]
QUADG = [1, 2, 3, 4]


def loo_iso(s, y, ev):
    from sklearn.isotonic import IsotonicRegression
    p = np.zeros_like(s, dtype=np.float64)
    for e in np.unique(ev):
        tr = ev != e
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(s[tr], y[tr])
        p[ev == e] = iso.predict(s[ev == e])
    return p


def report(name, s, y, eg, wrong):
    line = f"{name:22s} AUC {auc_rank(y, s):.3f} (E-wt {auc_rank(y, s, eg):.3f})"
    for lo, hi, lab in [(5, 20, "5-20"), (20, 100, "20-100"), (100, 1e9, ">=100")]:
        m = (eg >= lo) & (eg < hi)
        if m.sum() > 10 and len(set(y[m])) == 2:
            line += f"  {lab}:{auc_rank(y[m], s[m]):.3f}"
    print(line)


def catch(name, s, y, eg, wrong, q):
    fl = s <= np.quantile(s, q)
    cw = fl & wrong
    e_caught = eg[cw].sum() / max(eg[wrong].sum(), 1e-9)
    print(f"  {name:22s} flags {fl.sum():5d}  wrong caught {cw.sum():4d}/{wrong.sum()}"
          f"  wrong-E caught {100*e_caught:5.1f}%"
          f"  purity {100*cw.sum()/max(fl.sum(),1):4.1f}%")


def main():
    X, G, y, eg, ev = load(sys.argv[1])
    wrong = y == 0
    print(f"{len(y)} clusters, {wrong.sum()} wrong "
          f"({100*eg[wrong].sum()/eg.sum():.2f}% of energy)")
    s_tr = loo("logit", X[:, TRAD], y, ev)
    s_qd = loo("logit", G[:, QUADG], y, ev)
    p_tr = loo_iso(s_tr, y, ev)
    p_qd = loo_iso(s_qd, y, ev)
    s_stack = loo("logit", np.column_stack([s_tr, s_qd]), y, ev)
    combos = [
        ("traditional alone", p_tr),
        ("quadratic alone", p_qd),
        ("MIN (worse-of-two)", np.minimum(p_tr, p_qd)),
        ("product", p_tr * p_qd),
        ("mean", 0.5 * (p_tr + p_qd)),
        ("learned 2-in stack", s_stack),
    ]
    for name, s in combos:
        report(name, s, y, eg, wrong)
    for q in (0.10, 0.20):
        print(f"\ncatch table at {int(q*100)}% flag rate:")
        for name, s in combos:
            catch(name, s, y, eg, wrong, q)


if __name__ == "__main__":
    main()
