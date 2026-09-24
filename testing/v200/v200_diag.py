"""Where does the v2.0 light-only optimum beat or lose to the chain, cluster by cluster?

python v200_diag.py results.json
Per cluster (from the run's diagnostics): E, light significance, self-scan margin, decidable
flag, own-dominated channel count n_dom, v200 time, chain time, truth time.  Tables:
 1. energy class x outcome (both right / v200 only / chain only / both wrong), energy-weighted;
 2. for clusters where v200 and chain disagree: P(v200 right) vs n_dom and vs margin;
 3. decidable clusters: P(own placement right) vs n_dom, per energy class.
Truth is used here only to read the diagnostics, never by the solver."""
import json
import sys
from collections import defaultdict

import numpy as np

CUT = 10.0
EB = [(0, 1.5, "<1.5"), (1.5, 5, "1.5-5"), (5, 20, "5-20"), (20, 100, "20-100"), (100, 1e9, ">=100")]
DOMB = [(0, 0, "0"), (1, 2, "1-2"), (3, 5, "3-5"), (6, 1e9, ">=6")]
MARB = [(0, 25, "<25"), (25, 100, "25-100"), (100, 1000, "100-1k"), (1000, 1e18, ">1k")]


def bucket(v, table):
    for lo, hi, name in table:
        if lo <= v <= hi if lo == hi else lo <= v < hi:
            return name
    return table[-1][2]


def main(path):
    rows = []
    for r in json.load(open(path)):
        for c in r["clusters"]:
            rows.append(c)
    R = np.array([[c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8]] for c in rows], dtype=float)
    E, sig, margin, decid, ndom, tv, tc, tt = R.T
    ok = np.isfinite(tt) & (tv >= 0) & (tc >= 0)
    rv = np.abs(tv - tt) <= CUT; rc = np.abs(tc - tt) <= CUT
    print(f"clusters with light: {len(rows)}, scored {int(ok.sum())}")
    print("\n1. outcome by energy class (energy-weighted MeV):")
    print(f"{'class':>8} {'both ok':>9} {'v200 only':>10} {'chain only':>11} {'both wrong':>11} {'v200 %wrong':>12} {'chain %wrong':>13}")
    for lo, hi, name in EB:
        m = ok & (E >= lo) & (E < hi)
        tot = E[m].sum()
        print(f"{name:>8} {E[m & rv & rc].sum():9.0f} {E[m & rv & ~rc].sum():10.0f} {E[m & ~rv & rc].sum():11.0f} {E[m & ~rv & ~rc].sum():11.0f} "
              f"{100*E[m & ~rv].sum()/max(tot,1e-9):12.2f} {100*E[m & ~rc].sum()/max(tot,1e-9):13.2f}")
    dis = ok & (np.abs(tv - tc) > 3)
    print(f"\n2. disagreements (|t_v200 - t_chain| > 3): {int(dis.sum())} clusters, {E[dis].sum():.0f} MeV")
    print(f"{'n_dom':>6} {'n':>6} {'E MeV':>8} {'P(v200 right)':>14} {'P(chain right)':>15}")
    for lo, hi, name in DOMB:
        m = dis & (ndom >= lo) & (ndom <= hi)
        if not m.any():
            continue
        print(f"{name:>6} {int(m.sum()):6d} {E[m].sum():8.0f} {100*E[m & rv].sum()/max(E[m].sum(),1e-9):14.1f} {100*E[m & rc].sum()/max(E[m].sum(),1e-9):15.1f}")
    print(f"{'margin':>7} {'n':>6} {'E MeV':>8} {'P(v200 right)':>14} {'P(chain right)':>15}")
    for lo, hi, name in MARB:
        m = dis & (margin >= lo) & (margin < hi)
        if not m.any():
            continue
        print(f"{name:>7} {int(m.sum()):6d} {E[m].sum():8.0f} {100*E[m & rv].sum()/max(E[m].sum(),1e-9):14.1f} {100*E[m & rc].sum()/max(E[m].sum(),1e-9):15.1f}")
    print("\n3. decidable clusters: P(v200 placement right) by n_dom x energy class (E-weighted %, n):")
    print(f"{'class':>8} " + " ".join(f"{name:>14}" for _, _, name in DOMB))
    for lo, hi, name in EB:
        cells = []
        for dlo, dhi, dname in DOMB:
            m = ok & (decid > 0) & (E >= lo) & (E < hi) & (ndom >= dlo) & (ndom <= dhi)
            cells.append(f"{100*E[m & rv].sum()/max(E[m].sum(),1e-9):6.1f} ({int(m.sum()):5d})" if m.any() else f"{'-':>14}")
        print(f"{name:>8} " + " ".join(cells))
    print("\n   same for the chain on those clusters:")
    for lo, hi, name in EB:
        cells = []
        for dlo, dhi, dname in DOMB:
            m = ok & (decid > 0) & (E >= lo) & (E < hi) & (ndom >= dlo) & (ndom <= dhi)
            cells.append(f"{100*E[m & rc].sum()/max(E[m].sum(),1e-9):6.1f} ({int(m.sum()):5d})" if m.any() else f"{'-':>14}")
        print(f"{name:>8} " + " ".join(cells))
    und = ok & (decid == 0)
    print(f"\n4. undecidable clusters: {int(und.sum())}, {E[und].sum():.0f} MeV; v200 right {100*E[und & rv].sum()/max(E[und].sum(),1e-9):.1f} %, chain right {100*E[und & rc].sum()/max(E[und].sum(),1e-9):.1f} %")


if __name__ == "__main__":
    main(sys.argv[1])
