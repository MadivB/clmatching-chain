"""Compare two overflow_rescue_pipeline output dirs on their common events.

Usage: python expand_ab_score.py v040expA_out v040expB_out
Prints NS/all-TPC efficiency, phase-3 stage diagnostics and cohort accuracies.
"""
import glob
import json
import sys

import numpy as np


def load(d):
    rows = {}
    for f in glob.glob(f"{d}/groups_worker*.jsonl"):
        for line in open(f):
            r = json.loads(line)
            rows[(r["file"].split(".")[-3], r["event"])] = r
    return rows


def wmean(vals, wts):
    v, w = np.asarray(vals, float), np.asarray(wts, float)
    ok = np.isfinite(v) & np.isfinite(w)
    return float((v[ok] * w[ok]).sum() / max(w[ok].sum(), 1e-9))


def report(name, rows, common):
    m = [rows[c]["m_rescued"] for c in common]
    te = [r["truth_energy"] for r in [rows[c] for c in common]]
    ns_te = [x.get("ns_total_eff") for x in m]
    ns_acc = [x.get("ns_acc") for x in m]
    fin = [rows[c]["acc_rescued"] for c in common]
    ex = [rows[c]["phase3"].get("expand") or {} for c in common]
    e_acc = [x.get("p3e_acc_new") for x in m]
    e_frac = [x.get("p3e_new_efrac") for x in m]
    p_acc = [x.get("p3_acc_new") for x in m]
    p_frac = [x.get("p3_new_efrac") for x in m]
    n_asn = sum(x.get("n_assigned", 0) for x in ex)
    e_asn = sum(x.get("e_assigned", 0.0) for x in ex)
    mx = [rows[c]["phase3"].get("matrix") or {} for c in common]
    print(f"{name} ({len(common)} common events):")
    print(f"  NS total-eff {wmean(ns_te, te):.4f}   NS acc {wmean(ns_acc, te):.4f}   "
          f"all-TPC final {wmean(fin, te):.4f}")
    print(f"  expansion: {n_asn} clusters / {e_asn/1000:.1f} GeV assigned; "
          f"cohort acc {wmean(e_acc, e_frac):.3f} on {np.nanmean(np.array(e_frac, float))*100:.2f}% of E")
    print(f"  phase3 new-cohort acc {wmean(p_acc, p_frac):.3f} on "
          f"{np.nanmean(np.array(p_frac, float))*100:.2f}% of E")
    print(f"  matrix: proposed {sum(x.get('n_proposed',0) for x in mx)} "
          f"assigned {sum(x.get('n_assigned',0) for x in mx)}   "
          f"expand stats: ej={sum(x.get('ejections',0) for x in ex)} "
          f"newt0={sum(x.get('new_t0',0) for x in ex)} "
          f"dis={sum(x.get('dissents',0) for x in ex)} "
          f"frz={sum(x.get('freezes',0) for x in ex)}")


def main():
    da, db = sys.argv[1], sys.argv[2]
    ra, rb = load(da), load(db)
    common = sorted(set(ra) & set(rb))
    only_a, only_b = sorted(set(ra) - set(rb)), sorted(set(rb) - set(ra))
    if only_a or only_b:
        print(f"non-common events dropped: A-only {only_a}  B-only {only_b}")
    report(da, ra, common)
    report(db, rb, common)
    # paired per-event delta on NS total-eff
    d = [(rb[c]["m_rescued"].get("ns_total_eff") or np.nan)
         - (ra[c]["m_rescued"].get("ns_total_eff") or np.nan) for c in common]
    d = np.array(d, float)
    d = d[np.isfinite(d)]
    print(f"paired per-event delta NS total-eff (B - A): mean {d.mean():+.4f}  "
          f"median {np.median(d):+.4f}  better/worse/tied "
          f"{(d > 1e-6).sum()}/{(d < -1e-6).sum()}/{(np.abs(d) <= 1e-6).sum()}")


if __name__ == "__main__":
    main()
