"""Shower-eat A/B: baseline dir vs --shower-eat dir on common events.

Usage: python se_ab_score.py v101LE_out v105SE_out
Headlines (NS acc / all-TPC final), the LE-blip guard (le_acc/le_efrac),
latency, per-event deltas, and the eat ledger with its truth audit.
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


def headline(rows, common):
    m = [rows[c]["m_rescued"] for c in common]
    te = [rows[c]["truth_energy"] for c in common]
    return {
        "ns_acc": wmean([x.get("ns_acc") for x in m], te),
        "all": wmean([rows[c]["acc_rescued"] for c in common], te),
        "ns_te": wmean([x.get("ns_total_eff") for x in m], te),
        "le_acc": wmean([x.get("le_acc") for x in m],
                        [x.get("le_efrac", 0.0) or 0.0 for x in m]),
        "le_efrac": float(np.nanmean([x.get("le_efrac", np.nan) for x in m])),
        "t_ev": float(np.mean([rows[c]["t_event_s"] for c in common])),
    }


def main():
    da, db = sys.argv[1], sys.argv[2]
    ra, rb = load(da), load(db)
    common = sorted(set(ra) & set(rb))
    print(f"common events: {len(common)}  ({da} vs {db})")
    ha, hb = headline(ra, common), headline(rb, common)
    print(f"{'':14s}{'baseline':>10s}{'shower-eat':>12s}{'delta':>9s}")
    for k, lab in [("ns_acc", "NS acc"), ("all", "all-TPC"),
                   ("ns_te", "NS tot-eff"), ("le_acc", "LE acc"),
                   ("le_efrac", "LE efrac"), ("t_ev", "s/event")]:
        print(f"{lab:14s}{ha[k]:10.4f}{hb[k]:12.4f}{hb[k]-ha[k]:+9.4f}")

    up = down = 0
    worst = []
    for c in common:
        d = rb[c]["acc_rescued"] - ra[c]["acc_rescued"]
        if d > 1e-9:
            up += 1
        elif d < -1e-9:
            down += 1
        worst.append((d, c))
    print(f"per-event all-TPC: {up} up / {down} down / "
          f"{len(common)-up-down} tied")
    for d, c in sorted(worst)[:3]:
        print(f"  worst ev {c}: {d:+.4f}")

    # eat ledger
    tot = {"expl_seeds": 0, "emrg_groups": 0, "n_eaten": 0, "e_eaten": 0.0,
           "veto": 0, "misfit": 0, "weak": 0, "ok": 0, "wrong": 0}
    fs, fe, aud = [], [], []
    for c in common:
        se = rb[c].get("shower_eat")
        if not se:
            continue
        for part in ("explicit", "emergent"):
            st = se.get(part) or {}
            tot["n_eaten"] += st.get("n_eaten", 0)
            tot["e_eaten"] += st.get("e_eaten", 0.0)
            tot["veto"] += st.get("n_veto_contact", 0)
            tot["misfit"] += st.get("n_misfit", 0)
            tot["weak"] += st.get("n_weak_want", 0)
            fs += st.get("fulfill_start", [])
            fe += st.get("fulfill_end", [])
        tot["expl_seeds"] += (se.get("explicit") or {}).get("n_seeds", 0)
        tot["emrg_groups"] += se.get("n_groups", 0)
        tot["ok"] += se.get("eaten_ok", 0)
        tot["wrong"] += se.get("eaten_wrong", 0)
        aud += se.get("audit", [])
    print(f"\neat ledger: {tot['expl_seeds']} explicit seeds, "
          f"{tot['emrg_groups']} emergent groups; "
          f"ate {tot['n_eaten']} clusters / {tot['e_eaten']/1000:.2f} GeV")
    print(f"  guards: contact-veto {tot['veto']}  misfit {tot['misfit']}  "
          f"weak-want {tot['weak']}")
    na = tot["ok"] + tot["wrong"]
    if na:
        print(f"  truth audit: {tot['ok']}/{na} eaten correctly "
              f"({100*tot['ok']/na:.1f}%)")
        e_ok = sum(a["e"] for a in aud if a.get("ok"))
        e_all = sum(a["e"] for a in aud) or 1e-9
        print(f"  energy-weighted: {100*e_ok/e_all:.1f}% of eaten energy correct")
    if fs:
        print(f"  fulfillment: start med {np.median(fs):.2f} -> "
              f"end med {np.median(fe):.2f}  "
              f"(reached 60%: {sum(1 for x in fe if x >= 0.6)}/{len(fe)})")


if __name__ == "__main__":
    main()
