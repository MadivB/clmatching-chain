"""Per-stage performance report from a harness output dir.

Usage: python stage_report.py <out_dir> [<out_dir2> ...]
Aggregates the instrumented row keys (pool_stats, rescue_audit, phase3.*,
m_rescued trajectory) into the per-stage table: placements, corrections,
truth-audited correctness, and pool batch/iteration statistics.
"""
import glob
import json
import sys

import numpy as np


def load(dirs):
    rows = []
    for d in dirs:
        for f in glob.glob(f"{d}/groups_worker*.jsonl"):
            for line in open(f):
                rows.append(json.loads(line))
    return rows


def wmean(rows, key, wkey="truth_energy", sub="m_rescued"):
    v = [(r[sub].get(key) if sub else r.get(key)) for r in rows]
    w = [r[wkey] for r in rows]
    v, w = np.asarray(v, float), np.asarray(w, float)
    ok = np.isfinite(v)
    if not ok.any():
        return float("nan")
    return float((v[ok] * w[ok]).sum() / w[ok].sum())


def cohort_acc(rows, acc_key, frac_key):
    """Energy-weighted cohort accuracy: sum(TE*frac*acc)/sum(TE*frac).
    Events with frac ~ 0 or missing acc contribute nothing."""
    num = den = 0.0
    for r in rows:
        te = float(r["truth_energy"])
        f = r["m_rescued"].get(frac_key) or 0.0
        a = r["m_rescued"].get(acc_key)
        if f > 0 and a is not None and np.isfinite(a):
            num += te * f * a
            den += te * f
    return (num / den) if den > 0 else float("nan"), den


def main():
    rows = load(sys.argv[1:])
    n = len(rows)
    print(f"=== per-stage performance over {n} events ({', '.join(sys.argv[1:])}) ===\n")

    # ---- stage 2 + 3.1: merged permutation pool ----
    ps = [r.get("pool_stats") for r in rows if r.get("pool_stats")]
    if ps:
        hist = {}
        for p in ps:
            for k, v in (p.get("batch_size_hist") or {}).items():
                hist[k] = hist.get(k, 0) + v
        print("STAGE 2+3.1 (Phase 1 + merged permutation pool), per event:")
        print(f"  showers {np.mean([p['n_showers'] for p in ps]):.1f} | "
              f">=3-TPC tracks {np.mean([p['n_multi3'] for p in ps]):.1f} | "
              f"pool size {np.mean([p['n_pool'] for p in ps]):.1f}")
        print(f"  batches/event {np.mean([p['n_batches'] for p in ps]):.1f} "
              f"(= placement iterations); scans/event {np.mean([p['n_scans'] for p in ps]):.0f}")
        print("  batch-size histogram (avg per event): "
              + "  ".join(f"{k}-cluster: {hist[k]/len(ps):.1f}"
                          for k in sorted(hist, key=int)))
        pt_, pb = sum(p["perm_total"] for p in ps), sum(p["perm_better"] for p in ps)
        print(f"  permutation batches where a non-energy order won: "
              f"{pb}/{pt_} ({100*pb/max(pt_,1):.0f}%)")
        print(f"  defer-gate deferrals/event {np.mean([p['n_deferred'] for p in ps]):.1f} | "
              f"pool swap-pass swaps/event {np.mean([p['swap_pass_swaps'] for p in ps]):.1f} | "
              f"pool rescue-pass flagged/moved/event "
              f"{np.mean([p['rescue_pass_flagged'] for p in ps]):.1f}/"
              f"{np.mean([p['rescue_pass_moved'] for p in ps]):.1f}")
        print(f"  pool time/event {np.mean([p['time_s'] for p in ps]):.1f}s\n")

    # ---- stage 3.2: trigger + saving ----
    audits = [a for r in rows for a in (r.get("rescue_audit") or [])]
    moves = [a for a in audits if a["kind"] == "move"]
    fixed = sum(1 for a in moves if not a.get("was_ok") and a.get("now_ok"))
    broke = sum(1 for a in moves if a.get("was_ok") and not a.get("now_ok"))
    neut = len(moves) - fixed - broke
    det = [a for a in audits if a["kind"] == "detach"]
    print("STAGE 3.2 (overflow trigger + saving):")
    print(f"  moves {len(moves)} total ({len(moves)/n:.1f}/event): "
          f"FIXED {fixed}, BROKE {broke}, neutral {neut} "
          f"(net {fixed-broke:+d}); moved energy "
          f"{sum(a['e'] for a in moves)/max(n,1):.0f} MeV/event")
    print(f"  detached parts {len(det)} ({len(det)/n:.1f}/event) -> re-placed in 3.3\n")

    # ---- stages 3.3 + 3.5 ----
    print("STAGE 3.3+3.5 (P2 leftover scan + V2, police inline):")
    print(f"  post-V2 accuracy among assigned {100*wmean(rows,'v2_acc'):.2f}% "
          f"at coverage {100*wmean(rows,'v2_afrac'):.1f}%\n")

    # ---- stage 3.9 expansion ----
    ex = [r["phase3"].get("expand") or {} for r in rows if r.get("phase3")]
    e_asn = sum(x.get("e_assigned", 0) for x in ex)
    print("STAGE 3.9 (family expansion):")
    print(f"  assigned {sum(x.get('n_assigned',0) for x in ex)} clusters "
          f"({sum(x.get('n_assigned',0) for x in ex)/n:.0f}/event), "
          f"{e_asn/max(n,1):.0f} MeV/event")
    _ea, _ = cohort_acc(rows, "p3e_acc_new", "p3e_new_efrac")
    print(f"  cohort accuracy (how much of it was RIGHT) {100*_ea:.1f}% "
          f"-> wrong {100-100*_ea:.1f}%")
    print(f"  ejections {sum(x.get('ejections',0) for x in ex)}, "
          f"new-t0 openings {sum(x.get('new_t0',0) for x in ex)}, "
          f"dissents {sum(x.get('dissents',0) for x in ex)}\n")

    # ---- stage 4.1 matrix ----
    mx = [r["phase3"].get("matrix") or {} for r in rows if r.get("phase3")]
    print("STAGE 4.1 (error-matrix tail):")
    print(f"  proposed {sum(x.get('n_proposed',0) for x in mx)/n:.0f}/event, "
          f"assigned {sum(x.get('n_assigned',0) for x in mx)/n:.0f}/event")
    num = den = tot = 0.0
    for r in rows:
        te = float(r["truth_energy"]); m = r["m_rescued"]
        fa_i = m.get("p3_new_efrac") or 0.0
        fe_i = m.get("p3e_new_efrac") or 0.0
        aa_i = m.get("p3_acc_new") if fa_i > 0 else 0.0
        ae_i = m.get("p3e_acc_new") if fe_i > 0 else 0.0
        num += te * ((aa_i or 0.0) * fa_i - (ae_i or 0.0) * fe_i)
        den += te * (fa_i - fe_i)
        tot += te
    if den / max(tot, 1e-9) > 1e-4:
        print(f"  matrix+force cohort accuracy {100*num/den:.1f}% on "
              f"{100*den/tot:.1f}% of event energy\n")
    else:
        print("  matrix+force cohort: negligible energy (expansion covered "
              "all new placements)\n")

    # ---- stage 4.5 swap fix ----
    sw = [r["phase3"].get("swap") or {} for r in rows if r.get("phase3")]
    print("STAGE 4.5 (per-TPC swap fix):")
    print(f"  swaps {sum(x.get('n_swaps',0) for x in sw)} total: "
          f"GOOD {sum(x.get('n_good',0) for x in sw)}, "
          f"BAD {sum(x.get('n_bad',0) for x in sw)} "
          f"(net {sum(x.get('n_good',0) for x in sw)-sum(x.get('n_bad',0) for x in sw):+d}); "
          f"{sum(x.get('e_swapped',0) for x in sw)/max(n,1):.0f} MeV/event touched\n")

    # ---- stage 4.9 force ----
    fc = [r["phase3"].get("force") or {} for r in rows if r.get("phase3")]
    print("STAGE 4.9 (force coverage):")
    print(f"  imaged {sum(x.get('n_img',0) for x in fc)/n:.0f} + no-image "
          f"{sum(x.get('n_noimg',0) for x in fc)/n:.0f} clusters + noise "
          f"{sum(x.get('n_noise',0) for x in fc)/n:.0f} hits/event -> coverage 100%")
    fnum = fden = 0.0
    for r in rows:
        f3 = (r.get("phase3") or {}).get("force") or {}
        a = f3.get("force_acc_new"); fr = f3.get("force_new_efrac") or 0.0
        if a is not None and np.isfinite(a) and fr > 0:
            fnum += float(r["truth_energy"]) * fr * a
            fden += float(r["truth_energy"]) * fr
    if fden > 0:
        print(f"  force cohort accuracy {100*fnum/fden:.1f}% "
              f"(energy-weighted)\n")

    print(f"FINAL: NS total-eff {100*wmean(rows,'ns_total_eff'):.2f}% | "
          f"all-TPC {100*wmean(rows,'acc_rescued',sub=None,wkey='truth_energy'):.2f}%")


if __name__ == "__main__":
    main()
