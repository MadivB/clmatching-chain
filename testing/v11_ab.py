"""v1.1 paired A/B ledger: <ref_dir> <arm_dir> [<arm_dir> ...]

Row-level (jsonl): NS total-eff and all-TPC accuracy, energy-weighted exactly
as stage_report.py's FINAL line, paired per (file, event) with better/worse
splits; per-file numbers; per-stage medians; pool statistics.
--dumps: add the hit-level wrong-energy breakdown by cluster category from
full_dumps (slower)."""
import glob, json, os, sys
import numpy as np

CUT = 10.0


def load(d):
    """d may be a comma-separated list of output dirs (merged, e.g. the v1.0
    reference = v138FS_out for files 1-3 + v141PT_out for files 4-10)."""
    rows = {}
    for dd in str(d).split(","):
        for f in glob.glob(f"{dd}/groups_worker*.jsonl"):
            for line in open(f):
                r = json.loads(line)
                rows[(os.path.basename(str(r["file"]))[-20:], int(r["event"]))] = r
    return rows


def wmean(rows, key, sub="m_rescued"):
    v = np.array([(r[sub].get(key) if sub else r.get(key)) for r in rows], float)
    w = np.array([r["truth_energy"] for r in rows], float)
    ok = np.isfinite(v)
    return float((v[ok] * w[ok]).sum() / w[ok].sum())


def ledger(ref, arm, name):
    keys = sorted(set(ref) & set(arm))
    R = [ref[k] for k in keys]; A = [arm[k] for k in keys]
    out = {"n": len(keys)}
    for label, key, sub in (("NS", "ns_total_eff", "m_rescued"), ("allTPC", "acc_rescued", None)):
        r0, a0 = wmean(R, key, sub), wmean(A, key, sub)
        dr = [((a[sub].get(key) if sub else a.get(key)) or 0) - ((r[sub].get(key) if sub else r.get(key)) or 0) for a, r in zip(A, R)]
        dr = np.array(dr)
        out[label] = (r0, a0, int((dr > 1e-6).sum()), int((dr < -1e-6).sum()))
    # LE class if present
    if all("le_acc" in r["m_rescued"] for r in R + A):
        out["LE"] = (wmean(R, "le_acc"), wmean(A, "le_acc"))
    # timing
    st_r = {}; st_a = {}
    for r in R:
        for k, v in (r.get("stage_t") or {}).items(): st_r.setdefault(k, []).append(v)
    for a in A:
        for k, v in (a.get("stage_t") or {}).items(): st_a.setdefault(k, []).append(v)
    out["t_ev"] = (np.median([r["t_event_s"] for r in R]), np.median([a["t_event_s"] for a in A]))
    out["pool_t"] = (np.median(st_r.get("2+3.1_phase1_backbone", [0])), np.median(st_a.get("2+3.1_phase1_backbone", [0])))
    out["tot_t"] = (np.median([sum(v for v in (r.get("stage_t") or {}).values()) for r in R]),
                    np.median([sum(v for v in (a.get("stage_t") or {}).values()) for a in A]))
    ps = [a.get("pool_stats") or {} for a in A]
    out["pool"] = {
        "batches": np.mean([p.get("n_batches", 0) for p in ps]),
        "scans": np.mean([p.get("n_scans", 0) for p in ps]),
        "leaves": np.mean([p.get("n_perm_leaves", 0) for p in ps]),
        "perm_better": f"{sum(p.get('perm_better',0) for p in ps)}/{sum(p.get('perm_total',0) for p in ps)}",
        "deferred": np.mean([p.get("n_deferred", 0) for p in ps]),
        "components": np.mean([p.get("n_components", 0) for p in ps]) if any("n_components" in p for p in ps) else None,
        "edges": np.mean([p.get("n_edges", 0) for p in ps]) if any("n_edges" in p for p in ps) else None,
        "chunked": np.mean([p.get("n_chunked", 0) for p in ps]) if any("n_chunked" in p for p in ps) else None,
        "gain_med": np.nanmedian([p.get("gain_med", np.nan) for p in ps]) if any("gain_med" in p for p in ps) else None,
        "gain_edge": np.nanmean([p.get("gain_frac_at_edge", np.nan) for p in ps]) if any("gain_frac_at_edge" in p for p in ps) else None,
        "p1_gain_med": np.nanmedian([(p.get("p1_gain") or {}).get("med") or np.nan for p in ps]) if any("p1_gain" in p for p in ps) else None,
    }
    # per file
    out["files"] = {}
    for fk in sorted(set(k[0] for k in keys)):
        Rf = [ref[k] for k in keys if k[0] == fk]; Af = [arm[k] for k in keys if k[0] == fk]
        out["files"][fk] = (wmean(Rf, "ns_total_eff"), wmean(Af, "ns_total_eff"), wmean(Rf, "acc_rescued", None), wmean(Af, "acc_rescued", None))
    return out


def dump_breakdown(d):
    cats = ["noise", "<1.5 no-tmpl", "img 1.5-5", "img 5-20", "img 20-100", "img>=100"]
    T = np.zeros((len(cats), 2))
    files = sorted(f for dd in str(d).split(",") for f in glob.glob(f"{dd}/full_dumps/*.npz"))
    for ff in files:
        z = np.load(ff, allow_pickle=True)
        e = np.clip(z["e"], 0, None).astype(np.float64)
        ts = z["ts_final"].astype(np.float64); tru = z["truth_t0"].astype(np.float64)
        labs = z["labels"].astype(int)
        ok_t = np.isfinite(tru) & (tru >= 0); assigned = np.isfinite(ts) & (ts >= 0)
        wrong = ok_t & (~assigned | (np.abs(ts - tru) > CUT))
        imaged = set(int(l) for l in np.unique(z["amp_lab"]))
        cat = np.zeros(labs.size, int)
        for L in np.unique(labs[labs >= 0]):
            m = labs == L; E = e[m].sum()
            cat[m] = 1 if int(L) not in imaged else 2 if E < 5 else 3 if E < 20 else 4 if E < 100 else 5
        for k in range(len(cats)):
            mk = (cat == k) & ok_t
            T[k, 0] += e[mk].sum(); T[k, 1] += e[mk & wrong].sum()
    return cats, T


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    do_dumps = "--dumps" in sys.argv
    ref = load(args[0])
    print(f"reference: {args[0]} ({len(ref)} events)")
    if do_dumps:
        cats, T0 = dump_breakdown(args[0])
    for d in args[1:]:
        arm = load(d)
        L = ledger(ref, arm, d)
        print(f"\n=== {d}  ({L['n']} paired events) ===")
        for lab in ("NS", "allTPC"):
            r0, a0, nb, nw = L[lab]
            print(f"  {lab:7s} ref {100*r0:6.2f}  arm {100*a0:6.2f}  delta {100*(a0-r0):+.3f} pp  events better/worse {nb}/{nw}")
        if "LE" in L:
            print(f"  LE      ref {100*L['LE'][0]:6.2f}  arm {100*L['LE'][1]:6.2f}")
        for fk, (r_ns, a_ns, r_all, a_all) in L["files"].items():
            print(f"    {fk}: NS {100*r_ns:.2f} -> {100*a_ns:.2f} ({100*(a_ns-r_ns):+.2f}) | all {100*r_all:.2f} -> {100*a_all:.2f} ({100*(a_all-r_all):+.2f})")
        print(f"  time/ev median: total {L['tot_t'][0]:.0f} -> {L['tot_t'][1]:.0f} s | stage 2+3.1 {L['pool_t'][0]:.1f} -> {L['pool_t'][1]:.1f} s")
        p = L["pool"]
        print(f"  pool: batches {p['batches']:.0f}, scans {p['scans']:.0f}, table-leaves {p['leaves']:.0f}, joint-beats-greedy {p['perm_better']}, deferred {p['deferred']:.1f}"
              + (f", components {p['components']:.0f}, edges {p['edges']:.0f}, chunked {p['chunked']:.1f}" if p["components"] is not None else "")
              + (f", gain med {p['gain_med']:.3f} (at edge {p['gain_edge']:.0%}), phase-1 gain med {p['p1_gain_med']:.3f}" if p["gain_med"] is not None else ""))
        if do_dumps:
            cats, T1 = dump_breakdown(d)
            print("  wrong E by category (GeV): " + " | ".join(f"{c}: {T0[k,1]/1e3:.2f}->{T1[k,1]/1e3:.2f}" for k, c in enumerate(cats)))
            print(f"  total wrong: {T0[:,1].sum()/1e3:.2f} -> {T1[:,1].sum()/1e3:.2f} GeV of {T1[:,0].sum()/1e3:.0f}")


if __name__ == "__main__":
    main()
