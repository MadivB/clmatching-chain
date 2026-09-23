"""Evaluate in-chain per-piece records (row['phase3']['pieces']) against truth from the dumps.
python v121_piece_eval.py <arm_dir>
Per piece: fitted delay dt (piece apex time - core apex time), true delay from the dump (robust core
apex reference), implied beta.  Reports delay resolution by true-delay bin and by piece energy."""
import glob, json, os, sys
import numpy as np
C, NS = 29.9792458, 16.0


def rob(v):
    return 1.4826 * np.median(np.abs(v - np.median(v))) if v.size else np.nan


def main(d):
    rows = [json.loads(l) for f in glob.glob(f"{d}/groups_worker*.jsonl") for l in open(f)]
    dumps = {(os.path.basename(str(r["file"]))[-20:-13], int(r["event"])): None for r in rows}
    out = []; n_pieces = n_fit = 0
    for r in rows:
        pcs = (r.get("phase3") or {}).get("pieces") or []
        stp = (r.get("phase3") or {}).get("precision") or {}
        n_pieces += stp.get("n_piece", 0); n_fit += stp.get("n_piece_fit", 0)
        if not pcs:
            continue
        import re
        tag = re.search(r"(\d{7})\.FLOW", str(r["file"])).group(1)
        f = f"{d}/full_dumps/{tag}_ev{int(r['event'])}.npz"
        if not os.path.exists(f):
            continue
        z = np.load(f, allow_pickle=True)
        tru = z["truth_t0"].astype(float); ts = z["ts_final"].astype(float); e = np.clip(z["e"], 0, None)
        labs = z["labels"].astype(int); tpc = z["tpc"]; xyz = np.stack([z["x"], z["y"], z["z"]], 1).astype(float)
        # chain tick per hit is not stored after stage P; use the piece record's t_chain and the labels
        for pc in pcs:
            m = np.isin(labs, pc["labels"]) & (tpc == pc["tpc"]) & np.isfinite(tru) & (tru >= 0)
            if not m.any():
                continue
            # core = hits of this TPC whose final time is within 1.5 ticks of the core apex time + their propagation... approximate:
            # core hits = same TPC, not in the piece, |ts - t0_core| <= 3 ticks (v1.2 times are apex + d/c, within ~3 ticks of the apex time)
            mc = (tpc == pc["tpc"]) & ~np.isin(labs, pc["labels"]) & np.isfinite(ts) & (np.abs(ts - pc["t0_core"]) <= 3.0) & np.isfinite(tru) & (tru >= 0)
            if mc.sum() < 10:
                continue
            o = np.argsort(tru[mc]); n2 = max(3, int(0.02 * o.size))
            apex = np.average(xyz[mc][o[:n2]], axis=0, weights=e[mc][o[:n2]])
            t_apex = float(np.median(tru[mc] * NS - np.linalg.norm(xyz[mc] - apex, axis=1) / C))
            dt_true = float(np.average(tru[m], weights=e[m])) * NS - t_apex
            if abs(dt_true) > 160:
                continue
            out.append((pc["E"], pc["dt_ns"], dt_true, pc["dist_cm"], pc["sig_ns"], pc["n_ch"]))
    R = np.array(out); dd = R[:, 1] - R[:, 2]
    print(f"{d}: {len(rows)} events, pieces {n_pieces} ({n_pieces/len(rows):.0f}/ev), light-resolved {n_fit} ({n_fit/len(rows):.1f}/ev); evaluable {len(R)}")
    print(f"  fitted - true delay: median {np.median(dd):+.2f} ns, robust {rob(dd):.2f}, |err|<10 ns {np.mean(np.abs(dd)<10):.2f}")
    for lo, hi in ((-160, 10), (10, 30), (30, 160)):
        s = (R[:, 2] >= lo) & (R[:, 2] < hi)
        if s.sum() > 2:
            b_fit = R[s, 3] / (C * np.maximum(R[s, 1], 0.5)); b_true = R[s, 3] / (C * np.maximum(R[s, 2], 0.5))
            print(f"  true delay [{lo},{hi}) ns: n={s.sum():4d}, E-sum {R[s,0].sum()/1e3:.2f} GeV | fitted delay median {np.median(R[s,1]):+.1f} vs true {np.median(R[s,2]):+.1f} | err robust {rob(dd[s]):.1f} ns, std {dd[s].std():.1f} | sigma_p median {np.median(R[s,4]):.2f}")
    for lo, hi in ((5, 20), (20, 100), (100, 1e9)):
        s = (R[:, 0] >= lo) & (R[:, 0] < hi)
        if s.sum() > 2:
            print(f"  piece E [{lo},{hi}): n={s.sum():4d} | err robust {rob(dd[s]):.1f} ns | n_ch median {np.median(R[s,5]):.0f}")


if __name__ == "__main__":
    main(sys.argv[1])
