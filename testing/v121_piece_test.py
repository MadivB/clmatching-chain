"""Offline truth test of per-fragment free-time fits (stage P with pieces) on chain dumps.
python v121_piece_test.py <dump_dir> <flow_file> <calib.json> [max_events] [rounds]"""
import glob, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v120_precision as vp
C, NS = 29.9792458, 16.0


def rob(v):
    return 1.4826 * np.median(np.abs(v - np.median(v)))


def wsd(v, w):
    mu = np.average(v, weights=w); return np.sqrt(np.average((v - mu) ** 2, weights=w))


def main(d, flow, calib, max_events, rounds):
    files = sorted(glob.glob(f"{d}/full_dumps/*.npz"))[:max_events]
    rows = []; hb, ha, hw = [], [], []
    for f in files:
        new_t, sig, st, z = vp.run_on_dump(f, flow, calib, rounds=rounds, pieces=True, collect=True)
        print(os.path.basename(f), {k: st[k] for k in ("n_fam", "n_fit", "n_piece", "n_piece_fit", "time_s")}, flush=True)
        tru = z["truth_t0"].astype(float); ts = z["ts_final"].astype(float); e = np.clip(z["e"], 0, None)
        labs = z["labels"].astype(int); tpc = z["tpc"]
        tsr = np.round(np.where(np.isfinite(ts), ts, -1))
        for pc in st["pieces"]:
            m = np.isin(labs, pc["labels"]) & (tpc == pc["tpc"]) & np.isfinite(tru) & (tru >= 0)
            if not m.any():
                continue
            mc = (tpc == pc["tpc"]) & (tsr == np.round(pc["t_chain"])) & ~np.isin(labs, pc["labels"]) & np.isfinite(tru) & (tru >= 0)
            if mc.sum() < 10:
                continue
            o = np.argsort(tru[mc]); n2 = max(3, int(0.02 * o.size))
            xyz_ = np.stack([z["x"], z["y"], z["z"]], 1).astype(float)
            apex_ = np.average(xyz_[mc][o[:n2]], axis=0, weights=e[mc][o[:n2]])
            t_apex = float(np.median(tru[mc] * NS - np.linalg.norm(xyz_[mc] - apex_, axis=1) / C))   # robust apex time
            dt_true = float(np.average(tru[m], weights=e[m])) * NS - t_apex
            if abs(dt_true) > 160.0:        # mis-assigned late deposit (outside the assignment window): not a resolution test
                continue
            rows.append((pc["E"], pc["dt_ns"], dt_true, pc["dist_cm"], pc["sig_ns"], pc["n_ch"]))
            ok = m & np.isfinite(ts) & (ts >= 0) & (np.abs(ts - tru) <= 10)
            hb.append((ts[ok] - tru[ok]) * NS); ha.append((new_t[ok] - tru[ok]) * NS); hw.append(e[ok])
    R = np.array(rows); dd = R[:, 1] - R[:, 2]
    print(f"pieces fitted, correctly assigned (|true delay| <= 160 ns): {len(R)}; fitted - true delay: median {np.median(dd):+.1f} ns, robust {rob(dd):.1f}, std {dd.std():.1f} ns; |err|<10 ns frac {np.mean(np.abs(dd)<10):.2f}")
    for lo, hi in ((5, 20), (20, 100), (100, 1e9)):
        s = (R[:, 0] >= lo) & (R[:, 0] < hi)
        if s.sum() > 3:
            print(f"  piece E [{lo},{hi}): n={s.sum():4d} | delay resolution robust {rob(dd[s]):.1f} ns (std {dd[s].std():.1f}) | true delay >10 ns {np.mean(R[s,2]>10):.2f} | sigma_p median {np.median(R[s,4]):.2f} ns, n_ch median {np.median(R[s,5]):.0f}")
    slow = R[:, 2] > 20
    if slow.sum() > 3:
        print(f"  pieces with true delay > 20 ns: n={slow.sum()}, fitted delay median {np.median(R[slow,1]):.1f} vs true {np.median(R[slow,2]):.1f} ns; resolution robust {rob(dd[slow]):.1f} ns")
    hb, ha, hw = map(np.concatenate, (hb, ha, hw))
    print(f"fragment hits (E-wt): stage E std {wsd(hb,hw):.1f} robust {rob(hb):.1f} -> pieces std {wsd(ha,hw):.1f} robust {rob(ha):.1f} ns")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 4, int(sys.argv[5]) if len(sys.argv) > 5 else 1)
