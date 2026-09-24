"""Hit-level A/B between two dump arms (same events): where do the final times differ?
Buckets of |ts_arm - ts_ref| in ticks: <=0.5 (identical / sub-tick noise), 0.5-2 (sub-tick
timing), 2-10 (still inside the correctness window), >10 (assignment flips); plus phase-1
identity, veto identity, base-image difference, and energy-weighted efficiency of both arms.
python v122_ab_breakdown.py <arm_dir> <ref_dir>"""
import glob, os, sys
import numpy as np
CUT = 10.0


def main(arm, ref):
    tot = np.zeros(4); totE = np.zeros(4); n_ev = 0; effs = []
    print(f"{'event':>14} {'<=0.5':>7} {'0.5-2':>6} {'2-10':>5} {'>10':>4} | {'E 0.5-2':>8} {'E 2-10':>7} {'E >10':>6} | {'p1 same':>7} {'veto':>5} {'|dBase|':>8} | {'eff arm':>8} {'eff ref':>8}")
    for f in sorted(glob.glob(f"{arm}/full_dumps/*.npz")):
        r = os.path.join(ref, "full_dumps", os.path.basename(f))
        if not os.path.exists(r):
            continue
        a = np.load(f, allow_pickle=True); b = np.load(r, allow_pickle=True)
        ta = np.where(np.isfinite(a["ts_final"]), a["ts_final"], -1.0); tb = np.where(np.isfinite(b["ts_final"]), b["ts_final"], -1.0)
        e = np.clip(a["e"], 0, None).astype(float); tru = a["truth_t0"].astype(float)
        d = np.abs(ta - tb)
        bins = [(-1, 0.5), (0.5, 2), (2, 10), (10, 1e9)]
        cnt = [int(((d > lo) & (d <= hi)).sum()) for lo, hi in bins]
        Es = [float(e[(d > lo) & (d <= hi)].sum()) for lo, hi in bins]
        tot += cnt; totE += Es; n_ev += 1
        p1 = bool(np.allclose(np.nan_to_num(a["ts_p1"], nan=-1), np.nan_to_num(b["ts_p1"], nan=-1)))
        veto = bool(np.array_equal(a["veto_mask"], b["veto_mask"]))
        db = float(np.abs(a["base_image"].astype(np.float32) - b["base_image"].astype(np.float32)).max())
        def eff(ts):
            ok = np.isfinite(tru) & (ts >= 0)
            return e[ok & (np.abs(ts - tru) <= CUT)].sum() / max(e[ok].sum(), 1e-9)
        ea, eb = eff(ta), eff(tb); effs.append((ea, eb, e[np.isfinite(tru) & (ta >= 0)].sum()))
        print(f"{os.path.basename(f)[:-4]:>14} {cnt[0]:7d} {cnt[1]:6d} {cnt[2]:5d} {cnt[3]:4d} | {Es[1]:8.0f} {Es[2]:7.0f} {Es[3]:6.0f} | {str(p1):>7} {str(veto):>5} {db:8.1f} | {ea:8.4f} {eb:8.4f}")
    E = np.array(effs)
    w = E[:, 2]
    print(f"{'TOTAL':>14} {int(tot[0]):7d} {int(tot[1]):6d} {int(tot[2]):5d} {int(tot[3]):4d} | {totE[1]:8.0f} {totE[2]:7.0f} {totE[3]:6.0f} |"
          f"  E-weighted eff arm {np.sum(E[:,0]*w)/w.sum():.4f} ref {np.sum(E[:,1]*w)/w.sum():.4f}  ({n_ev} events)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
