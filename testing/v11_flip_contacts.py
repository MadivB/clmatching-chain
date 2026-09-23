"""Would a 3-cm contact prior have prevented arm-B's breaks?  For every cluster
>= 5 MeV that changed time between <ref> and <arm> (label-identical events),
test whether it is in 3-cm contact with another cluster whose REFERENCE time is
(a) within 2 ticks of the cluster's truth time, (b) within 2 ticks of the
cluster's NEW (arm) time.  python v11_flip_contacts.py <ref_dir> <arm_dir>"""
import glob, os, sys
import numpy as np
from scipy.spatial import cKDTree
CUT, R, SAME_T = 10.0, 3.0, 2.0


def main(ref_dir, arm_dir):
    ref = {os.path.basename(f): f for f in glob.glob(f"{ref_dir}/full_dumps/*.npz")}
    arm = {os.path.basename(f): f for f in glob.glob(f"{arm_dir}/full_dumps/*.npz")}
    rows = []
    for k in sorted(set(ref) & set(arm)):
        zr = np.load(ref[k], allow_pickle=True); za = np.load(arm[k], allow_pickle=True)
        labs = zr["labels"].astype(int)
        if labs.size != za["labels"].size or not np.array_equal(labs, za["labels"].astype(int)):
            continue
        e = np.clip(zr["e"], 0, None).astype(np.float64)
        tru = zr["truth_t0"].astype(np.float64); tsr = zr["ts_final"].astype(np.float64); tsa = za["ts_final"].astype(np.float64)
        xyz = np.stack([zr["x"], zr["y"], zr["z"]], 1).astype(np.float64)
        ok_t = np.isfinite(tru) & (tru >= 0)
        tree = cKDTree(xyz)
        lab_t = {}
        for L in np.unique(labs[labs >= 0]):
            v = tsr[labs == L]; v = v[np.isfinite(v) & (v >= 0)]
            lab_t[int(L)] = float(np.median(v)) if v.size else np.nan
        for L in np.unique(labs[labs >= 0]):
            m = labs == L
            E = e[m].sum()
            if E < 5:
                continue
            va = tsa[m]; va = va[np.isfinite(va) & (va >= 0)]
            if not va.size or not np.isfinite(lab_t[int(L)]):
                continue
            ta = float(np.median(va)); tr_ = lab_t[int(L)]
            if abs(ta - tr_) <= 0.5:
                continue
            mt = m & ok_t
            if not mt.any():
                continue
            truth = float(np.median(tru[mt]))
            was = e[mt & (np.abs(tsr - tru) <= CUT)].sum() / e[mt].sum()
            now = e[mt & (np.abs(tsa - tru) <= CUT)].sum() / e[mt].sum()
            h = np.flatnonzero(m)
            if h.size > 3000:
                h = h[np.linspace(0, h.size - 1, 3000).astype(int)]
            nb = tree.query_ball_point(xyz[h], R)
            idx = np.unique(np.concatenate([np.asarray(v, int) for v in nb]))
            nl = set(int(v) for v in np.unique(labs[idx]) if v >= 0 and int(v) != int(L))
            ct = [lab_t[l] for l in nl if np.isfinite(lab_t.get(l, np.nan))]
            c_truth = any(abs(t - truth) <= SAME_T for t in ct)
            c_new = any(abs(t - ta) <= SAME_T for t in ct)
            c_old = any(abs(t - tr_) <= SAME_T for t in ct)
            rows.append((E, was, now, c_truth, c_new, c_old, len(nl)))
    r = np.array(rows, float)
    dE = r[:, 0] * (r[:, 2] - r[:, 1])
    broke = dE < -0.5; fixed = dE > 0.5
    print(f"moved clusters >=5 MeV: {len(r)} | broke {broke.sum()} ({dE[broke].sum()/1e3:.2f} GeV) | fixed {fixed.sum()} (+{dE[fixed].sum()/1e3:.2f} GeV)")
    for name, s in (("BROKE", broke), ("FIXED", fixed)):
        if not s.any():
            continue
        print(f"  {name}: has a 3-cm contact cluster at the TRUTH time {r[s,3].mean():.0%} (E-wt {np.average(r[s,3], weights=r[s,0]):.0%}) | at the NEW time {r[s,4].mean():.0%} | at the OLD time {r[s,5].mean():.0%} | any contact {np.mean(r[s,6]>0):.0%}")
    # what a contact prior would do: among broke, contact at truth & not at new -> prevented; among fixed, contact at old (wrong) & not at truth -> would have blocked the fix
    prev = broke & (r[:, 3] == 1) & (r[:, 4] == 0)
    blk = fixed & (r[:, 5] == 1) & (r[:, 3] == 0)
    print(f"  contact prior would PREVENT {prev.sum()} breaks ({-dE[prev].sum()/1e3:.2f} GeV) and BLOCK {blk.sum()} fixes ({dE[blk].sum()/1e3:.2f} GeV); ambiguous (contact at both) breaks {np.sum(broke & (r[:,3]==1) & (r[:,4]==1))}, fixes {np.sum(fixed & (r[:,3]==1) & (r[:,5]==1))}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
