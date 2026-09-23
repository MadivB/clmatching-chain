"""Cluster-level flip census between two arms:  python v11_flips.py <ref_dir> <arm_dir> [n_show]

For every event present in both dumps and every cluster whose median t0
changed by > 0.5 tick: energy, truth, was_ok/now_ok, |dt|, and whether the
new time coincides with an existing family in the reference (joined) or is a
new time.  Reports fixed/broke energy by cluster-energy band and lists the
biggest flips."""
import glob, os, sys
import numpy as np

CUT = 10.0


def med_t(ts, labs, L):
    v = ts[labs == L]; v = v[np.isfinite(v) & (v >= 0)]
    return float(np.median(v)) if v.size else np.nan


def main(ref_dir, arm_dir, n_show=15):
    ref = {os.path.basename(f): f for f in glob.glob(f"{ref_dir}/full_dumps/*.npz")}
    arm = {os.path.basename(f): f for f in glob.glob(f"{arm_dir}/full_dumps/*.npz")}
    common = sorted(set(ref) & set(arm))
    rows = []
    for k in common:
        zr = np.load(ref[k], allow_pickle=True); za = np.load(arm[k], allow_pickle=True)
        labs = zr["labels"].astype(int)
        if labs.size != za["labels"].size or not np.array_equal(labs, za["labels"].astype(int)):
            print("label mismatch, skipping", k); continue
        e = np.clip(zr["e"], 0, None).astype(np.float64)
        tru = zr["truth_t0"].astype(np.float64); tsr = zr["ts_final"].astype(np.float64); tsa = za["ts_final"].astype(np.float64)
        tpc = zr["tpc"].astype(int)
        ok_t = np.isfinite(tru) & (tru >= 0)
        imaged = set(int(l) for l in np.unique(zr["amp_lab"]))
        for L in np.unique(labs[labs >= 0]):
            m = labs == L
            tr_, ta_ = med_t(tsr, labs, L), med_t(tsa, labs, L)
            if not (np.isfinite(tr_) and np.isfinite(ta_)) or abs(tr_ - ta_) <= 0.5:
                continue
            mt = m & ok_t
            if not mt.any():
                continue
            E = e[m].sum(); truth = float(np.median(tru[mt]))
            was = e[mt & (np.abs(tsr - tru) <= CUT)].sum() / max(e[mt].sum(), 1e-9)
            now = e[mt & (np.abs(tsa - tru) <= CUT)].sum() / max(e[mt].sum(), 1e-9)
            c = int(np.bincount(tpc[m]).argmax())
            others = (tpc == c) & (labs != L) & np.isfinite(tsr) & (tsr >= 0)
            joined = bool(np.any(np.abs(tsr[others] - ta_) <= CUT))
            rows.append((k, int(L), E, truth, tr_, ta_, was, now, int(L) in imaged, joined, c))
    if not rows:
        print("no flips"); return
    E = np.array([r[2] for r in rows]); was = np.array([r[6] for r in rows]); now = np.array([r[7] for r in rows])
    dE = E * (now - was)
    print(f"{len(common)} common events; {len(rows)} clusters changed time; net correct-energy change {dE.sum()/1e3:+.3f} GeV")
    for lo, hi in [(0, 1.5), (1.5, 5), (5, 20), (20, 100), (100, 1e9)]:
        s = (E >= lo) & (E < hi)
        if s.any():
            fx = (dE[s] > 0).sum(); br = (dE[s] < 0).sum()
            print(f"  E [{lo},{hi}): moved {s.sum():5d} | fixed {fx:4d} (+{dE[s][dE[s]>0].sum()/1e3:.3f} GeV) | broke {br:4d} ({dE[s][dE[s]<0].sum()/1e3:.3f} GeV) | net {dE[s].sum()/1e3:+.3f}")
    joined = np.array([r[9] for r in rows])
    print(f"  moved to an existing reference family: {joined.mean():.0%} of moves; net there {dE[joined].sum()/1e3:+.3f} GeV; to a new time {dE[~joined].sum()/1e3:+.3f} GeV")
    order = np.argsort(dE)
    print(f"\n  biggest BROKE ({n_show}):  event | label | E MeV | truth | ref t | arm t | was/now | joined")
    for i in order[:n_show]:
        r = rows[i]
        if dE[i] >= 0: break
        print(f"    {r[0][:14]} L{r[1]:5d} {r[2]:8.1f} {r[3]:7.1f} {r[4]:7.1f} {r[5]:7.1f} {r[6]:.2f}/{r[7]:.2f} {'J' if r[9] else 'N'} tpc{r[10]}")
    print(f"  biggest FIXED ({n_show}):")
    for i in order[::-1][:n_show]:
        r = rows[i]
        if dE[i] <= 0: break
        print(f"    {r[0][:14]} L{r[1]:5d} {r[2]:8.1f} {r[3]:7.1f} {r[4]:7.1f} {r[5]:7.1f} {r[6]:.2f}/{r[7]:.2f} {'J' if r[9] else 'N'} tpc{r[10]}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 15)
