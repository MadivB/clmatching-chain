"""Hit-level flip census (no label pairing needed):  python v11_hitflips.py <ref_dir> <arm_dir>
Per hit: correct in ref vs correct in arm; fixed/broke energy by the REF
cluster's energy band, and whether the arm time coincides with a ref family."""
import glob, os, sys
import numpy as np
CUT = 10.0


def main(ref_dir, arm_dir):
    ref = {os.path.basename(f): f for f in glob.glob(f"{ref_dir}/full_dumps/*.npz")}
    arm = {os.path.basename(f): f for f in glob.glob(f"{arm_dir}/full_dumps/*.npz")}
    bands = [(0, 1.5), (1.5, 5), (5, 20), (20, 100), (100, 1e9)]
    T = {b: [0.0, 0.0] for b in bands}; T["noise"] = [0.0, 0.0]
    joined = [0.0, 0.0]; newt = [0.0, 0.0]; n = 0
    for k in sorted(set(ref) & set(arm)):
        zr = np.load(ref[k], allow_pickle=True); za = np.load(arm[k], allow_pickle=True)
        if zr["e"].size != za["e"].size:
            continue
        n += 1
        e = np.clip(zr["e"], 0, None).astype(np.float64); tru = zr["truth_t0"].astype(np.float64)
        tsr = zr["ts_final"].astype(np.float64); tsa = za["ts_final"].astype(np.float64)
        labs = zr["labels"].astype(int); tpc = zr["tpc"].astype(int)
        ok_t = np.isfinite(tru) & (tru >= 0)
        okr = ok_t & np.isfinite(tsr) & (tsr >= 0) & (np.abs(tsr - tru) <= CUT)
        oka = ok_t & np.isfinite(tsa) & (tsa >= 0) & (np.abs(tsa - tru) <= CUT)
        fixed = ok_t & ~okr & oka; broke = ok_t & okr & ~oka
        lab_E = {}
        for L in np.unique(labs[labs >= 0]):
            lab_E[int(L)] = e[labs == L].sum()
        cE = np.array([lab_E.get(int(L), 0.0) for L in labs])
        for b in bands:
            m = (labs >= 0) & (cE >= b[0]) & (cE < b[1])
            T[b][0] += e[m & fixed].sum(); T[b][1] += e[m & broke].sum()
        T["noise"][0] += e[(labs < 0) & fixed].sum(); T["noise"][1] += e[(labs < 0) & broke].sum()
        # destination of changed hits: an existing ref family in the same TPC?
        ch = ok_t & np.isfinite(tsa) & np.isfinite(tsr) & (np.abs(tsa - tsr) > 0.5)
        for c in np.unique(tpc[ch]):
            mc = ch & (tpc == c)
            fam = np.unique(np.round(tsr[(tpc == c) & np.isfinite(tsr) & (tsr >= 0)]))
            if fam.size == 0:
                continue
            d = np.abs(tsa[mc][:, None] - fam[None, :]).min(axis=1)
            jm = d <= CUT
            idx = np.flatnonzero(mc)
            joined[0] += e[idx[jm] ][fixed[idx[jm]]].sum(); joined[1] += e[idx[jm]][broke[idx[jm]]].sum()
            newt[0] += e[idx[~jm]][fixed[idx[~jm]]].sum(); newt[1] += e[idx[~jm]][broke[idx[~jm]]].sum()
    tf = sum(v[0] for v in T.values()); tb = sum(v[1] for v in T.values())
    print(f"{n} events | hit-level: fixed {tf/1e3:.3f} GeV, broke {tb/1e3:.3f} GeV, net {(tf-tb)/1e3:+.3f} GeV")
    for b, v in T.items():
        print(f"  {str(b):18s} fixed {v[0]/1e3:6.3f}  broke {v[1]/1e3:6.3f}  net {(v[0]-v[1])/1e3:+.3f}")
    print(f"  moved to an existing ref family: fixed {joined[0]/1e3:.3f} broke {joined[1]/1e3:.3f} | to a new time: fixed {newt[0]/1e3:.3f} broke {newt[1]/1e3:.3f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
