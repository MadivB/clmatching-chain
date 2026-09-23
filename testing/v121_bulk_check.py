"""Offline 39-event check: bulk per-hit resolution with the piece layer on vs off (module == in-chain)."""
import glob, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v120_precision as vp
F1 = "/global/cfs/cdirs/dunepro/people/abooth/nd-production/output/MiniProdN5/run-ndlar-flow/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift/FLOW/0000000/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.0000001.FLOW.hdf5"
files = sorted(glob.glob("/pscratch/sd/y/yuxuan/v11arms/CERT_K1/full_dumps/*.npz"))
def rob(v, w):
    o = np.argsort(v); cw = np.cumsum(w[o]); med = v[o][np.searchsorted(cw, 0.5 * cw[-1])]; d = np.abs(v - med); o2 = np.argsort(d); cw2 = np.cumsum(w[o2]); return 1.4826 * d[o2][np.searchsorted(cw2, 0.5 * cw2[-1])]
def wsd(v, w):
    mu = np.average(v, weights=w); return np.sqrt(np.average((v - mu) ** 2, weights=w))
for mode in (False, True):
    rN, w = [], []; npf = 0; tt = 0.0
    for f in files:
        new_t, sig, st, z = vp.run_on_dump(f, F1, "v120_calib_lowi.json", rounds=3, pieces=mode, collect=mode)
        npf += st.get("n_piece_fit", 0); tt += st["time_s"]
        tru = z["truth_t0"].astype(float); ts = z["ts_final"].astype(float); e = np.clip(z["e"], 0, None)
        ok = np.isfinite(tru) & (tru >= 0) & np.isfinite(ts) & (ts >= 0) & (np.abs(ts - tru) <= 5) & np.isfinite(new_t)
        rN.append((new_t[ok] - tru[ok]) * 16); w.append(e[ok])
    rN, w = np.concatenate(rN), np.concatenate(w)
    print(f"pieces={mode}: 39 events per-hit std {wsd(rN,w):.2f} robust {rob(rN,w):.2f} ns | light-resolved pieces {npf/len(files):.1f}/ev | {tt/len(files):.1f} s/ev CPU", flush=True)
