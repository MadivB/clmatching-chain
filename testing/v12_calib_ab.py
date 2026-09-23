"""Out-of-sample calibration A/B for stage P: no calibration vs file-1 calibration vs
low-intensity calibration, on dumps of files 2-3 (seen by neither).  python v12_calib_ab.py"""
import glob, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v120_precision as vp
F1 = "/global/cfs/cdirs/dunepro/people/abooth/nd-production/output/MiniProdN5/run-ndlar-flow/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift/FLOW/0000000/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.0000001.FLOW.hdf5"
HERE = os.path.dirname(os.path.abspath(__file__))
files = sorted(glob.glob("/pscratch/sd/y/yuxuan/v11arms/CERT_K1/full_dumps/000000[23]_ev*.npz"))
def rob(v, ww):
    o = np.argsort(v); cw = np.cumsum(ww[o]); med = v[o][np.searchsorted(cw, 0.5 * cw[-1])]; d = np.abs(v - med); o2 = np.argsort(d); cw2 = np.cumsum(ww[o2]); return 1.4826 * d[o2][np.searchsorted(cw2, 0.5 * cw2[-1])]
def wsd(v, ww):
    mu = np.average(v, weights=ww); return np.sqrt(np.average((v - mu) ** 2, weights=ww))
for name, calib in (("none", None), ("dev13", os.path.join(HERE, "v120_calib_dev13.json")), ("lowi", os.path.join(HERE, "v120_calib_lowi.json"))):
    rE, rN, w, fE, fN, fw = [], [], [], [], [], []
    for f in files:
        new_t, sig, st, z = vp.run_on_dump(f, F1, calib, rounds=3)
        tru = z["truth_t0"].astype(float); ts = z["ts_final"].astype(float); e = np.clip(z["e"], 0, None); tpc = z["tpc"]
        ok = np.isfinite(tru) & (tru >= 0) & np.isfinite(ts) & (ts >= 0) & (np.abs(ts - tru) <= 5) & (np.abs(new_t - ts) > 1e-9)
        rE.append((ts[ok] - tru[ok]) * 16); rN.append((new_t[ok] - tru[ok]) * 16); w.append(e[ok])
        for c in np.unique(tpc[ok]):
            for tv in np.unique(ts[ok & (tpc == c)]):
                m = ok & (tpc == c) & (ts == tv)
                if e[m].sum() >= 5:
                    fE.append(np.average((ts[m] - tru[m]) * 16, weights=e[m])); fN.append(np.average((new_t[m] - tru[m]) * 16, weights=e[m])); fw.append(e[m].sum())
    rE, rN, w = map(np.concatenate, (rE, rN, w)); fE, fN, fw = map(np.array, (fE, fN, fw))
    mu = np.average(rN, weights=w); clip = np.abs(rN - mu) < 20
    print(f"calib={name:6s} ({len(files)} ev files 2-3): per-hit stage E {wsd(rE,w):.2f}/{rob(rE,w):.2f} -> P {wsd(rN,w):.2f}/{rob(rN,w):.2f} ns (clipped std {wsd(rN[clip],w[clip]):.2f}) | per-family {wsd(fE,fw):.2f}/{rob(fE,fw):.2f} -> {wsd(fN,fw):.2f}/{rob(fN,fw):.2f} ns", flush=True)
