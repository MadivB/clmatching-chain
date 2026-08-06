"""Grade the improved per-cluster components (cache2) vs v1, per band, with complexity
normalization and the TPC blend.  v1 baselines: light 0.50/0.62/0.80/0.85, blend
0.54/0.66/0.81/0.92 (bands 1-5/5-20/20-100/>=100 MeV)."""
import glob, os, pickle, numpy as np

HERE = os.path.abspath(os.path.dirname(__file__))
R = pickle.load(open(os.path.join(HERE, "cluster_conf_cache2.pkl"), "rb"))
M = pickle.load(open(os.path.join(HERE, "master_v04.pkl"), "rb")); T = M["TPC"]
tpc_conf = {(T["file"][i], int(T["event"][i]), int(T["tpc"][i])): float(T["conf"][i])
            for i in range(len(T["conf"]))}
dom = {}
for fp in sorted(glob.glob(os.path.join(HERE, "v030v3_out", "dumps", "*.npz"))):
    d = np.load(fp, allow_pickle=True)
    ft = os.path.basename(fp).split("_ev")[0]; ev = int(d["event"])
    lab = np.asarray(d["labels"]); tpc = np.asarray(d["tpc"]); e = np.asarray(d["e"], np.float64)
    order = np.argsort(lab, kind="stable"); ls = lab[order]; ts = tpc[order]; es = e[order]
    bnd = np.r_[0, np.flatnonzero(np.diff(ls)) + 1, len(ls)]
    for i in range(len(bnd) - 1):
        L = int(ls[bnd[i]])
        if L < 0: continue
        sl = slice(bnd[i], bnd[i + 1])
        cs, inv = np.unique(ts[sl], return_inverse=True)
        ew = np.zeros(len(cs)); np.add.at(ew, inv, es[sl])
        dom[(ft, ev, L)] = int(cs[np.argmax(ew)])
    d.close()

wf = np.array([r["wf"] for r in R]); E = np.array([r["E"] for r in R])
bad = wf > 0.5; ok = wf < 0.05; sel = bad | ok
tc = np.array([tpc_conf.get((r["file"], r["event"], dom.get((r["file"], r["event"], r["L"]), -1)), np.nan) for r in R])
has = np.isfinite(tc)


def rank(x): return np.argsort(np.argsort(np.asarray(x, float))).astype(float) / len(x)
def auc(sc, yy):
    sc = np.asarray(sc); pos = sc[yy == 1]; neg = sc[yy == 0]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    o = np.argsort(np.concatenate([pos, neg])); rk = np.empty_like(o, float); rk[o] = np.arange(1, len(o) + 1)
    return (rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
def presid(v, ctrls):
    r = rank(v); X = np.c_[[np.ones_like(r)] + [rank(c) for c in ctrls]].T
    return r - X @ np.linalg.lstsq(X, r, rcond=None)[0]


ctrls = [np.log(np.maximum(E, 1)), np.array([r["N"] for r in R], float),
         np.array([r["bright"] for r in R]), np.array([r["chi2_loc"] for r in R]),
         np.array([r["nfl"] for r in R], float)]
V = {"z_min (v1)": -np.array([r["z_min"] for r in R]),
     "z_own (v1)": -np.array([r["z_own"] for r in R]),
     "z_min_fl (t0-list)": -np.array([r["zfl_raw"] for r in R]),
     "displaced (amount)": np.array([r["displaced"] for r in R])}
V["light v1 (min+own)"] = rank(V["z_min (v1)"]) + rank(V["z_own (v1)"])
V["light v2 (fl+disp+own)"] = rank(V["z_min_fl (t0-list)"]) + rank(V["displaced (amount)"]) + rank(V["z_own (v1)"])
V["light v2 cx-norm"] = rank(presid(V["z_min_fl (t0-list)"], ctrls)) + rank(presid(V["displaced (amount)"], ctrls)) + rank(presid(V["z_own (v1)"], ctrls))
tcr = rank(np.where(has, tc, np.nanmedian(tc)))
V["BLEND v1"] = tcr + rank(V["light v1 (min+own)"])
V["BLEND v2"] = tcr + rank(V["light v2 (fl+disp+own)"])
V["BLEND v2 cx"] = tcr + rank(V["light v2 cx-norm"])

bands = [(1, 5), (5, 20), (20, 100), (100, 1e12)]
s0 = sel & has
print(f"graded n={int(s0.sum())} WRONG={int(bad[s0].sum())}   bands: 1-5 / 5-20 / 20-100 / >=100 MeV")
for nm, v in V.items():
    line = f"{nm:>24}  "
    for lo, hi in bands:
        s = s0 & (E >= lo) & (E < hi); yy = bad[s].astype(int)
        line += f" {auc(v[s], yy):.3f}" if yy.sum() >= 5 else "   -  "
    print(line)
print("DONE")
