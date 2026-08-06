"""Round 2: per-cluster confidence = own-light evidence (E>=20 MeV works) + per-TPC
CONTEXT (v0.4 confidence) for the light-blind small clusters.  Join clusters to their
dominant TPC, grade TPC-conf as a per-cluster predictor per band, and build the blend."""
import glob, os, pickle, numpy as np

HERE = os.path.abspath(os.path.dirname(__file__))
R = pickle.load(open(os.path.join(HERE, "cluster_conf_cache.pkl"), "rb"))
M = pickle.load(open(os.path.join(HERE, "master_v04.pkl"), "rb")); T = M["TPC"]
tpc_conf = {(T["file"][i], int(T["event"][i]), int(T["tpc"][i])): float(T["conf"][i])
            for i in range(len(T["conf"]))}

# dominant TPC per cluster (lazy npz pass: only small arrays touched)
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
print(f"dominant-TPC join: {len(dom)} clusters")

wf = np.array([r["wf"] for r in R]); E = np.array([r["E"] for r in R])
bad = wf > 0.5; ok = wf < 0.05; sel = bad | ok
tc = np.array([tpc_conf.get((r["file"], r["event"], dom.get((r["file"], r["event"], r["L"]), -1)), np.nan)
               for r in R])
has_tc = np.isfinite(tc)
print(f"clusters with TPC-conf: {int(has_tc.sum())} / {len(R)} "
      f"(missing = TPC outside the [100,1400] MeV confidence band)")


def rank(x): return np.argsort(np.argsort(np.asarray(x, float))).astype(float) / len(x)
def auc(sc, yy):
    sc = np.asarray(sc); pos = sc[yy == 1]; neg = sc[yy == 0]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    o = np.argsort(np.concatenate([pos, neg])); rk = np.empty_like(o, float); rk[o] = np.arange(1, len(o) + 1)
    return (rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


own = -np.array([r["z_own"] for r in R])          # own-site excess (badness)
zmin = -np.array([r["z_min"] for r in R])
cl_light = rank(own) + rank(zmin)                  # cluster-light evidence

s = sel & has_tc
y = bad[s].astype(int)
print(f"\ngraded set with TPC-conf: n={int(s.sum())} WRONG={int(y.sum())}")
print(f"{'predictor':>22} {'AUC all':>8}", "  per band (1-5 / 5-20 / 20-100 / >=100):")
bands = [(1, 5), (5, 20), (20, 100), (100, 1e12)]
for nm, v in [("TPC conf (context)", tc), ("cluster light", cl_light),
              ("blend rank-mean", rank(np.where(has_tc, tc, np.nanmedian(tc))) + rank(cl_light))]:
    line = f"{nm:>22} {auc(v[s], y):>8.3f}  "
    for lo, hi in bands:
        sb = s & (E >= lo) & (E < hi); yb = bad[sb].astype(int)
        line += f" {auc(v[sb], yb):.3f}" if yb.sum() >= 5 else "   -  "
    print(line)

# E-adaptive blend: weight cluster-light evidence by energy (it only informs >=20 MeV)
wE = np.clip((np.log10(np.maximum(E, 1)) - np.log10(20)) / (np.log10(200) - np.log10(20)), 0, 1)
blend = wE * rank(cl_light) + (1 - wE) * rank(np.where(has_tc, tc, np.nanmedian(tc)))
line = f"{'E-adaptive blend':>22} {auc(blend[s], y):>8.3f}  "
for lo, hi in bands:
    sb = s & (E >= lo) & (E < hi); yb = bad[sb].astype(int)
    line += f" {auc(blend[sb], yb):.3f}" if yb.sum() >= 5 else "   -  "
print(line)
print("DONE")
