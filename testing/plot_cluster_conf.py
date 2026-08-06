"""Per-cluster confidence deliverable figure: per-band AUC of TPC-context vs
cluster-light vs blend, + the covariance-row score distribution for big clusters."""
import glob, os, pickle, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.abspath(os.path.dirname(__file__))
R = pickle.load(open(os.path.join(HERE, "cluster_conf_cache.pkl"), "rb"))
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


light = rank(-np.array([r["z_own"] for r in R])) + rank(-np.array([r["z_min"] for r in R]))
blend = rank(np.where(has, tc, np.nanmedian(tc))) + rank(light)
bands = [(1, 5, "1-5"), (5, 20, "5-20"), (20, 100, "20-100"), (100, 1e12, ">=100")]
preds = [("per-TPC conf (context)", tc, "#9e9e9e"), ("cluster light (z_min+z_own)", light, "#4C78A8"),
         ("blend (rank-mean)", blend, "black")]

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.4))
xw = 0.26
for j, (nm, v, col) in enumerate(preds):
    xs, ys = [], []
    for i, (lo, hi, lab_) in enumerate(bands):
        s = sel & has & (E >= lo) & (E < hi); yy = bad[s].astype(int)
        if yy.sum() >= 5:
            xs.append(i + (j - 1) * xw); ys.append(auc(v[s], yy))
    axL.bar(xs, ys, width=xw, color=col, edgecolor="k", label=nm)
    for xx, yv in zip(xs, ys): axL.text(xx, yv + 0.005, f"{yv:.2f}", ha="center", fontsize=8)
axL.set_xticks(range(len(bands))); axL.set_xticklabels([b[2] for b in bands])
axL.set_xlabel("cluster energy [MeV]"); axL.set_ylabel("AUC  (WRONG>=50% vs OK<5%)")
axL.set_ylim(0.45, 1.0); axL.axhline(0.5, color="r", ls=":", lw=.8)
axL.legend(fontsize=8, loc="upper left"); axL.grid(axis="y", alpha=.3, ls=":")
axL.set_title("per-cluster confidence: AUC by energy band")

s = sel & has & (E >= 20)
zmin = np.array([r["z_min"] for r in R])
b_ = np.linspace(-12, 2, 50)
axR.hist(zmin[s & ok], bins=b_, density=True, alpha=.6, color="tab:green", label="OK clusters (<5% wrong)")
axR.hist(zmin[s & bad], bins=b_, density=True, alpha=.6, color="tab:red", label="WRONG clusters (>=50%)")
axR.set_xlabel("z_min  =  min_j' m(j0, j')   [anchored covariance row]")
axR.set_ylabel("density"); axR.legend(fontsize=9)
axR.set_title("E >= 20 MeV: anticorrelation evidence separates wrong clusters")
fig.suptitle("Per-cluster charge-light confidence (52.9k clusters, 39 events, v0.4)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = os.path.join(HERE, "deck_figs", "cluster_confidence.png")
fig.savefig(out, dpi=130); print("wrote", out)
