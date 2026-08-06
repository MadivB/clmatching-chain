"""Per-cluster confidence 2D plots in the per-TPC-study style:
 (a) square-bin 2D: confidence vs mis-assigned energy %  (all clusters)
 (b) same, E >= 20 MeV (where the cluster's own light informs)
 (c) cluster t0 assignment accuracy vs confidence, per energy band
"""
import glob, os, pickle, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

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
tc = np.array([tpc_conf.get((r["file"], r["event"], dom.get((r["file"], r["event"], r["L"]), -1)), np.nan) for r in R])
has = np.isfinite(tc)


def rank(x): return np.argsort(np.argsort(np.asarray(x, float))).astype(float) / len(x)


try:                                                     # improved v2 blend if available
    B2 = pickle.load(open(os.path.join(HERE, "cluster_blend_v2.pkl"), "rb"))
    conf = rank(B2["blend"]); wf = np.asarray(B2["wf"]); E = np.asarray(B2["E"])   # self-consistent rows
except Exception:
    light = rank(-np.array([r["z_own"] for r in R])) + rank(-np.array([r["z_min"] for r in R]))
    conf = rank(rank(np.where(has, tc, np.nanmedian(tc))) + rank(light))


def panel_2d(ax, m, title):
    h = ax.hist2d(conf[m], 100 * wf[m], bins=[np.linspace(0, 1, 42), np.linspace(0, 100, 42)],
                  norm=LogNorm(), cmap="viridis")
    cx, mn = [], []
    for i in range(14):
        lo, hi = i / 14, (i + 1) / 14
        s = m & (conf >= lo) & (conf < hi)
        if s.sum() >= 30:
            cx.append(conf[s].mean()); mn.append(100 * np.average(wf[s], weights=E[s]))
    ax.plot(cx, mn, "-", color="white", lw=2.6); ax.plot(cx, mn, "o", color="black", ms=4,
                                                         label="binned mean (E-weighted)")
    ax.set_xlabel("confidence  (per-cluster blend)"); ax.set_ylabel("mis-assigned energy [% of cluster]")
    ax.set_title(title); ax.legend(loc="upper left", fontsize=8)
    return h


fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
h0 = panel_2d(ax[0], np.ones(len(R), bool), "confidence vs mis-assigned energy")
fig.colorbar(h0[3], ax=ax[0], label="clusters / bin")
h1 = panel_2d(ax[1], E >= 20, "confidence vs mis-assigned energy  (E >= 20 MeV)")
fig.colorbar(h1[3], ax=ax[1], label="clusters / bin")

# (c) assignment accuracy vs confidence per band
bands = [(1, 5, "1-5 MeV", "#c0c0c0"), (5, 20, "5-20 MeV", "#F58518"),
         (20, 100, "20-100 MeV", "#4C78A8"), (100, 1e12, ">=100 MeV", "black")]
for lo, hi, lb, col in bands:
    m = (E >= lo) & (E < hi)
    cx, acc = [], []
    for i in range(10):
        clo, chi_ = i / 10, (i + 1) / 10
        s = m & (conf >= clo) & (conf < chi_)
        if s.sum() >= 40:
            cx.append((clo + chi_) / 2); acc.append(100 * (wf[s] < 0.5).mean())
    ax[2].plot(cx, acc, "o-", color=col, lw=2, ms=4, label=lb)
ax[2].set_xlabel("confidence  (per-cluster blend)"); ax[2].set_ylabel("t0 assignment accuracy [%]")
ax[2].set_title("cluster t0 accuracy vs confidence"); ax[2].legend(fontsize=8)
ax[2].grid(alpha=.3, ls=":"); ax[2].set_ylim(0, 102)
fig.tight_layout()
out = os.path.join(HERE, "deck_figs", "cluster_confidence_2d.png")
fig.savefig(out, dpi=130); print("wrote", out)
