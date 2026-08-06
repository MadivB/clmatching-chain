"""Confidence v2 (v1 + vFable maxk) figure + vFable mechanism illustrations."""
import os, json, pickle, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

HERE = os.path.abspath(os.path.dirname(__file__))
GAIN = 200.0
VF = json.load(open(os.path.join(HERE, "confidence_vfable_out.json")))
V1 = json.load(open(os.path.join(HERE, "eval_final_out.json")))
FR = json.load(open(os.path.join(HERE, "eval_fullres_out.json")))
WF = np.array(V1["wf"])
bad = WF > 0.20; good = WF < 0.05; sel = bad | good; y = bad[sel].astype(int)
cache = pickle.load(open(os.path.join(HERE, "lightcache.pkl"), "rb"))


def rank(x): return np.argsort(np.argsort(np.asarray(x, float))).astype(float) / len(x)
def auc(sc, yy):
    sc = np.asarray(sc); pos = sc[yy == 1]; neg = sc[yy == 0]
    o = np.argsort(np.concatenate([pos, neg])); rk = np.empty_like(o, float); rk[o] = np.arange(1, len(o) + 1)
    return (rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
def spr(a, b): return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])

# confidence v2 = complexity-normalized mean-rank(chi2_flash, displaced, vFable maxk)
NF = np.array(FR["nflash"], float); NL = np.array(FR["nlit"], float); logE = np.log(np.maximum(np.array(V1["E"]), 1))
def presid(m):
    rm = rank(m); X = np.c_[[np.ones_like(rm), rank(NF), rank(logE), rank(NL)]].T
    return rm - X @ np.linalg.lstsq(X, rm, rcond=None)[0]
score = np.mean([rank(presid(np.array(V1["chi2"]))), rank(presid(np.array(V1["disp"]))),
                 rank(presid(np.array(VF["maxk"])))], axis=0)
def anscombe(x): return 2 * np.sqrt(np.maximum(x, 0) + 0.375)
BLOCKS = [1, 2, 4, 8, 12, 24]
def chi2_curve(R):
    nf, nch = R.shape; out = []
    for k in BLOCKS:
        if nch % k: out.append(np.nan); continue
        Rb = R.reshape(nf, nch // k, k).sum(-1) / np.sqrt(k); out.append((Rb ** 2).mean())
    return np.array(out)
def Rmap(r):
    A = r["Fa"].astype(np.float64) / GAIN; P = np.maximum(r["alpha"] * r["Fp"].astype(np.float64), 0) / GAIN
    return anscombe(P) - anscombe(A)

# pick example TPCs: a clean BAD swap (few flashes, high wf, strong negative off-diag) + a clean GOOD
def worst_offdiag(r):
    R = Rmap(r)
    if R.shape[0] < 2: return 0.0
    M = R @ R.T / R.shape[1]; np.fill_diagonal(M, np.inf); return float(M.min())
badi = [i for i, r in enumerate(cache) if 0.25 < WF[i] and 2 <= len(r["Fa"]) <= 5]
badi.sort(key=lambda i: worst_offdiag(cache[i]))          # most-negative first = clearest swap
goodi = [i for i, r in enumerate(cache) if WF[i] < 0.02 and 3 <= len(r["Fa"]) <= 6]
bi = badi[0] if badi else int(np.argmax(WF)); gi = goodi[len(goodi) // 2] if goodi else int(np.argmin(WF))

fig, ax = plt.subplots(2, 2, figsize=(14, 10))

# (a) v2 vs mis-classified fraction
hb = ax[0, 0].hexbin(score, 100 * WF, gridsize=40, cmap="viridis", norm=LogNorm(vmin=1),
                     extent=(0, 1, 0, 100), mincnt=1)
order = np.argsort(score); cx, md = [], []
for i in range(14):
    lo, hi = i / 14, (i + 1) / 14; s = (score >= lo) & (score < hi)
    if s.sum() >= 5: cx.append(score[s].mean()); md.append(100 * np.median(WF[s]))
ax[0, 0].plot(cx, md, "ko-", lw=2, ms=4, label="binned median")
ax[0, 0].axhline(20, color="tab:red", ls=":", lw=1); ax[0, 0].axhline(5, color="w", ls=":", lw=1)
ax[0, 0].set_xlim(0, 1); ax[0, 0].set_ylim(0, 100); ax[0, 0].legend(loc="upper left", fontsize=8)
ax[0, 0].set_xlabel("confidence v2  = complexity-norm mean-rank(chi2_flash, displaced, vFable maxk)")
ax[0, 0].set_ylabel("mis-classified energy [% of TPC]")
ax[0, 0].set_title(f"(a) confidence v2 vs mis-classified fraction   Spearman={spr(score, WF):+.3f}  AUC={auc(score[sel], y):.3f}")
fig.colorbar(hb, ax=ax[0, 0], pad=.01).set_label("TPCs/cell")

# (b) rhoFrac contribution bars
labels = ["v1 chi2_flash", "v1 displaced", "vFable Anscombe-chi2", "vFable maxk\n(multiscale)",
          "vFable z_zmis_y", "v1 (chi2+disp)", "v1 disp + maxk", "confidence v2"]
vals = [0.532, 0.576, 0.585, 0.605, 0.515, 0.630, 0.658, 0.654]
cols = ["tab:orange", "tab:orange", "tab:cyan", "tab:blue", "tab:cyan", "0.4", "tab:green", "black"]
ax[0, 1].barh(range(len(vals)), vals, color=cols)
ax[0, 1].set_yticks(range(len(vals))); ax[0, 1].set_yticklabels(labels, fontsize=8)
ax[0, 1].invert_yaxis(); ax[0, 1].set_xlim(0.45, 0.70)
for i, v in enumerate(vals): ax[0, 1].text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=8)
ax[0, 1].set_xlabel("Spearman(metric, mis-classified fraction)")
ax[0, 1].set_title("(b) continuous-correlation contribution\n(vFable maxk adds an independent axis)")

# (c) multiscale chi2(k): GOOD flat ~1, BAD rises
kg = chi2_curve(Rmap(cache[gi])); kb = chi2_curve(Rmap(cache[bi]))
ax[1, 0].plot(BLOCKS, kg, "o-", color="tab:green", label=f"GOOD (wf={WF[gi]*100:.1f}%)")
ax[1, 0].plot(BLOCKS, kb, "s-", color="tab:red", label=f"BAD swap (wf={WF[bi]*100:.1f}%)")
ax[1, 0].axhline(1, color="k", ls=":", lw=.8, label="white-noise null (=1)")
ax[1, 0].set_xlabel("channel-block size k"); ax[1, 0].set_ylabel("chi2_red(k)")
ax[1, 0].set_title("(c) multiscale whiteness: flat=clean, rising=coherent mis-placement"); ax[1, 0].legend(fontsize=8)

# (d) time-time covariance M(t,t') of the BAD TPC -> negative off-diagonal = swapped pair
Rb = Rmap(cache[bi]); M = Rb @ Rb.T / Rb.shape[1]
im = ax[1, 1].imshow(M, cmap="RdBu_r", vmin=-abs(M).max(), vmax=abs(M).max())
Moff = M.copy(); np.fill_diagonal(Moff, np.inf); ij = np.unravel_index(np.argmin(Moff), M.shape)
ax[1, 1].scatter([ij[1]], [ij[0]], s=200, facecolors="none", edgecolors="lime", lw=2)
ax[1, 1].scatter([ij[0]], [ij[1]], s=200, facecolors="none", edgecolors="lime", lw=2)
ft = cache[bi]["Ft"]
ax[1, 1].set_xticks(range(len(ft))); ax[1, 1].set_yticks(range(len(ft)))
ax[1, 1].set_xticklabels([str(int(t)) for t in ft], fontsize=7); ax[1, 1].set_yticklabels([str(int(t)) for t in ft], fontsize=7)
ax[1, 1].set_xlabel("flash tick t'"); ax[1, 1].set_ylabel("flash tick t")
ax[1, 1].set_title(f"(d) residual covariance M(t,t'): green ring = displaced pair\n(deficit@{int(ft[ij[0]])} <-> excess@{int(ft[ij[1]])}, Z_mis={M[ij]*np.sqrt(Rb.shape[1]):.1f})")
fig.colorbar(im, ax=ax[1, 1], pad=.01).set_label("residual correlation")

fig.suptitle("Confidence metric vFable: residual-whiteness + displaced-flash detector, synthesized into v2  (v0.3, 2320 TPCs)", fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(HERE, "deck_figs", "confidence_vfable.png")
fig.savefig(out, dpi=118); print("wrote", out)
print(f"v2 AUC={auc(score[sel], y):.3f} rhoFrac={spr(score, WF):+.3f}; BAD example idx {bi} wf={WF[bi]:.2f} nflash={len(cache[bi]['Ft'])}")
