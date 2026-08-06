"""Round 7: remove the COMPLEXITY confound found in the diagnosis.  Busy/bright TPCs
have high chi2 & displaced-light even when correctly placed (ML imperfection accrues
over many flashes).  Score each metric RELATIVE to what's normal for its complexity
(nflash, energy, lit-channels) -- all truth-free.  Two truth-free normalizations:
 (A) partial-rank residual vs [nflash, logE, n_lit],
 (B) within-complexity-bin percentile.
"""
import os, pickle, json, numpy as np

HERE = os.path.abspath(os.path.dirname(__file__))
GAIN = 200.0
cache = pickle.load(open(os.path.join(HERE, "lightcache.pkl"), "rb"))
K = 3.0


def feats(r):
    A = r["Fa"].astype(np.float64); P = r["Fp"].astype(np.float64) * r["alpha"]
    nf = len(A)
    if nf < 1:
        return 0.0, 0.0, 0, 0.0, 0
    ampA = A.sum(1); red = ((P - A) ** 2 / (GAIN * np.maximum(A, 100.0))).sum(1) / A.shape[1]
    w = np.sqrt(np.maximum(ampA, 0) / GAIN)
    chi2 = float((red * w).sum() / w.sum()) if w.sum() > 0 else 0.0
    R = A - P; sig = np.sqrt(GAIN * np.maximum(A, 900.0)); Rt = np.where(np.abs(R) > K * sig, R, 0.0)
    disp = float(np.minimum(np.maximum(Rt, 0).sum(0), np.maximum(-Rt, 0).sum(0)).sum() / max(A.sum(), 1.0))
    tot = float(A.sum())
    nlit = int((A.sum(0) > 5 * np.sqrt(GAIN * 100.0)).sum())
    return chi2, disp, nf, tot, nlit


CH, DP, NF, TOT, NL = [], [], [], [], []
for r in cache:
    c, d, nf, tot, nl = feats(r); CH.append(c); DP.append(d); NF.append(nf); TOT.append(tot); NL.append(nl)
CH = np.array(CH); DP = np.array(DP); NF = np.array(NF); TOT = np.array(TOT); NL = np.array(NL)
WF = np.array([r["wf"] for r in cache]); EE = np.array([r["E"] for r in cache])
bad = WF > 0.20; good = WF < 0.05; sel = bad | good; y = bad[sel].astype(int)
logE = np.log(np.maximum(EE, 1)); logT = np.log(np.maximum(TOT, 1))


def rank(x): return np.argsort(np.argsort(np.asarray(x, float))).astype(float) / len(x)
def auc(sc, yy):
    sc = np.asarray(sc); pos = sc[yy == 1]; neg = sc[yy == 0]
    o = np.argsort(np.concatenate([pos, neg])); rk = np.empty_like(o, float); rk[o] = np.arange(1, len(o) + 1)
    return (rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
def spr(a, b): return float(np.corrcoef(rank(a), rank(b))[0, 1])


def partial_resid(metric, ctrls):
    """rank-residual of metric after regressing out ctrl ranks (truth-free)."""
    rm = rank(metric); X = np.c_[[np.ones_like(rm)] + [rank(c) for c in ctrls]].T
    beta = np.linalg.lstsq(X, rm, rcond=None)[0]
    return rm - X @ beta


print("confound check (Spearman among GOOD only):")
g = good
for nm, v in [("nflash", NF), ("logE", logE), ("nlit", NL)]:
    print(f"   corr(chi2,{nm})|GOOD={spr(CH[g], v[g]):+.3f}  corr(disp,{nm})|GOOD={spr(DP[g], v[g]):+.3f}")
print(f"   corr(nflash, wf) ALL = {spr(NF, WF):+.3f}   (genuine complexity->error link?)\n")

# baseline
zc0, zd0 = rank(CH), rank(DP); combo0 = 0.5 * (zc0 + zd0)
print(f"{'variant':>28} {'AUC':>7} {'rhoFrac':>9}")
print(f"{'raw combo mean(z)':>28} {auc(combo0[sel], y):>7.3f} {spr(combo0, WF):>+9.3f}")

# (A) partial-rank residual vs complexity
ctrls = [NF, logE, NL]
ac = partial_resid(CH, ctrls); ad = partial_resid(DP, ctrls)
for nm, arr in [("chi2 resid|complexity", ac), ("disp resid|complexity", ad)]:
    print(f"{nm:>28} {auc(arr[sel], y):>7.3f} {spr(arr, WF):>+9.3f}")
comboA = 0.5 * (rank(ac) + rank(ad))
print(f"{'combo resid mean(z)':>28} {auc(comboA[sel], y):>7.3f} {spr(comboA, WF):>+9.3f}")
comboAmax = np.maximum(rank(ac), rank(ad))
print(f"{'combo resid max(z)':>28} {auc(comboAmax[sel], y):>7.3f} {spr(comboAmax, WF):>+9.3f}")

# (B) within-nflash-bin percentile
def within_bin_pct(metric, binvar, nb=8):
    out = np.zeros_like(metric, float); order = np.argsort(binvar)
    edges = np.quantile(binvar, np.linspace(0, 1, nb + 1))
    for i in range(nb):
        m = (binvar >= edges[i]) & (binvar <= edges[i + 1]) if i == nb - 1 else (binvar >= edges[i]) & (binvar < edges[i + 1])
        if m.sum() > 1: out[m] = rank(metric[m])
    return out
bc = within_bin_pct(CH, NF); bd = within_bin_pct(DP, NF)
comboB = 0.5 * (bc + bd)
print(f"{'combo within-nflash-bin':>28} {auc(comboB[sel], y):>7.3f} {spr(comboB, WF):>+9.3f}")

# combine residual-normalized chi2 & disp with a light complexity prior kept out
print("\nspecificity (GOOD FPR at matched BAD recall):")
def fpr_at(s, tpr):
    ss = s[sel]; bp = ss[y == 1]; gn = ss[y == 0]; thr = np.quantile(bp, 1 - tpr); return (gn >= thr).mean()
print(f"   {'recall':>7} {'rawcombo':>9} {'residcombo':>11}")
for tpr in [0.70, 0.80, 0.90]:
    print(f"   {tpr:>7.2f} {100*fpr_at(combo0, tpr):>8.1f}% {100*fpr_at(comboA, tpr):>10.1f}%")

json.dump({"chi2": list(CH), "disp": list(DP), "nflash": list(map(int, NF)), "E": list(EE),
           "resid_chi2": list(ac), "resid_disp": list(ad), "comboA": list(comboA), "wf": list(WF)},
          open(os.path.join(HERE, "eval_metrics7_out.json"), "w"))
print("DONE")
