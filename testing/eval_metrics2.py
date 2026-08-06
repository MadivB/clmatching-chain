"""Round 2: UN-normalized, energy-scaled similarity metrics (structure x amplitude),
since the target is mis-classified ENERGY.  Also: honest confound check -- how much of
any metric's correlation with absolute mis-classified MeV is just TPC energy E?
Reports Spearman vs |MeV|, vs fraction, and PARTIAL Spearman vs |MeV| controlling for E.
"""
import os, pickle, numpy as np

HERE = os.path.abspath(os.path.dirname(__file__))
GAIN = 200.0
cache = pickle.load(open(os.path.join(HERE, "lightcache.pkl"), "rb"))


def _n(v):
    s = v.sum(); return v / s if s > 0 else v


def flash_scores(A, P):
    A = np.maximum(A, 0.0); P = np.maximum(P, 0.0)
    nA = np.linalg.norm(A); nP = np.linalg.norm(P)
    cos = (A @ P / (nA * nP)) if nA > 0 and nP > 0 else 0.0
    a = _n(A); p = _n(P)
    tv = 0.5 * np.abs(a - p).sum() if a.sum() > 0 and p.sum() > 0 else 1.0
    muA, muP = A.mean(), P.mean(); vA, vP = A.var(), P.var(); cov = ((A - muA) * (P - muP)).mean()
    Lr = max(A.max(), P.max(), 1.0); c1 = (0.01 * Lr) ** 2; c2 = (0.03 * Lr) ** 2
    ssim = ((2 * muA * muP + c1) * (2 * cov + c2)) / ((muA ** 2 + muP ** 2 + c1) * (vA + vP + c2))
    dist = float(np.linalg.norm(P - A) / (nA + nP)) if (nA + nP) > 0 else 0.0
    return dict(cosbad=1 - cos, tv=tv, ssimbad=1 - float(ssim), dist=dist,
                l1=float(np.abs(P - A).sum()), l2=float(np.linalg.norm(P - A)))


M = {k: [] for k in ["abs_misplaced", "abs_deficit", "abs_excess", "abs_cosmis", "abs_distw",
                     "abs_ssimw", "abs_tvw", "abs_l1", "abs_l2", "sqrt_cosmis", "log_cosmis",
                     "chi2_flash"]}
WF, EE = [], []
for r in cache:
    A_all = r["Fa"].astype(np.float64); P_all = r["Fp"].astype(np.float64) * r["alpha"]
    if len(A_all) == 0:
        WF.append(r["wf"]); EE.append(r["E"])
        for k in M: M[k].append(0.0)
        continue
    ampA = A_all.sum(1); ampP = P_all.sum(1)
    cosmis = 0.0; distw = 0.0; ssimw = 0.0; tvw = 0.0; l1 = 0.0; l2sq = 0.0
    for A, P, aA in zip(A_all, P_all, ampA):
        s = flash_scores(A, P)
        cosmis += aA * s["cosbad"]; distw += aA * s["dist"]; ssimw += aA * s["ssimbad"]
        tvw += aA * s["tv"]; l1 += s["l1"]; l2sq += s["l2"] ** 2
    M["abs_misplaced"].append(float(np.abs(ampA - ampP).sum()))
    M["abs_deficit"].append(float(np.maximum(ampA - ampP, 0).sum()))
    M["abs_excess"].append(float(np.maximum(ampP - ampA, 0).sum()))
    M["abs_cosmis"].append(float(cosmis)); M["abs_distw"].append(float(distw))
    M["abs_ssimw"].append(float(ssimw)); M["abs_tvw"].append(float(tvw))
    M["abs_l1"].append(float(l1)); M["abs_l2"].append(float(np.sqrt(l2sq)))
    M["sqrt_cosmis"].append(float(np.sqrt(max(cosmis, 0))))
    M["log_cosmis"].append(float(np.log1p(max(cosmis, 0))))
    # flash-windowed chi2/dof (apples-to-apples with whole-TPC chi2s=0.685)
    num = 0.0; dof = 0
    for A, P in zip(A_all, P_all):
        v = GAIN * np.maximum(A, 100.0); num += float((((P - A) ** 2) / v).sum()); dof += len(A)
    M["chi2_flash"].append(num / max(dof, 1))
    WF.append(r["wf"]); EE.append(r["E"])

WF = np.array(WF); EE = np.array(EE); WE = WF * EE
bad = WF > 0.20; good = WF < 0.05; sel = bad | good; y = bad[sel].astype(int)


def rank(x): return np.argsort(np.argsort(np.asarray(x, float))).astype(float)
def spr(a, b): return float(np.corrcoef(rank(a), rank(b))[0, 1])
def partial_spr(a, b, ctrl):
    """Spearman(a,b) controlling for ctrl, via rank-residuals."""
    ra, rb, rc = rank(a), rank(b), rank(ctrl)
    rc1 = np.c_[np.ones_like(rc), rc]
    ea = ra - rc1 @ np.linalg.lstsq(rc1, ra, rcond=None)[0]
    eb = rb - rc1 @ np.linalg.lstsq(rc1, rb, rcond=None)[0]
    return float(np.corrcoef(ea, eb)[0, 1])
def auc(sc, yy):
    sc = np.asarray(sc); pos = sc[yy == 1]; neg = sc[yy == 0]
    o = np.argsort(np.concatenate([pos, neg])); rk = np.empty_like(o, float); rk[o] = np.arange(1, len(o) + 1)
    return (rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

print(f"{len(WF)} TPCs. BAD={int(bad.sum())} GOOD={int(good.sum())}")
print(f"CONFOUND baseline: Spearman(E, |MeV|) = {spr(EE, WE):+.3f}   (any E-correlated metric inherits this)\n")
print(f"{'metric':>15} {'AUC':>7} {'rhoFrac':>9} {'rhoMeV':>8} {'partialMeV|E':>13}")
res = []
# include the whole-TPC chi2_scaled reference from quality_final.json if present
ref = None
qf = os.path.join(HERE, "quality_final.json")
if os.path.exists(qf):
    import json
    Q = json.load(open(qf)); ref = np.array([q["chi2s"] for q in Q])
    if len(ref) == len(WF):
        M["chi2s_wholeTPC"] = list(ref)
for name, vals in M.items():
    v = np.array(vals, float)
    if v.std() == 0: continue
    res.append((auc(v[sel], y), spr(v, WF), spr(v, WE), partial_spr(v, WE, EE), name))
res.sort(key=lambda r: -r[2])
for a, rf, rm, pm, n in res:
    print(f"{n:>15} {a:>7.3f} {rf:>+9.3f} {rm:>+8.3f} {pm:>+13.3f}")
print(f"\nBEST rhoMeV: {res[0][4]}   BEST AUC: {max(res,key=lambda r:r[0])[4]}   "
      f"BEST partial(MeV|E): {max(res,key=lambda r:r[3])[4]}")
print("DONE")
