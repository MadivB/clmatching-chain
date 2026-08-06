"""Round 3: refine the winner chi2_flash (chi2 evaluated only at the t0 list).
Study the cross-flash aggregation (cell-avg vs flash-avg vs amp/snr-weighted), a
structural combo, and confirm generalization on a held-out EVENT split.
"""
import os, pickle, numpy as np

HERE = os.path.abspath(os.path.dirname(__file__))
GAIN = 200.0
cache = pickle.load(open(os.path.join(HERE, "lightcache.pkl"), "rb"))


def _n(v):
    s = v.sum(); return v / s if s > 0 else v


def ssim_bad(A, P):
    A = np.maximum(A, 0.0); P = np.maximum(P, 0.0)
    muA, muP = A.mean(), P.mean(); vA, vP = A.var(), P.var(); cov = ((A - muA) * (P - muP)).mean()
    Lr = max(A.max(), P.max(), 1.0); c1 = (0.01 * Lr) ** 2; c2 = (0.03 * Lr) ** 2
    return 1.0 - float(((2 * muA * muP + c1) * (2 * cov + c2)) / ((muA ** 2 + muP ** 2 + c1) * (vA + vP + c2)))


rows = []
for r in cache:
    A_all = r["Fa"].astype(np.float64); P_all = r["Fp"].astype(np.float64) * r["alpha"]
    key = (r["file"], r["event"]); wf = r["wf"]; E = r["E"]
    if len(A_all) == 0:
        rows.append(dict(key=key, wf=wf, E=E, cellavg=0, flashavg=0, ampavg=0, snravg=0, ssimw=0)); continue
    ampA = A_all.sum(1)
    fchi = []; fdof = []
    for A, P in zip(A_all, P_all):
        v = GAIN * np.maximum(A, 100.0)
        fchi.append(float(((P - A) ** 2 / v).sum())); fdof.append(len(A))
    fchi = np.array(fchi); fdof = np.array(fdof); red = fchi / fdof
    cellavg = fchi.sum() / fdof.sum()                       # current winner (0.894)
    flashavg = red.mean()                                    # democratic per flash
    wsum = ampA.sum()
    ampavg = float((red * ampA).sum() / wsum) if wsum > 0 else 0.0
    wsnr = np.sqrt(np.maximum(ampA, 0) / GAIN); ssum = wsnr.sum()
    snravg = float((red * wsnr).sum() / ssum) if ssum > 0 else 0.0
    ssimw = float(sum(a * ssim_bad(A, P) for A, P, a in zip(A_all, P_all, ampA)))
    rows.append(dict(key=key, wf=wf, E=E, cellavg=cellavg, flashavg=flashavg,
                     ampavg=ampavg, snravg=snravg, ssimw=ssimw))

WF = np.array([r["wf"] for r in rows]); EE = np.array([r["E"] for r in rows]); WE = WF * EE
bad = WF > 0.20; good = WF < 0.05; sel = bad | good; y = bad[sel].astype(int)


def rank(x): return np.argsort(np.argsort(np.asarray(x, float))).astype(float)
def spr(a, b): return float(np.corrcoef(rank(a), rank(b))[0, 1])
def partial(a, b, c):
    ra, rb, rc = rank(a), rank(b), rank(c); X = np.c_[np.ones_like(rc), rc]
    ea = ra - X @ np.linalg.lstsq(X, ra, rcond=None)[0]; eb = rb - X @ np.linalg.lstsq(X, rb, rcond=None)[0]
    return float(np.corrcoef(ea, eb)[0, 1])
def auc(sc, yy):
    sc = np.asarray(sc); pos = sc[yy == 1]; neg = sc[yy == 0]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    o = np.argsort(np.concatenate([pos, neg])); rk = np.empty_like(o, float); rk[o] = np.arange(1, len(o) + 1)
    return (rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

print(f"{len(WF)} TPCs. BAD={int(bad.sum())} GOOD={int(good.sum())}\n")
print(f"{'variant':>12} {'AUC':>7} {'rhoFrac':>9} {'rhoMeV':>8} {'partMeV|E':>10}")
for v in ["cellavg", "flashavg", "ampavg", "snravg", "ssimw"]:
    s = np.array([r[v] for r in rows])
    print(f"{v:>12} {auc(s[sel], y):>7.3f} {spr(s, WF):>+9.3f} {spr(s, WE):>+8.3f} {partial(s, WE, EE):>+10.3f}")

# ---- structural combo: cellavg + normalized ssimw, via 5-fold logistic CV (BAD/GOOD) ----
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    Xc = np.c_[[r["cellavg"] for r in rows], [r["ssimw"] for r in rows]][sel]
    Xc = (Xc - Xc.mean(0)) / (Xc.std(0) + 1e-9)
    p = cross_val_predict(LogisticRegression(class_weight="balanced", max_iter=2000), Xc, y, cv=5,
                          method="predict_proba")[:, 1]
    print(f"\ncombo[cellavg+ssimw] 5-fold CV AUC = {auc(p, y):.3f}")
except Exception as e:
    print("sklearn:", e)

# ---- generalization: held-out EVENT split (train events -> pick nothing; just check AUC stability) ----
keys = sorted(set(r["key"] for r in rows))
half = set(keys[::2])
inA = np.array([r["key"] in half for r in rows])
for grp, mask in [("splitA", inA), ("splitB", ~inA)]:
    m = mask & sel; yy = bad[m].astype(int); s = np.array([r["cellavg"] for r in rows])[m]
    print(f"chi2_flash cellavg  {grp}: AUC={auc(s, yy):.3f}  (BAD={int(yy.sum())} GOOD={int((yy==0).sum())})")
print("DONE")
