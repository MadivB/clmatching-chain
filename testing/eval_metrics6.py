"""Round 6: push the metric further.
 (1) diagnose current-combo failures (missed BAD / false-positive GOOD),
 (2) PAIRWISE shape-matched transport: does the deficit pattern at t_A match the
     EXCESS pattern at t_B (same cluster moved t_A->t_B)? more specific than the
     per-channel aggregate,
 (3) a cross-validated CONTINUOUS combination trained on the fraction over ALL 2320
     TPCs (not the 37-BAD binary -> less overfitting).
"""
import os, pickle, json, numpy as np

HERE = os.path.abspath(os.path.dirname(__file__))
GAIN = 200.0
cache = pickle.load(open(os.path.join(HERE, "lightcache.pkl"), "rb"))
K = 3.0


def feats(r):
    A = r["Fa"].astype(np.float64); P = r["Fp"].astype(np.float64) * r["alpha"]; Ft = r["Ft"].astype(float)
    nf = len(A)
    if nf < 1:
        return dict(chi2=0, disp=0, pair_ov=0, pair_dot=0, pair_cos=0, pair_time=0, nflash=0)
    ampA = A.sum(1); red = ((P - A) ** 2 / (GAIN * np.maximum(A, 100.0))).sum(1) / A.shape[1]
    w = np.sqrt(np.maximum(ampA, 0) / GAIN)
    chi2 = float((red * w).sum() / w.sum()) if w.sum() > 0 else 0.0
    R = A - P; sig = np.sqrt(GAIN * np.maximum(A, 900.0)); Rt = np.where(np.abs(R) > K * sig, R, 0.0)
    Def = np.maximum(Rt, 0); Exc = np.maximum(-Rt, 0)          # (nf,120) thresholded deficit/excess
    tot = max(A.sum(), 1.0)
    disp = float(np.minimum(Def.sum(0), Exc.sum(0)).sum() / tot)   # current per-channel metric
    # PAIRWISE: deficit at flash i vs excess at flash j (i != j)
    ov = 0.0; dot = 0.0; coss = 0.0; timew = 0.0
    for i in range(nf):
        di = Def[i]; ndi = np.linalg.norm(di)
        if ndi == 0: continue
        for j in range(nf):
            if i == j: continue
            ej = Exc[j]; nej = np.linalg.norm(ej)
            if nej == 0: continue
            m = np.minimum(di, ej).sum()                       # channel-matched overlap i->j
            ov += m
            dot += float(di @ ej) / tot                        # spatial coincidence (same channels)
            c = float(di @ ej / (ndi * nej))                   # SHAPE match (same footprint)
            coss += c * m
            timew += m * abs(Ft[i] - Ft[j])                    # weighted by temporal separation
    return dict(chi2=chi2, disp=disp, pair_ov=ov / tot, pair_dot=dot,
                pair_cos=coss / tot, pair_time=timew / tot / 100.0, nflash=nf)


F = [feats(r) for r in cache]
WF = np.array([r["wf"] for r in cache]); EE = np.array([r["E"] for r in cache]); WE = WF * EE
keys = [(r["file"], r["event"], r["tpc"]) for r in cache]
bad = WF > 0.20; good = WF < 0.05; sel = bad | good; y = bad[sel].astype(int)


def auc(sc, yy):
    sc = np.asarray(sc); pos = sc[yy == 1]; neg = sc[yy == 0]
    o = np.argsort(np.concatenate([pos, neg])); rk = np.empty_like(o, float); rk[o] = np.arange(1, len(o) + 1)
    return (rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
def spr(a, b): return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])

print("=== single-metric scan ===")
print(f"{'metric':>10} {'AUC':>7} {'rhoFrac':>9}")
cols = ["chi2", "disp", "pair_ov", "pair_dot", "pair_cos", "pair_time"]
for c in cols:
    v = np.array([f[c] for f in F])
    print(f"{c:>10} {auc(v[sel], y):>7.3f} {spr(v, WF):>+9.3f}")

# current best combo for diagnosis
zc = np.argsort(np.argsort([f["chi2"] for f in F])).astype(float) / len(F)
zd = np.argsort(np.argsort([f["disp"] for f in F])).astype(float) / len(F)
combo = 0.5 * (zc + zd)
print("\n=== FAILURE DIAGNOSIS (combo mean(z)) ===")
thr = np.quantile(combo[sel][y == 1], 0.15)   # ~85% recall threshold
missed = [(keys[i], round(WF[i], 3), round(EE[i], 0), F[i]["nflash"], round(zc[i], 2), round(zd[i], 2))
          for i in range(len(F)) if bad[i] and combo[i] < thr]
fps = [(keys[i], round(WF[i], 3), round(EE[i], 0), F[i]["nflash"], round(zc[i], 2), round(zd[i], 2))
       for i in range(len(F)) if good[i] and combo[i] >= thr]
print(f"MISSED BAD (n={len(missed)}): [key, wf, E, nflash, z_chi2, z_disp]")
for m in missed: print("   ", m)
print(f"FALSE-POSITIVE GOOD (n={len(fps)}, showing 12 highest-combo):")
for m in sorted(fps, key=lambda t: -(zc[keys.index(t[0])] + zd[keys.index(t[0])]))[:12]: print("   ", m)

# (3) continuous CV combination on the FRACTION target (all TPCs)
try:
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import cross_val_predict, KFold
    X = np.array([[f[c] for c in cols] for f in F]); X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    tgt = np.sqrt(WF)     # compress skew
    kf = KFold(5, shuffle=True, random_state=0)
    pred = cross_val_predict(RidgeCV(alphas=[0.1, 1, 10, 100]), X, tgt, cv=kf)
    print(f"\n=== CV continuous combo (Ridge on sqrt(fraction), all {len(F)} TPCs) ===")
    print(f"   CV Spearman(pred, wf) = {spr(pred, WF):+.3f}   AUC(BAD/GOOD) = {auc(pred[sel], y):.3f}")
    # compare to best hand combo
    print(f"   (hand combo mean(z): rhoFrac {spr(combo, WF):+.3f}  AUC {auc(combo[sel], y):.3f})")
except Exception as e:
    print("sklearn:", e)

json.dump({c: [f[c] for f in F] for c in cols} | {"wf": list(WF), "E": list(EE)},
          open(os.path.join(HERE, "eval_metrics6_out.json"), "w"))
print("DONE")
