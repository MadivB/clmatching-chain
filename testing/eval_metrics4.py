"""Round 4: the DISPLACED-LIGHT metric (user's conservation insight, 2026-07-09).
A swap = deficit at some channels at one t0 matched by EXCESS at the SAME channels at
another t0 (same SiPMs, light assigned to the wrong time).  Per channel, across the
t0 list, the matched deficit/excess = light that MOVED between t0s = mis-classified.
Uniform fluctuations / the 0.78 systematic are deficit-only -> min(deficit,excess)~0,
so this is naturally immune to them.  Graded vs MC-truth mis-classification energy.
"""
import os, pickle, json, numpy as np

HERE = os.path.abspath(os.path.dirname(__file__))
GAIN = 200.0
cache = pickle.load(open(os.path.join(HERE, "lightcache.pkl"), "rb"))

M = {k: [] for k in ["disp_abs", "disp_frac", "disp_thr_abs", "disp_thr_frac",
                     "disp_pair", "disp_frac_sig", "net_deficit_frac", "chi2_flash_snr"]}
WF, EE = [], []
for r in cache:
    A = r["Fa"].astype(np.float64)               # (nflash,120) actual
    P = r["Fp"].astype(np.float64) * r["alpha"]  # (nflash,120) alpha-scaled predicted
    WF.append(r["wf"]); EE.append(r["E"])
    if len(A) < 1:
        for k in M: M[k].append(0.0)
        continue
    R = A - P                                     # residual per (flash, channel); >0 deficit, <0 excess
    pos = np.maximum(R, 0).sum(0)                 # (120,) total deficit per channel across t0s
    neg = np.maximum(-R, 0).sum(0)               # (120,) total excess per channel across t0s
    disp = np.minimum(pos, neg)                   # matched displaced light per channel
    tot = max(A.sum(), 1.0)
    M["disp_abs"].append(float(disp.sum()))
    M["disp_frac"].append(float(disp.sum() / tot))
    M["net_deficit_frac"].append(float(np.abs(pos - neg).sum() / tot))   # UN-matched (benign-ish) for contrast
    # significance-thresholded ("clearly" deficit/excess): only count |R| > 2 sigma_cell
    sig = np.sqrt(GAIN * np.maximum(A, 900.0))    # per-cell window noise (9 ticks -> floor 900)
    Rt = np.where(np.abs(R) > 2 * sig, R, 0.0)
    post = np.maximum(Rt, 0).sum(0); negt = np.maximum(-Rt, 0).sum(0); dispt = np.minimum(post, negt)
    M["disp_thr_abs"].append(float(dispt.sum()))
    M["disp_thr_frac"].append(float(dispt.sum() / tot))
    # significance-weighted displaced fraction: displaced/(local noise) summed
    M["disp_frac_sig"].append(float((disp / np.sqrt(GAIN * np.maximum(pos + neg, 900.0))).sum() / max(len(A), 1)))
    # explicit pairwise: deficit at flash i meets excess at flash j on same channels
    pair = 0.0
    for i in range(len(A)):
        for j in range(len(A)):
            if i == j: continue
            pair += float((np.maximum(R[i], 0) * np.maximum(-R[j], 0)).sum())
    M["disp_pair"].append(float(np.sqrt(max(pair, 0)) / tot))
    # reference: chi2_flash snr-weighted
    ampA = A.sum(1); red = ((P - A) ** 2 / (GAIN * np.maximum(A, 100.0))).sum(1) / A.shape[1]
    w = np.sqrt(np.maximum(ampA, 0) / GAIN)
    M["chi2_flash_snr"].append(float((red * w).sum() / w.sum()) if w.sum() > 0 else 0.0)

WF = np.array(WF); EE = np.array(EE); WE = WF * EE
bad = WF > 0.20; good = WF < 0.05; sel = bad | good; y = bad[sel].astype(int)


def rank(x): return np.argsort(np.argsort(np.asarray(x, float))).astype(float)
def spr(a, b): return float(np.corrcoef(rank(a), rank(b))[0, 1])
def partial(a, b, c):
    ra, rb, rc = rank(a), rank(b), rank(c); X = np.c_[np.ones_like(rc), rc]
    ea = ra - X @ np.linalg.lstsq(X, ra, rcond=None)[0]; eb = rb - X @ np.linalg.lstsq(X, rb, rcond=None)[0]
    return float(np.corrcoef(ea, eb)[0, 1])
def auc(sc, yy):
    sc = np.asarray(sc); pos = sc[yy == 1]; neg = sc[yy == 0]
    o = np.argsort(np.concatenate([pos, neg])); rk = np.empty_like(o, float); rk[o] = np.arange(1, len(o) + 1)
    return (rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

print(f"{len(WF)} TPCs. BAD={int(bad.sum())} GOOD={int(good.sum())}  (ref chi2_flash_snr: AUC0.903 rhoFrac0.532 partial0.560)\n")
print(f"{'metric':>18} {'AUC':>7} {'rhoFrac':>9} {'rhoMeV':>8} {'partMeV|E':>10}")
res = []
for name, vals in M.items():
    v = np.array(vals, float)
    if v.std() == 0: continue
    res.append((auc(v[sel], y), spr(v, WF), spr(v, WE), partial(v, WE, EE), name))
res.sort(key=lambda r: -r[0])
for a, rf, rm, pm, n in res:
    print(f"{n:>18} {a:>7.3f} {rf:>+9.3f} {rm:>+8.3f} {pm:>+10.3f}")

# combine displaced + chi2_flash (do they complement?) via 5-fold CV
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    for feats in [["disp_frac"], ["chi2_flash_snr"], ["disp_frac", "chi2_flash_snr"],
                  ["disp_thr_frac", "chi2_flash_snr"], ["disp_frac", "disp_thr_frac", "chi2_flash_snr"]]:
        X = np.c_[[M[f] for f in feats]].T[sel]; X = (X - X.mean(0)) / (X.std(0) + 1e-9)
        p = cross_val_predict(LogisticRegression(class_weight="balanced", max_iter=2000), X, y, cv=5,
                              method="predict_proba")[:, 1]
        print(f"  CV AUC {'+'.join(feats):>40} = {auc(p, y):.3f}")
except Exception as e:
    print("sklearn:", e)

json.dump({k: list(map(float, M[k])) for k in M} | {"wf": list(map(float, WF)), "E": list(map(float, EE))},
          open(os.path.join(HERE, "eval_metrics4_out.json"), "w"))
print("DONE")
