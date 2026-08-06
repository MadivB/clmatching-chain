"""Round 5: is displaced-light MORE SPECIFIC than chi2_flash (the real point)?
Sweep the 'clearly' threshold; compare GOOD false-positive rate at matched BAD-recall;
check complementarity (which BAD each catches); test a physically-motivated MAX/OR
combination instead of a fitted one.
"""
import os, pickle, numpy as np

HERE = os.path.abspath(os.path.dirname(__file__))
GAIN = 200.0
cache = pickle.load(open(os.path.join(HERE, "lightcache.pkl"), "rb"))


def displaced_frac(A, P, k):
    R = A - P
    sig = np.sqrt(GAIN * np.maximum(A, 900.0))
    Rt = np.where(np.abs(R) > k * sig, R, 0.0)
    pos = np.maximum(Rt, 0).sum(0); neg = np.maximum(-Rt, 0).sum(0)
    return float(np.minimum(pos, neg).sum() / max(A.sum(), 1.0))


chi2 = []; WF = []; EE = []; keys = []
disp = {k: [] for k in [1.0, 1.5, 2.0, 2.5, 3.0]}
for r in cache:
    A = r["Fa"].astype(np.float64); P = r["Fp"].astype(np.float64) * r["alpha"]
    WF.append(r["wf"]); EE.append(r["E"]); keys.append((r["file"], r["event"], r["tpc"]))
    if len(A) < 1:
        chi2.append(0.0)
        for k in disp: disp[k].append(0.0)
        continue
    ampA = A.sum(1); red = ((P - A) ** 2 / (GAIN * np.maximum(A, 100.0))).sum(1) / A.shape[1]
    w = np.sqrt(np.maximum(ampA, 0) / GAIN)
    chi2.append(float((red * w).sum() / w.sum()) if w.sum() > 0 else 0.0)
    for k in disp: disp[k].append(displaced_frac(A, P, k))

chi2 = np.array(chi2); WF = np.array(WF); EE = np.array(EE); WE = WF * EE
bad = WF > 0.20; good = WF < 0.05; sel = bad | good; y = bad[sel].astype(int)


def auc(sc, yy):
    sc = np.asarray(sc); pos = sc[yy == 1]; neg = sc[yy == 0]
    o = np.argsort(np.concatenate([pos, neg])); rk = np.empty_like(o, float); rk[o] = np.arange(1, len(o) + 1)
    return (rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
def spr(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])
def fpr_at_tpr(score, target):
    """threshold giving >=target recall on BAD; return GOOD false-positive rate there."""
    s = score[sel]; bpos = s[y == 1]; gneg = s[y == 0]
    thr = np.quantile(bpos, 1 - target)     # catches ~target fraction of BAD
    return float((gneg >= thr).mean()), thr

print(f"{len(WF)} TPCs. BAD={int(bad.sum())} GOOD={int(good.sum())}\n")
print("threshold sweep for displaced_frac:")
for k in sorted(disp):
    d = np.array(disp[k])
    print(f"  k={k}:  AUC={auc(d[sel], y):.3f}  rhoFrac={spr(d, WF):+.3f}")
best_k = max(disp, key=lambda k: auc(np.array(disp[k])[sel], y))
D = np.array(disp[best_k])
print(f"\nbest displaced threshold k={best_k}  (AUC {auc(D[sel], y):.3f})")
print(f"chi2_flash_snr AUC {auc(chi2[sel], y):.3f}\n")

print("SPECIFICITY: GOOD false-positive rate at matched BAD-recall (lower=better):")
print(f"  {'BAD recall':>11} {'chi2 FPR':>9} {'disp FPR':>9}")
for tpr in [0.70, 0.80, 0.90, 0.95]:
    fc, _ = fpr_at_tpr(chi2, tpr); fd, _ = fpr_at_tpr(D, tpr)
    print(f"  {tpr:>11.2f} {100*fc:>8.1f}% {100*fd:>8.1f}%")

# complementarity at 85% recall each
_, tc = fpr_at_tpr(chi2, 0.85); _, td = fpr_at_tpr(D, 0.85)
cb = (chi2 >= tc); db = (D >= td)
badkeys = [k for k, b in zip(keys, bad) if b]
both = sum(1 for k, b in zip(keys, bad) if b and cb[keys.index(k)] and db[keys.index(k)])
conly = sum(1 for i, b in enumerate(bad) if b and cb[i] and not db[i])
donly = sum(1 for i, b in enumerate(bad) if b and db[i] and not cb[i])
neither = sum(1 for i, b in enumerate(bad) if b and not cb[i] and not db[i])
print(f"\nBAD caught @85% recall each: both={both}  chi2-only={conly}  disp-only={donly}  neither={neither}")

# physically-motivated combos (rank-space, no fitting): max of z-scores; sum; product
def z(x): return (np.argsort(np.argsort(x)) / len(x))
zc = z(chi2); zd = z(D)
for name, comb in [("max(z)", np.maximum(zc, zd)), ("mean(z)", 0.5 * (zc + zd)),
                   ("chi2*disp", zc * zd), ("chi2+0.5disp", zc + 0.5 * zd)]:
    print(f"  combo {name:>14}: AUC={auc(comb[sel], y):.3f}  rhoFrac={spr(comb, WF):+.3f}")
print("DONE")
