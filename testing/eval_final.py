"""Finalize: chi2_flash + displaced-light(k) + unfitted rank combo.  Confirm the
threshold, validate the combo on a held-out EVENT split, save per-TPC values.
"""
import os, pickle, json, numpy as np

HERE = os.path.abspath(os.path.dirname(__file__))
GAIN = 200.0
cache = pickle.load(open(os.path.join(HERE, "lightcache.pkl"), "rb"))


def compute(r, k):
    A = r["Fa"].astype(np.float64); P = r["Fp"].astype(np.float64) * r["alpha"]
    if len(A) < 1:
        return 0.0, 0.0
    ampA = A.sum(1); red = ((P - A) ** 2 / (GAIN * np.maximum(A, 100.0))).sum(1) / A.shape[1]
    w = np.sqrt(np.maximum(ampA, 0) / GAIN)
    c2 = float((red * w).sum() / w.sum()) if w.sum() > 0 else 0.0
    R = A - P; sig = np.sqrt(GAIN * np.maximum(A, 900.0)); Rt = np.where(np.abs(R) > k * sig, R, 0.0)
    pos = np.maximum(Rt, 0).sum(0); neg = np.maximum(-Rt, 0).sum(0)
    disp = float(np.minimum(pos, neg).sum() / max(A.sum(), 1.0))
    return c2, disp


chi2 = []; WF = []; EE = []; keys = []
dispk = {k: [] for k in [2.5, 3.0, 3.5, 4.0]}
for r in cache:
    c2, _ = compute(r, 3.0); chi2.append(c2); WF.append(r["wf"]); EE.append(r["E"])
    keys.append((r["file"], r["event"]))
    for k in dispk: dispk[k].append(compute(r, k)[1])
chi2 = np.array(chi2); WF = np.array(WF); EE = np.array(EE); WE = WF * EE
bad = WF > 0.20; good = WF < 0.05; sel = bad | good; y = bad[sel].astype(int)


def auc(sc, yy):
    sc = np.asarray(sc); pos = sc[yy == 1]; neg = sc[yy == 0]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    o = np.argsort(np.concatenate([pos, neg])); rk = np.empty_like(o, float); rk[o] = np.arange(1, len(o) + 1)
    return (rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
def spr(a, b): return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])
def zt(x): return np.argsort(np.argsort(x)).astype(float) / len(x)

for k in sorted(dispk):
    d = np.array(dispk[k]); print(f"displaced k={k}: AUC={auc(d[sel], y):.3f} rhoFrac={spr(d, WF):+.3f}")
K = 3.0; D = np.array(dispk[K])
zc, zd = zt(chi2), zt(D)
combo_max = np.maximum(zc, zd); combo_mean = 0.5 * (zc + zd)
print(f"\nchi2_flash        AUC={auc(chi2[sel], y):.3f} rhoFrac={spr(chi2, WF):+.3f}")
print(f"displaced k={K}    AUC={auc(D[sel], y):.3f} rhoFrac={spr(D, WF):+.3f}")
print(f"combo max(z)      AUC={auc(combo_max[sel], y):.3f} rhoFrac={spr(combo_max, WF):+.3f}")
print(f"combo mean(z)     AUC={auc(combo_mean[sel], y):.3f} rhoFrac={spr(combo_mean, WF):+.3f}")

# held-out EVENT split validation (z-scores recomputed WITHIN each split -> honest)
ukeys = sorted(set(keys)); halfA = set(ukeys[::2])
inA = np.array([kk in halfA for kk in keys])
print("\nEVENT-SPLIT validation (z recomputed within split):")
for grp, mask in [("splitA", inA), ("splitB", ~inA)]:
    zc2 = np.argsort(np.argsort(chi2[mask])).astype(float) / mask.sum()
    zd2 = np.argsort(np.argsort(D[mask])).astype(float) / mask.sum()
    cm = np.maximum(zc2, zd2); mn = 0.5 * (zc2 + zd2)
    s = sel[mask]; yy = bad[mask][s].astype(int)
    print(f"  {grp} (BAD={int(yy.sum())} GOOD={int((yy==0).sum())}): "
          f"chi2={auc(chi2[mask][s], yy):.3f} disp={auc(D[mask][s], yy):.3f} "
          f"max={auc(cm[s], yy):.3f} mean={auc(mn[s], yy):.3f}")

json.dump({"chi2": list(map(float, chi2)), "disp": list(map(float, D)),
           "combo_max": list(map(float, combo_max)), "combo_mean": list(map(float, combo_mean)),
           "wf": list(map(float, WF)), "E": list(map(float, EE))},
          open(os.path.join(HERE, "eval_final_out.json"), "w"))
print("\nsaved eval_final_out.json"); print("DONE")
