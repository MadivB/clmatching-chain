"""Grade vGPT (Q_noise/Q_patch/Q_dipole/Q_shift), the calibrated Q_failure (Sec 10),
and combinations with v1/v2.  All off saved jsons (no recompute)."""
import os, json, numpy as np
from scipy.special import ndtri   # inverse standard-normal CDF (Phi^-1)

HERE = os.path.abspath(os.path.dirname(__file__))
G = json.load(open(os.path.join(HERE, "vgpt_out.json")))
V1 = json.load(open(os.path.join(HERE, "eval_final_out.json")))
VF = json.load(open(os.path.join(HERE, "confidence_vfable_out.json")))
FR = json.load(open(os.path.join(HERE, "eval_fullres_out.json")))
WF = np.array(G["wf"]); EE = np.array(G["E"])
bad = WF > 0.20; good = WF < 0.05; sel = bad | good; y = bad[sel].astype(int)
NF = np.array(FR["nflash"], float); NL = np.array(FR["nlit"], float); lE = np.log(np.maximum(EE, 1))


def rank(x): return np.argsort(np.argsort(np.asarray(x, float))).astype(float) / len(x)
def auc(sc, yy):
    sc = np.asarray(sc); pos = sc[yy == 1]; neg = sc[yy == 0]
    o = np.argsort(np.concatenate([pos, neg])); rk = np.empty_like(o, float); rk[o] = np.arange(1, len(o) + 1)
    return (rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
def spr(a, b): return float(np.corrcoef(rank(a), rank(b))[0, 1])
def presid(m):
    r = rank(m); X = np.c_[[np.ones_like(r), rank(NF), rank(lE), rank(NL)]].T
    return r - X @ np.linalg.lstsq(X, r, rcond=None)[0]

# calibrated Q_failure (Sec 10): z_k = Phi^-1(F_good(Q_k)); Q_fail = z_n^2+z_p^2+lam z_d^2
def zcal(Q):
    ref = np.sort(Q[good]); F = np.clip(np.searchsorted(ref, Q, "right") / (len(ref) + 1), 1e-4, 1 - 1e-4)
    return ndtri(F)
zn, zp, zd, zs = (zcal(np.array(G[k])) for k in ["Qnoise", "Qpatch", "Qdipole", "Qshift"])

print(f"{len(WF)} TPCs BAD={int(bad.sum())} (v2 ref AUC0.935 rhoFrac0.654)")
print(f"{'metric':>16} {'AUC':>7} {'rhoFrac':>9}")
for k in ["Qnoise", "Qpatch", "Qdipole", "Qshift"]:
    v = np.array(G[k]); print(f"{k:>16} {auc(v[sel], y):>7.3f} {spr(v, WF):>+9.3f}")
for lam in [1.0, 2.0, 4.0]:
    Qf = zn ** 2 + zp ** 2 + lam * zd ** 2
    print(f"{'Q_failure lam=%.0f' % lam:>16} {auc(Qf[sel], y):>7.3f} {spr(Qf, WF):>+9.3f}")

# combinations (complexity-normalized mean-rank)
P = {"chi2": np.array(V1["chi2"]), "disp": np.array(V1["disp"]), "maxk": np.array(VF["maxk"]),
     "Qnoise": np.array(G["Qnoise"]), "Qdipole": np.array(G["Qdipole"]), "Qshift": np.array(G["Qshift"])}
pr = {k: rank(presid(v)) for k, v in P.items()}
print(f"\n{'combo (complexity-norm)':>28} {'AUC':>7} {'rhoFrac':>9}")
for nm, ks in [("v2 = chi2+disp+maxk", ["chi2", "disp", "maxk"]),
               ("v2+Qdipole", ["chi2", "disp", "maxk", "Qdipole"]),
               ("v2+Qnoise", ["chi2", "disp", "maxk", "Qnoise"]),
               ("v2+Qshift", ["chi2", "disp", "maxk", "Qshift"]),
               ("disp+maxk+Qdipole", ["disp", "maxk", "Qdipole"]),
               ("chi2+disp+maxk+Qdipole+Qshift", ["chi2", "disp", "maxk", "Qdipole", "Qshift"])]:
    c = np.mean([pr[k] for k in ks], axis=0)
    print(f"{nm:>28} {auc(c[sel], y):>7.3f} {spr(c, WF):>+9.3f}")

# does vGPT's dipole catch BAD that v2 misses? complementarity
zc, zdsp, zmk = pr["chi2"], pr["disp"], pr["maxk"]; v2 = (zc + zdsp + zmk) / 3
zdip = pr["Qdipole"]
def thr_at(s, tpr): return np.quantile(s[sel][y == 1], 1 - tpr)
tv, td = thr_at(v2, 0.85), thr_at(zdip, 0.85)
cb = v2 >= tv; db = zdip >= td
both = int((bad & cb & db).sum()); v2only = int((bad & cb & ~db).sum()); donly = int((bad & ~cb & db).sum())
print(f"\nBAD@85%: v2&dipole both={both} v2-only={v2only} dipole-only={donly}")
print("DONE")
