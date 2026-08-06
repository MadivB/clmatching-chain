"""Confidence metric vFable: residual-whiteness + displaced-flash detector, Poisson-
calibrated by parametric bootstrap.  Operates on the per-flash per-channel light in
lightcache.pkl (Anscombe residuals), grades every layer against MC-truth mis-class E.

Layers (per TPC):
  1. Anscombe residual  R = 2(sqrt(P_pe+3/8) - sqrt(A_pe+3/8))   [~N(0,1) under Poisson null]
  2. chi2_red = mean(R^2)                                         [overall goodness of fit]
  3. multiscale whiteness  max_k chi2_red(k)  over channel-blocks [spatial coherence]
  4. Z_mis = min_{i!=j} M(i,j)*sqrt(Nch),  M=R R^T/Nch            [displaced flash: deficit@i, excess@j]
     + y-windowed variant (worst sliding channel band)
  5. parametric bootstrap: A* ~ Poisson(P_pe) -> null of each stat -> calibrated z per TPC
"""
import os, pickle, json, numpy as np

HERE = os.path.abspath(os.path.dirname(__file__))
GAIN = 200.0
BLOCKS = [1, 2, 4, 8, 12, 24]          # channel-block sizes (divide 120)
YWIN = 30                               # sliding y-window width (channels)
B = 160                                 # bootstrap draws
rng = np.random.default_rng(0)
cache = pickle.load(open(os.path.join(HERE, "lightcache.pkl"), "rb"))


def anscombe(x):
    return 2.0 * np.sqrt(np.maximum(x, 0.0) + 0.375)


def stats_from_R(R):
    """R: (...,nf,nch).  Returns chi2_red, maxk (multiscale), zmis (min off-diag*sqrtNch),
    zmis_y (worst y-window).  Works batched over a leading bootstrap axis if present."""
    nf, nch = R.shape[-2], R.shape[-1]
    chi2 = (R ** 2).mean(axis=(-2, -1))
    # multiscale over channels
    maxk = chi2.copy()
    for k in BLOCKS:
        if k == 1 or nch % k:
            continue
        Rb = R.reshape(*R.shape[:-1], nch // k, k).sum(-1) / np.sqrt(k)
        maxk = np.maximum(maxk, (Rb ** 2).mean(axis=(-2, -1)))
    # time-time covariance M = R R^T / nch  -> most-negative off-diagonal
    if nf >= 2:
        M = np.matmul(R, np.swapaxes(R, -1, -2)) / nch          # (...,nf,nf)
        eye = np.eye(nf, dtype=bool)
        Moff = np.where(eye, np.inf, M)
        zmis = Moff.min(axis=(-2, -1)) * np.sqrt(nch)
        # y-windowed: slide 30-ch bands, worst (most negative) off-diagonal
        zy = np.full(chi2.shape, np.inf)
        for s in range(0, nch - YWIN + 1, YWIN // 2):
            Rw = R[..., s:s + YWIN]
            Mw = np.matmul(Rw, np.swapaxes(Rw, -1, -2)) / YWIN
            mw = np.where(eye, np.inf, Mw).min(axis=(-2, -1)) * np.sqrt(YWIN)
            zy = np.minimum(zy, mw)
    else:
        zmis = np.zeros(chi2.shape); zy = np.zeros(chi2.shape)
    return chi2, maxk, zmis, zy


rows = []
for r in cache:
    Fa = r["Fa"].astype(np.float64); Fp = r["Fp"].astype(np.float64)
    if len(Fa) < 1:
        rows.append(None); continue
    A_pe = Fa / GAIN
    P_pe = np.maximum(r["alpha"] * Fp, 0.0) / GAIN                # alpha-corrected predicted PE
    R = anscombe(P_pe) - anscombe(A_pe)                          # (nf,nch)
    c0, m0, z0, zy0 = stats_from_R(R)
    # parametric bootstrap null: A* ~ Poisson(P_pe)
    Astar = rng.poisson(np.broadcast_to(P_pe, (B,) + P_pe.shape)).astype(np.float64)
    Rstar = anscombe(P_pe)[None] - anscombe(Astar)              # (B,nf,nch)
    cB, mB, zB, zyB = stats_from_R(Rstar)
    def zc(x, xb):     # calibrated z (bootstrap mean/std)
        s = xb.std(); return float((x - xb.mean()) / s) if s > 0 else 0.0
    rows.append(dict(
        chi2=float(c0), maxk=float(m0), zmis=float(z0), zmis_y=float(zy0),
        z_chi2=zc(c0, cB), z_maxk=zc(m0, mB),
        z_zmis=-zc(z0, zB),         # more-negative-than-null -> positive badness
        z_zmis_y=-zc(zy0, zyB),
        wf=r["wf"], E=r["E"], nflash=len(Fa)))

# align with truth
WF = np.array([r["wf"] for r in cache]); EE = np.array([r["E"] for r in cache])
bad = WF > 0.20; good = WF < 0.05; sel = bad | good; y = bad[sel].astype(int)
ef = json.load(open(os.path.join(HERE, "eval_final_out.json")))
v1_chi2 = np.array(ef["chi2"]); v1_disp = np.array(ef["disp"])


def rank(x): return np.argsort(np.argsort(np.asarray(x, float))).astype(float) / len(x)
def auc(sc, yy):
    sc = np.asarray(sc); pos = sc[yy == 1]; neg = sc[yy == 0]
    o = np.argsort(np.concatenate([pos, neg])); rk = np.empty_like(o, float); rk[o] = np.arange(1, len(o) + 1)
    return (rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
def spr(a, b): return float(np.corrcoef(rank(a), rank(b))[0, 1])


def col(name):
    return np.array([(rw[name] if rw is not None else 0.0) for rw in rows])

print(f"{len(rows)} TPCs  BAD={int(bad.sum())} GOOD={int(good.sum())}  B={B} bootstrap")
print(f"(v1 ref: chi2_flash AUC0.903, displaced 0.919, combo 0.935/rhoFrac0.599)\n")
print(f"{'vFable layer':>16} {'AUC':>7} {'rhoFrac':>9}")
for nm in ["chi2", "maxk", "zmis", "zmis_y", "z_chi2", "z_maxk", "z_zmis", "z_zmis_y"]:
    v = col(nm)
    print(f"{nm:>16} {auc(v[sel], y):>7.3f} {spr(v, WF):>+9.3f}")

# Fisher-combine the three calibrated z's (as one-sided gaussian p-values)
from math import erfc, sqrt, log
def sf(z): return 0.5 * erfc(z / sqrt(2))
zc_, zm_, zz_ = col("z_chi2"), col("z_maxk"), col("z_zmis")
fisher = np.array([-2 * sum(log(max(sf(z), 1e-12)) for z in trip) for trip in zip(zc_, zm_, zz_)])
print(f"\n{'combo':>22} {'AUC':>7} {'rhoFrac':>9}")
print(f"{'Fisher(chi2,maxk,zmis)':>22} {auc(fisher[sel], y):>7.3f} {spr(fisher, WF):>+9.3f}")
zmax = np.maximum.reduce([rank(zc_), rank(zm_), rank(zz_)])
print(f"{'max-z(3)':>22} {auc(zmax[sel], y):>7.3f} {spr(zmax, WF):>+9.3f}")
# vFable's displaced detector alone (z_zmis) vs v1 displaced
print(f"\nvFable z_zmis vs v1 displaced (both displaced-flash detectors):")
print(f"   z_zmis   AUC={auc(col('z_zmis')[sel], y):.3f} rhoFrac={spr(col('z_zmis'), WF):+.3f}")
print(f"   v1 disp  AUC={auc(v1_disp[sel], y):.3f} rhoFrac={spr(v1_disp, WF):+.3f}")
# combine vFable with v1
for nm, comb in [("v1combo+z_zmis", rank(v1_chi2) + rank(v1_disp) + rank(col('z_zmis'))),
                 ("v1combo+Fisher", rank(v1_chi2) + rank(v1_disp) + rank(fisher)),
                 ("all5 mean-rank", rank(v1_chi2) + rank(v1_disp) + rank(zc_) + rank(zm_) + rank(zz_))]:
    print(f"   {nm:>16} AUC={auc(comb[sel], y):.3f} rhoFrac={spr(comb, WF):+.3f}")

json.dump({nm: list(map(float, col(nm))) for nm in ["chi2", "maxk", "zmis", "zmis_y", "z_chi2", "z_maxk", "z_zmis", "z_zmis_y"]}
          | {"fisher": list(map(float, fisher)), "wf": list(map(float, WF)), "E": list(map(float, EE))},
          open(os.path.join(HERE, "confidence_vfable_out.json"), "w"))
print("DONE")
