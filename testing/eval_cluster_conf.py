"""Grade per-cluster confidence variants against MC truth (cluster_conf_cache.pkl).
Variants (all truth-free):
  z_min    : most-negative covariance-row entry  min_j' m(j0,j')   [misplacement evidence]
  z_maxabs : max |m(j0,j')|
  s_neg    : summed negative mass of the row
  z_own    : own-site excess (-sum r(j0); excess = over-explaining = suspicious)
  chi2_loc : local whiteness at own site
  combos   : rank-mean pairs
Targets: cluster WRONG (>=50% energy mis-assigned) vs OK; continuous wf; breakdowns by
n_tpc (multi-TPC normalization check) and by energy band.
"""
import os, pickle, numpy as np

HERE = os.path.abspath(os.path.dirname(__file__))
R = pickle.load(open(os.path.join(HERE, "cluster_conf_cache.pkl"), "rb"))
wf = np.array([r["wf"] for r in R]); E = np.array([r["E"] for r in R])
ntpc = np.array([r["n_tpc"] for r in R]); split = np.array([r["split"] for r in R])
bad = wf > 0.5; ok = wf < 0.05
print(f"{len(R)} clusters | WRONG(>=50%)={int(bad.sum())} OK(<5%)={int(ok.sum())} "
      f"| multi-TPC={int((ntpc>1).sum())} split-t0={int(split.sum())}")


def rank(x): return np.argsort(np.argsort(np.asarray(x, float))).astype(float) / len(x)
def auc(sc, yy):
    sc = np.asarray(sc); pos = sc[yy == 1]; neg = sc[yy == 0]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    o = np.argsort(np.concatenate([pos, neg])); rk = np.empty_like(o, float); rk[o] = np.arange(1, len(o) + 1)
    return (rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
def spr(a, b): return float(np.corrcoef(rank(a), rank(b))[0, 1])


# badness orientation: higher = worse
V = {"z_min": -np.array([r["z_min"] for r in R]),            # more negative = worse
     "z_maxabs": np.array([r["z_maxabs"] for r in R]),
     "s_neg": -np.array([r["s_neg"] for r in R]),
     "z_own_excess": -np.array([r["z_own"] for r in R]),     # excess (negative z_own) = worse
     "chi2_loc": np.array([r["chi2_loc"] for r in R])}
V["combo_min_own"] = rank(V["z_min"]) + rank(V["z_own_excess"])
V["combo_min_chi2"] = rank(V["z_min"]) + rank(V["chi2_loc"])
V["combo_all"] = rank(V["z_min"]) + rank(V["z_own_excess"]) + rank(V["chi2_loc"])

sel = bad | ok; y = bad[sel].astype(int)
print(f"\n{'variant':>16} {'AUC':>7} {'rhoWF':>8}")
for k, v in V.items():
    print(f"{k:>16} {auc(v[sel], y):>7.3f} {spr(v, wf):>+8.3f}")

best = max(V, key=lambda k: auc(V[k][sel], y))
print(f"\nBEST: {best}")
for nm, m in [("single-TPC", ntpc == 1), ("multi-TPC", ntpc > 1), ("split-t0", split)]:
    s = sel & m; yy = bad[s].astype(int)
    if yy.sum() >= 5:
        print(f"  {nm:>11}: AUC={auc(V[best][s], yy):.3f}  (WRONG={int(yy.sum())}, n={int(s.sum())})")
print("energy bands:")
for lo, hi in [(1, 5), (5, 20), (20, 100), (100, 1e9)]:
    s = sel & (E >= lo) & (E < hi); yy = bad[s].astype(int)
    if yy.sum() >= 5:
        print(f"  {lo:>5.0f}-{hi:<6.0f}: AUC={auc(V[best][s], yy):.3f}  (WRONG={int(yy.sum())}, n={int(s.sum())})")
print("DONE")
