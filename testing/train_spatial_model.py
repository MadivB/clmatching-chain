"""Train + persist the v4 spatial score for in-chain use (PASS 2 trading)."""
import glob, sys
import numpy as np
import joblib
from scipy.spatial import cKDTree
from spatial_score import TRUTH_GAP, cluster_shape, emedian
from spatial_score4 import FEATURES2, RNG, pair_features2
from spatial_trade import build_families, load_event, shapes_and_times, BIG_MEV

dumps = sys.argv[1] if len(sys.argv) > 1 else "v060dump_out/dumps"
n_train = int(sys.argv[2]) if len(sys.argv) > 2 else 40
files = [f for f in sorted(glob.glob(f"{dumps}/*.npz")) if "0000001_" not in f][:n_train]
X, y = [], []
for fp in files:
    ev = load_event(fp)
    shp, ttime, atime, cmask = shapes_and_times(ev)
    F = build_families(ev, shp, cmask, ttime, TRUTH_GAP)
    if not F:
        continue
    f_ids = sorted(F); f_mu = np.stack([F[g]["mu"] for g in f_ids])
    cands = [l for l in shp if l in ttime and float(ev["e"][cmask[l]].sum()) < BIG_MEV]
    if len(cands) > 90:
        cands = list(RNG.choice(cands, size=90, replace=False))
    for c in cands:
        for g in [f_ids[i] for i in np.argsort(np.linalg.norm(f_mu - shp[c]["mu"], axis=1))[:4]]:
            ft = pair_features2(shp[c], F[g])
            X.append([ft[k] for k in FEATURES2])
            y.append(1 if abs(ttime[c] - F[g]["t0"]) <= TRUTH_GAP else 0)
X, y = np.array(X, float), np.array(y, int)
from sklearn.ensemble import HistGradientBoostingClassifier
clf = HistGradientBoostingClassifier(max_iter=600, learning_rate=0.08, max_depth=6,
                                     early_stopping=True, random_state=0).fit(X, y)
joblib.dump({"model": clf, "features": FEATURES2,
             "train": f"{len(y)} pairs / {len(files)} events (files != 0000001)"},
            "spatial_score_v4.joblib")
print(f"saved spatial_score_v4.joblib ({len(y)} pairs, {y.mean()*100:.0f}% same)")
