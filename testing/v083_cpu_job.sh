#!/bin/bash
HERE=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing
cd $HERE
PY=/global/common/software/nersc9/pytorch/2.8.0/bin/python
$PY -u v083_transfer.py
echo "=== importance ==="
$PY -u - <<'PYEOF'
import numpy as np, sys
sys.path.insert(0, "/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing")
from v083_prune import ALL_FEATS
z = np.load("v083_prune_rows.npz", allow_pickle=True)
X, y, fil = z["X"], z["y"], z["fil"]
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
tr = fil != "0000001"
clf = HistGradientBoostingClassifier(max_iter=200, max_depth=6, learning_rate=0.08)
clf.fit(X[tr], y[tr])
sub = np.flatnonzero(~tr)
if len(sub) > 8000:
    sub = np.random.default_rng(0).choice(sub, 8000, replace=False)
r = permutation_importance(clf, X[sub], y[sub], n_repeats=5, random_state=0,
                           scoring="roc_auc", n_jobs=16)
order = np.argsort(-r.importances_mean)
print("permutation importance (AUC drop, OOS file 1):")
for i in order[:15]:
    print(f"  {ALL_FEATS[i]:12s} {r.importances_mean[i]:+.4f} +- {r.importances_std[i]:.4f}")
PYEOF
echo V083_CPU_DONE
