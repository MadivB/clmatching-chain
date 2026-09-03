"""v0.10.1: full-stack test of the per-channel + per-cluster quadratic blocks.

Arms answer two questions:
  1. does scan28 beat scan8/scan22 in combination?
  2. can the per-CLUSTER quadratic evidence REPLACE the per-group block
     (user 2026-09-03: quadratic-informed confidence per cluster, not per
     family/group)?

Usage: python subtick_conf_study5.py v126MG_out
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subtick_conf_study2 import load, loo, auc_rank, fit_model
from scan_margin import SCAN_NAMES as SCAN28

SCAN8 = SCAN28[:8]


def main():
    outdir = sys.argv[1]
    X, G, y, eg, ev, keys = load(outdir, with_keys=True)
    wrong = y == 0
    print(f"{len(y)} clusters, {wrong.sum()} wrong "
          f"({100*eg[wrong].sum()/eg.sum():.2f}% E)")
    sm = np.load(os.path.join(outdir, "scan_margin_feats.npz"), allow_pickle=True)
    names = list(sm["names"])

    def block(feat_names):
        cols = [names.index(k) for k in feat_names]
        lut = {(t, int(l)): f for t, l, f in zip(sm["tags"], sm["lab"], sm["F"][:, cols])}
        S = np.zeros((len(y), len(feat_names) + 1))
        for i, k in enumerate(keys):
            f = lut.get(k)
            if f is not None:
                S[i, :-1] = np.nan_to_num(f, nan=0.0, posinf=1e9, neginf=-1e9)
                S[i, -1] = 1.0
        return S

    S8, S28 = block(SCAN8), block(SCAN28)
    print(f"scan present for {int(S28[:, -1].sum())}/{len(y)} clusters")
    AG = np.hstack([X, G])
    arms = [("v0.10.0 (AG+S8)", np.hstack([AG, S8])),
            ("S28 only", S28),
            ("X40+S28 (no group)", np.hstack([X, S28])),
            ("AG+S28 (full)", np.hstack([AG, S28]))]
    best = (None, -1, None, None, None)
    for name, F in arms:
        for kind in ("hgb",):
            s = loo(kind, F, y, ev)
            a, aw = auc_rank(y, s), auc_rank(y, s, eg)
            line = f"{name:20s} {kind:4s} AUC {a:.3f} (E-wt {aw:.3f})"
            for lo, hi, lab in [(5, 20, "5-20"), (20, 100, "20-100"), (100, 1e9, ">=100")]:
                m = (eg >= lo) & (eg < hi)
                if m.sum() > 10 and len(set(y[m])) == 2:
                    line += f"  {lab}:{auc_rank(y[m], s[m]):.3f}"
            print(line, flush=True)
            if a > best[1]:
                best = (name, a, kind, F, s)
    name, a, kind, F, s = best
    print(f"\nWINNER: {name} / {kind} (AUC {a:.3f})")
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(s, y)
    p = iso.predict(s)
    print("reliability (calibrated OOF, E-weighted):")
    for lo, hi in [(0, .2), (.2, .5), (.5, .8), (.8, .95), (.95, 1.001)]:
        m = (p >= lo) & (p < hi)
        if m.sum():
            fc = eg[m & (y == 1)].sum() / eg[m].sum()
            print(f"  [{lo:.2f},{hi:.2f}): n={m.sum():5d}  E-frac correct {fc*100:5.1f}%")
    import joblib
    scanlist = SCAN28 if F.shape[1] > 55 or name == "S28 only" else SCAN8
    joblib.dump({"kind": kind, "arm": name, "model": fit_model(kind, F, y),
                 "iso": iso, "auc_loo": a, "scan_feats": list(scanlist),
                 "no_group": name == "X40+S28 (no group)",
                 "scan_only": name == "S28 only"},
                "subtick_conf_model5.joblib")
    np.savez("subtick_conf_oof5.npz", s=s, p=p, y=y, eg=eg, ev=ev)
    print("saved subtick_conf_model5.joblib + subtick_conf_oof5.npz")


if __name__ == "__main__":
    main()
