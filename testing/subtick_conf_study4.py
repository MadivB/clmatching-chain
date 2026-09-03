"""v0.10.1 candidate: does the per-channel quadratic block (flatness +
location votes) add on top of the v0.10.0 winner?

Originally: v0.9.3 final combination: study2's winner (40 cluster feats + 10 group
feats, HGB) PLUS the scan-curve landscape block (deep-research features:
Delta-chi2-unit depth/margin, PDA null posterior, competitor count, valley
width, argmin displacement).

Needs: v126MG_out with dumps + full_dumps (amp rows) and
scan_margin_feats.npz (run scan_margin.py first).

Arms: winner-repeat / scan-only / winner+scan.  LOO by event, isotonic on
the winner, complementarity vs the missed-by-both population, saves
subtick_conf_model3.joblib.

Usage: python subtick_conf_study3.py v126MG_out
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subtick_conf_study2 import load, loo, auc_rank

SCAN8 = ["depth_n", "margin2_n", "rel_margin", "n_comp", "post",
         "post_null", "width", "d_place"]
from scan_margin import SCAN_NAMES as SCAN22   # 8 + 7 per-channel


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

    S8, S22 = block(SCAN8), block(SCAN22)
    print(f"scan features present for {int(S22[:, -1].sum())}/{len(y)} clusters")
    AG = np.hstack([X, G])
    arms = [("winner+scan8 (v0.10.0)", np.hstack([AG, S8])),
            ("scan22 only", S22),
            ("winner + scan22", np.hstack([AG, S22]))]
    best = (None, -1, None, None, None)
    for name, F in arms:
        for kind in ("logit", "hgb"):
            s = loo(kind, F, y, ev)
            a, aw = auc_rank(y, s), auc_rank(y, s, eg)
            line = f"{name:18s} {kind:5s} AUC {a:.3f} (E-wt {aw:.3f})"
            for lo, hi, lab in [(5, 20, "5-20"), (20, 100, "20-100"), (100, 1e9, ">=100")]:
                m = (eg >= lo) & (eg < hi)
                if m.sum() > 10 and len(set(y[m])) == 2:
                    line += f"  {lab}:{auc_rank(y[m], s[m]):.3f}"
            print(line, flush=True)
            if a > best[1]:
                best = (name, a, kind, F, s)
    name, a, kind, F, s = best
    print(f"\nWINNER: {name} / {kind} (AUC {a:.3f})")
    # does the PC block recover clusters the v0.10.0 winner missed?
    s_ag = loo("hgb", np.hstack([AG, S8]), y, ev)
    s_sc = loo("hgb", S22, y, ev)
    q = 0.10
    fl_ag = s_ag <= np.quantile(s_ag, q)
    fl_sc = s_sc <= np.quantile(s_sc, q)
    print(f"at 10% flag: winner catches {int((fl_ag & wrong).sum())}, "
          f"scan-only catches {int((fl_sc & wrong).sum())}, "
          f"scan-only NEW (not winner-flagged) {int((fl_sc & ~fl_ag & wrong).sum())}")
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
    from subtick_conf_study2 import fit_model
    joblib.dump({"kind": kind, "arm": name, "model": fit_model(kind, F, y),
                 "iso": iso, "auc_loo": a, "scan_feats": (SCAN22 if F.shape[1] > 60 else SCAN8)},
                "subtick_conf_model4.joblib")
    np.savez("subtick_conf_oof4.npz", s=s, p=p, y=y, eg=eg, ev=ev,
             s_ag=s_ag, s_sc=s_sc)
    print("saved subtick_conf_model4.joblib + subtick_conf_oof4.npz")


if __name__ == "__main__":
    main()
