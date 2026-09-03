"""Feature reduction to <= 20 (user 2026-09-03: 79 features is way too much).

1. Permutation importance computed on HELD-OUT events (per LOO fold, on the
   fold's own out-of-sample rows, E-weighted AUC drop), summed across folds.
2. LOO evaluation of top-k subsets (k = 10, 15, 20) + the full baseline.
3. Best <= 20 subset -> HGB + isotonic -> subtick_conf_model6.joblib with
   feat_cols/feat_names for serving.

Usage: python subtick_conf_study6.py v126MG_out
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v083_prune as vp
from subtick_conf_study2 import load, loo, auc_rank, fit_model, score_model, GNAMES
from scan_margin import SCAN_NAMES as SCAN28


def main():
    outdir = sys.argv[1]
    X, G, y, eg, ev, keys = load(outdir, with_keys=True)
    wrong = y == 0
    sm = np.load(os.path.join(outdir, "scan_margin_feats.npz"), allow_pickle=True)
    names_s = list(sm["names"])
    cols = [names_s.index(k) for k in SCAN28]
    lut = {(t, int(l)): f for t, l, f in zip(sm["tags"], sm["lab"], sm["F"][:, cols])}
    S = np.zeros((len(y), len(SCAN28) + 1))
    for i, k in enumerate(keys):
        f = lut.get(k)
        if f is not None:
            S[i, :-1] = np.nan_to_num(f, nan=0.0, posinf=1e9, neginf=-1e9)
            S[i, -1] = 1.0
    F = np.hstack([X, G, S])
    NAMES = list(vp.ALL_FEATS) + list(GNAMES) + list(SCAN28) + ["has_scan"]
    assert F.shape[1] == len(NAMES)
    print(f"{len(y)} clusters, {wrong.sum()} wrong; {F.shape[1]} features")

    # held-out permutation importance, summed over LOO folds
    rng = np.random.default_rng(11)
    imp = np.zeros(F.shape[1])
    for e in np.unique(ev):
        tr, te = ev != e, ev == e
        mdl = fit_model("hgb", F[tr], y[tr])
        s0 = score_model(mdl, F[te])
        a0 = auc_rank(y[te], s0, eg[te])
        if not np.isfinite(a0):
            continue
        Fe = F[te].copy()
        for j in range(F.shape[1]):
            saved = Fe[:, j].copy()
            drops = []
            for _ in range(2):
                Fe[:, j] = rng.permutation(saved)
                drops.append(a0 - auc_rank(y[te], score_model(mdl, Fe), eg[te]))
            Fe[:, j] = saved
            imp[j] += float(np.mean(drops))
        print(f"fold {e} done", flush=True)
    order = np.argsort(imp)[::-1]
    print("\ntop 25 by held-out permutation importance (E-wt AUC drop):")
    for j in order[:25]:
        print(f"  {NAMES[j]:18s} {imp[j]:+.4f}")

    def report(name, s):
        line = f"{name:22s} AUC {auc_rank(y, s):.3f} (E-wt {auc_rank(y, s, eg):.3f})"
        for lo, hi, lab in [(5, 20, "5-20"), (20, 100, "20-100"), (100, 1e9, ">=100")]:
            m = (eg >= lo) & (eg < hi)
            if m.sum() > 10 and len(set(y[m])) == 2:
                line += f"  {lab}:{auc_rank(y[m], s[m]):.3f}"
        print(line, flush=True)
        return auc_rank(y, s)

    s_full = loo("hgb", F, y, ev)
    report("FULL 79f (baseline)", s_full)
    best = (None, -1, None)
    for k in (10, 15, 20):
        idx = sorted(order[:k])
        s = loo("hgb", F[:, idx], y, ev)
        a = report(f"top-{k}", s)
        if k <= 20 and a > best[1]:
            best = (idx, a, s)
    idx, a, s = best
    print(f"\nCHOSEN: {len(idx)} features (AUC {a:.3f} vs full {auc_rank(y, s_full):.3f})")
    print("features:", [NAMES[j] for j in idx])
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
    joblib.dump({"kind": "hgb", "arm": f"top{len(idx)}",
                 "model": fit_model("hgb", F[:, idx], y), "iso": iso,
                 "auc_loo": a, "scan_feats": list(SCAN28),
                 "feat_cols": [int(j) for j in idx],
                 "feat_names": [NAMES[j] for j in idx]},
                "subtick_conf_model6.joblib")
    np.savez("subtick_conf_oof6.npz", s=s, p=p, y=y, eg=eg, ev=ev, imp=imp,
             names=NAMES)
    print("saved subtick_conf_model6.joblib + subtick_conf_oof6.npz")


if __name__ == "__main__":
    main()
