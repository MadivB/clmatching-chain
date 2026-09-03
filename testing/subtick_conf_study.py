"""v0.9.2 confidence study: 0-to-1 per-group t0 confidence.

Combines the two evidence families per t0 group:
  quadratic-fit  : sig_t (curvature), fit status, chi2/dof at min, contrast
  waveform-match : mean |z| at the peak tick, frac |z|>2, cos(pred, residual)
  shared priors  : log E, n TPCs, log n hits
Label: group PURITY = E-frac of member hits with |t - truth| <= 10 ticks;
wrong group = purity < 0.5 ("impure t0").

Honest 1-file evaluation: leave-one-event-out CV logistic regression;
reports AUC (each family alone vs combined), energy-banded AUC, and a
reliability (calibration) table. Final model on all events is saved to
subtick_conf_model.json for the in-chain 0-1 output.

Usage: python subtick_conf_study.py v123CS_out/dumps
"""
import glob
import json
import sys

import numpy as np

NAMES = ["logE", "sig_t", "fit_status", "chi2pd", "contrast",
         "mean_absz", "frac_z2", "cos_res", "n_tpc", "log_nhits"]
QUAD = [1, 2, 3, 4]
WAVE = [5, 6, 7]
SHARED = [0, 8, 9]


def load(dumps):
    X, y, eg, ev = [], [], [], []
    for i, f in enumerate(sorted(glob.glob(f"{dumps}/*.npz"))):
        d = np.load(f)
        cf, cm = d["conf_feat"], d["conf_meta"]
        if not cf.size:
            continue
        tot = cm[:, 2] + cm[:, 3]
        m = tot > 0
        X.append(cf[m])
        y.append((cm[m, 2] / tot[m]) >= 0.5)
        eg.append(cm[m, 1])
        ev.append(np.full(int(m.sum()), i))
    return (np.concatenate(X), np.concatenate(y).astype(int),
            np.concatenate(eg), np.concatenate(ev))


def logit_fit(X, y, l2=1.0, iters=500, lr=0.5):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    w = np.zeros(Z.shape[1])
    b = float(np.log(max(y.mean(), 1e-6) / max(1 - y.mean(), 1e-6)))
    n = len(y)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(Z @ w + b)))
        gw = Z.T @ (p - y) / n + l2 * w / n
        gb = float(np.mean(p - y))
        w -= lr * gw
        b -= lr * gb
    return {"w": w, "b": b, "mu": mu, "sd": sd}


def logit_score(mdl, X):
    Z = (X - mdl["mu"]) / mdl["sd"]
    return 1.0 / (1.0 + np.exp(-(Z @ mdl["w"] + mdl["b"])))


def auc(y, s, w=None):
    w = np.ones_like(s) if w is None else w
    order = np.argsort(s, kind="mergesort")
    y2, w2 = y[order], w[order]
    r = np.cumsum(w2) - w2 / 2.0
    wp, wn = w2[y2 == 1].sum(), w2[y2 == 0].sum()
    if wp == 0 or wn == 0:
        return np.nan
    return float((r[y2 == 1] * w2[y2 == 1]).sum() / w2[y2 == 1].sum()
                 - (r[y2 == 0] * w2[y2 == 0]).sum() / w2[y2 == 0].sum()) / (r[-1] + w2[-1] / 2) + 0.5


def wilcox_auc(y, s, w):
    # weighted AUC via rank statistic (midranks)
    from scipy.stats import rankdata
    r = rankdata(s)
    wp = w[y == 1].sum(); wn = w[y == 0].sum()
    if wp == 0 or wn == 0:
        return np.nan
    # unweighted-rank, energy-weighted average of ranks
    rp = np.average(r[y == 1], weights=w[y == 1])
    rn = np.average(r[y == 0], weights=w[y == 0])
    n = len(y)
    return float(0.5 + (rp - rn) / (2.0 * n) * 2.0 / 1.0 / 1.0) if n else np.nan


def cv_scores(X, y, ev, cols):
    s = np.zeros(len(y))
    for e in np.unique(ev):
        tr = ev != e
        mdl = logit_fit(X[tr][:, cols], y[tr])
        s[ev == e] = logit_score(mdl, X[ev == e][:, cols])
    return s


def main():
    X, y, eg, ev = load(sys.argv[1])
    print(f"{len(y)} groups, wrong (purity<0.5): {(y==0).sum()} "
          f"({100*(y==0).mean():.1f}% by count, "
          f"{100*eg[y==0].sum()/eg.sum():.1f}% by energy)")
    from scipy.stats import rankdata
    arms = [("quadratic-only", QUAD + SHARED),
            ("waveform-only", WAVE + SHARED),
            ("COMBINED", QUAD + WAVE + SHARED)]
    s_comb = None
    for name, cols in arms:
        s = cv_scores(X, y, ev, cols)
        r = rankdata(s)
        a = auc(y, r)
        print(f"{name:16s} LOO-AUC {a:.3f}", end="")
        for lo, hi, lab in [(5, 20, "5-20"), (20, 100, "20-100"), (100, 1e9, ">=100 MeV")]:
            m = (eg >= lo) & (eg < hi)
            if m.sum() > 10 and len(set(y[m])) == 2:
                print(f"  {lab}: {auc(y[m], rankdata(s[m])):.3f}", end="")
        print()
        if name == "COMBINED":
            s_comb = s
    # reliability of the combined 0-1 score
    print("\nreliability (combined score -> measured correct fraction, E-weighted):")
    bins = [0, .2, .4, .6, .8, .9, .95, 1.001]
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (s_comb >= lo) & (s_comb < hi)
        if m.sum():
            fc = eg[m & (y == 1)].sum() / eg[m].sum()
            print(f"  [{lo:.2f},{hi:.2f}): n={m.sum():5d}  E-frac correct {fc*100:5.1f}%")
    # final model on all events
    mdl = logit_fit(X[:, QUAD + WAVE + SHARED], y)
    out = {"cols": QUAD + WAVE + SHARED,
           "names": [NAMES[i] for i in QUAD + WAVE + SHARED],
           "w": mdl["w"].tolist(), "b": mdl["b"],
           "mu": mdl["mu"].tolist(), "sd": mdl["sd"].tolist()}
    json.dump(out, open("subtick_conf_model.json", "w"), indent=1)
    imp = sorted(zip(out["names"], np.abs(mdl["w"])), key=lambda t: -t[1])
    print("\n|coef| ranking:", ", ".join(f"{n}={v:.2f}" for n, v in imp))
    print("model -> subtick_conf_model.json")


if __name__ == "__main__":
    main()
