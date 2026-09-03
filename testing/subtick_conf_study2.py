"""v0.9.3 confidence study: cluster-level, three evidence families, to optimal.

Per assigned CLUSTER (the object the user reasons about):
  spatial block   : the 29 v0.8.3 orphan-ness features (d_eu, best_lat, ...)
  light-conf block: z_min / z_own / dz / chi2_loc / n_tpc (+ logE basics)
  group block     : the 10 v0.9.2 quadratic+waveform features of its t0 group
Label: cluster wrong = E-frac of member hits outside +-10 ticks > 0.5.

Arms x models (logistic, HGB), leave-one-event-out; isotonic on OOF scores
for calibration. Prints count- and E-weighted AUC, banded AUC; saves the
winning calibrated model.

Usage: python subtick_conf_study2.py v125CD_out
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v083_prune as vp

GNAMES = ["g_logE", "g_sig_t", "g_fit_status", "g_chi2pd", "g_contrast",
          "g_mean_absz", "g_frac_z2", "g_cos_res", "g_n_tpc", "g_log_nhits"]
SP = [i for i, n in enumerate(vp.ALL_FEATS) if n not in vp.CONF_FEATS]  # spatial+basic
LC = [i for i, n in enumerate(vp.ALL_FEATS) if n in vp.CONF_FEATS + ["logE", "lognh"]]


def load(outdir, with_keys=False):
    rows_X, rows_g, rows_y, rows_e, rows_ev = [], [], [], [], []
    rows_key = []
    subs = sorted(glob.glob(f"{outdir}/dumps/*.npz"))
    for iev, sf in enumerate(subs):
        tag = os.path.basename(sf)
        ff = f"{outdir}/full_dumps/{tag}"
        if not os.path.exists(ff):
            continue
        sd, fdmp = np.load(sf), np.load(ff, allow_pickle=True)
        n = sd["ts_after"].size
        if fdmp["e"].size != n:
            continue
        conf = json.loads(str(fdmp["cluster_conf"]))
        ev = {"x": fdmp["x"], "y": fdmp["y"], "z": fdmp["z"], "e": fdmp["e"],
              "tpc": sd["hit_tpc"], "labels": fdmp["labels"],
              "ts_final": sd["ts_after"].astype(np.float64),
              "truth_t0": sd["truth"].astype(np.float64)}
        crows = vp.cluster_rows(ev, conf)
        # group features lookup: gi -> feature row
        gf = {int(m[0]): f for f, m in zip(sd["conf_feat"], sd["conf_meta"])}
        gid, labs = sd["group_id"], fdmp["labels"]
        ts, tru, e = ev["ts_final"], ev["truth_t0"], np.clip(fdmp["e"], 0, None)
        for r in crows:
            l = r["label"]
            m = labs == l
            fin = m & np.isfinite(tru) & np.isfinite(ts) & (ts >= 0)
            etot = float(e[fin].sum())
            if etot <= 0:
                continue
            ew = float(e[fin & (np.abs(ts - tru) > 10.0)].sum())
            gis = gid[m & (gid >= 0)]
            grow = None
            if gis.size:
                vals, cnt = np.unique(gis, return_counts=True)
                grow = gf.get(int(vals[np.argmax(cnt)]))
            rows_X.append(np.asarray(r["x"], np.float64))
            rows_g.append(np.asarray(grow, np.float64) if grow is not None
                          else np.full(len(GNAMES), np.nan))
            rows_y.append(1.0 if ew / etot <= 0.5 else 0.0)   # 1 = correct
            rows_e.append(etot)
            rows_ev.append(iev)
            rows_key.append((tag, int(l)))
    X = np.array(rows_X)
    G = np.array(rows_g)
    base = (np.nan_to_num(X, nan=0.0), np.nan_to_num(G, nan=0.0),
            np.array(rows_y, int), np.array(rows_e), np.array(rows_ev, int))
    return base + (rows_key,) if with_keys else base


def auc_rank(y, s, w=None):
    """Exact (weighted) AUC: P(s_pos > s_neg) with pair weight w_p*w_n,
    ties counted half."""
    w = np.ones_like(s, dtype=float) if w is None else np.asarray(w, float)
    order = np.argsort(s, kind="mergesort")
    y2, w2, s2 = y[order], w[order], s[order]
    wp, wn = w2[y2 == 1].sum(), w2[y2 == 0].sum()
    if wp == 0 or wn == 0:
        return np.nan
    # group ties: within a tie group, pos-neg pairs count 0.5
    num = 0.0
    cum_n = 0.0
    i = 0
    n = len(s2)
    while i < n:
        j = i
        tp = tn = 0.0
        while j < n and s2[j] == s2[i]:
            if y2[j] == 1:
                tp += w2[j]
            else:
                tn += w2[j]
            j += 1
        num += tp * (cum_n + 0.5 * tn)
        cum_n += tn
        i = j
    return float(num / (wp * wn))


def fit_model(kind, X, y):
    if kind == "logit":
        mu, sd = X.mean(0), X.std(0) + 1e-9
        Z = (X - mu) / sd
        w = np.zeros(Z.shape[1])
        b = float(np.log(max(y.mean(), 1e-6) / max(1 - y.mean(), 1e-6)))
        for _ in range(600):
            p = 1.0 / (1.0 + np.exp(-(Z @ w + b)))
            w -= 0.5 * (Z.T @ (p - y) / len(y) + w / len(y))
            b -= 0.5 * float(np.mean(p - y))
        return ("logit", w, b, mu, sd)
    from sklearn.ensemble import HistGradientBoostingClassifier
    m = HistGradientBoostingClassifier(max_iter=250, max_depth=4,
                                       learning_rate=0.08, min_samples_leaf=20,
                                       l2_regularization=1.0, random_state=0)
    m.fit(X, y)
    return ("hgb", m)


def score_model(mdl, X):
    if mdl[0] == "logit":
        _, w, b, mu, sd = mdl
        return 1.0 / (1.0 + np.exp(-(((X - mu) / sd) @ w + b)))
    return mdl[1].predict_proba(X)[:, 1]


def loo(kind, X, y, ev):
    s = np.zeros(len(y))
    for e in np.unique(ev):
        tr = ev != e
        s[ev == e] = score_model(fit_model(kind, X[tr], y[tr]), X[ev == e])
    return s


def main():
    X, G, y, eg, ev = load(sys.argv[1])
    print(f"{len(y)} clusters, wrong: {(y==0).sum()} "
          f"({100*(y==0).mean():.1f}% count, {100*eg[y==0].sum()/eg.sum():.2f}% energy)")
    arms = [
        ("v092 group-only", G),
        ("light-conf block", X[:, LC]),
        ("spatial+basic", X[:, SP]),
        ("cluster 40f", X),
        ("ALL (40f+group)", np.hstack([X, G])),
    ]
    best = (None, -1, None, None)
    for name, F in arms:
        for kind in ("logit", "hgb"):
            s = loo(kind, F, y, ev)
            a = auc_rank(y, s)
            aw = auc_rank(y, s, eg)
            line = f"{name:18s} {kind:5s} AUC {a:.3f} (E-wt {aw:.3f})"
            for lo, hi, lab in [(5, 20, "5-20"), (20, 100, "20-100"), (100, 1e9, ">=100")]:
                m = (eg >= lo) & (eg < hi)
                if m.sum() > 10 and len(set(y[m])) == 2:
                    line += f"  {lab}:{auc_rank(y[m], s[m]):.3f}"
            print(line)
            if a > best[1]:
                best = (name, a, kind, F, s)
    name, a, kind, F, s = best
    print(f"\nWINNER: {name} / {kind} (AUC {a:.3f}) -> isotonic calibration")
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(s, y)
    p = iso.predict(s)
    print("reliability (calibrated OOF, E-weighted):")
    for lo, hi in [(0, .2), (.2, .5), (.5, .8), (.8, .95), (.95, 1.001)]:
        m = (p >= lo) & (p < hi)
        if m.sum():
            fc = eg[m & (y == 1)].sum() / eg[m].sum()
            print(f"  [{lo:.2f},{hi:.2f}): n={m.sum():5d}  E-frac correct {fc*100:5.1f}%")
    import joblib
    final = fit_model(kind, F, y)
    joblib.dump({"kind": kind, "arm": name, "model": final, "iso": iso,
                 "auc_loo": a}, "subtick_conf_model2.joblib")
    np.savez("subtick_conf_oof.npz", s=s, p=p, y=y, eg=eg, ev=ev)
    print("saved subtick_conf_model2.joblib + subtick_conf_oof.npz")


if __name__ == "__main__":
    main()
