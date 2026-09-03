"""Head-to-head: traditional (anchored-covariance swap-aware) confidence vs
quadratic-fit confidence, per cluster — which is stronger, and do they tag
DIFFERENT bad clusters?

traditional = z_min / z_own / dz / chi2_loc / n_tpc  (confidence_cluster.py:
  covariance row anchored at the placed tick — z_min = "this cluster's light
  appears elsewhere / a swap partner exists on the same channels")
quadratic  = sig_t / fit_status / chi2pd / curve contrast of the cluster's
  t0 group (the v0.9 stage-6 fit evidence)

Both scored by LOO-by-event logistic (pure blocks - no logE in either, so we
compare information, not the energy prior), isotonic-calibrated for the
plots. Complementarity: at matched flag rates, which truth-wrong clusters
does each catch?

Usage: python subtick_conf_compare.py v125CD_out
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v083_prune as vp
from subtick_conf_study2 import load, loo, auc_rank

TRAD = [vp.ALL_FEATS.index(k) for k in ("z_min", "z_own", "dz", "chi2_loc", "n_tpc")]
QUADG = [1, 2, 3, 4]     # g_sig_t, g_fit_status, g_chi2pd, g_contrast
C_OK, C_BAD, C_REF = "#2471a3", "#c0392b", "#999999"


def calib_panel(ax, s, y, eg, title):
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(s, y)
    p = iso.predict(s)
    ax.scatter(p, 1 - y + np.random.uniform(-0.02, 0.02, len(y)), s=6 + 2 * np.sqrt(eg),
               alpha=0.12, color=C_OK, lw=0)
    bins = np.linspace(0, 1.0001, 11)
    bx, by, be = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() < 3:
            continue
        w = eg[m]
        q = float(np.average(1 - y[m], weights=w))
        bx.append(0.5 * (lo + hi)); by.append(q)
        be.append(np.sqrt(max(q * (1 - q), 1e-4) / m.sum()))
    ax.errorbar(bx, by, yerr=be, color=C_BAD, lw=2, marker="o", ms=5, capsize=3,
                label="E-weighted wrong-frac", zorder=5)
    xx = np.linspace(0, 1, 20)
    ax.plot(xx, 1 - xx, "--", color=C_REF, lw=1.2, label="perfect (1 - conf)")
    ax.set_xlabel("calibrated confidence"); ax.set_ylabel("wrong fraction (MC truth)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.06, 1.06)
    return p


def main():
    X, G, y, eg, ev = load(sys.argv[1])
    wrong = y == 0
    print(f"{len(y)} clusters, {wrong.sum()} wrong ({100*eg[wrong].sum()/eg.sum():.2f}% E)")
    s_tr = loo("logit", X[:, TRAD], y, ev)
    s_qd = loo("logit", G[:, QUADG], y, ev)
    for name, s in [("traditional (swap-aware)", s_tr), ("quadratic-fit", s_qd)]:
        line = f"{name:26s} AUC {auc_rank(y, s):.3f} (E-wt {auc_rank(y, s, eg):.3f})"
        for lo, hi, lab in [(5, 20, "5-20"), (20, 100, "20-100"), (100, 1e9, ">=100")]:
            m = (eg >= lo) & (eg < hi)
            if m.sum() > 10 and len(set(y[m])) == 2:
                line += f"  {lab}:{auc_rank(y[m], s[m]):.3f}"
        print(line)

    # complementarity at matched flag rate (flag the lowest-score 10% each)
    q = 0.10
    fl_tr = s_tr <= np.quantile(s_tr, q)
    fl_qd = s_qd <= np.quantile(s_qd, q)
    cw_tr, cw_qd = fl_tr & wrong, fl_qd & wrong
    both = (cw_tr & cw_qd).sum()
    only_tr = (cw_tr & ~cw_qd).sum()
    only_qd = (cw_qd & ~cw_tr).sum()
    missed = (wrong & ~fl_tr & ~fl_qd).sum()
    print(f"\nflagging the lowest {int(q*100)}% by each metric "
          f"({fl_tr.sum()} / {fl_qd.sum()} clusters):")
    print(f"  wrong clusters caught by BOTH        : {both}")
    print(f"  caught ONLY by traditional           : {only_tr}")
    print(f"  caught ONLY by quadratic             : {only_qd}")
    print(f"  wrong clusters missed by both        : {missed}  (of {wrong.sum()})")
    r = np.corrcoef(s_tr, s_qd)[0, 1]
    rw = np.corrcoef(s_tr[wrong], s_qd[wrong])[0, 1] if wrong.sum() > 2 else np.nan
    print(f"  score correlation: all {r:.2f}, wrong-only {rw:.2f}")

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
    p_tr = calib_panel(axes[0], s_tr, y, eg,
                       "TRADITIONAL: anchored-covariance swap-aware\n(z_min/z_own/dz/chi2_loc)")
    p_qd = calib_panel(axes[1], s_qd, y, eg,
                       "QUADRATIC-FIT only\n(contrast/chi2pd/sig_t/status of the t0 group)")
    ax = axes[2]
    ax.scatter(p_tr[~wrong], p_qd[~wrong], s=5, alpha=0.10, color=C_OK, lw=0,
               label=f"correct ({int((~wrong).sum())})")
    ax.scatter(p_tr[wrong], p_qd[wrong], s=26, alpha=0.85, color=C_BAD, lw=0,
               marker="x", label=f"WRONG ({int(wrong.sum())})")
    ax.plot([0, 1], [0, 1], "--", color=C_REF, lw=1)
    ax.set_xlabel("traditional confidence"); ax.set_ylabel("quadratic confidence")
    ax.set_title(f"complementarity  (caught only-trad {only_tr} / only-quad {only_qd} / both {both})",
                 fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("which evidence tags the bad clusters? file 0000001, LOO scores, cluster level")
    fig.tight_layout()
    out = "subtick_out/v093_conf_compare.png"
    fig.savefig(out, dpi=140)
    print("FIGURE:", out)


if __name__ == "__main__":
    main()
