"""Variance-aware confidence A/B (user idea): does adding the ML light
model's PREDICTED std to the residual normalization (quadrature with the
amplitude/Poisson term) sharpen the pruning decision?

Paired comparison on the v091STD dumps (same events, same clusters):
  plain:  var = GAIN * max(A, 100)
  +std:   var = GAIN * max(A, 100) + std_pred^2
Reports per-band AUCs of dz / z_own / z_min and the fixed current rule
(E<2, d_eu>20, dz>0.25, z_own>-2, all-3) under both variants.

Usage: python v083_std_ab.py [dump_dir=v091STD_out/dumps]
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from confidence_cluster import cluster_scores
from v083_prune import ALL_FEATS, cluster_rows
from sptl_v081 import auc_midrank

IX = {k: ALL_FEATS.index(k) for k in ("logE", "d_eu", "dz", "z_min", "z_own")}


def _proc(fp):
    d = np.load(fp, allow_pickle=True)
    ev = {k: d[k] for k in ("x", "y", "z", "e", "tpc", "labels",
                            "ts_final", "truth_t0")}
    actual = d["full_light"].astype(np.float64)
    base = d["base_image"].astype(np.float64)
    # dumped field is VARIANCE (sigma^2, return_variance=True) -> take sqrt
    std = (np.sqrt(d["full_light_std"].astype(np.float64))
           if "full_light_std" in d.files else None)
    args = (ev["labels"], ev["tpc"], ev["y"].astype(np.float64),
            np.clip(ev["e"].astype(np.float64), 0, None),
            ev["ts_final"].astype(np.float64), actual, base)
    conf_p = cluster_scores(*args)
    conf_s = cluster_scores(*args, std=std) if std is not None else None
    rp = cluster_rows(ev, conf_p)
    rs = cluster_rows(ev, conf_s) if conf_s is not None else None
    return rp, rs


def main():
    import multiprocessing as mp
    dump_dir = sys.argv[1] if len(sys.argv) > 1 else "v091STD_out/dumps"
    files = sorted(glob.glob(os.path.join(dump_dir, "*.npz")))
    with mp.Pool(min(13, len(files))) as pool:
        parts = pool.map(_proc, files)
    if parts and parts[0][1] is None:
        print("NO full_light_std in dumps — rerun the dump job first")
        return
    RP = sum((p[0] for p in parts), [])
    RS = sum((p[1] for p in parts), [])
    assert len(RP) == len(RS)
    Xp = np.array([r["x"] for r in RP], float)
    Xs = np.array([r["x"] for r in RS], float)
    y = np.array([r["wrong"] for r in RP], float)
    e = np.array([r["e"] for r in RP])
    nh = np.array([r["nh"] for r in RP], float)
    ns = np.array([r["ns"] for r in RP])
    E = 10 ** Xp[:, IX["logE"]]
    print(f"{len(RP)} paired clusters, wrong rate {y.mean():.3f}")
    for lo, hi in ((0.5, 2), (2, 5), (5, 20), (20, 100)):
        m = (E >= lo) & (E < hi)
        line = [f"E {lo:>4}-{hi:<4}:"]
        for k in ("dz", "z_own", "z_min"):
            vp, vs = Xp[m, IX[k]], Xs[m, IX[k]]
            fp_, fs_ = np.isfinite(vp), np.isfinite(vs)
            ap = auc_midrank(y[m][fp_], vp[fp_]) if fp_.sum() > 100 else np.nan
            as_ = auc_midrank(y[m][fs_], vs[fs_]) if fs_.sum() > 100 else np.nan
            line.append(f"{k}: {ap:.3f}->{as_:.3f}")
        print("  " + "  ".join(line))

    def rule(X):
        with np.errstate(invalid="ignore"):
            return ((E < 2.0) & (X[:, IX["d_eu"]] > 20)
                    & (X[:, IX["dz"]] > 0.25) & (X[:, IX["z_own"]] > -2))

    print(f"\ncurrent rule (E<2, d_eu>20, dz>0.25, z_own>-2, ALL):")
    for name, X in (("plain", Xp), ("+std", Xs)):
        pr = rule(X)
        pur = (e[pr] * y[pr]).sum() / max(e[pr].sum(), 1e-9)
        sel = ns
        keep = sel & ~pr
        a0 = 1 - (e[sel] * y[sel]).sum() / max(e[sel].sum(), 1e-9)
        a1 = 1 - (e[keep] * y[keep]).sum() / max(e[keep].sum(), 1e-9)
        print(f"  {name:6s}: pruned {pr.sum():4d} cl, {nh[pr].sum()/nh.sum():.2%} hits, "
              f"purity {pur:.0%}, NS acc {100*a0:.2f}->{100*a1:.2f}")
    # grid-retune dz/z_own thresholds per variant (2 free thresholds)
    for name, X in (("plain", Xp), ("+std", Xs)):
        best = None
        for T_dz in (0.1, 0.25, 0.5, 1, 2):
            for T_z in (-3, -2, -1, 0, 1):
                with np.errstate(invalid="ignore"):
                    pr = ((E < 2.0) & (X[:, IX["d_eu"]] > 20)
                          & (X[:, IX["dz"]] > T_dz) & (X[:, IX["z_own"]] > T_z))
                we = (e[pr] * y[pr]).sum()
                obj = we - (e[pr].sum() - we)
                if pr.sum() and (best is None or obj > best[0]):
                    best = (obj, T_dz, T_z, pr)
        _, T_dz, T_z, pr = best
        pur = (e[pr] * y[pr]).sum() / max(e[pr].sum(), 1e-9)
        keep = ns & ~pr
        a0 = 1 - (e[ns] * y[ns]).sum() / max(e[ns].sum(), 1e-9)
        a1 = 1 - (e[keep] * y[keep]).sum() / max(e[keep].sum(), 1e-9)
        print(f"  retuned {name:6s} (dz>{T_dz}, z_own>{T_z}): pruned {pr.sum():4d}, "
              f"{nh[pr].sum()/nh.sum():.2%} hits, purity {pur:.0%}, "
              f"NS {100*a0:.2f}->{100*a1:.2f}")


if __name__ == "__main__":
    main()
