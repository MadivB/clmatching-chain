"""Per-CLUSTER charge-light confidence (v0.4.1 candidate output).

Score = evidence that a cluster's t0 assignment is wrong, from light alone:
  - per TPC (once): standardized windowed residual  rW (unit-variance cells)
  - per cluster part (label, TPC): support channels from the hit-y -> channel-row map;
    covariance row anchored at the PLACED tick over its support,
        m(j') = sum_{i in sup} rW_i(j0) * rW_i(j') / sqrt(N)
    -> z_min (most-negative anticorrelation = its light appears elsewhere),
       z_own (own-site deficit(+)/excess(-)), chi2_loc (local whiteness).
  - multi-TPC clusters: parts are stacked AFTER standardization, so the only
    normalization is the total cell count (same footing as single-TPC).
Recommended per-cluster confidence for consumers = rank-blend with the per-TPC v0.4
confidence: cluster-light evidence informs E >~ 20 MeV; TPC context covers the rest.

Performance: one O(n_ch x n_tick) pass per TPC + one small matvec per cluster part;
hits grouped by (label, TPC) with a single argsort (no per-label masks).
"""
from __future__ import annotations
import numpy as np

CLIP, GAIN, PEAK = 60780.0, 200.0, 105
FWIN, SEP, DIL = 4, 20, 3
YLO, YHI, NROW = -215.5, 81.7, 60


def standardized_residual(actual_tpc, base_tpc):
    """Unit-variance, saturation-zeroed, +-FWIN-windowed residual for ONE TPC.
    float32 throughout, no boolean-index copies (bandwidth-bound kernel)."""
    A = np.asarray(actual_tpc, np.float32)
    P = np.minimum(np.asarray(base_tpc, np.float32), np.float32(CLIP))
    ok = (A < np.float32(0.98 * CLIP)).astype(np.float32)   # 1 = usable cell
    Pm = P * ok
    alpha = float((Pm * A).sum() / max(float((Pm * P).sum()), 1e-9))
    r = (A - np.float32(alpha) * P)
    r *= ok / np.sqrt(np.float32(GAIN) * np.maximum(A, np.float32(100.0)))
    # centered +-FWIN window sum via cumsum: s(t) = cs[t+F] - cs[t-F-1] (edge-clamped)
    W = 2 * FWIN + 1
    cs = np.cumsum(r, axis=1, dtype=np.float32)
    csp = np.pad(cs, ((0, 0), (1, 0)))
    n = r.shape[1]; t = np.arange(n)
    lo = np.maximum(t - FWIN, 0); hi = np.minimum(t + FWIN, n - 1)
    return (csp[:, hi + 1] - csp[:, lo]) * np.float32(1.0 / np.sqrt(W))


def cluster_scores(labels, hit_tpc, hit_y, hit_e, ts, actual, base, min_hits=3, min_energy=1.0):
    """Per-cluster confidence components for one event.

    labels/hit_tpc/hit_y/hit_e/ts : per-hit arrays (ts = assigned t0)
    actual/base                    : (n_light, n_ch, n_tick) light images
    Returns dict label -> {z_min, z_own, chi2_loc, E, n_tpc}.
    """
    labels = np.asarray(labels); hit_tpc = np.asarray(hit_tpc)
    hit_y = np.asarray(hit_y, np.float64); hit_e = np.asarray(hit_e, np.float64)
    ts = np.asarray(ts, np.float64)
    n_light, n_ch, n_tick = actual.shape

    # ---- group hits by (label, tpc) with one argsort ----
    ok = (labels >= 0) & (hit_tpc < n_light) & np.isfinite(ts)
    key = labels[ok].astype(np.int64) * 128 + hit_tpc[ok].astype(np.int64)
    idx = np.flatnonzero(ok)[np.argsort(key, kind="stable")]
    ks = labels[idx].astype(np.int64) * 128 + hit_tpc[idx].astype(np.int64)
    bnd = np.r_[0, np.flatnonzero(np.diff(ks)) + 1, len(ks)]

    RW = {}                                                  # lazy per-TPC standardization
    parts: dict[int, list] = {}
    for i in range(len(bnd) - 1):
        sl = idx[bnd[i]:bnd[i + 1]]
        if len(sl) < 2:
            continue
        L = int(labels[sl[0]]); c = int(hit_tpc[sl[0]])
        t0 = float(np.median(ts[sl]))
        r0 = int(np.clip((hit_y[sl].min() - YLO) / (YHI - YLO) * NROW, 0, NROW - 1))
        r1 = int(np.clip((hit_y[sl].max() - YLO) / (YHI - YLO) * NROW, 0, NROW - 1))
        lo, hi = max(0, r0 - DIL), min(NROW - 1, r1 + DIL)
        ch = np.r_[np.arange(lo, hi + 1), NROW + np.arange(lo, hi + 1)]
        parts.setdefault(L, []).append((c, t0, ch, float(hit_e[sl].sum())))

    out = {}
    for L, pl in parts.items():
        E = sum(p[3] for p in pl)
        if E < min_energy or sum(1 for _ in pl) == 0:
            continue
        curve = np.zeros(n_tick); own = []; N = 0; chi2 = 0.0; nloc = 0
        for c, t0, ch, ec in pl:
            tick = int(round(t0)) + PEAK
            if not (0 <= tick < n_tick):
                continue
            if c not in RW:
                RW[c] = standardized_residual(actual[c], base[c])
            v0 = RW[c][ch, tick]
            curve += v0 @ RW[c][ch]
            own.append(v0); N += len(ch)
            wlo, whi = max(0, tick - FWIN), min(n_tick, tick + FWIN + 1)
            chi2 += float((RW[c][ch, wlo:whi] ** 2).sum()); nloc += len(ch) * (whi - wlo)
        if N == 0:
            continue
        own = np.concatenate(own)
        curve /= np.sqrt(N)
        for c, t0, ch, ec in pl:
            tick = int(round(t0)) + PEAK
            curve[max(0, tick - SEP):tick + SEP + 1] = 0.0
        out[L] = dict(z_min=float(curve.min()), z_own=float(own.sum() / np.sqrt(N)),
                      chi2_loc=chi2 / max(nloc, 1), E=float(E), n_tpc=len(pl))
    return out
