"""Charge-Light matching CONFIDENCE metric (v0.4 output).

Confidence v2 = complexity-normalized combination of three truth-free per-TPC light
discrepancy statistics, each computed at the TPC's flash t0 list:
  chi2_flash : SNR-weighted per-flash predicted-vs-actual chi2/dof
  displaced  : conservation-matched deficit<->excess light (the swap fingerprint), 3sigma
  maxk       : multiscale residual-whiteness (Anscombe residuals, channel blocks)
Higher = worse match (more likely mis-placed light).

Design for production:
  components(A, P, t0_ticks) -> per-TPC (chi2_flash, displaced, maxk, nflash, nlit)   [fast, vectorized]
  Calibration.fit(comp_df, complexity) then .score(...)  -> confidence scalar
Fit the Calibration ONCE on a trusted MC sample (stores the complexity regression and
the empirical rank transforms); apply per-TPC on data.  For a self-contained batch, fit
and score on the same run (relative confidence within the run).
"""
import numpy as np

GAIN, PEAK, CLIP = 200.0, 105, 60780.0
BLOCKS = (1, 2, 4, 8, 12, 24)
FWIN = 4                       # +-ticks per flash window


try:
    from confidence_fast import find_peaks_nb as _fp_nb, HAVE_NUMBA as _NB   # 14x numba fast path
except Exception:
    _NB = False


def find_peaks(trace, frac=0.06, min_sep=8):
    """Local-maxima flash finder (matches the validated build_lightcache peaks()).
    Uses the numba kernel when available, else an equivalent numpy path."""
    trace = np.asarray(trace, np.float64)
    if _NB:
        return _fp_nb(np.ascontiguousarray(trace), float(frac), int(min_sep))
    if trace.max() <= 0:
        return np.empty(0, int)
    thr = frac * trace.max(); t = trace; n = len(t); idx = np.arange(2, n - 2)
    m = (t[idx] > thr) & (t[idx] >= t[idx - 1]) & (t[idx] > t[idx + 1]) & (t[idx] >= t[idx - 2])
    cand = idx[m]
    if cand.size == 0:
        return cand
    keep = [cand[0]]                                   # greedy min-separation merge
    for c in cand[1:]:
        if c - keep[-1] < min_sep:
            if t[c] > t[keep[-1]]:
                keep[-1] = c
        else:
            keep.append(c)
    return np.array(keep, int)


def _anscombe(x):
    return 2.0 * np.sqrt(np.maximum(x, 0.0) + 0.375)


def t0_list(A, placed_ticks=()):
    """candidate flash ticks = actual-light peaks UNION placed cluster ticks."""
    cand = list(find_peaks(A.sum(0)))
    for p in placed_ticks:
        if 0 <= p < A.shape[1] and all(abs(p - q) >= 8 for q in cand):
            cand.append(int(p))
    return sorted(set(int(c) for c in cand))


def components(A, P_raw, ticks, alpha=None):
    """Per-TPC confidence components from actual A and predicted P (both [n_ch, n_tick]).
    ticks = flash t0 list.  alpha auto-fit (removes the ML amplitude systematic)."""
    A = np.asarray(A, float); P_raw = np.minimum(np.asarray(P_raw, float), CLIP)
    n_ch, n_t = A.shape
    if alpha is None:
        m = A < 0.98 * CLIP; a = A[m]; p = P_raw[m]
        alpha = float((p @ a) / max(p @ p, 1e-9))
    P = alpha * P_raw
    if not len(ticks):
        return dict(chi2_flash=0.0, displaced=0.0, maxk=0.0, nflash=0, nlit=0)
    # per-flash window channel vectors (saturation-zeroed)
    Fa = np.empty((len(ticks), n_ch)); Fp = np.empty_like(Fa)
    for i, tk in enumerate(ticks):
        lo, hi = max(0, tk - FWIN), min(n_t, tk + FWIN + 1)
        aw = A[:, lo:hi]; pw = P[:, lo:hi]; sat = aw.max(1) >= 0.98 * CLIP
        av = aw.sum(1); pv = pw.sum(1); av[sat] = 0; pv[sat] = 0
        Fa[i] = av; Fp[i] = pv
    ampA = Fa.sum(1); tot = max(Fa.sum(), 1.0)
    # chi2_flash (SNR-weighted per-flash reduced chi2)
    red = ((Fp - Fa) ** 2 / (GAIN * np.maximum(Fa, 100.0))).sum(1) / n_ch
    w = np.sqrt(np.maximum(ampA, 0) / GAIN)
    chi2 = float((red * w).sum() / w.sum()) if w.sum() > 0 else 0.0
    # displaced-light (per-channel matched deficit/excess, 3 sigma "clearly")
    R = Fa - Fp; sig = np.sqrt(GAIN * np.maximum(Fa, 900.0)); Rt = np.where(np.abs(R) > 3 * sig, R, 0.0)
    disp = float(np.minimum(np.maximum(Rt, 0).sum(0), np.maximum(-Rt, 0).sum(0)).sum() / tot)
    # maxk (multiscale whiteness on Anscombe residuals, channel blocks); Fp already alpha-scaled
    Rw = _anscombe(np.maximum(Fp, 0) / GAIN) - _anscombe(Fa / GAIN)
    maxk = float((Rw ** 2).mean())
    for k in BLOCKS:
        if k == 1 or n_ch % k:
            continue
        Rb = Rw.reshape(len(ticks), n_ch // k, k).sum(-1) / np.sqrt(k)
        maxk = max(maxk, float((Rb ** 2).mean()))
    nlit = int((Fa.sum(0) > 5 * np.sqrt(GAIN * 100.0) * np.sqrt(len(ticks))).sum())
    return dict(chi2_flash=chi2, displaced=disp, maxk=maxk, nflash=len(ticks), nlit=nlit)


def _rank(x):
    return np.argsort(np.argsort(np.asarray(x, float))).astype(float) / len(x)


class Calibration:
    """complexity-normalized rank combination of the three components -> confidence."""
    def __init__(self, keys=("chi2_flash", "displaced", "maxk")):
        self.keys = keys

    def _design(self, nflash, logE, nlit):
        return np.c_[np.ones_like(logE), _rank(nflash), _rank(logE), _rank(nlit)]

    def fit_score(self, comp, nflash, E, nlit):
        """Fit + score on the same batch (relative confidence). comp: dict of arrays."""
        logE = np.log(np.maximum(E, 1.0)); X = self._design(nflash, logE, nlit)
        parts = []
        for k in self.keys:
            r = _rank(comp[k]); beta = np.linalg.lstsq(X, r, rcond=None)[0]
            parts.append(_rank(r - X @ beta))                 # complexity-residual, re-ranked
        return np.mean(parts, axis=0)
