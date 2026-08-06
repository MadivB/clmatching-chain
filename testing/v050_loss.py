"""v0.5.0 signed residual loss (EXPERIMENTAL — v0.4.1 chi2 remains the default).

User design (2026-07-16): make the loss DIRECTIONAL. With r = clip(pred) - actual:
  cubic   : L = sum w * r^3            (deficit -> negative = attractive;
                                        overflow -> positive = repulsive)
  signed2 : L = sum w * sign(r) * r^2  (same sign structure, square magnitude)

Noise mask: ticks where BOTH clip(pred) < NOISE_ADC and actual < NOISE_ADC are
excluded — both sides agree "only noise here", and (crucially for odd powers)
the ~100k noise-floor ticks would otherwise contribute random-signed junk that
swamps real flash terms instead of cancelling like even powers do.

Weights: the v0.4.1 Poisson weights w = 1/(GAIN*max(actual,100)) by default
(WEIGHT_POW=2 in sigma units); WEIGHT_POW=3 gives the scale-free sum of z^3.

Scans: the UNMASKED cubic curve decomposes exactly into three FFT
correlations, (D + sP)^3 = D^3 + 3 D^2 sP + 3 D sP^2 + sP^3 with D = B - A,
so full-range shift scans stay cheap.  The t-independent D^3 term and the
mask are irrelevant for argmin *ranking* wherever the template halo is small,
so coarse curves are unmasked; every ACCEPT/COMPARE decision (batch-ordering
loss, Stage-A configuration loss, expansion delta) uses the exact MASKED
pointwise loss.  signed2 has no FFT form: its curves reuse the cubic coarse
curve for ranking; pointwise decisions are exact.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

GAIN = 200.0
CLIP = 60780.0
NOISE_ADC = 200.0
SEARCH = 800


def poisson_w(actual: np.ndarray, weight_pow: int = 2) -> np.ndarray:
    sig2 = GAIN * np.maximum(actual, 100.0)
    if weight_pow == 2:
        return 1.0 / sig2
    return 1.0 / sig2 ** (weight_pow / 2.0)


def point_loss(pred, actual, w=None, mode="cubic", noise_adc=NOISE_ADC):
    """Exact masked signed loss of a full model vs actual (any shape)."""
    p = np.minimum(np.asarray(pred, np.float64), CLIP)
    a = np.asarray(actual, np.float64)
    r = p - a
    if w is None:
        w = poisson_w(a)
    keep = (p >= noise_adc) | (a >= noise_adc)
    if mode == "cubic":
        v = r * r * r
    elif mode == "signed2":
        v = np.sign(r) * r * r
    elif mode == "chi2":
        v = r * r
        keep = np.ones_like(keep)
    else:
        raise ValueError(mode)
    return float(np.sum((w * v)[keep]))


def delta_loss(base, add, actual, w=None, mode="cubic", noise_adc=NOISE_ADC):
    """Raw masked loss(base+add) - loss(base) (diagnostic; see improvement)."""
    return (point_loss(np.asarray(base) + np.asarray(add), actual, w, mode, noise_adc)
            - point_loss(base, actual, w, mode, noise_adc))


def loss_key(L, mode="cubic"):
    """Decision transform: signed losses are DRIVEN TOWARD ZERO (deficit is
    negative, overflow positive, balance is 0), so every argmin/accept
    comparison uses |L|.  chi2 is >= 0, so this is the identity there and
    v0.4.1 semantics are the exact special case."""
    return abs(L) if mode in ("cubic", "signed2") else L


def improvement(base, add, actual, w=None, mode="cubic", noise_adc=NOISE_ADC):
    """The accept/score primitive: |loss(base+add)| - |loss(base)| (masked).
    < 0 means the addition moves the model TOWARD balance -> accept.
    For chi2 this reduces to the plain delta chi2 of v0.4.1."""
    la = point_loss(np.asarray(base) + np.asarray(add), actual, w, mode, noise_adc)
    lb = point_loss(base, actual, w, mode, noise_adc)
    return loss_key(la, mode) - loss_key(lb, mode)


def scan_cubic(pred, resid, w, max_shift=SEARCH):
    """Unmasked cubic curve over shifts t=0..max_shift (FFT, 3 correlations).

    pred: (n_ch, n_ticks) template rows; resid = base - actual (same rows);
    w: weights (same rows).  curve(t) = sum w*(resid + shift_t(pred))^3.
    Same row-stacking convention as v030_expansion.weighted_scan.
    """
    P = np.asarray(pred, np.float64)
    D = np.asarray(resid, np.float64)
    W = np.asarray(w, np.float64)
    n = P.shape[1]
    sl = slice(n - 1, n - 1 + max_shift + 1)
    c1 = fftconvolve(W * D * D, P[:, ::-1], mode="full", axes=1)[:, sl].sum(axis=0)
    c2 = fftconvolve(W * D, (P * P)[:, ::-1], mode="full", axes=1)[:, sl].sum(axis=0)
    c3 = fftconvolve(W, (P * P * P)[:, ::-1], mode="full", axes=1)[:, sl].sum(axis=0)
    return float(np.sum(W * D ** 3)) + 3.0 * c1 + 3.0 * c2 + c3


def scan_curve(pred, resid, w, max_shift=SEARCH, mode="cubic"):
    """Drop-in replacement for weighted_scan: returns a full curve whose argmin
    ranking follows the signed loss (unmasked coarse; see module docstring).
    chi2 falls back to the quadratic curve; signed2 reuses the cubic curve."""
    if mode == "chi2":
        P = np.asarray(pred, np.float64)
        D = np.asarray(resid, np.float64)
        W = np.asarray(w, np.float64)
        n = P.shape[1]
        sl = slice(n - 1, n - 1 + max_shift + 1)
        t1 = fftconvolve(W, (P * P)[:, ::-1], mode="full", axes=1)[:, sl].sum(axis=0)
        t2 = fftconvolve(W * D, P[:, ::-1], mode="full", axes=1)[:, sl].sum(axis=0)
        return t1 + 2.0 * t2 + float(np.sum(W * D * D))
    return scan_cubic(pred, resid, w, max_shift)


def refine_argmin(pred, base, actual, w, curve, top_k=32, mode="cubic",
                  noise_adc=NOISE_ADC):
    """Exact masked evaluation of the top_k coarse-curve candidates (+-1 tick),
    ranked and selected by loss_key (|L| for signed modes);
    returns (best_t, best_masked_loss_key)."""
    rank = np.abs(curve) if mode in ("cubic", "signed2") else curve
    order = np.argsort(rank)[:top_k]
    cand = sorted(set(int(t) for t0 in order for t in (t0 - 1, t0, t0 + 1)
                      if 0 <= t < curve.size))
    P = np.asarray(pred, np.float64)
    n = P.shape[1]
    best = None
    for t in cand:
        sft = np.zeros_like(P)
        if t < n:
            sft[:, t:] = P[:, :n - t]
        val = loss_key(point_loss(np.asarray(base) + sft, actual, w, mode,
                                  noise_adc), mode)
        if best is None or val < best[1]:
            best = (t, val)
    return best
