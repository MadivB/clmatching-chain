"""numba-JIT fast paths for the confidence engine (the 'C++-module' maneuver).
find_peaks_nb and displaced_nb replace the Python/loop-heavy inner kernels; they are
drop-in for the classical matcher's flash-detection and residual loops too.
Falls back to numpy transparently if numba is unavailable."""
import numpy as np
try:
    from numba import njit
    _HAVE = True
except Exception:
    _HAVE = False
    def njit(*a, **k):
        def deco(f): return f
        return deco if not a else a[0]


@njit(cache=True, fastmath=True)
def find_peaks_nb(trace, frac, min_sep):
    mx = 0.0
    for v in trace:
        if v > mx:
            mx = v
    out = np.empty(len(trace), np.int64)
    if mx <= 0.0:
        return out[:0]
    thr = frac * mx
    n = 0
    for t in range(2, len(trace) - 2):
        v = trace[t]
        if v > thr and v >= trace[t - 1] and v > trace[t + 1] and v >= trace[t - 2]:
            if n > 0 and t - out[n - 1] < min_sep:
                if v > trace[out[n - 1]]:
                    out[n - 1] = t
            else:
                out[n] = t; n += 1
    return out[:n]


@njit(cache=True, fastmath=True)
def displaced_nb(Fa, Fp, gain):
    """per-channel matched deficit/excess (3 sigma), summed / total actual.  Fa,Fp:(nf,nch)."""
    nf, nch = Fa.shape
    pos = np.zeros(nch); neg = np.zeros(nch); tot = 0.0
    for j in range(nch):
        for i in range(nf):
            a = Fa[i, j]; r = a - Fp[i, j]; s = 3.0 * np.sqrt(gain * (a if a > 900.0 else 900.0))
            if r > s:
                pos[j] += r
            elif -r > s:
                neg[j] += -r
            tot += a
    d = 0.0
    for j in range(nch):
        d += pos[j] if pos[j] < neg[j] else neg[j]
    return d / (tot if tot > 1.0 else 1.0)


def find_peaks(trace, frac=0.06, min_sep=8):
    return find_peaks_nb(np.ascontiguousarray(trace, np.float64), float(frac), int(min_sep))


HAVE_NUMBA = _HAVE
