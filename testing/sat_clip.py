"""Per-event light saturation ceiling for the testing-side chain modules (v1.2.2).

The harness calls ``set_ceiling(ceil)`` once per event with the (n_tpc, 120) array built by
``M5p1.light_saturation`` (32767 minus the per-channel dynamic baseline).  Every module that
used the constant ``CLIP = 60780`` now asks this registry for the bound of the channels it
is looking at.  When nothing has been set (old scripts, dumps without raw samples) the
legacy constant is returned, so behaviour is unchanged there.

    from sat_clip import clip_rows, clip_col, ceiling
    np.minimum(model_rows, CLIP)      ->  clip_rows(model_rows, c, ch)     # model (n_ch, T)
    np.minimum(column, CLIP)          ->  clip_col(column, c, ch)          # one tick, (n_ch,)
    a < 0.98 * CLIP                   ->  a < 0.98 * ceiling(c, ch)        # (n_ch,) broadcast
"""
from __future__ import annotations

import numpy as np

LEGACY_CLIP = 60780.0
_CEIL: np.ndarray | None = None


def set_ceiling(ceil) -> None:
    """Register the per-event ceiling (n_tpc, n_ch) or clear it with None."""
    global _CEIL
    _CEIL = None if ceil is None else np.asarray(ceil, dtype=np.float64)


def get_ceiling() -> np.ndarray | None:
    return _CEIL


def ceiling(c=None, ch=None):
    """Ceiling for TPC ``c`` (n_ch,) or for its channel subset ``ch``; scalar when unset."""
    if _CEIL is None:
        return LEGACY_CLIP
    if c is None:
        return _CEIL
    v = _CEIL[int(c)]
    return v if ch is None else v[ch]          # index array, bool mask, list or slice


def ceiling_rows(c, ch=None):
    """Ceiling shaped (n_ch, 1) for broadcasting against a (n_ch, T) block."""
    v = ceiling(c, ch)
    return v if np.isscalar(v) else v[:, None]


def clip_rows(x, c, ch=None):
    """np.minimum(x, ceiling) for x of shape (n_ch, T) (channels of TPC c, subset ch)."""
    return np.minimum(x, ceiling_rows(c, ch))


def clip_col(x, c, ch=None):
    """np.minimum(x, ceiling) for a single-tick column x of shape (n_ch,)."""
    return np.minimum(x, ceiling(c, ch))


def clip_full(x):
    """np.minimum(x, ceiling) for the full (n_tpc, n_ch, T) image."""
    if _CEIL is None:
        return np.minimum(x, LEGACY_CLIP)
    return np.minimum(x, _CEIL[..., None])


def clip_scalar() -> float:
    """One number for paths that need a scalar bound (raw prediction clip): the event minimum."""
    return LEGACY_CLIP if _CEIL is None else float(_CEIL.min())


__all__ = ["LEGACY_CLIP", "set_ceiling", "get_ceiling", "ceiling", "ceiling_rows",
           "clip_rows", "clip_col", "clip_full", "clip_scalar"]
