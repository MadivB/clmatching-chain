"""Stage 4.8 residual-flash filling ("reverse construction", user 2026-10-01).

Instead of asking each small cluster where it wants to go, start from the LIGHT that still needs filling:
  1. residual flashes per light TPC = peaks of the summed residual (actual - everything rendered) with prominence
     >= k_prom x pre-trigger sigma, not within fam_tol ticks of an existing family time (those are deficits of placed
     families, not new flashes), and NOT on the tail of a large flash: a peak is kept only if the residual there is
     at least tail_ratio of the modelled light (a template-mismatch bump on a big pulse's tail is a few % of the model)
     and its summed residual amplitude is >= min_adc.
  2. for each residual flash, brightest first: look at the templated small clusters (small_min <= E < small_max) of the
     TPC that are not already there and score each by the cosine between its predicted light pattern and the residual
     pattern at the flash time on a RESTRICTED channel set: the K channels nearest to the cluster (mode "near"), the K
     brightest predicted channels ("top") or the 90 % support ("sup").  Take the best; accept iff
        cos >= cos_min,  cos > cos at the cluster's current slot (own light added back) + dcos_min,
        residual / template amplitude on those channels in [ratio_lo, ratio_hi]  (the light there is at least a
        good part of what the cluster would emit),
        optional blip-rehome gates: own / others' light at the current slot >= glare_min (deep-glare deposits are
        the host's spray) and distance to the co-timed host >= host_min_cm;
     move it (re-render), recompute the residual at the flash, and repeat while the flash still has light and another
     cluster passes (max_per_flash).
No truth in any decision.  ~0.1 s per spill.
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
from scipy.signal import find_peaks

PEAK = 105; HW = 2
_T = None


def _template(pulse_path):
    global _T
    if _T is None:
        p = np.load(pulse_path).astype(np.float64).ravel(); _T = p / p.max()
    return _T


def render(amp, t, n, T):
    s = np.arange(n); sh = np.interp(s - t, np.arange(T.size), T, left=0.0, right=0.0)
    return amp[:, None] * sh[None, :]


def peak_pattern(R, t):
    i = int(np.clip(round(t) + PEAK, HW, R.shape[1] - HW - 1)); return R[:, i - HW:i + HW + 1].mean(1)


def channel_set(amp, chan_pos, centroid, mode, K, veto):
    ok = np.flatnonzero(~veto & (amp > 0)) if mode != "near" else np.flatnonzero(~veto)
    if ok.size == 0:
        return ok
    if mode == "near":
        d = np.linalg.norm(chan_pos[ok] - centroid[None, :], axis=1); return ok[np.argsort(d)[:K]]
    if mode == "top":
        return ok[np.argsort(-amp[ok])[:K]]
    o = ok[np.argsort(-amp[ok])]; c = np.cumsum(amp[o]); k = int(np.searchsorted(c, 0.9 * c[-1])) + 1
    return o[:max(k, 3)]


def score(R, amp, chans, t):
    p = peak_pattern(R, t)[chans]; a = amp[chans]
    na = np.linalg.norm(a); npv = np.linalg.norm(p)
    if na <= 0 or npv <= 0:
        return -1.0, 0.0
    return float(a @ p / (na * npv)), float(max(a @ p, 0.0) / (a @ a))      # cosine, residual/template amplitude


def residual_flashes(R, B, veto, fam_t, *, k_prom, fam_tol, tail_ratio, min_adc):
    chans = np.flatnonzero(~veto)
    if chans.size == 0:
        return np.zeros(0), np.zeros(0)
    rs = R[chans].sum(0); bs = B[chans].sum(0); sig = max(float(rs[:90].std()), 1.0)
    p, _ = find_peaks(rs, prominence=k_prom * sig, distance=4)
    t = (p - PEAK).astype(float); keep = (t >= 0) & (t <= 885)
    p, t = p[keep], t[keep]
    if t.size == 0:
        return t, np.zeros(0)
    amp = rs[p]
    good = amp >= min_adc
    good &= amp >= tail_ratio * np.maximum(bs[p], 0.0)
    if fam_t.size:
        good &= np.min(np.abs(t[:, None] - fam_t[None, :]), 1) > fam_tol
    return t[good], amp[good]


def run_resid_fill(*, ts, labels, energy, xyz, tpc, amp, actual, base, veto, chan_pos, render_cb: Callable,
                   pulse_path, small_min=0.5, small_max=20.0, mode="near", K=8, cos_min=0.9, dcos_min=0.1,
                   ratio_lo=0.5, ratio_hi=20.0, k_prom=10.0, fam_tol=5.0, tail_ratio=0.3, min_adc=600.0,
                   max_per_flash=4, glare_min=None, host_min_cm=None, truth_t0=None):
    """ts: per-hit times (modified in place); amp: {(label, tpc): (120,)}; actual/base: (n_light,120,1000) (base
    modified by render_cb); veto (n_light,120); chan_pos {tpc: (120,3)}; render_cb(L, c, t_old, t_new) updates base."""
    t0 = time.time(); T = _template(pulse_path)
    labels = np.asarray(labels); energy = np.clip(np.asarray(energy, float), 0, None); tpc = np.asarray(tpc)
    st = {"n_flash": 0, "n_flash_tail_cut": 0, "n_moved": 0, "e_moved": 0.0, "fixed_e": 0.0, "broke_e": 0.0, "audit": []}
    n_light, n_ch, n_s = actual.shape
    lab_hits = {}
    for L in np.unique(labels[labels >= 0]):
        lab_hits[int(L)] = np.flatnonzero(labels == L)
    cE = {L: energy[h].sum() for L, h in lab_hits.items()}
    cXYZ = {L: xyz[h].mean(0) for L, h in lab_hits.items()}
    for c in range(n_light):
        if c not in chan_pos:
            continue
        mc = tpc == c
        fin = np.isfinite(ts) & (ts >= 0)
        fam_t = np.unique(np.round(ts[mc & fin & (labels >= 0)]))
        R = actual[c].astype(np.float64) - base[c].astype(np.float64)
        f_t, f_a = residual_flashes(R, base[c].astype(np.float64), veto[c], fam_t, k_prom=k_prom, fam_tol=fam_tol, tail_ratio=tail_ratio, min_adc=min_adc)
        if f_t.size == 0:
            continue
        st["n_flash"] += int(f_t.size)
        # candidate clusters in this TPC
        cands = []
        for L in np.unique(labels[mc & (labels >= 0)]):
            L = int(L)
            if (L, c) not in amp or not (small_min <= cE[L] < small_max):
                continue
            h = lab_hits[L]; tv = ts[h]; tv = tv[np.isfinite(tv) & (tv >= 0)]
            if tv.size == 0:
                continue
            a = amp[(L, c)]; ch = channel_set(a, chan_pos[c], cXYZ[L], mode, K, veto[c])
            if ch.size < 3:
                continue
            cands.append([L, float(np.median(tv)), a, ch])
        if not cands:
            continue
        moved = set()
        for fi in np.argsort(-f_a):
            tf = f_t[fi]
            for _ in range(max_per_flash):
                best = None
                for L, tcur, a, ch in cands:
                    if L in moved or abs(tcur - tf) <= fam_tol:
                        continue
                    cs, ratio = score(R, a, ch, tf)
                    if cs < cos_min or not (ratio_lo <= ratio <= ratio_hi):
                        continue
                    own_cur = render(a, tcur, n_s, T)
                    cs_cur, _ = score(R + own_cur, a, ch, tcur)
                    if cs <= cs_cur + dcos_min:
                        continue
                    if glare_min is not None:          # buried in a host's glare at the current slot = the host's own spray: leave it
                        others = np.maximum(peak_pattern(base[c] - own_cur, tcur)[ch], 0).sum()
                        if a[ch].sum() / max(others, 1e-9) < glare_min:
                            continue
                    if host_min_cm is not None:        # spray sits close to its parent: a cluster touching its co-timed host stays
                        hm = mc & fin & (labels >= 0) & (labels != L) & (np.abs(ts - tcur) <= 3)
                        if hm.any():
                            dh = np.min(np.linalg.norm(xyz[hm][:, None, :] - xyz[lab_hits[L]][None, :, :], axis=2))
                            if dh < host_min_cm:
                                continue
                    if best is None or cs > best[0]:
                        best = (cs, L, tcur, a, ratio, cs_cur)
                if best is None:
                    break
                cs, L, tcur, a, ratio, cs_cur = best
                h = lab_hits[L]
                render_cb(L, c, tcur, tf)
                R -= render(a, tf, n_s, T) - render(a, tcur, n_s, T)
                ts[h] = tf; moved.add(L)
                st["n_moved"] += 1; st["e_moved"] += float(cE[L])
                if truth_t0 is not None:
                    tr = truth_t0[h]; okh = np.isfinite(tr)
                    was = float((energy[h][okh] * (np.abs(tcur - tr[okh]) <= 10)).sum()); now = float((energy[h][okh] * (np.abs(tf - tr[okh]) <= 10)).sum())
                    st["fixed_e"] += max(now - was, 0.0); st["broke_e"] += max(was - now, 0.0)
                    st["audit"].append((c, L, float(cE[L]), tcur, float(tf), cs, cs_cur, ratio, was, now))
                for cd in cands:
                    if cd[0] == L:
                        cd[1] = float(tf)
    st["time_s"] = round(time.time() - t0, 3)
    return st
