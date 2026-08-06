"""Residual-driven per-TPC reassignment (an alternative to pairwise swaps).

User's idea (2026-07-09): don't force a swap.  Instead, per TPC, look at the light
RESIDUAL R = actual - predicted.  Where R>0 there is a DEFICIT (missing light);
where R<0 there is EXCESS (a cluster over-explaining -> it is wrongly placed).
For each deficit flash, find the cluster PART whose per-channel template best
matches the MISSING-light pattern, and MOVE it (only its hits in this TPC) to the
deficit flash IF that lowers this TPC's light loss.

Why this beats the pairwise-swap-chi2 (which broke TPC61): a correctly-placed
cluster has no EXCESS at its flash (only the uniform ~22% ML-amplitude deficit),
so it never registers as a source; only a genuinely over-placed cluster (a bigger
cluster sitting on a smaller cluster's flash) shows negative residual and gets
moved.  The move (not a swap) also handles: filling an EMPTY flash, and splitting
a cluster (moving only its part in one TPC).
"""
from __future__ import annotations
import os, sys
from typing import Any, Callable, Optional
import numpy as np

_HERE = os.path.abspath(os.path.dirname(__file__))
for p in (_HERE, os.path.join(_HERE, "..", "CLMatching_AlphaRelease"),
          os.path.join(_HERE, "..", "CLMatching_AlphaRelease", "M5p1")):
    ap = os.path.abspath(p)
    if ap not in sys.path:
        sys.path.append(ap)
from pulse_shapes import timeinterpolation  # noqa: E402

CLIP, GAIN, PEAK = 60780.0, 200.0, 105
FWIN = 4
WIN_LO, WIN_HI = PEAK - 15, PEAK + 50
MIN_SEP = 20


def _deficit_peaks(rt, thresh, min_sep=8, n=8):
    """positive-residual (deficit) peaks in the channel-summed residual, t0 units."""
    out = []
    for t in range(2, len(rt) - 2):
        if rt[t] > thresh and rt[t] >= rt[t-1] and rt[t] > rt[t+1] and rt[t] >= rt[t-2]:
            if out and t - out[-1][0] < min_sep:
                if rt[t] > out[-1][1]:
                    out[-1] = (t, rt[t])
            else:
                out.append((t, rt[t]))
    return sorted(out, key=lambda p: -p[1])[:n]


def run_residual_reassign(ns: dict[str, Any], *,
                          hit_tpc: np.ndarray,
                          cluster_to_tpcs: dict,          # signature compat (unused)
                          min_energy: float = 25.0,
                          deficit_frac: float = 0.15,      # deficit peak >= this * max actual
                          min_match_cos: float = 0.9,      # cluster must match the missing pattern
                          chi2_ratio_max: float = 0.9,     # and the move must lower this-TPC loss
                          max_passes: int = 4,
                          truth_fn: Optional[Callable] = None,
                          verbose: bool = False) -> dict:
    image_maps = ns["imageMaps"]
    actual = np.asarray(ns["fullLightWaveform"], np.float64)
    base = np.asarray(ns["baseImage"], np.float64)
    n_light, n_ch, n_tick = actual.shape
    labels = np.asarray(ns["labels_global"])
    ts = np.asarray(ns["hit_timestamps"], np.float64).copy()
    e_pos = np.clip(np.asarray(ns["Eset"], np.float64), 0.0, None)
    hit_tpc = np.asarray(hit_tpc)

    _T, _S = {}, {}
    def tmpl(L, c):
        k = (int(L), int(c))
        if k not in _T:
            P = np.asarray(image_maps[k], np.float64); _T[k] = P
            s = P.sum(1); _S[k] = s > 0.02 * s.max() if s.max() > 0 else np.zeros(P.shape[0], bool)
        return _T[k]

    # per-TPC parts (label, t0, E) with a template
    parts_by_tpc: dict[int, list] = {}
    asg = np.isfinite(ts) & (ts >= 0)
    for c in np.unique(hit_tpc[hit_tpc < n_light]):
        c = int(c)
        mc = hit_tpc == c
        for L in np.unique(labels[mc & (labels >= 0)]):
            if (int(L), c) not in image_maps:
                continue
            m = mc & (labels == L) & asg
            if not m.any() or float(e_pos[m].sum()) < min_energy:
                continue
            tmpl(int(L), c)
            parts_by_tpc.setdefault(c, []).append([int(L), float(np.median(ts[m])), float(e_pos[m].sum())])
    if not parts_by_tpc:
        return {"n_parts": 0, "n_moves": 0, "passes": 0, "e_moved": 0.0,
                "n_good": 0, "n_bad": 0, "audit": []}

    def prof(arr, t0):
        t = int(round(t0)) + PEAK
        return arr[:, max(0, t - FWIN):min(n_tick, t + FWIN + 1)].sum(1)
    def cos(u, v, m):
        u = u[~m]; v = v[~m]; nu = np.linalg.norm(u); nv = np.linalg.norm(v)
        return float(u @ v / (nu * nv)) if nu and nv else 0.0

    def tpc_chi2(c, extra_lab=None, extra_from=None, extra_to=None):
        """support-masked chi2 over the (extra_lab moved) light; if extra_lab is
        None just current base.  Uses union support of that cluster + window."""
        pred = base[c]
        if extra_lab is not None:
            P = tmpl(extra_lab, c)
            pred = base[c] - timeinterpolation(P, shift=extra_from, baseline=0) \
                           + timeinterpolation(P, shift=extra_to, baseline=0)
            supp = _S[(extra_lab, c)]
            lo = max(0, int(round(min(extra_from, extra_to))) + WIN_LO)
            hi = min(n_tick, int(round(max(extra_from, extra_to))) + WIN_HI)
        else:
            return None
        pred = np.clip(pred, 0.0, CLIP)
        a = actual[c][np.ix_(supp, np.arange(lo, hi))]
        pr = np.minimum(pred[np.ix_(supp, np.arange(lo, hi))], CLIP)
        base_pr = np.minimum(base[c][np.ix_(supp, np.arange(lo, hi))], CLIP)
        v = GAIN * np.maximum(a, 100.0); msk = a < 0.98 * CLIP
        cur = float((((base_pr - a) ** 2 / v)[msk]).sum())
        new = float((((pr - a) ** 2 / v)[msk]).sum())
        return cur, new

    _tpc_truth = False
    if truth_fn is not None:
        try:
            truth_fn(-1, -1); _tpc_truth = True
        except TypeError:
            _tpc_truth = False
        except Exception:
            _tpc_truth = True

    audit = []; n_moves = 0; e_moved = 0.0; passes = 0
    for _p in range(max_passes):
        passes += 1
        moves = []      # (gain, c, L, t_from, t_to)
        for c, parts in parts_by_tpc.items():
            R = actual[c] - base[c]
            rt = R.sum(0)
            thr = deficit_frac * actual[c].sum(0).max()
            for t_d, _h in _deficit_peaks(rt, thr):
                t_d = t_d - PEAK
                Rd = prof(R, t_d)                       # missing-light pattern
                sat = prof(actual[c], t_d) >= 0.98 * CLIP
                # best-matching cluster PART not already at this flash
                best = None
                for L, tL, EL in parts:
                    if abs(tL - t_d) < MIN_SEP:
                        continue
                    cm = cos(tmpl(L, c).sum(1), Rd, sat)
                    if best is None or cm > best[0]:
                        best = (cm, L, tL, EL)
                if best is None or best[0] < min_match_cos:
                    continue
                cm, L, tL, EL = best
                cur, new = tpc_chi2(c, L, tL, float(t_d))
                if new < cur * chi2_ratio_max:
                    moves.append((cur / max(new, 1e-9), c, L, tL, float(t_d), EL, cm))
        if not moves:
            break
        moves.sort(reverse=True)
        used = set(); applied = 0
        for gain, c, L, t_from, t_to, EL, cm in moves:
            if (c, L) in used:
                continue
            P = tmpl(L, c)
            base[c] += (- timeinterpolation(P, shift=t_from, baseline=0)
                        + timeinterpolation(P, shift=t_to, baseline=0))
            ts[(hit_tpc == c) & (labels == L)] = t_to
            for pr in parts_by_tpc[c]:
                if pr[0] == L:
                    pr[1] = t_to
            rec = {"tpc": c, "L": L, "from": round(t_from, 1), "to": round(t_to, 1),
                   "E": round(EL, 1), "match_cos": round(cm, 3), "chi2_gain": round(gain, 2)}
            if truth_fn is not None:
                tr = truth_fn(int(L), int(c)) if _tpc_truth else truth_fn(int(L))
                if tr is not None:
                    rec["truth"] = round(tr, 1)
                    rec["good"] = bool(abs(t_to - tr) < abs(t_from - tr))
            audit.append(rec)
            used.add((c, L)); applied += 1; n_moves += 1; e_moved += EL
        if verbose:
            print(f"[resid] pass {passes}: {applied} moves ({len(moves)} candidates)", flush=True)
        if applied == 0:
            break

    ns["hit_timestamps"] = ts.astype(np.float32)
    ns["baseImage"] = base.astype(np.float32)
    n_good = sum(1 for r in audit if r.get("good") is True)
    n_bad = sum(1 for r in audit if r.get("good") is False)
    return {"n_parts": sum(len(v) for v in parts_by_tpc.values()), "n_moves": n_moves,
            "passes": passes, "e_moved": round(e_moved, 1), "n_good": n_good,
            "n_bad": n_bad, "audit": audit}

