"""Per-TPC pairwise t0-SWAP correction for the v0.3.0 charge-light matcher.

Redesign (2026-07-09, after the user's key insight): the swap must act on
cluster-PARTS *within a single charge-TPC*, not on whole multi-TPC clusters.
We cannot trust the multi-TPC clustering — one label can hold two interactions,
and the pipeline legitimately splits a cluster's hits across TPCs (each hit's t0
is independent).  So the unit is (label, TPC), the part's t0 is the median of
that label's hits in that TPC, and a swap exchanges the t0s of TWO such parts in
the same TPC.

Metric = support-masked per-TPC light chi2 of the JOINT swap.  A single move
fails (leaves a light deficit at the vacated flash -> chi2 worse); swapping both
parts fills each other's flash.  Validated on the FIXED base_image (the earlier
"chi2 is degenerate" conclusion was an artifact of the stale-base_image dump
bug): true swaps have chi2(swap)/chi2(current) median 0.70, false/correct pairs
median 1.11, and accepting ratio < 0.9 gives ~64% recall at 0% false-accept on
the templated-pair sample.  Best-improvement-first, non-conflicting, iterated.
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
from sat_clip import clip_rows, ceiling, ceiling_rows, LEGACY_CLIP  # noqa: E402

CLIP = LEGACY_CLIP   # legacy constant; math goes through sat_clip
GAIN = 200.0
PEAK = 105
MIN_SEP = 20      # a swap pair's part-t0s must differ by > this (distinct flashes)
SUPP_FRAC = 0.02  # template-support channel threshold (fraction of peak)
WIN_LO, WIN_HI = PEAK - 15, PEAK + 50   # tick window (relative to a t0) for the chi2


def run_swap_correction(ns: dict[str, Any], *,
                        hit_tpc: np.ndarray,
                        cluster_to_tpcs: dict,          # kept for signature compat (unused)
                        min_energy: float = 25.0,
                        chi2_ratio_max: float = 0.9,     # accept swap iff chi2_swap < this * chi2_cur
                        spatial_gate: bool = True,        # AND require both parts to prefer the swap
                        spatial_min: float = 0.0,         #   spatially (rejects chi2 deficit-fills)
                        min_cos_gain: float = 0.0,        # legacy arg, ignored
                        max_passes: int = 3,
                        truth_fn: Optional[Callable[[int], float]] = None,
                        verbose: bool = False) -> dict:
    """Reassign co-located cluster-PARTS (per charge-TPC) by swapping the t0s of
    pairs whose joint exchange lowers the support-masked per-TPC light chi2.
    Mutates ns['hit_timestamps'] and ns['baseImage']."""
    image_maps = ns["imageMaps"]
    actual = np.asarray(ns["fullLightWaveform"], np.float64)
    base = np.asarray(ns["baseImage"], np.float64)
    n_light, n_ch, n_tick = actual.shape
    labels = np.asarray(ns["labels_global"])
    ts = np.asarray(ns["hit_timestamps"], np.float64).copy()
    e_pos = np.clip(np.asarray(ns["Eset"], np.float64), 0.0, None)
    hit_tpc = np.asarray(hit_tpc)

    # cache per-(label, tpc) template + its support channels
    _tmpl, _supp = {}, {}
    def tmpl(L, c):
        k = (int(L), int(c))
        if k not in _tmpl:
            P = np.asarray(image_maps[k], np.float64)
            _tmpl[k] = P
            s = P.sum(1)
            _supp[k] = s > SUPP_FRAC * s.max() if s.max() > 0 else np.zeros(P.shape[0], bool)
        return _tmpl[k]

    # build parts: per charge-TPC, list of (label, part_t0, energy) that have a template
    assigned = np.isfinite(ts) & (ts >= 0)
    parts_by_tpc: dict[int, list] = {}
    for c in np.unique(hit_tpc[hit_tpc < n_light]):
        c = int(c)
        mc = hit_tpc == c
        for L in np.unique(labels[mc & (labels >= 0)]):
            if (int(L), c) not in image_maps:
                continue
            m = mc & (labels == L) & assigned
            if not m.any():
                continue
            E = float(e_pos[m].sum())
            if E < min_energy:
                continue
            tmpl(int(L), c)  # warm cache
            parts_by_tpc.setdefault(c, []).append([int(L), float(np.median(ts[m])), E])
    if not any(len(v) >= 2 for v in parts_by_tpc.values()):
        return {"n_parts": 0, "n_swaps": 0, "passes": 0, "e_swapped": 0.0,
                "n_good": 0, "n_bad": 0, "audit": []}

    def part_chi2(c, A, B, tA, tB, aAt, aBt):
        PA, PB = tmpl(A, c), tmpl(B, c)
        act = actual[c]
        bw = base[c] - timeinterpolation(PA, shift=tA, baseline=0) - timeinterpolation(PB, shift=tB, baseline=0)
        pred = np.clip(bw + timeinterpolation(PA, shift=aAt, baseline=0)
                       + timeinterpolation(PB, shift=aBt, baseline=0), 0.0, ceiling_rows(c))
        supp = _supp[(A, c)] | _supp[(B, c)]
        lo = max(0, int(round(min(tA, tB))) + WIN_LO)
        hi = min(n_tick, int(round(max(tA, tB))) + WIN_HI)
        if not supp.any() or hi <= lo:
            return 0.0
        a = act[np.ix_(supp, np.arange(lo, hi))]
        pr = clip_rows(pred[np.ix_(supp, np.arange(lo, hi))], c, supp)
        v = GAIN * np.maximum(a, 100.0)
        m = a < 0.98 * ceiling_rows(c, supp)
        return float((((pr - a) ** 2 / v)[m]).sum())

    # spatial gate: the shape-normalized per-channel cosine is immune to the ML
    # amplitude systematic that fools chi2 into "deficit-filling" a flash with a
    # bigger correct cluster.  A correctly-placed part matches ITS flash's spatial
    # pattern, so it will NOT prefer the swap -> the gate vetoes truth-wrong swaps.
    _fpc: dict[tuple, tuple] = {}
    def _flash(c, t0):
        k = (int(c), int(round(t0)))
        if k not in _fpc:
            a = actual[c]; t = int(round(t0)) + PEAK
            col = a[:, max(0, t - 4):min(n_tick, t + 5)]
            _fpc[k] = (col.sum(1), col.max(1) >= 0.98 * ceiling(c))
        return _fpc[k]
    def _cos(u, v, m):
        u = u[~m]; v = v[~m]; nu = np.linalg.norm(u); nv = np.linalg.norm(v)
        return float(u @ v / (nu * nv)) if nu and nv else 0.0
    def spatial_gains(c, A, B, tA, tB):
        pA = tmpl(A, c).sum(1); pB = tmpl(B, c).sum(1)
        fA, sA = _flash(c, tA); fB, sB = _flash(c, tB)
        gA = _cos(pA, fB, sB) - _cos(pA, fA, sA)   # A prefers B's flash spatially?
        gB = _cos(pB, fA, sA) - _cos(pB, fB, sB)   # B prefers A's flash spatially?
        return gA, gB

    def apply_swap(c, A, B, tA, tB):
        PA, PB = tmpl(A, c), tmpl(B, c)
        base[c] += (- timeinterpolation(PA, shift=tA, baseline=0) - timeinterpolation(PB, shift=tB, baseline=0)
                    + timeinterpolation(PA, shift=tB, baseline=0) + timeinterpolation(PB, shift=tA, baseline=0))
        ts[(hit_tpc == c) & (labels == A)] = tB
        ts[(hit_tpc == c) & (labels == B)] = tA

    def part_truth(c, L):
        if truth_fn is None:
            return None
        return truth_fn(int(L), int(c)) if _truth_takes_tpc else truth_fn(int(L))

    # detect whether truth_fn accepts a tpc arg (per-TPC truth) or just a label
    _truth_takes_tpc = False
    if truth_fn is not None:
        try:
            truth_fn(-1, -1); _truth_takes_tpc = True
        except TypeError:
            _truth_takes_tpc = False
        except Exception:
            _truth_takes_tpc = True

    audit = []
    n_swaps = 0; e_swapped = 0.0; passes = 0
    for _p in range(max_passes):
        passes += 1
        cands = []           # (ratio, c, A, B, tA, tB, EA, EB)
        for c, parts in parts_by_tpc.items():
            for i in range(len(parts)):
                for j in range(i + 1, len(parts)):
                    A, tA, EA = parts[i]; B, tB, EB = parts[j]
                    if abs(tA - tB) < MIN_SEP:
                        continue
                    if spatial_gate:
                        gA, gB = spatial_gains(c, A, B, tA, tB)
                        if not (gA > spatial_min and gB > spatial_min):
                            continue   # a part is spatially content where it is -> not a real swap
                    cur = part_chi2(c, A, B, tA, tB, tA, tB)
                    if cur <= 0:
                        continue
                    swp = part_chi2(c, A, B, tA, tB, tB, tA)
                    ratio = swp / cur
                    if ratio < chi2_ratio_max:
                        cands.append((ratio, c, A, B, tA, tB, EA, EB))
        if not cands:
            break
        cands.sort(key=lambda r: r[0])
        used = set(); applied = 0
        for ratio, c, A, B, tA, tB, EA, EB in cands:
            if (c, A) in used or (c, B) in used:
                continue
            rec = {"tpc": c, "A": A, "B": B, "tA": round(tA, 1), "tB": round(tB, 1),
                   "EA": round(EA, 1), "EB": round(EB, 1), "chi2_ratio": round(ratio, 3)}
            if truth_fn is not None:
                trA, trB = part_truth(c, A), part_truth(c, B)
                if trA is not None and trB is not None:
                    err_b = abs(tA - trA) + abs(tB - trB)
                    err_a = abs(tB - trA) + abs(tA - trB)
                    rec["truthA"] = round(trA, 1); rec["truthB"] = round(trB, 1)
                    rec["good"] = bool(err_a < err_b)
            apply_swap(c, A, B, tA, tB)
            # update part t0s in the working list
            for pr in parts_by_tpc[c]:
                if pr[0] == A: pr[1] = tB
                elif pr[0] == B: pr[1] = tA
            used.add((c, A)); used.add((c, B))
            n_swaps += 1; applied += 1; e_swapped += EA + EB
            audit.append(rec)
        if verbose:
            print(f"[swap] pass {passes}: {applied} per-TPC swaps ({len(cands)} candidates)", flush=True)
        if applied == 0:
            break

    ns["hit_timestamps"] = ts.astype(np.float32)
    ns["baseImage"] = base.astype(np.float32)
    n_good = sum(1 for r in audit if r.get("good") is True)
    n_bad = sum(1 for r in audit if r.get("good") is False)
    n_parts = sum(len(v) for v in parts_by_tpc.values())
    return {"n_parts": n_parts, "n_swaps": n_swaps, "passes": passes,
            "e_swapped": round(e_swapped, 1), "n_good": n_good, "n_bad": n_bad,
            "audit": audit}
