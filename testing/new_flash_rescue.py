"""Residual NEW-FLASH rescue, v2 — BLOB-CENTRIC (user direction 2026-07-10).

v1 (flash-centric) peak-found the CHANNEL-SUMMED residual, so only big-flash deficits
surfaced and small blobs (which hold ~58% of wrong energy) stayed invisible; a bare
loss-drop acceptance also let ~27% wrong moves through.  v2 inverts the loop:

For each SMALL BLOB (compact cluster, tracks excluded) currently placed somewhere:
  1. restrict the residual R = actual - clip(base) to the blob's OWN support channels
     (from its light template) across all its light-TPCs -> a per-tick trace where
     even a 3-5 MeV blob is locally significant;
  2. peak-find THAT trace -> candidate new t0s (deficits in exactly its channels);
  3. gates (v3, after the raw-excess gate of v2 proved neighbor-confounded — it moved
     correctly-placed blobs because OTHER clusters' excess bled into their channels):
     (a) target deficit >= deficit_k * sigma AND >= tgt_deficit_frac of the blob's own
         light amplitude;
     (b) target SHAPE: the deficit's channel profile must MATCH the blob's template
         profile (cosine >= cos_min) — not merely overlap it (kills "big foreign
         deficit hoovers up blobs");
     (c) source REMOVAL test: removing the blob's OWN light must IMPROVE the source
         fit by >= rem_frac of its own power — exact and per-blob (a correctly placed
         blob's removal always WORSENS the fit, even with the 0.78 ML systematic), so
         correct placements are protected regardless of what neighbors do;
  4. move to the candidate with the largest weighted-loss DROP (must be < 0).
Greedy best-first across blobs, base updated after each move, multiple passes.
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
from sat_clip import ceiling_rows, LEGACY_CLIP  # noqa: E402

GAIN, PEAK = 200.0, 105
CLIP = LEGACY_CLIP   # legacy constant; math goes through sat_clip
FWIN = 4


def _peaks(tr, thr, min_sep=12, n=12):
    out = []
    for t in range(2, len(tr) - 2):
        if tr[t] > thr and tr[t] >= tr[t - 1] and tr[t] > tr[t + 1] and tr[t] >= tr[t - 2]:
            if out and t - out[-1][0] < min_sep:
                if tr[t] > out[-1][1]:
                    out[-1] = (t, tr[t])
            else:
                out.append((t, tr[t]))
    return sorted(out, key=lambda p: -p[1])[:n]


def run_new_flash_rescue(ns: dict[str, Any], *, hit_tpc: np.ndarray,
                         x: np.ndarray, y: np.ndarray, z: np.ndarray,
                         blob_extent: float = 30.0, min_energy: float = 3.0,
                         min_sep: int = 20, deficit_k: float = 3.0,
                         tgt_deficit_frac: float = 0.3, cos_min: float = 0.80,
                         rem_frac: float = 0.10, src_empty_max: float = 1.0,
                         max_passes: int = 3,
                         truth_fn: Optional[Callable] = None, verbose: bool = False) -> dict:
    image_maps = ns["imageMaps"]
    actual = np.asarray(ns["fullLightWaveform"], np.float64)
    base = np.asarray(ns["baseImage"], np.float64)
    n_light, n_ch, n_tick = actual.shape
    labels = np.asarray(ns["labels_global"])
    ts = np.asarray(ns["hit_timestamps"], np.float64).copy()
    e_pos = np.clip(np.asarray(ns["Eset"], np.float64), 0.0, None)
    hit_tpc = np.asarray(hit_tpc); x = np.asarray(x); y = np.asarray(y); z = np.asarray(z)
    asg = np.isfinite(ts) & (ts >= 0)
    NOISE_CELL = GAIN * 100.0                       # per-cell variance floor

    # ---- collect small blobs: compact, assigned, with templates; per-blob support ----
    blobs: dict[int, dict] = {}
    for L in np.unique(labels[labels >= 0]):
        m = (labels == L) & asg
        if m.sum() < 3:
            continue
        ext = float(np.sqrt(np.ptp(x[m]) ** 2 + np.ptp(y[m]) ** 2 + np.ptp(z[m]) ** 2))
        if ext > blob_extent:                       # track -> not allowed
            continue
        E = float(e_pos[m].sum())
        if E < min_energy:
            continue
        t0s = ts[m]
        if float(np.ptp(t0s)) > 3.0:                # split-t0 blob: skip (ambiguous removal)
            continue
        sup = {}; T_amp = 0.0
        for c in range(n_light):
            k = (int(L), int(c))
            if k not in image_maps:
                continue
            P = np.asarray(image_maps[k], np.float64)
            ch = np.flatnonzero(P.max(axis=1) > 1.0)
            if ch.size == 0:
                continue
            sup[c] = (ch, P)
            pk = int(np.argmax(P[ch].sum(0)))
            lo, hi = max(0, pk - FWIN), min(n_tick, pk + FWIN + 1)
            T_amp += float(P[ch][:, lo:hi].sum())   # blob's own light amplitude (window sum)
        if not sup or T_amp <= 0:
            continue
        blobs[int(L)] = {"t0": float(np.median(t0s)), "E": E, "sup": sup, "T": T_amp}
    stats = {"n_blobs": len(blobs), "n_deficit_cand": 0, "n_src_gated": 0,
             "n_moves": 0, "e_moved": 0.0, "passes": 0, "n_good": 0, "n_bad": 0, "audit": []}
    if not blobs:
        return stats

    def win_sums(L, tick):
        """per-cell-clipped deficit/excess/actual sums over the blob's support window."""
        lo, hi = max(0, tick - FWIN), min(n_tick, tick + FWIN + 1)
        D = Ex = A = 0.0; ncell = 0
        for c, (ch, _P) in blobs[L]["sup"].items():
            Aw = actual[c][ch][:, lo:hi]
            Bw = np.clip(base[c][ch][:, lo:hi], 0.0, ceiling_rows(c, ch))
            R = Aw - Bw
            D += float(np.maximum(R, 0).sum()); Ex += float(np.maximum(-R, 0).sum())
            A += float(Aw.sum()); ncell += R.size
        sig = np.sqrt(max(GAIN * A, NOISE_CELL * ncell))
        return D, Ex, A, sig

    def tgt_shape_cos(L, tick):
        """cosine between the blob's template channel profile and the deficit channel
        profile at the target window (over its support channels, all its TPCs)."""
        lo, hi = max(0, tick - FWIN), min(n_tick, tick + FWIN + 1)
        tv, dv = [], []
        for c, (ch, P) in blobs[L]["sup"].items():
            pk = int(np.argmax(P[ch].sum(0)))
            plo, phi = max(0, pk - FWIN), min(n_tick, pk + FWIN + 1)
            tv.append(P[ch][:, plo:phi].sum(1))                     # template profile
            R = actual[c][ch][:, lo:hi] - np.clip(base[c][ch][:, lo:hi], 0.0, ceiling_rows(c, ch))
            dv.append(np.maximum(R, 0).sum(1))                      # deficit profile
        t = np.concatenate(tv); d = np.concatenate(dv)
        nt = np.linalg.norm(t); nd = np.linalg.norm(d)
        return float(t @ d / (nt * nd)) if nt > 0 and nd > 0 else 0.0

    def src_removal_gain(L, t0):
        """(loss_without_blob - loss_with_blob, own_power) at the source window.
        Negative gain = removal improves = blob is over-explaining = movable."""
        lo = max(0, int(round(t0)) + PEAK - 10)
        hi = min(n_tick, int(round(t0)) + PEAK + 45)
        gain = 0.0; own_pow = 0.0
        for c, (ch, P) in blobs[L]["sup"].items():
            own = timeinterpolation(P, shift=t0, baseline=0)[ch][:, lo:hi]
            B = base[c][ch][:, lo:hi]; A = actual[c][ch][:, lo:hi]
            w = 1.0 / (GAIN * np.maximum(A, 100.0))
            gain += float(np.sum(w * ((np.clip(B - own, 0.0, ceiling_rows(c, ch)) - A) ** 2
                                      - (np.clip(B, 0.0, ceiling_rows(c, ch)) - A) ** 2)))
            own_pow += float(np.sum(w * own ** 2))
        return gain, own_pow

    def move_delta(L, t_from, t_to):
        """total weighted-loss change of moving blob L (all its TPC templates)."""
        tot = 0.0
        for c, (ch, P) in blobs[L]["sup"].items():
            bw = base[c] - timeinterpolation(P, shift=t_from, baseline=0)
            new = np.clip(bw + timeinterpolation(P, shift=t_to, baseline=0), 0.0, ceiling_rows(c))
            cur = np.clip(base[c], 0.0, ceiling_rows(c))
            lo = max(0, int(round(min(t_from, t_to))) + PEAK - 30)
            hi = min(n_tick, int(round(max(t_from, t_to))) + PEAK + 60)
            A = actual[c][np.ix_(ch, np.arange(lo, hi))]
            w = 1.0 / (GAIN * np.maximum(A, 100.0))
            tot += float(np.sum(w * ((new[np.ix_(ch, np.arange(lo, hi))] - A) ** 2
                                     - (cur[np.ix_(ch, np.arange(lo, hi))] - A) ** 2)))
        return tot

    _tpc_truth = False
    if truth_fn is not None:
        try:
            truth_fn(-1, -1); _tpc_truth = True
        except TypeError:
            _tpc_truth = False
        except Exception:
            _tpc_truth = True

    used: set[int] = set()
    for _p in range(max_passes):
        stats["passes"] += 1
        proposals = []                       # (delta, L, t_from, t_to, diag)
        for L, b in blobs.items():
            if L in used:
                continue
            # candidate targets = ACTUAL-light peaks in the blob's own channels (real
            # flashes).  v4 peak-found the RESIDUAL trace, which favors quiet-region
            # wiggles (significance is easier at low sigma) and landed moves 10-90
            # ticks off the true flash; truth t0s live at actual flash peaks.
            tr = np.zeros(n_tick)
            for c, (ch, _P) in b["sup"].items():
                tr += actual[c][ch].sum(0)
            nch_tot = sum(len(ch) for ch, _ in b["sup"].values())
            thr = 3.0 * np.sqrt(NOISE_CELL * max(nch_tot, 1))          # per-tick noise floor
            cur_tick = int(round(b["t0"])) + PEAK
            # source gates (per blob).  (1) EMPTINESS — this pass only rescues blobs
            # DISCOVERING a new flash: a correctly placed blob has actual light ~1.3x
            # its own amplitude at its source (0.78 ML systematic), so requiring the
            # source to be light-poor protects correct placements even when neighbors
            # over-predict (which confounds any residual-based test there).
            _Ds, _Exs, A_src, _sig_s = win_sums(L, cur_tick)
            if A_src > src_empty_max * b["T"]:
                stats["n_src_gated"] += 1
                continue
            # (2) REMOVAL — the blob's own light must be over-explaining where it sits
            rem, own_pow = src_removal_gain(L, b["t0"])
            if rem > -rem_frac * max(own_pow, 1e-9):
                stats["n_src_gated"] += 1
                continue
            best = None
            for tick, _h in _peaks(tr, thr):
                if abs(tick - cur_tick) < min_sep:
                    continue
                D, _Ex, _A, sig = win_sums(L, tick)
                if D < deficit_k * sig or D < tgt_deficit_frac * b["T"]:
                    continue                                           # not a clear deficit for THIS blob
                stats["n_deficit_cand"] += 1
                cs = tgt_shape_cos(L, tick)
                if cs < cos_min:
                    continue                                           # deficit is not THIS blob's footprint
                t_new = float(tick - PEAK)
                d = move_delta(L, b["t0"], t_new)
                if d < 0 and (best is None or d < best[0]):
                    best = (d, t_new, D / max(sig, 1e-9), cs)
            if best is not None:
                proposals.append((best[0], L, b["t0"], best[1], best[2], best[3]))
        if not proposals:
            break
        proposals.sort(key=lambda r: r[0])
        applied = 0
        for delta, L, t_from, t_to, dsig, tcos in proposals:
            if L in used:
                continue
            for c, (ch, P) in blobs[L]["sup"].items():
                base[c] += (-timeinterpolation(P, shift=t_from, baseline=0)
                            + timeinterpolation(P, shift=t_to, baseline=0))
            ts[(labels == L) & asg] = t_to
            blobs[L]["t0"] = t_to
            rec = {"L": int(L), "from": round(t_from, 1), "to": round(t_to, 1),
                   "E": round(blobs[L]["E"], 1), "deficit_sig": round(dsig, 1),
                   "tgt_cos": round(tcos, 2), "delta_loss": round(delta, 1)}
            if truth_fn is not None:
                trv = truth_fn(int(L), -1) if _tpc_truth else truth_fn(int(L))
                if trv is not None:
                    rec["truth"] = round(trv, 1)
                    rec["good"] = bool(abs(t_to - trv) < abs(t_from - trv))
            stats["audit"].append(rec); used.add(L)
            applied += 1; stats["n_moves"] += 1; stats["e_moved"] += blobs[L]["E"]
        if verbose:
            print(f"[newflash2] pass {stats['passes']}: {applied} moves "
                  f"({len(proposals)} proposals)", flush=True)
        if applied == 0:
            break

    ns["hit_timestamps"] = ts.astype(np.float32)
    ns["baseImage"] = base.astype(np.float32)
    stats["e_moved"] = round(stats["e_moved"], 1)
    stats["n_good"] = sum(1 for r in stats["audit"] if r.get("good") is True)
    stats["n_bad"] = sum(1 for r in stats["audit"] if r.get("good") is False)
    return stats
