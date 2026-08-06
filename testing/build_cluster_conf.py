"""Per-CLUSTER confidence, extraction pass (user request 2026-07-10).

Per event:
  - per TPC: standardized residual  r_hat = (A - alpha*P) / sqrt(GAIN*max(A,100)),
    saturated cells zeroed; windowed (+-FWIN, unit-variance-normalized) via cumsum.
  - per cluster PART (label, TPC): support channels from the HIT-y -> channel-row map
    (works for every cluster; no template needed), placed tick j0 = median reco t0;
    then the covariance row anchored at j0 over its support:
        m(j') = (1/N) sum_{i in sup} rW_i(j0) * rW_i(j')
    plus own-site excess/deficit and local chi2.
  - multi-TPC clusters: parts are STACKED after standardization (time axis shared),
    so normalization is just the total cell count — same footing as single-TPC.
Saves a per-cluster feature table (cluster_conf_cache.pkl) for fast variant scans.
"""
import glob, os, pickle, time, numpy as np

HERE = os.path.abspath(os.path.dirname(__file__))
DUMPS = os.path.join(HERE, "v030v3_out", "dumps")
CLIP, GAIN, PEAK, CUT = 60780.0, 200.0, 105, 10
FWIN = 4                                  # +- ticks per window
SEP = 20                                  # exclusion around own t0 for the j' scan
DIL = 3                                   # support dilation in channel rows
YLO, YHI = -215.5, 81.7
NROW = 60


def row_of(yv):
    return np.clip(((yv - YLO) / (YHI - YLO) * NROW).astype(int), 0, NROW - 1)


rows = []
t_compute = 0.0; n_events = 0
for fp in sorted(glob.glob(os.path.join(DUMPS, "*.npz"))):
    d = np.load(fp, allow_pickle=True)
    ft = os.path.basename(fp).split("_ev")[0]; ev = int(d["event"])
    lab = np.asarray(d["labels"]); tpc = np.asarray(d["tpc"]); e = np.asarray(d["e"], np.float64)
    yv = np.asarray(d["y"], np.float64)
    tsf = np.asarray(d["ts_final"], np.float64); tru = np.asarray(d["truth_t0"], np.float64)
    full = np.asarray(d["full_light"], np.float32); base = np.asarray(d["base_image"], np.float32)
    n_light, n_ch, n_tick = full.shape
    t0c = time.time()

    # ---- per-TPC standardized, windowed residual (built once) ----
    RW = {}                                   # tpc -> (n_ch, n_tick) unit-variance windowed residual
    for c in np.unique(tpc[tpc < n_light]):
        c = int(c)
        A = full[c].astype(np.float64); P = np.minimum(base[c].astype(np.float64), CLIP)
        sat = A >= 0.98 * CLIP; m = ~sat
        a = A[m]; p = P[m]; alpha = float((p @ a) / max(p @ p, 1e-9))
        r = (A - alpha * P) / np.sqrt(GAIN * np.maximum(A, 100.0))
        r[sat] = 0.0
        cs = np.cumsum(np.pad(r, ((0, 0), (1, 0))), axis=1)
        W = 2 * FWIN + 1
        rw = (cs[:, W:] - cs[:, :-W]) / np.sqrt(W)          # window sum, re-unit-variance
        RW[c] = np.pad(rw, ((0, 0), (FWIN, FWIN + n_tick - rw.shape[1] - FWIN)))[:, :n_tick] \
            if rw.shape[1] != n_tick else rw
        if RW[c].shape[1] < n_tick:                          # safety pad
            RW[c] = np.pad(RW[c], ((0, 0), (0, n_tick - RW[c].shape[1])))

    # ---- per-cluster parts ----
    for L in np.unique(lab[lab >= 0]):
        mL = lab == L
        okL = mL & np.isfinite(tru)
        eL = float(e[mL].sum())
        if mL.sum() < 3 or eL < 1.0 or not okL.any():
            continue
        e_wrong = float(e[okL & np.isfinite(tsf) & (np.abs(tsf - tru) > CUT)].sum())
        e_tru = float(e[okL].sum())
        parts = []
        for c in np.unique(tpc[mL]):
            c = int(c)
            if c not in RW:
                continue
            mp = mL & (tpc == c)
            if mp.sum() < 2:
                continue
            t0 = float(np.median(tsf[mp][np.isfinite(tsf[mp])])) if np.isfinite(tsf[mp]).any() else None
            if t0 is None:
                continue
            r0, r1 = row_of(np.array([yv[mp].min()]))[0], row_of(np.array([yv[mp].max()]))[0]
            lo, hi = max(0, r0 - DIL), min(NROW - 1, r1 + DIL)
            ch = np.r_[np.arange(lo, hi + 1), NROW + np.arange(lo, hi + 1)]      # both walls
            parts.append((c, t0, ch, float(e[mp].sum())))
        if not parts:
            continue
        # ---- stacked covariance row over all parts (unit-var cells -> count-normalized) ----
        curve = np.zeros(n_tick); own = []; N = 0
        chi2_loc = 0.0; nloc = 0
        split = float(np.ptp([p[1] for p in parts])) > 3.0
        for c, t0, ch, ec in parts:
            tick = int(round(t0)) + PEAK
            if not (0 <= tick < n_tick):
                continue
            v0 = RW[c][ch, tick]                            # (nch,) residual at own site
            curve += v0 @ RW[c][ch]                          # sum_i r_i(j0) r_i(j')
            own.append(v0); N += len(ch)
            wlo, whi = max(0, tick - FWIN), min(n_tick, tick + FWIN + 1)
            chi2_loc += float((RW[c][ch, wlo:whi] ** 2).sum()); nloc += len(ch) * (whi - wlo)
        if N == 0:
            continue
        own = np.concatenate(own)
        z_own = float(own.sum() / np.sqrt(N))                # >0 deficit / <0 excess at own site
        curve /= np.sqrt(N)                                  # unit-normal null per j'
        # mask own neighborhood
        for c, t0, ch, ec in parts:
            tick = int(round(t0)) + PEAK
            curve[max(0, tick - SEP):tick + SEP + 1] = 0.0
        z_min = float(curve.min())                           # most-negative anticorrelation
        z_maxabs = float(np.abs(curve).max())
        s_neg = float(np.minimum(curve, 0.0).sum())          # summed negative mass
        rows.append(dict(file=ft, event=ev, L=int(L), E=eL, n_tpc=len(parts), split=split,
                         wf=e_wrong / max(e_tru, 1e-9),
                         z_min=z_min, z_maxabs=z_maxabs, s_neg=s_neg, z_own=z_own,
                         chi2_loc=chi2_loc / max(nloc, 1)))
    t_compute += time.time() - t0c; n_events += 1
    d.close(); del full, base
    print(f"  [{n_events}/39] {ft} ev{ev}: {len(rows)} clusters | compute {1000*(t_compute/n_events):.0f} ms/ev",
          flush=True)

pickle.dump(rows, open(os.path.join(HERE, "cluster_conf_cache.pkl"), "wb"))
print(f"\nWROTE cluster_conf_cache.pkl: {len(rows)} clusters, "
      f"compute {1000*t_compute/max(n_events,1):.0f} ms/event (excl. I/O)")
print("DONE", flush=True)
