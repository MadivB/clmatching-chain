"""Per-cluster confidence v2 extraction — improved components using the per-TPC lessons:
  z_min_fl : covariance row evaluated ONLY at flash peaks of the support-restricted
             actual light (t0-list lesson: kills look-elsewhere dilution)
  displaced: AMOUNT-based matched min(excess@placed, deficit@flash) per channel,
             3-sigma 'clearly' thresholded, ADC, normalized by cluster energy
  site_bright, nflash_sup, N : complexity regressors (chi2_loc kept as regressor too)
plus the v1 components (z_min all-j', z_own) for comparison.
"""
import glob, os, pickle, time, numpy as np

HERE = os.path.abspath(os.path.dirname(__file__))
DUMPS = os.path.join(HERE, "v030v3_out", "dumps")
CLIP, GAIN, PEAK, CUT = 60780.0, 200.0, 105, 10
FWIN, SEP, DIL = 4, 20, 3
YLO, YHI, NROW = -215.5, 81.7, 60
W = 2 * FWIN + 1


def wsum(arr):
    cs = np.cumsum(arr, axis=1, dtype=np.float32)
    csp = np.pad(cs, ((0, 0), (1, 0)))
    n = arr.shape[1]; t = np.arange(n)
    lo = np.maximum(t - FWIN, 0); hi = np.minimum(t + FWIN, n - 1)
    return csp[:, hi + 1] - csp[:, lo]


def peaks_vec(tr, thr, min_sep=12, n=12):
    out = []
    for t in range(2, len(tr) - 2):
        if tr[t] > thr and tr[t] >= tr[t - 1] and tr[t] > tr[t + 1] and tr[t] >= tr[t - 2]:
            if out and t - out[-1][0] < min_sep:
                if tr[t] > out[-1][1]:
                    out[-1] = (t, tr[t])
            else:
                out.append((t, tr[t]))
    return [p[0] for p in sorted(out, key=lambda p: -p[1])[:n]]


rows = []; t_comp = 0.0; nev = 0
for fp in sorted(glob.glob(os.path.join(DUMPS, "*.npz"))):
    d = np.load(fp, allow_pickle=True)
    ft = os.path.basename(fp).split("_ev")[0]; ev = int(d["event"])
    lab = np.asarray(d["labels"]); tpc = np.asarray(d["tpc"]); e = np.asarray(d["e"], np.float64)
    yv = np.asarray(d["y"], np.float64)
    tsf = np.asarray(d["ts_final"], np.float64); tru = np.asarray(d["truth_t0"], np.float64)
    full = np.asarray(d["full_light"], np.float32); base = np.asarray(d["base_image"], np.float32)
    n_light, n_ch, n_tick = full.shape
    t0c = time.time()

    RW = {}; RA = {}; SA = {}                      # standardized | raw-ADC resid | actual win-sums
    for c in np.unique(tpc[tpc < n_light]):
        c = int(c)
        A = full[c].astype(np.float32); P = np.minimum(base[c].astype(np.float32), np.float32(CLIP))
        okm = (A < np.float32(0.98 * CLIP)).astype(np.float32)
        Pm = P * okm
        alpha = float((Pm * A).sum() / max(float((Pm * P).sum()), 1e-9))
        r = (A - np.float32(alpha) * P)
        RA[c] = wsum(r * okm)                                       # ADC windowed residual
        SA[c] = wsum(A)                                             # actual windowed
        RW[c] = wsum(r * okm / np.sqrt(np.float32(GAIN) * np.maximum(A, np.float32(100.0)))) / np.float32(np.sqrt(W))

    # group (label, tpc) via argsort
    okh = (lab >= 0) & (tpc < n_light) & np.isfinite(tsf)
    key = lab[okh].astype(np.int64) * 128 + tpc[okh].astype(np.int64)
    idx = np.flatnonzero(okh)[np.argsort(key, kind="stable")]
    ks = lab[idx].astype(np.int64) * 128 + tpc[idx].astype(np.int64)
    bnd = np.r_[0, np.flatnonzero(np.diff(ks)) + 1, len(ks)]
    parts = {}
    for i in range(len(bnd) - 1):
        sl = idx[bnd[i]:bnd[i + 1]]
        if len(sl) < 2:
            continue
        L = int(lab[sl[0]]); c = int(tpc[sl[0]])
        r0 = int(np.clip((yv[sl].min() - YLO) / (YHI - YLO) * NROW, 0, NROW - 1))
        r1 = int(np.clip((yv[sl].max() - YLO) / (YHI - YLO) * NROW, 0, NROW - 1))
        lo, hi = max(0, r0 - DIL), min(NROW - 1, r1 + DIL)
        ch = np.r_[np.arange(lo, hi + 1), NROW + np.arange(lo, hi + 1)]
        parts.setdefault(L, []).append((c, float(np.median(tsf[sl])), ch, float(e[sl].sum())))

    # truth per label (energy fractions) via same grouping on label only
    for L, pl in parts.items():
        mL = lab == L
        okL = mL & np.isfinite(tru)
        eL = float(e[mL].sum())
        if not okL.any() or eL < 1.0:
            continue
        e_wr = float(e[okL & np.isfinite(tsf) & (np.abs(tsf - tru) > CUT)].sum())
        wf = e_wr / max(float(e[okL].sum()), 1e-9)
        curve = np.zeros(n_tick); own = []; N = 0
        chi2 = 0.0; nloc = 0; bright = 0.0
        disp_best = 0.0; zfl_min = 0.0; nfl = 0
        for c, t0, ch, ec in pl:
            tick = int(round(t0)) + PEAK
            if not (0 <= tick < n_tick):
                continue
            v0 = RW[c][ch, tick]
            curve += v0 @ RW[c][ch]
            own.append(v0); N += len(ch)
            chi2 += float((v0 ** 2).sum()); nloc += len(ch)
            bright += float(SA[c][ch, tick].sum())
            # flash peaks of the support-restricted actual trace
            tra = SA[c][ch].sum(0)
            pks = [p for p in peaks_vec(tra, 0.05 * tra.max()) if abs(p - tick) >= SEP]
            nfl += len(pks)
            if pks:
                # standardized covariance at flash ticks only
                zfl = min(float(v0 @ RW[c][ch, p]) for p in pks)
                zfl_min += zfl                                   # sum across parts (stacked later)
                # AMOUNT displaced: matched min(excess@own, deficit@flash) per channel, 3sigma
                sig_own = np.sqrt(GAIN * np.maximum(SA[c][ch, tick], 900.0))
                exc = np.maximum(-RA[c][ch, tick], 0.0); exc[exc < 3 * sig_own] = 0.0
                for p in pks:
                    sig_t = np.sqrt(GAIN * np.maximum(SA[c][ch, p], 900.0))
                    dfc = np.maximum(RA[c][ch, p], 0.0); dfc[dfc < 3 * sig_t] = 0.0
                    disp_best = max(disp_best, float(np.minimum(exc, dfc).sum()))
        if N == 0:
            continue
        own = np.concatenate(own)
        curve /= np.sqrt(N)
        for c, t0, ch, ec in pl:
            tick = int(round(t0)) + PEAK
            curve[max(0, tick - SEP):tick + SEP + 1] = 0.0
        rows.append(dict(file=ft, event=ev, L=int(L), E=eL, n_tpc=len(pl), wf=wf,
                         z_min=float(curve.min()), z_own=float(own.sum() / np.sqrt(N)),
                         z_min_fl=zfl_min / np.sqrt(max(N, 1)) * np.sqrt(N / max(len(own), 1)) if N else 0.0,
                         zfl_raw=zfl_min, displaced=disp_best / max(eL, 1e-9),
                         chi2_loc=chi2 / max(nloc, 1), bright=bright / max(eL, 1e-9),
                         N=N, nfl=nfl))
    t_comp += time.time() - t0c; nev += 1
    d.close(); del full, base
    print(f"  [{nev}/39] {len(rows)} clusters | {1000*t_comp/nev:.0f} ms/ev", flush=True)

pickle.dump(rows, open(os.path.join(HERE, "cluster_conf_cache2.pkl"), "wb"))
print(f"WROTE cluster_conf_cache2.pkl: {len(rows)} clusters, {1000*t_comp/max(nev,1):.0f} ms/event")
print("DONE", flush=True)
