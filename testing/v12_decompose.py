"""v1.2 phase 0: decompose the per-hit timing residual of the v1.1 chain into
(a) family offset vs within-family spread, (b) the truth propagation term
(apex clock: t_dep(h) = t_apex + |r_h - r_apex| / c), (c) the light time-of-
flight of the family to its support channels.  Truth is used ONLY to measure
what a physics model could recover.   python v12_decompose.py <dump_dir> <flow_file>
"""
import glob, os, sys
import numpy as np
HERE = os.path.abspath(os.path.dirname(__file__))
BASE = os.path.join(HERE, "..", "CLMatching_AlphaRelease")
for p in (BASE, os.path.join(BASE, "M5p1"), os.path.join(BASE, "NewMLSection"), HERE):
    if p not in sys.path:
        sys.path.insert(0, os.path.abspath(p))
from subtick_refine import build_sipm_positions

C = 29.9792458      # cm/ns
N_LAR = 2.237       # c / c_LAr, c_LAr = 13.4 cm/ns (2x2 convention)
NS_PER_TICK = 16.0
CUT = 10.0


def wstd(v, w):
    m = np.average(v, weights=w)
    return float(np.sqrt(np.average((v - m) ** 2, weights=w))), float(m)


def robust(v, w=None):
    """weighted robust sigma = 1.4826 * weighted MAD"""
    if w is None:
        w = np.ones_like(v)
    o = np.argsort(v); cw = np.cumsum(w[o]); med = v[o][np.searchsorted(cw, 0.5 * cw[-1])]
    d = np.abs(v - med); o2 = np.argsort(d); cw2 = np.cumsum(w[o2])
    return 1.4826 * float(d[o2][np.searchsorted(cw2, 0.5 * cw2[-1])])


def main(dump_dir, flow):
    sipm = build_sipm_positions(flow)
    rows = []           # per family
    hit_res, hit_res_prop, hit_w = [], [], []
    for ff in sorted(glob.glob(f"{dump_dir}/full_dumps/*.npz")):
        if os.path.basename(flow)[-20:-13] not in os.path.basename(ff):
            pass
        z = np.load(ff, allow_pickle=True)
        e = np.clip(z["e"], 0, None).astype(np.float64)
        ts = z["ts_final"].astype(np.float64); tru = z["truth_t0"].astype(np.float64)
        labs = z["labels"].astype(int); tpc = z["tpc"].astype(int)
        xyz = np.stack([z["x"], z["y"], z["z"]], 1).astype(np.float64)
        ok = np.isfinite(ts) & (ts >= 0) & np.isfinite(tru) & (tru >= 0)
        al, at, ar = z["amp_lab"].astype(int), z["amp_tpc"].astype(int), z["amp_row"].astype(np.float64)
        veto = z["veto_mask"]
        amp = {}
        for i in range(len(al)):
            amp[(int(al[i]), int(at[i]))] = ar[i]
        # families = exact final t0 value (post stage E) within one light TPC
        vals = np.unique(ts[ok])
        for c in np.unique(tpc[ok]):
            if c not in sipm:
                continue
            mc = ok & (tpc == c)
            for tv in np.unique(ts[mc]):
                m = mc & (ts == tv)
                E = e[m].sum()
                if E < 20 or m.sum() < 20:
                    continue
                r = (ts[m] - tru[m]) * NS_PER_TICK          # ns
                good = np.abs(r) <= CUT * NS_PER_TICK
                if e[m][good].sum() < 0.9 * E:
                    continue                                 # keep clean families
                idx = np.flatnonzero(m)[good]
                w = e[idx]; rr = r[good]
                # (a) offset and within spread
                s_in, off = wstd(rr, w)
                # (b) apex clock in truth: apex = E-weighted position of the earliest 2% of truth time
                tt = tru[idx] * NS_PER_TICK
                o = np.argsort(tt); n2 = max(3, int(0.02 * o.size))
                apex = np.average(xyz[idx][o[:n2]], axis=0, weights=w[o[:n2]])
                t_apex = float(np.average(tt[o[:n2]], weights=w[o[:n2]]))
                d_over_c = np.linalg.norm(xyz[idx] - apex, axis=1) / C          # ns
                dt_truth = tt - t_apex
                # slope of truth deposition time vs distance/c (1 = apex clock exact)
                A = np.vstack([d_over_c, np.ones_like(d_over_c)]).T
                slope = float(np.linalg.lstsq(A * np.sqrt(w)[:, None], dt_truth * np.sqrt(w), rcond=None)[0][0]) if np.ptp(d_over_c) > 1e-6 else np.nan
                s_prop_truth, _ = wstd(dt_truth - d_over_c, w)                 # truth spread not explained by apex clock
                s_prop_truth_raw, _ = wstd(dt_truth, w)                        # truth within-family spread
                # what the chain would gain: per-hit time = t_fam + d/c instead of t_fam
                rr_prop = rr + (d_over_c - float(np.average(d_over_c, weights=w)))   # per-hit time = t_fam + d/c (apex clock), mean-centered
                s_in_prop, _ = wstd(rr_prop, w)
                # (c) light ToF: family support channels in this light TPC
                labset = np.unique(labs[idx]); pat = np.zeros(120)
                for L in labset:
                    if (int(L), int(c)) in amp:
                        pat += amp[(int(L), int(c))]
                pat[veto[c]] = 0
                if pat.max() <= 0:
                    tof_mean = np.nan; n_sup = 0
                else:
                    order = np.argsort(pat)[::-1]; cs = np.cumsum(pat[order]); sup = order[: int(np.searchsorted(cs, 0.9 * cs[-1])) + 1]
                    n_sup = sup.size
                    P = sipm[c][sup]                                          # (n_sup, 3)
                    # onset per channel = min over hits of (|r_h - apex|/c + n |r_h - p_i|/c)
                    hs = idx if idx.size <= 4000 else idx[np.linspace(0, idx.size - 1, 4000).astype(int)]
                    dprop = np.linalg.norm(xyz[hs] - apex, axis=1) / C
                    dd = np.linalg.norm(xyz[hs][:, None, :] - P[None, :, :], axis=2) * N_LAR / C   # (nh, n_sup)
                    onset = (dprop[:, None] + dd).min(axis=0)                  # ns per channel
                    # the perceiver/template fit sees the bulk of the light, not only the onset:
                    # amplitude-weighted mean arrival over hits and channels as a second proxy
                    mean_arr = np.average((dprop[:, None] + dd), axis=0, weights=e[hs])
                    tof_mean = float(np.average(onset, weights=pat[sup]))
                    tof_bulk = float(np.average(mean_arr, weights=pat[sup]))
                rows.append((E, off, s_in, s_in_prop, slope, s_prop_truth_raw, s_prop_truth, tof_mean, tof_bulk if pat.max() > 0 else np.nan, n_sup, m.sum(), float(np.ptp(d_over_c))))
                hit_res.append(rr); hit_res_prop.append(rr_prop); hit_w.append(w)
    R = np.array(rows, float)
    hr = np.concatenate(hit_res); hp = np.concatenate(hit_res_prop); hw = np.concatenate(hit_w)
    print(f"{len(R)} clean families (>=20 MeV, >=90% correct), {hr.size} hits")
    print(f"per-hit residual (E-wt): std {wstd(hr, hw)[0]:.2f} ns, robust {robust(hr, hw):.2f} ns | after removing the apex-clock propagation pattern within families: std {wstd(hp, hw)[0]:.2f}, robust {robust(hp, hw):.2f} ns")
    w = R[:, 0]
    print(f"family OFFSET spread (E-wt std over families): {wstd(R[:,1], w)[0]:.2f} ns (mean offset {np.average(R[:,1], weights=w):+.2f} ns)")
    print(f"WITHIN-family spread (E-wt mean of per-family std): {np.average(R[:,2], weights=w):.2f} ns -> after propagation correction {np.average(R[:,3], weights=w):.2f} ns")
    print(f"truth check of the apex clock: slope(t_dep vs d/c) median {np.nanmedian(R[:,4]):.2f} (1 = exact); truth within-family spread {np.average(R[:,5], weights=w):.2f} ns -> unexplained by apex clock {np.average(R[:,6], weights=w):.2f} ns; family extent d/c p50 {np.median(R[:,11]):.1f} ns")
    ok = np.isfinite(R[:, 7])
    if ok.any():
        A = np.vstack([R[ok, 7], np.ones(ok.sum())]).T
        coef = np.linalg.lstsq(A * np.sqrt(w[ok])[:, None], R[ok, 1] * np.sqrt(w[ok]), rcond=None)[0]
        resid = R[ok, 1] - (coef[0] * R[ok, 7] + coef[1])
        A2 = np.vstack([R[ok, 8], np.ones(ok.sum())]).T
        coef2 = np.linalg.lstsq(A2 * np.sqrt(w[ok])[:, None], R[ok, 1] * np.sqrt(w[ok]), rcond=None)[0]
        resid2 = R[ok, 1] - (coef2[0] * R[ok, 8] + coef2[1])
        print(f"light ToF vs family offset: onset-ToF (amp-wt over support, {np.median(R[ok,9]):.0f} ch median): slope {coef[0]:.2f}, offset spread {wstd(R[ok,1], w[ok])[0]:.2f} -> {wstd(resid, w[ok])[0]:.2f} ns after linear removal | bulk-arrival proxy: slope {coef2[0]:.2f}, -> {wstd(resid2, w[ok])[0]:.2f} ns")
        print(f"  ToF ranges: onset-ToF p10/p50/p90 {np.percentile(R[ok,7],10):.1f}/{np.percentile(R[ok,7],50):.1f}/{np.percentile(R[ok,7],90):.1f} ns; bulk {np.percentile(R[ok,8],10):.1f}/{np.percentile(R[ok,8],50):.1f}/{np.percentile(R[ok,8],90):.1f} ns")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
