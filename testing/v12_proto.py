"""v1.2 precision-timing PROTOTYPE (offline, on chain dumps).

For every stage-E family (hits sharing one final t0, split by light TPC):
  * support channels = the channels holding 90 % of the family's own predicted
    light in that TPC (not a fixed count);
  * per-channel DELAY-AWARE model: the family's arrival-time density on the
    channel, rho_i(tau) = sum_h q_h w_ih delta(tau - |r_h - apex|/c - n d_ih/c)
    (w_ih ~ 1/d^2 visibility proxy, 0.5 ns bins), convolved with the single
    average pulse template -> the per-channel waveform shape including onset
    delay and intra-family spread (2x2 pushforward model, discrete hits);
  * pile-up: the fit runs on the RESIDUAL actual - (other families' model at
    their refined times) and channels are down-weighted by their pile-up
    fraction;
  * per-channel time y_i from a 0.25-tick scan + quadratic (amplitude
    profiled out), family apex time = weighted median over channels
    (robust to pile-up outliers) after the slew term beta*log10(A);
  * apex choice = the candidate (two principal-axis ends, centroid) with the
    lowest total per-channel loss; per-hit deposition time = t0 + |r_h-apex|/c.
Self-consistent slew calibration: pass 1 (beta=0) measures the per-channel
residual vs log10 A across all families; pass 2 applies it.  Truth is used
only in the final evaluation (per-hit / per-family sigma vs stage E).
python v12_proto.py <dump_dir> <flow_file> [max_events]
"""
import glob, os, sys, time
import numpy as np
HERE = os.path.abspath(os.path.dirname(__file__))
BASE = os.path.join(HERE, "..", "CLMatching_AlphaRelease")
for p in (BASE, os.path.join(BASE, "M5p1"), os.path.join(BASE, "NewMLSection"), HERE):
    if p not in sys.path:
        sys.path.insert(0, os.path.abspath(p))
from subtick_refine import build_sipm_positions

C, N_LAR, NS, PEAK, CUT, GAIN = 29.9792458, 2.237, 16.0, 105, 10.0, 200.0
TMPL = np.load(os.path.join(BASE, "assets", "avg_pulse.npy")).astype(np.float64)
TMPL = TMPL / TMPL.max()
FINE = 0.5                                    # ns
NFINE = int(TMPL.size * NS / FINE)            # 32000
TMPL_F = np.interp(np.arange(NFINE) * FINE / NS, np.arange(TMPL.size), TMPL)
ARR_LO, ARR_HI = -16.0, 144.0                 # ns: arrival-density window relative to the apex time
NB = int((ARR_HI - ARR_LO) / FINE)
WIN_LO, WIN_HI = -20, 40                      # ticks around the predicted peak used in the fit
SCAN = np.arange(-16, 17) * 0.25              # ticks
MAX_HITS = 2500


def _quad(L, i0):
    if i0 < 4 or i0 > L.size - 5:
        return np.nan, np.nan
    x = np.arange(-4, 5) * 0.25; y = L[i0 - 4: i0 + 5]
    A = np.vstack([x * x, x, np.ones_like(x)]).T
    a, b, c0 = np.linalg.lstsq(A, y, rcond=None)[0]
    if a <= 0:
        return np.nan, np.nan
    xs = -b / (2 * a)
    if abs(xs) > 0.75:
        return np.nan, np.nan
    return SCAN[i0] + xs, a


def wmedian(v, w):
    o = np.argsort(v); cw = np.cumsum(w[o])
    return float(v[o][np.searchsorted(cw, 0.5 * cw[-1])])


def family_fit(hits_xyz, hits_e, apex, chans, sipm_pos, actual, others, w_tick, t_chain, pile, beta, amp_pat):
    """Per-channel delay-aware fits for one family in one light TPC.
    Returns (t0_apex [ticks], loss_total, per-channel records)."""
    dprop = np.linalg.norm(hits_xyz - apex, axis=1) / C                      # ns
    P = sipm_pos[chans]
    dd = np.linalg.norm(hits_xyz[:, None, :] - P[None, :, :], axis=2)         # cm (nh, nc)
    tau = dprop[:, None] + dd * N_LAR / C                                     # ns arrival rel. apex time
    wv = (hits_e[:, None] / np.maximum(dd, 5.0) ** 2)                         # visibility proxy
    col0 = int(round(t_chain))                                                # chain tick
    ks = np.arange(col0 + PEAK + WIN_LO, col0 + PEAK + WIN_HI + 1)            # fit window ticks
    ks = ks[(ks >= 0) & (ks < actual.shape[1])]
    recs = []
    tot = 0.0
    # arrival densities for all channels at once -> (nc, NB)
    b = np.clip(((tau - ARR_LO) / FINE).astype(int), 0, NB - 1)
    rho = np.zeros((len(chans), NB))
    for j in range(len(chans)):
        np.add.at(rho[j], b[:, j], wv[:, j])
    # fine convolution with the template (rho lives on [ARR_LO, ARR_HI), template on [0, NFINE*FINE))
    nfft = 1 << int(np.ceil(np.log2(NB + NFINE)))
    conv = np.fft.irfft(np.fft.rfft(rho, nfft, axis=1) * np.fft.rfft(TMPL_F, nfft)[None, :], nfft, axis=1)[:, : NB + NFINE - 1]
    # conv index m corresponds to fine time ARR_LO + m*FINE relative to the apex time (template start at arrival)
    for j, ch in enumerate(chans):
        R = actual[ch, ks] - others[ch, ks]
        w = w_tick[ch, ks]
        Ls = np.empty(SCAN.size); amps = np.empty(SCAN.size)
        for si, s in enumerate(SCAN):
            # model at tick k for apex time (t_chain + s) ticks: fine time = (k - t_chain - s)*NS - ARR_LO
            m_idx = ((ks - t_chain - s) * NS - ARR_LO) / FINE
            M = np.interp(m_idx, np.arange(conv.shape[1]), conv[j], left=0.0, right=0.0)
            den = float(np.sum(w * M * M))
            a = float(np.sum(w * R * M)) / den if den > 0 else 0.0
            a = max(a, 0.0)
            amps[si] = a
            Ls[si] = float(np.sum(w * (R - a * M) ** 2))
        i0 = int(np.argmin(Ls))
        y, curv = _quad(Ls, i0)
        tot += Ls[i0]
        if not np.isfinite(y):
            continue
        A = amps[i0] * conv[j].max()
        if A < 300:
            continue
        recs.append((int(ch), y, curv, A, pile[j]))
    if len(recs) < 3:
        return np.nan, tot, recs
    r = np.array(recs, dtype=float)
    y = r[:, 1] - (beta * np.log10(np.maximum(r[:, 3], 1.0))) / NS          # slew correction [ticks]
    wgt = np.sqrt(r[:, 3]) * np.clip(1.0 - r[:, 4], 0.05, 1.0)
    return t_chain + wmedian(y, wgt), tot, recs      # y are shifts relative to the chain time


def run(dump_dir, flow, max_events, beta):
    sipm = build_sipm_positions(flow)
    files = sorted(glob.glob(f"{dump_dir}/full_dumps/*.npz"))[:max_events]
    slew_rows = []
    res_E, res_new, res_w = [], [], []
    fam_E, fam_new, fam_w = [], [], []
    n_fam = n_fit = 0
    t_start = time.time()
    for ff in files:
        z = np.load(ff, allow_pickle=True)
        e = np.clip(z["e"], 0, None).astype(np.float64)
        ts = z["ts_final"].astype(np.float64); tru = z["truth_t0"].astype(np.float64)
        labs = z["labels"].astype(int); tpc = z["tpc"].astype(int)
        xyz = np.stack([z["x"], z["y"], z["z"]], 1).astype(np.float64)
        ok = np.isfinite(ts) & (ts >= 0)
        full = z["full_light"].astype(np.float64); veto = z["veto_mask"]
        n_light, n_ch, n_ticks = full.shape
        w_tick = 1.0 / (GAIN * np.maximum(full, 100.0))
        al, at, ar = z["amp_lab"].astype(int), z["amp_tpc"].astype(int), z["amp_row"].astype(np.float64)
        amp = {(int(al[i]), int(at[i])): ar[i] for i in range(len(al))}
        # label -> refined time (median of its hits)
        lab_t = {}
        for L in np.unique(labs[labs >= 0]):
            v = ts[labs == L]; v = v[np.isfinite(v) & (v >= 0)]
            if v.size:
                lab_t[int(L)] = float(np.median(v))
        # re-render every family's light at its refined time per TPC: model[c] = sum_L amp_L,c x T(t - t_L)
        tick_idx = np.arange(n_ticks)
        model = np.zeros_like(full)
        for (L, c), a_vec in amp.items():
            if L not in lab_t or not (0 <= c < n_light):
                continue
            sh = np.interp(tick_idx - lab_t[L], np.arange(TMPL.size), TMPL, left=0.0, right=0.0)
            model[c] += a_vec[:, None] * sh[None, :]
        new_t = ts.copy()
        for c in np.unique(tpc[ok]):
            if c not in sipm:
                continue
            mc = ok & (tpc == c)
            for tv in np.unique(ts[mc]):
                m = mc & (ts == tv)
                E = e[m].sum()
                if E < 5 or m.sum() < 10:
                    continue
                n_fam += 1
                idx = np.flatnonzero(m)
                pat = np.zeros(n_ch)
                for L in np.unique(labs[idx]):
                    if (int(L), int(c)) in amp:
                        pat += amp[(int(L), int(c))]
                pat[veto[c]] = 0
                if pat.max() <= 1.0:
                    continue
                order = np.argsort(pat)[::-1]; cs = np.cumsum(pat[order])
                chans = order[: int(np.searchsorted(cs, 0.9 * cs[-1])) + 1]
                chans = chans[pat[chans] > 1.0]
                # others = full model minus this family's own render at its chain time
                own = np.zeros((n_ch, n_ticks))
                sh = np.interp(tick_idx - tv, np.arange(TMPL.size), TMPL, left=0.0, right=0.0)
                own[chans] = pat[chans, None] * sh[None, :]
                others = model[c] - own
                col = int(round(tv)) + PEAK
                pile = np.array([max(others[ch, col], 0.0) / max(pat[ch], 1e-9) for ch in chans]) if 0 <= col < n_ticks else np.zeros(len(chans))
                hs = idx if idx.size <= MAX_HITS else idx[np.linspace(0, idx.size - 1, MAX_HITS).astype(int)]
                hx, he = xyz[hs], e[hs]
                # apex candidates: principal-axis ends + centroid
                cen = np.average(hx, axis=0, weights=he)
                u = np.linalg.svd((hx - cen) * np.sqrt(he)[:, None], full_matrices=False)[2][0]
                proj = (hx - cen) @ u
                o = np.argsort(proj); n2 = max(3, int(0.02 * o.size))
                cands = [np.average(hx[o[:n2]], axis=0, weights=he[o[:n2]]), np.average(hx[o[-n2:]], axis=0, weights=he[o[-n2:]]), cen]
                best = None
                for apex in cands:
                    t0, tot, recs = family_fit(hx, he, apex, chans, sipm[c], full[c], others, w_tick[c], tv, pile, beta, pat)
                    if best is None or tot < best[1]:
                        best = (t0, tot, recs, apex)
                t0, tot, recs, apex = best
                if not np.isfinite(t0):
                    continue
                n_fit += 1
                dprop_all = np.linalg.norm(xyz[idx] - apex, axis=1) / C / NS    # ticks
                new_t[idx] = t0 + dprop_all
                for (ch, y, curv, A, pl) in recs:
                    if pl < 0.1:
                        slew_rows.append(((y - (t0 - tv)) * NS, np.log10(max(A, 1.0))))
                # evaluation on correctly assigned hits of this family
                okt = np.isfinite(tru[idx]) & (tru[idx] >= 0) & (np.abs(ts[idx] - tru[idx]) <= 5)
                if okt.sum() >= 5:
                    rE = (ts[idx][okt] - tru[idx][okt]) * NS
                    rN = (new_t[idx][okt] - tru[idx][okt]) * NS
                    res_E.append(rE); res_new.append(rN); res_w.append(e[idx][okt])
                    fam_E.append(np.average(rE, weights=e[idx][okt])); fam_new.append(np.average(rN, weights=e[idx][okt])); fam_w.append(E)
    rE = np.concatenate(res_E); rN = np.concatenate(res_new); w = np.concatenate(res_w)
    def rob(v, ww):
        o = np.argsort(v); cw = np.cumsum(ww[o]); med = v[o][np.searchsorted(cw, 0.5 * cw[-1])]
        d = np.abs(v - med); o2 = np.argsort(d); cw2 = np.cumsum(ww[o2]); return 1.4826 * d[o2][np.searchsorted(cw2, 0.5 * cw2[-1])]
    def wsd(v, ww):
        mu = np.average(v, weights=ww); return np.sqrt(np.average((v - mu) ** 2, weights=ww))
    fE, fN, fw = map(np.array, (fam_E, fam_new, fam_w))
    print(f"beta={beta:+.2f} ns/decade | {len(files)} events, families {n_fam}, fitted {n_fit}, {time.time()-t_start:.0f} s")
    print(f"  per-hit  (core, E-wt): stage E std {wsd(rE,w):.2f} robust {rob(rE,w):.2f} ns  ->  v1.2 proto std {wsd(rN,w):.2f} robust {rob(rN,w):.2f} ns")
    print(f"  per-family offset (E-wt): stage E std {wsd(fE,fw):.2f} robust {rob(fE,fw):.2f} ns  ->  v1.2 proto std {wsd(fN,fw):.2f} robust {rob(fN,fw):.2f} ns")
    S = np.array(slew_rows)
    coef = np.polyfit(S[:, 1], S[:, 0], 1) if len(S) > 100 else (np.nan, np.nan)
    print(f"  self-consistent slew from per-channel residuals (clean channels, n={len(S)}): {coef[0]:+.2f} ns/decade")
    return coef[0]


if __name__ == "__main__":
    d, f = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 13
    b1 = run(d, f, n, 0.0)
    if np.isfinite(b1):
        run(d, f, n, b1)
