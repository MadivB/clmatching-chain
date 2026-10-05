"""v1.2 stage P: per-hit precision timing for ND (port of the 2x2 v2 line, ND-adapted).

Runs after stage E on the final assignment (membership is never touched).
For every family (hits sharing one final t0, per light TPC):
  * support channels = channels holding 90 % of the family's own predicted light
    (no fixed channel count);
  * per-channel DELAY-AWARE model = the family's light-arrival density on that
    channel (discrete hits, apex clock t_dep = t0 + |r - apex|/c, then n d/c to
    the SiPM; 1/d^2 visibility proxy; 0.5 ns bins) convolved with the single
    average pulse template; c_LAr = 13.4 cm/ns fixed;
  * PILE-UP: the fit runs on actual - (all other families' model at their times);
    channels are down-weighted by their pile-up fraction;
  * per-channel time from a 0.25-tick scan (+-3 ticks) with the amplitude
    profiled out and an analytic parabola; per-channel-index delay k_idx and
    per-detector-type slew beta_T (both calibrated IN SITU by alternating
    medians, no truth: v120_calibrate.py) removed; the family apex time = the
    weighted median over channels (w = sqrt(A) x (1 - pile)), robust to pile-up;
  * apex = the candidate (two principal-axis ends, centroid) with the lowest
    total per-channel loss; per-hit deposition time = t0 + |r_h - apex| / c;
  * per-family sigma_t = 1.4826 MAD of the corrected channel times / sqrt(n).
Core works on plain arrays (run offline on dumps or in the chain via the
adapters at the bottom).
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

_HERE = os.path.abspath(os.path.dirname(__file__))
_BASE = os.path.join(_HERE, "..", "CLMatching_AlphaRelease")

C = 29.9792458           # cm/ns
N_LAR = 2.237            # c / c_LAr (c_LAr = 13.4 cm/ns, 2x2 convention, FIXED)
NS = 16.0                # ns per matching tick
PEAK = 105               # template peak sample after the pulse start
GAIN = 200.0
FINE = 0.5               # ns
ARR_LO, ARR_HI = -16.0, 144.0
NB = int((ARR_HI - ARR_LO) / FINE)
T_KEEP_NS = 3200.0       # the stored template has ~100 samples of pre-trigger padding; keep rise + fit reach
WIN_LO, WIN_HI = -20, 40 # fit window [ticks] around the predicted peak
SCAN = np.arange(-12, 13) * 0.25      # +-3 ticks: the chain time is the integer-scan argmin
QUAD_HW = 4              # 9-point parabola (0.25-tick steps)
MIN_A = 300.0            # per-channel fitted peak amplitude [ADC] to count a channel
MAX_HITS = 1000          # hits sampled for the arrival density
ROUNDS = 3               # pile-up model iterations: others re-rendered at their refined arrival-equivalent times
PIECE_R = 3.0            # cm: contact radius chaining clusters into pieces (the chain's hit-graph link radius)
PIECE_MIN_MEV = 5.0      # a displaced piece below this keeps the core's v = c time (light-blind)
SCAN_PIECE = np.arange(-40, 41) * 0.25   # +-10 ticks: a displaced fragment may sit anywhere in the assignment window

_TMPL = None


def _template():
    global _TMPL
    if _TMPL is None:
        t = np.load(os.path.join(_BASE, "assets", "avg_pulse.npy")).astype(np.float64)
        t = t / t.max()
        nf = int(t.size * NS / FINE)
        tf = np.interp(np.arange(nf) * FINE / NS, np.arange(t.size), t)[: int(T_KEEP_NS / FINE)]
        nfft = 1 << int(np.ceil(np.log2(NB + tf.size)))
        x = np.arange(-QUAD_HW, QUAD_HW + 1) * 0.25
        X = np.vstack([x * x, x, np.ones_like(x)]).T
        pinv = np.linalg.pinv(X)
        _TMPL = dict(t16=t, tf=tf, nfft=nfft, tf_f=np.fft.rfft(tf, nfft), pinv=pinv)
    return _TMPL


def load_calib(path):
    if not path or not os.path.exists(path):
        return dict(k_idx=np.zeros(120), beta_t={0: 0.0, 1: 0.0}, type_of=np.zeros(120, int), beta=0.0)
    d = json.load(open(path))
    return dict(k_idx=np.asarray(d.get("k_idx", np.zeros(120)), float),
                beta_t={int(k): float(v) for k, v in d.get("beta_t", {}).items()},
                type_of=np.asarray(d.get("type_of", np.zeros(120)), int),
                beta=float(d.get("beta", 0.0)))


def _wmedian(v, w):
    o = np.argsort(v); cw = np.cumsum(w[o])
    return float(v[o][np.searchsorted(cw, 0.5 * cw[-1])])


_TORCH = {"dev": None}


def _torch_dev():
    """'cuda' -> torch on GPU, 'tcpu' -> torch on CPU (test mode), 'cpu' -> numpy."""
    if _TORCH["dev"] is None:
        mode = os.environ.get("V12_TORCH", "auto")
        _TORCH["dev"] = "cpu"
        if mode == "cpu":
            _TORCH["dev"] = "tcpu"
        elif mode != "off":
            try:
                import torch
                if torch.cuda.is_available():
                    _TORCH["dev"] = "cuda"
            except Exception:
                pass
    return _TORCH["dev"]


_CONV_CACHE: dict = {}      # speed campaign 2026-10-05: the arrival-time histogram and its template convolution depend only on the
                            # family geometry (hits, apexes, channels, scan); rounds 2 and 3 of the precision fit reuse them (bit-identical)
_CONV_CACHE_ON = os.environ.get("CLM_PREC_CACHE", "1") != "0"


def precision_cache_reset():
    _CONV_CACHE.clear()


def _family_channels(hx, he, apexes, chans, sipm_c, R, W, ks, t_chain, calib, pile, scan=None):
    SCAN_ = SCAN if scan is None else scan
    """Vectorized per-channel delay-aware fits on the window residual R (nc, nk) with weights W,
    for ALL apex candidates at once (rows stacked).  Returns a list per candidate of
    (rows, total_loss): rows = (ch, shift_raw [ticks], curvature, A, pile, shift_corr [ticks])."""
    T = _template()
    nc = len(chans); na = len(apexes)
    key = None
    if _CONV_CACHE_ON:
        try:
            key = (float(t_chain), int(hx.shape[0]), float(hx[:, 0].sum()), float(he.sum()), tuple(int(c) for c in chans),
                   tuple(float(v) for a in apexes for v in np.asarray(a).ravel()), id(SCAN_), int(ks[0]), int(ks.size))
        except Exception:
            key = None
    hit = _CONV_CACHE.get(key) if key is not None else None
    if hit is None:
        P = sipm_c[chans]
        dd = np.linalg.norm(hx[:, None, :] - P[None, :, :], axis=2)         # (nh, nc) cm
        wv = he[:, None] / np.maximum(dd, 5.0) ** 2
        rho = np.zeros(na * nc * NB)
        for ai, apex in enumerate(apexes):
            dprop = np.linalg.norm(hx - apex, axis=1) / C                    # ns
            tau = dprop[:, None] + dd * N_LAR / C
            b = np.clip(((tau - ARR_LO) / FINE).astype(np.int64), 0, NB - 1)
            np.add.at(rho, ((ai * nc + np.arange(nc))[None, :] * NB + b).ravel(), wv.ravel())
        rho = rho.reshape(na * nc, NB)
    R = np.tile(R, (na, 1)); W = np.tile(W, (na, 1)); pile = np.tile(pile, na)
    chans_all = np.tile(chans, na); cand_of = np.repeat(np.arange(na), nc)
    # fine index of tick k for apex time (t_chain + s): ((k - t_chain - s)*NS - ARR_LO)/FINE  -> (nS, nk), same for all channels
    nconv = NB + T["tf"].size - 1
    if hit is None:
        m = ((ks[None, :] - t_chain - SCAN_[:, None]) * NS - ARR_LO) / FINE
        m = np.clip(m, 0, nconv - 1.001)
        i0 = np.floor(m).astype(np.int64); fr = m - i0
    if _torch_dev() in ("cuda", "tcpu"):
        import torch
        dev = "cuda" if _torch_dev() == "cuda" else "cpu"
        if hit is None:
            rho_t = torch.from_numpy(rho).to(dev)
            tf_f = _TORCH.get("tf_f_" + dev)
            if tf_f is None:
                tf_f = torch.fft.rfft(torch.from_numpy(T["tf"]).to(dev), T["nfft"]); _TORCH["tf_f_" + dev] = tf_f
            conv_t = torch.fft.irfft(torch.fft.rfft(rho_t, T["nfft"], dim=1) * tf_f[None, :], T["nfft"], dim=1)[:, :nconv]
            i0_t = torch.from_numpy(i0).to(dev); fr_t = torch.from_numpy(fr).to(dev)
            if key is not None:
                if len(_CONV_CACHE) > 20000:
                    _CONV_CACHE.clear()
                _CONV_CACHE[key] = ("t", conv_t, i0_t, fr_t)
        else:
            _, conv_t, i0_t, fr_t = hit
        M_t = conv_t[:, i0_t] * (1.0 - fr_t)[None] + conv_t[:, i0_t + 1] * fr_t[None]
        R_t = torch.from_numpy(R).to(dev); W_t = torch.from_numpy(W).to(dev)
        den_t = torch.einsum("ck,csk,csk->cs", W_t, M_t, M_t)
        num_t = torch.einsum("ck,ck,csk->cs", W_t, R_t, M_t)
        amps_t = torch.where(den_t > 0, torch.clamp(num_t / torch.clamp(den_t, min=1e-30), min=0.0), torch.zeros_like(den_t))
        resid_t = R_t[:, None, :] - amps_t[:, :, None] * M_t
        Ls = torch.einsum("ck,csk->cs", W_t, resid_t * resid_t).cpu().numpy()
        amps = amps_t.cpu().numpy(); conv_max = conv_t.max(dim=1).values.cpu().numpy()
    else:
        if hit is None:
            conv = np.fft.irfft(np.fft.rfft(rho, T["nfft"], axis=1) * T["tf_f"][None, :], T["nfft"], axis=1)[:, :nconv]
            if key is not None:
                if len(_CONV_CACHE) > 20000:
                    _CONV_CACHE.clear()
                _CONV_CACHE[key] = ("n", conv, i0, fr)
        else:
            _, conv, i0, fr = hit
        M = conv[:, i0] * (1.0 - fr)[None] + conv[:, i0 + 1] * fr[None]      # (nc, nS, nk)
        den = np.einsum("ck,csk,csk->cs", W, M, M)
        num = np.einsum("ck,ck,csk->cs", W, R, M)
        amps = np.where(den > 0, np.clip(num / np.maximum(den, 1e-30), 0.0, None), 0.0)
        resid = R[:, None, :] - amps[:, :, None] * M
        Ls = np.einsum("ck,csk->cs", W, resid * resid)                       # (nc, nS)
        conv_max = conv.max(axis=1)
    j0 = np.argmin(Ls, axis=1)
    lmin = Ls[np.arange(na * nc), j0]
    tots = [float(lmin[cand_of == ai].sum()) for ai in range(na)]
    ok = (j0 >= QUAD_HW) & (j0 <= SCAN_.size - 1 - QUAD_HW)
    out = [([], tots[ai]) for ai in range(na)]
    if not ok.any():
        return out
    idx = np.flatnonzero(ok)
    win = Ls[idx[:, None], (j0[idx][:, None] + np.arange(-QUAD_HW, QUAD_HW + 1)[None, :])]   # (n, 9)
    coef = win @ T["pinv"].T                                             # (n, 3): a, b, c
    a, bb = coef[:, 0], coef[:, 1]
    good = a > 0
    xs = np.where(good, -bb / (2.0 * np.maximum(a, 1e-30)), np.nan)
    good &= np.abs(xs) <= 0.75
    A = amps[idx, j0[idx]] * conv_max[idx]
    good &= A >= MIN_A
    for n_, ci in enumerate(idx):
        if not good[n_]:
            continue
        ch = int(chans_all[ci])
        y = float(SCAN_[j0[ci]] + xs[n_])
        bt = calib["beta_t"].get(int(calib["type_of"][ch]), calib["beta"])
        y_corr = y - (calib["k_idx"][ch] + bt * np.log10(max(A[n_], 1.0))) / NS
        out[int(cand_of[ci])][0].append((ch, y, float(a[n_]), float(A[n_]), float(pile[ci]), y_corr))
    return out


def _mean_arrival_ns(hx, he, apex, chans, sipm_c, pat):
    dprop = np.linalg.norm(hx - apex, axis=1) / C
    dd = np.linalg.norm(hx[:, None, :] - sipm_c[chans][None, :, :], axis=2)
    tau = dprop[:, None] + dd * N_LAR / C
    wv = he[:, None] / np.maximum(dd, 5.0) ** 2
    per_ch = (tau * wv).sum(0) / np.maximum(wv.sum(0), 1e-12)
    return float(np.average(per_ch, weights=np.maximum(pat[chans], 1e-9)))


def _pieces(idx, xyz, labels, e):
    """Split a family's hits into pieces: clusters chained by PIECE_R contact, biggest first.
    Returns list of hit-index arrays, core first."""
    from scipy.spatial import cKDTree
    labs = np.unique(labels[idx])
    hits = {int(L): idx[labels[idx] == L] for L in labs}
    E = {L: float(e[h].sum()) for L, h in hits.items()}
    trees = {L: cKDTree(xyz[h]) for L, h in hits.items()}
    unassigned = set(hits); comps = []
    while unassigned:
        seed = max(unassigned, key=lambda L: E[L])
        comp = {seed}; frontier = [seed]; unassigned.discard(seed)
        while frontier:
            cur = frontier.pop()
            for L in list(unassigned):
                dd, _ = trees[cur].query(xyz[hits[L]], k=1, distance_upper_bound=PIECE_R)
                if np.isfinite(dd).any():
                    comp.add(L); frontier.append(L); unassigned.discard(L)
        comps.append(np.concatenate([hits[L] for L in comp]))
    comps.sort(key=lambda h: -float(e[h].sum()))
    return comps


def precision_fit(xyz, e, tpc, labels, ts, amp, full, veto, sipm, calib, *,
                  min_fam_mev=5.0, min_hits=10, max_hits=MAX_HITS, collect=False, rounds=ROUNDS,
                  pieces=False):
    """Core.  amp: {(label, light_tpc): (120,) peak amplitudes}; full: (n_light,120,1000);
    sipm: {light_tpc: (120,3)}.  Returns (new_ts, hit_sigma_ticks, stats)."""
    t_start = time.time()
    T = _template()
    e = np.clip(np.asarray(e, np.float64), 0, None)
    ts = np.asarray(ts, np.float64); new_t = ts.copy()
    xyz = np.asarray(xyz, np.float64); tpc = np.asarray(tpc, np.int64); labels = np.asarray(labels, np.int64)
    n_light, n_ch, n_ticks = full.shape
    full = np.asarray(full, np.float64)
    w_tick = 1.0 / (GAIN * np.maximum(full, 100.0))
    ok = np.isfinite(ts) & (ts >= 0)
    sig = np.full(ts.size, np.nan)
    tick_idx = np.arange(n_ticks)
    lab_t = {}
    for L in np.unique(labels[labels >= 0]):
        v = ts[labels == L]; v = v[np.isfinite(v) & (v >= 0)]
        if v.size:
            lab_t[int(L)] = float(np.median(v))
    tpcs_needed = sorted(int(c) for c in np.unique(tpc[ok]) if int(c) in sipm and int(c) < n_light)
    st = {}
    fam_render = {}          # (tpc, chain t) -> arrival-equivalent refined time used to render this family (core)
    lab_render = {}          # (tpc, label) -> arrival-equivalent refined time for labels timed separately (pieces)
    n_rounds = max(1, int(rounds)); precision_cache_reset()
    for rnd in range(n_rounds):
        st = {"n_fam": 0, "n_fit": 0, "n_ch_used": 0, "e_fit": 0.0, "dt_med_ns": [], "sig_ns": [], "rows": [], "apex_win": [0, 0, 0],
              "n_piece": 0, "n_piece_fit": 0, "pieces": []}
        model = {c: np.zeros((n_ch, n_ticks)) for c in tpcs_needed}
        for (L, c), a_vec in amp.items():
            if int(c) in model and int(L) in lab_t:
                t_use = lab_render.get((int(c), int(L)), fam_render.get((int(c), lab_t[int(L)]), lab_t[int(L)]))
                sh = np.interp(tick_idx - t_use, np.arange(T["t16"].size), T["t16"], left=0.0, right=0.0)
                model[int(c)] += np.asarray(a_vec, np.float64)[:, None] * sh[None, :]
        for c in tpcs_needed:
            mc = ok & (tpc == c)
            for tv in np.unique(ts[mc]):
                m = mc & (ts == tv)
                E = float(e[m].sum())
                if E < min_fam_mev or m.sum() < min_hits:
                    continue
                st["n_fam"] += 1
                idx_all = np.flatnonzero(m)
                piece_list = _pieces(idx_all, xyz, labels, e) if pieces else [idx_all]
                idx = idx_all                             # the family fit is the certified v1.2 fit (all hits)
                pat = np.zeros(n_ch)
                for L in np.unique(labels[idx]):
                    a_vec = amp.get((int(L), int(c)))
                    if a_vec is not None:
                        pat += np.asarray(a_vec, np.float64)
                pat[np.asarray(veto[c], bool)] = 0.0
                if pat.max() <= 1.0:
                    continue
                order = np.argsort(pat)[::-1]; cs = np.cumsum(pat[order])
                chans = order[: int(np.searchsorted(cs, 0.9 * cs[-1])) + 1]
                chans = chans[pat[chans] > 1.0]
                col0 = int(round(tv)); col = col0 + PEAK
                if not (0 <= col < n_ticks):
                    continue
                ks = np.arange(col0 + PEAK + WIN_LO, col0 + PEAK + WIN_HI + 1)
                ks = ks[(ks >= 0) & (ks < n_ticks)]
                t_self = fam_render.get((int(c), float(tv)), tv)
                own_win = pat[chans, None] * np.interp(ks - t_self, np.arange(T["t16"].size), T["t16"], left=0.0, right=0.0)[None, :]
                others_win = model[c][chans][:, ks] - own_win                          # (nc, nk)
                own_col = pat[chans] * float(np.interp(col - t_self, np.arange(T["t16"].size), T["t16"]))
                pile = np.maximum(model[c][chans, col] - own_col, 0.0) / np.maximum(pat[chans], 1e-9)
                if pieces:
                    # the family's displaced pieces stay in 'others' at their current times; this fit is the CORE only
                    pass
                R_win = full[c][chans][:, ks] - others_win
                W_win = w_tick[c][chans][:, ks]
                hs = idx if idx.size <= max_hits else idx[np.linspace(0, idx.size - 1, max_hits).astype(int)]
                hx, he = xyz[hs], np.maximum(e[hs], 1e-6)
                cen = np.average(hx, axis=0, weights=he)
                u = np.linalg.svd((hx - cen) * np.sqrt(he)[:, None], full_matrices=False)[2][0]
                proj = (hx - cen) @ u
                o = np.argsort(proj); n2 = max(3, int(0.02 * o.size))
                cands = [np.average(hx[o[:n2]], axis=0, weights=he[o[:n2]]),
                         np.average(hx[o[-n2:]], axis=0, weights=he[o[-n2:]]), cen]
                res = _family_channels(hx, he, cands, chans, sipm[c], R_win, W_win, ks, tv, calib, pile)
                ai = int(np.argmin([tt for _, tt in res]))
                rows, tot = res[ai]; apex = cands[ai]
                st["apex_win"][ai] += 1
                if len(rows) < 3:
                    continue
                r = np.array(rows, dtype=float)
                wgt = np.sqrt(r[:, 3]) * np.clip(1.0 - r[:, 4], 0.05, 1.0)
                shift = _wmedian(r[:, 5], wgt)
                t0 = tv + shift
                mad = 1.4826 * float(np.median(np.abs(r[:, 5] - shift)))
                s_fam = mad / np.sqrt(len(rows))                                  # ticks
                fam_render[(int(c), float(tv))] = t0 + _mean_arrival_ns(hx, he, apex, chans, sipm[c], pat) / NS
                dprop = np.linalg.norm(xyz[idx] - apex, axis=1) / C / NS           # ticks
                new_t[idx] = t0 + dprop
                sig[idx] = s_fam
                st["n_fit"] += 1; st["n_ch_used"] += len(rows); st["e_fit"] += E
                st["dt_med_ns"].append(shift * NS); st["sig_ns"].append(s_fam * NS)
                if pieces and len(piece_list) > 1:
                    core_apex, core_t0 = apex, t0          # family time from the collective fit above
                    fam_arr = fam_render[(int(c), float(tv))]
                    for pidx in piece_list[1:]:
                        Ep = float(e[pidx].sum())
                        st["n_piece"] += 1                 # non-fitted pieces keep the family time and sigma
                        if Ep < PIECE_MIN_MEV:
                            continue
                        ppat = np.zeros(n_ch)
                        for L in np.unique(labels[pidx]):
                            a_vec = amp.get((int(L), int(c)))
                            if a_vec is not None:
                                ppat += np.asarray(a_vec, np.float64)
                        ppat[np.asarray(veto[c], bool)] = 0.0
                        if ppat.max() <= 1.0:
                            continue
                        po = np.argsort(ppat)[::-1]; pcs = np.cumsum(ppat[po])
                        pch = po[: int(np.searchsorted(pcs, 0.9 * pcs[-1])) + 1]; pch = pch[ppat[pch] > 1.0]
                        pks = np.arange(col0 + PEAK + WIN_LO - 10, col0 + PEAK + WIN_HI + 11)   # wider window for the wider scan
                        pks = pks[(pks >= 0) & (pks < n_ticks)]
                        # others for the piece = model (everything at current render times) minus this piece's own light at the chain tick
                        p_labels = [int(L) for L in np.unique(labels[pidx])]
                        p_tself = lab_render.get((int(c), p_labels[0]), fam_arr if rnd > 0 else tv)
                        p_own = ppat[pch, None] * np.interp(pks - p_tself, np.arange(T["t16"].size), T["t16"], left=0.0, right=0.0)[None, :]
                        p_others = model[c][pch][:, pks] - p_own
                        p_own_col = ppat[pch] * float(np.interp(col - p_tself, np.arange(T["t16"].size), T["t16"]))
                        p_pile = np.maximum(model[c][pch, col] - p_own_col, 0.0) / np.maximum(ppat[pch], 1e-9)
                        pR = full[c][pch][:, pks] - p_others; pW = w_tick[c][pch][:, pks]
                        phs = pidx if pidx.size <= max_hits else pidx[np.linspace(0, pidx.size - 1, max_hits).astype(int)]
                        phx, phe = xyz[phs], np.maximum(e[phs], 1e-6)
                        pcen = np.average(phx, axis=0, weights=phe)
                        if phx.shape[0] >= 3:
                            pu = np.linalg.svd((phx - pcen) * np.sqrt(phe)[:, None], full_matrices=False)[2][0]
                            pproj = (phx - pcen) @ pu; poo = np.argsort(pproj); pn2 = max(1, int(0.02 * poo.size))
                            pcands = [np.average(phx[poo[:pn2]], axis=0, weights=phe[poo[:pn2]]), np.average(phx[poo[-pn2:]], axis=0, weights=phe[poo[-pn2:]]), pcen]
                        else:
                            pcands = [pcen]
                        pres = _family_channels(phx, phe, pcands, pch, sipm[c], pR, pW, pks, tv, calib, p_pile, scan=SCAN_PIECE)
                        pai = int(np.argmin([tt for _, tt in pres])); prows, ptot = pres[pai]; papex = pcands[pai]
                        # a displaced piece is resolvable by light only on channels where its OWN predicted
                        # light is at least comparable to everything else there (pile < 1); fewer than 3 such
                        # channels -> keep the core's v = c time (flagged by sig = NaN)
                        prows = [r_ for r_ in prows if r_[4] < 1.0]
                        st["n_piece_dom"] = st.get("n_piece_dom", 0) + (1 if len(prows) >= 3 else 0)
                        if len(prows) < 3:
                            continue
                        pr = np.array(prows, dtype=float)
                        pwgt = np.sqrt(pr[:, 3]) * (1.0 - pr[:, 4])
                        pshift = _wmedian(pr[:, 5], pwgt); pt0 = tv + pshift
                        pmad = 1.4826 * float(np.median(np.abs(pr[:, 5] - pshift))); ps = pmad / np.sqrt(len(prows))
                        new_t[pidx] = pt0 + np.linalg.norm(xyz[pidx] - papex, axis=1) / C / NS
                        sig[pidx] = ps
                        p_arr = pt0 + _mean_arrival_ns(phx, phe, papex, pch, sipm[c], ppat) / NS
                        for L in p_labels:
                            lab_render[(int(c), int(L))] = p_arr
                        dist = float(np.linalg.norm(papex - core_apex))
                        dt_ns = (pt0 - core_t0) * NS
                        st["n_piece_fit"] += 1
                        if collect and rnd == n_rounds - 1:
                            st["pieces"].append(dict(tpc=int(c), t_chain=float(tv), E=Ep, n_ch=len(prows), t0_core=float(core_t0),
                                                     t0_piece=float(pt0), dt_ns=dt_ns, dist_cm=dist,
                                                     beta=(dist / (C * dt_ns) if dt_ns > 0.5 else None), sig_ns=float(ps * NS),
                                                     labels=[int(L) for L in np.unique(labels[pidx])]))
                if collect and rnd == n_rounds - 1:
                    st["rows"].append(dict(tpc=int(c), t_chain=float(tv), t0=float(t0), E=E, n_ch=len(rows),
                                           sig_ns=float(s_fam * NS), apex=[float(v) for v in apex],
                                           chan=[(int(a_), float(b_), float(d_), float(p_), float(yc)) for a_, b_, c_, d_, p_, yc in rows]))
    st["time_s"] = time.time() - t_start
    st["rounds"] = n_rounds
    st["dt_med_ns"] = float(np.median(st["dt_med_ns"])) if st["dt_med_ns"] else 0.0
    st["sig_med_ns"] = float(np.median(st["sig_ns"])) if st["sig_ns"] else 0.0
    st.pop("sig_ns")
    return new_t, sig, st


# ---------------------------------------------------------------- adapters
def run_precision(ns, *, flow_path, calib_path=None, veto_mask=None, **kw):
    """Chain adapter: refines ns['hit_timestamps'] in place (per-hit deposition times) and
    sets ns['hit_t0_sigma_p'] (ticks).  Uses the stage-1 amplitudes in ns['imageMaps']."""
    import sys
    for p in (_HERE, _BASE, os.path.join(_BASE, "M5p1"), os.path.join(_BASE, "NewMLSection")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from subtick_refine import build_sipm_positions
    sipm = build_sipm_positions(flow_path)
    calib = load_calib(calib_path)
    xyz = np.stack([np.asarray(ns["xset"], np.float64), np.asarray(ns["yset"], np.float64),
                    np.asarray(ns["zset"], np.float64)], axis=1)
    full = np.asarray(ns["fullLightWaveform"], np.float64)
    n_light = full.shape[0]
    amp = {}
    for (L, c), im in ns["imageMaps"].items():
        if 0 <= int(c) < n_light:
            amp[(int(L), int(c))] = np.asarray(im, np.float64).max(axis=1)
    if veto_mask is None:
        veto_mask = np.zeros(full.shape[:2], bool)
    new_t, sig, st = precision_fit(xyz, ns["Eset"], ns["hitTPCid"], ns["labels_global"], ns["hit_timestamps"],
                                   amp, full, veto_mask, sipm, calib, **kw)
    ns["hit_timestamps"] = new_t.astype(np.float32)
    ns["hit_t0_sigma_p"] = sig.astype(np.float32)
    return st


def run_on_dump(npz_path, flow_path, calib_path=None, **kw):
    """Offline adapter for chain dumps (v138-style npz)."""
    import sys
    for p in (_HERE, _BASE, os.path.join(_BASE, "M5p1"), os.path.join(_BASE, "NewMLSection")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from subtick_refine import build_sipm_positions
    z = np.load(npz_path, allow_pickle=True)
    sipm = build_sipm_positions(flow_path)
    calib = load_calib(calib_path)
    xyz = np.stack([z["x"], z["y"], z["z"]], 1).astype(np.float64)
    al, at, ar = z["amp_lab"].astype(int), z["amp_tpc"].astype(int), z["amp_row"].astype(np.float64)
    amp = {(int(al[i]), int(at[i])): ar[i] for i in range(len(al))}
    new_t, sig, st = precision_fit(xyz, z["e"], z["tpc"], z["labels"], z["ts_final"], amp,
                                   z["full_light"].astype(np.float64), z["veto_mask"], sipm, calib, **kw)
    return new_t, sig, st, z
