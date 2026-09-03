"""Scan-curve (loss-landscape) confidence features — the deep-research
direction: the quadratic confidence only sees the LOCAL curvature at the
minimum; the canonical assignment-confidence is the Delta-chi2 MARGIN to the
best COMPETING minimum plus the PDA posterior over candidate times including
an explicit null (no-light) hypothesis.

Per assigned cluster (amp_lab/amp_tpc/amp_row dumps = per-label per-channel
peaks; rank-1: image = (peak/tmpl_max) x shift(tmpl, t0)):

  L(t) = sum_ch [ -2 a_ch corr_ch(t) + a_ch^2 nvis(t) ]      (const dropped)
  corr_ch = FFT-correlate(actual_ch - others_ch, tmpl)

Corrections required for a faithful reconstruction (all measured, 2026-09-02):
  - vetoed channels excluded (chain's matching loss excludes them; CLIP=60780
    flat-tops break rank-1 there)
  - GROUP-SHARE RESCALE: co-timed clusters' individual renders are NOT
    additive (v097) — base_image holds the collective render; each member's
    per-channel peak is scaled so the super-group (t0s within 10 ticks) sum
    matches base at the anchor column, clipped to [0,1].

Features (sub-ms per cluster; empirical per-cluster temperature = active-
region residual variance, per the deep-research miscalibration findings):
  depth_n    : (L(no-light) - L(t*)) / varhat     -> is the cluster visible
  margin2_n  : (L(best competitor >10 ticks) - L(t*)) / varhat  -> ambiguity
  rel_margin : margin2 / depth
  n_comp     : competing local minima with L - L* < 0.5 * depth
  post       : posterior mass of t* over local minima (normalized curve)
  post_null  : PDA posterior of t* over {minima + null}
  width      : half-depth valley width (ticks)
  d_place    : |scan argmin - chain placement|
  d_truth    : (L(truth) - L(t*)) / varhat  [DIAGNOSTIC ONLY - never a feature]

Usage: python scan_margin.py v126MG_out [--eval]
  (bare: extract features -> <outdir>/scan_margin_feats.npz; --eval: join
   truth labels and report AUCs.)
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

TMPL = np.load(os.path.join(HERE, "..", "CLMatching_AlphaRelease", "assets",
                            "avg_pulse.npy")).astype(np.float64) / 999.0
TMPL_MAX = float(TMPL.max())
NT = TMPL.size            # 1000
TMAX = 896                # scan lags 0..895 (the chain's SEARCH_RANGE)
NVIS = np.array([np.square(TMPL[:NT - t]).sum() for t in range(TMAX)])
NFFT = 2048
TMPL_F = np.conj(np.fft.rfft(TMPL, NFFT))
SUP_THRESH = 1.0          # chain support definition: peak > 1 ADC
TOP_CH = 30

PC_STATS = ["n", "frac_clear", "wfrac_clear", "sig_med",
            "vote10", "vote_med", "vert_std"]
SCAN_NAMES = (["depth_n", "margin2_n", "rel_margin", "n_comp", "post",
               "post_null", "width", "d_place"]
              # per-channel quadratic block (user intuition 2026-09-03):
              # a FLAT per-channel quadratic = trouble; clear-minimum channels
              # should also AGREE on the location. pc_* = masked to predicted
              # peak >= 600 ADC (user: mask clearly-irrelevant channels);
              # pcu_* = unmasked (dim-channel votes still carry aggregate
              # signal - measured; model picks per regime).
              + [f"pc_{k}" for k in PC_STATS]
              + [f"pcu_{k}" for k in PC_STATS]
              # per-CLUSTER quadratic + waveform evidence (user 2026-09-03:
              # per-cluster instead of per-group): local curvature/contrast of
              # the cluster's own curve + z-stats at its own peak column
              + ["c_sig_t", "c_contrast", "c_meanz", "c_fracz2",
                 "c_cosres", "c_logvar"])
PC_HW = 8            # per-channel quadratic window: t0 +- 8 ticks
PC_MIN_ADC = 600.0   # PC block: only channels with predicted peak >= 600 ADC
                     # (user 2026-09-03: mask clearly-irrelevant dim channels
                     # — they are always "flat" and dilute the fractions)


def shift_place(vec, t0):
    """Chain placement: template shifted right by t0 (integer), zero-fill."""
    t0 = int(round(t0))
    out = np.zeros(NT)
    if t0 < NT:
        out[t0:] = vec[:NT - t0]
    return out


def local_minima(L):
    return np.flatnonzero((L[1:-1] <= L[:-2]) & (L[1:-1] <= L[2:])) + 1


def cluster_curve(A, O, a):
    """L(t) up to a t-independent constant, for channel rows of A/O, amps a.

    corr_ch(t) = sum_k R[t+k] tmpl[k] via FFT (NFFT=2048 >= 1999: no wrap
    for lags < 1049; we use 0..895)."""
    R = A - O
    RF = np.fft.rfft(R, NFFT, axis=1)
    corr = np.fft.irfft(RF * TMPL_F[None, :], NFFT, axis=1)[:, :TMAX]
    return (-2.0 * a[:, None] * corr + np.square(a)[:, None] * NVIS[None, :]).sum(0)


def _pc_stats(Lpc, apc, Rpc, wlo, whi, tt, t0):
    """7 per-channel quadratic statistics for one cluster: flatness (clear-
    minimum fraction, curvature significance) + location votes (agreement of
    per-channel minima with the placement)."""
    n_ch = Lpc.shape[0]
    if tt.size < 5 or n_ch == 0:
        return [float(n_ch), 0.0, 0.0, 0.0, 0.0, 200.0, float(PC_HW)]
    W = Lpc[:, wlo:whi]
    X = np.vander(tt, 3)                     # [x^2, x, 1]
    coef, *_ = np.linalg.lstsq(X, W.T, rcond=None)
    c2, c1 = coef[0], coef[1]
    vert = np.where(np.abs(c2) > 1e-12, -c1 / (2.0 * c2), np.inf)
    sig_r = np.sqrt(np.maximum(np.square(Rpc).mean(1), 1e-12))
    nscale = 2.0 * np.abs(apc) * sig_r * np.sqrt(NVIS[min(t0, TMAX - 1)])
    depth_win = W.max(1) - W.min(1)
    sig_c = depth_win / np.maximum(nscale, 1e-12)
    clear = (c2 > 0) & (np.abs(vert) <= PC_HW) & (sig_c > 2.0)
    wamp = np.maximum(apc, 0.0)
    wsum = max(wamp.sum(), 1e-12)
    gv = np.argmin(Lpc, axis=1)
    d_c = np.abs(gv - t0).astype(np.float64)
    order = np.argsort(d_c)
    cw = np.cumsum(wamp[order])
    vote_med = (float(np.clip(d_c[order][np.searchsorted(cw, 0.5 * cw[-1])],
                              0, 200)) if cw[-1] > 0 else 200.0)
    if clear.any():
        vw = wamp[clear] / max(wamp[clear].sum(), 1e-12)
        vmean = float((vert[clear] * vw).sum())
        vert_std = float(np.sqrt((vw * np.square(vert[clear] - vmean)).sum()))
    else:
        vert_std = float(PC_HW)
    return [float(n_ch), float(clear.mean()), float(wamp[clear].sum() / wsum),
            float(np.median(sig_c)), float(wamp[d_c <= 10].sum() / wsum),
            vote_med, vert_std]


def event_scan_feats_ref(base, act, vm, al, at, ar_raw, lab_t0):
    """Scan-landscape features for every label of one event.

    base/act: (n_tpc,120,1000) predicted-total / actual light
    vm: (n_tpc,120) veto mask; al/at/ar_raw: amp dump arrays;
    lab_t0: {label: chain integer placement tick}.
    Returns {label: (feats8 ndarray[SCAN_NAMES order], (Lc, varhat, i0))}.
    """
    ar = ar_raw.astype(np.float64).copy()
    for i in range(ar.shape[0]):
        ar[i][vm[int(at[i])]] = 0.0
    targmax = int(np.argmax(TMPL))
    grp = {}
    for i, (L, c) in enumerate(zip(al, at)):
        t0 = lab_t0.get(int(L))
        if t0 is not None:
            grp.setdefault(int(c), []).append((t0, i))
    for c, pairs in grp.items():
        pairs.sort()
        sg, cur = [], [pairs[0]]
        for p in pairs[1:]:
            if p[0] - cur[-1][0] <= 10:
                cur.append(p)
            else:
                sg.append(cur)
                cur = [p]
        sg.append(cur)
        for members in sg:
            anchor = max(members, key=lambda p: ar[p[1]].max())[0]
            col = anchor + targmax
            if col >= NT:
                continue
            pred = np.zeros(120)
            for t0, i in members:
                pred += ar[i] * (shift_place(TMPL, t0)[col] / TMPL_MAX)
            r = np.clip(base[c][:, col] / np.maximum(pred, 1e-9), 0.0, 1.0)
            for _, i in members:
                ar[i] = ar[i] * r
    out = {}
    for L in np.unique(al):
        L = int(L)
        if L not in lab_t0:
            continue
        t0 = lab_t0[L]
        sel = np.flatnonzero(al == L)
        A_rows, O_rows, a_rows = [], [], []
        for i in sel:
            c, row = int(at[i]), ar[i]
            a_vec = row / TMPL_MAX
            sup = np.flatnonzero(row > SUP_THRESH)
            if sup.size == 0:
                continue
            if sup.size > TOP_CH:
                sup = sup[np.argsort(row[sup])[::-1][:TOP_CH]]
            own = a_vec[sup, None] * shift_place(TMPL, t0)[None, :]
            A_rows.append(act[c][sup])
            O_rows.append(base[c][sup] - own)
            a_rows.append(a_vec[sup])
        if not A_rows:
            continue
        A = np.concatenate(A_rows)
        O = np.concatenate(O_rows)
        a = np.concatenate(a_rows)
        # per-channel curves L_ch(t) = -2 a corr(t) + a^2 nvis(t); cluster
        # curve is their sum (one FFT pass, shared with the PC block)
        R = A - O
        RF = np.fft.rfft(R, NFFT, axis=1)
        corr = np.fft.irfft(RF * TMPL_F[None, :], NFFT, axis=1)[:, :TMAX]
        Lch = -2.0 * a[:, None] * corr + np.square(a)[:, None] * NVIS[None, :]
        Lc = Lch.sum(0)
        i0 = int(np.argmin(Lc))
        L0 = Lc[i0]
        depth = -L0 if L0 < 0 else 0.0   # L(no light)=0 by construction
        # empirical per-cluster temperature: active-region residual variance
        own0 = a[:, None] * shift_place(TMPL, i0)[None, :]
        act_m = own0 > 0.01 * max(own0.max(), 1e-9)
        resid = (A - O) - own0
        varhat = float(np.square(resid[act_m]).mean()) if act_m.any() else 1.0
        varhat = max(varhat, 1e-9)
        mins = local_minima(Lc)
        comp = mins[np.abs(mins - i0) > 10]
        margin2 = float(Lc[comp].min() - L0) if comp.size else float(-L0)
        n_comp = int((Lc[comp] - L0 < 0.5 * depth).sum()) if comp.size else 0
        Ln = Lc / varhat
        cand = np.concatenate([[i0], comp]) if comp.size else np.array([i0])
        lw = np.concatenate([-0.5 * Ln[cand], [0.0]])   # null: L=0
        post_null = float(np.exp(lw[0] - np.logaddexp.reduce(lw)))
        lw2 = -0.5 * (Ln[cand] - Ln[i0])
        post = float(np.exp(lw2[0] - np.logaddexp.reduce(lw2)))
        width = float((Lc <= L0 + 0.5 * depth).sum())
        # ---- per-channel quadratic block (masked >=600 ADC + unmasked) ----
        wlo, whi = max(0, t0 - PC_HW), min(TMAX, t0 + PC_HW + 1)
        tt = np.arange(wlo, whi, dtype=np.float64) - t0
        keep = a * TMPL_MAX >= PC_MIN_ADC
        pc = _pc_stats(Lch[keep], a[keep], R[keep], wlo, whi, tt, t0)
        pcu = _pc_stats(Lch, a, R, wlo, whi, tt, t0)
        feats = np.array([depth / varhat, margin2 / varhat,
                          margin2 / max(depth, 1e-9), n_comp,
                          post, post_null, width, float(abs(i0 - t0))]
                         + pc + pcu)
        out[L] = (feats, (Lc, varhat, i0))
    return out


def extract(outdir):
    rows = []
    verified = False
    for sf in sorted(glob.glob(f"{outdir}/dumps/*.npz")):
        tag = os.path.basename(sf)
        ff = f"{outdir}/full_dumps/{tag}"
        if not os.path.exists(ff):
            continue
        sd = np.load(sf)
        fd = np.load(ff, allow_pickle=True)
        if "amp_lab" not in fd.files:
            print(f"  {tag}: no amp rows, skip")
            continue
        base, act = fd["base_image"].astype(np.float64), fd["full_light"].astype(np.float64)
        labs, e = fd["labels"], np.clip(fd["e"], 0, None)
        ts_bef = sd["ts_before"].astype(np.float64)   # chain integer times = base placement
        ts_aft = sd["ts_after"].astype(np.float64)
        tru = sd["truth"].astype(np.float64)
        al, at, ar = fd["amp_lab"], fd["amp_tpc"], fd["amp_row"]
        vm = fd["veto_mask"]
        lab_t0 = {}
        for L in np.unique(al):
            m = (labs == L) & np.isfinite(ts_bef) & (ts_bef >= 0)
            if m.any():
                v, c = np.unique(np.round(ts_bef[m]).astype(int), return_counts=True)
                lab_t0[int(L)] = int(v[np.argmax(c)])
        sfeats = event_scan_feats(base, act, vm, al, at, ar, lab_t0)
        if not verified:
            # superposition check on the busiest TPC, non-vetoed channels,
            # with the same veto+rescale the feature function applies
            ar2 = ar.astype(np.float64).copy()
            for i in range(ar2.shape[0]):
                ar2[i][vm[int(at[i])]] = 0.0
            tpc_pick = int(np.bincount(at).argmax())
            recon = np.zeros((120, NT))
            for L, c, row in zip(al, at, ar2):
                if int(c) == tpc_pick and int(L) in lab_t0:
                    recon += (row / TMPL_MAX)[:, None] * shift_place(TMPL, lab_t0[int(L)])[None, :]
            keep = ~vm[tpc_pick]
            err = np.abs(recon[keep] - base[tpc_pick][keep]).max()
            scale = max(base[tpc_pick][keep].max(), 1e-9)
            print(f"  raw superposition TPC {tpc_pick} (pre-rescale): "
                  f"max|recon-base| = {err:.0f} (peak {scale:.0f})")
            verified = True
        for L, (feats, (Lc, varhat, i0)) in sfeats.items():
            m = labs == L
            fin = m & np.isfinite(tru) & np.isfinite(ts_aft) & (ts_aft >= 0)
            etot = float(e[fin].sum())
            if etot <= 0:
                continue
            v_t, c_t = np.unique(np.round(tru[fin]).astype(int), return_counts=True)
            t_tru = int(v_t[np.argmax(c_t)])
            d_truth = (float(Lc[t_tru] - Lc[i0]) / varhat
                       if 0 <= t_tru < TMAX else np.nan)
            ew = float(e[fin & (np.abs(ts_aft - tru) > 10.0)].sum())
            rows.append((tag, L, etot, ew / etot) + tuple(feats) + (d_truth,))
    names = SCAN_NAMES + ["d_truth"]
    tags = np.array([r[0] for r in rows])
    lab = np.array([r[1] for r in rows], np.int32)
    eg = np.array([r[2] for r in rows])
    wf = np.array([r[3] for r in rows])
    F = np.array([r[4:] for r in rows], np.float64)
    np.savez(os.path.join(outdir, "scan_margin_feats.npz"),
             tags=tags, lab=lab, eg=eg, wfrac=wf, F=F, names=names)
    print(f"{len(rows)} clusters -> {outdir}/scan_margin_feats.npz")
    return tags, lab, eg, wf, F, names


def evaluate(outdir):
    from subtick_conf_study2 import auc_rank, loo
    d = np.load(os.path.join(outdir, "scan_margin_feats.npz"), allow_pickle=True)
    F, eg, wf, names = d["F"], d["eg"], d["wfrac"], list(d["names"])
    tags = d["tags"]
    y = (wf <= 0.5).astype(int)
    ev = np.unique(tags, return_inverse=True)[1]
    wrong = y == 0
    print(f"\n{len(y)} clusters, {wrong.sum()} wrong "
          f"({100*eg[wrong].sum()/eg.sum():.2f}% E)")
    print("single-feature AUCs (count / E-wt):")
    for i, n in enumerate(names):
        if n == "d_truth":
            continue
        s = F[:, i]
        for sgn in (1, -1):
            a = auc_rank(y, sgn * s)
            if a >= 0.5:
                print(f"  {n:12s}{'' if sgn>0 else ' (neg)'}: "
                      f"{a:.3f} / {auc_rank(y, sgn*s, eg):.3f}")
                break
    cols = [names.index(k) for k in SCAN_NAMES]
    Xs = np.nan_to_num(F[:, cols], nan=0.0, posinf=1e9, neginf=-1e9)
    Xs = np.column_stack([Xs, np.log10(np.clip(eg, 0.1, None))])
    s = loo("logit", Xs, y, ev)
    print(f"SCAN block logit LOO: AUC {auc_rank(y, s):.3f} "
          f"(E-wt {auc_rank(y, s, eg):.3f})")
    for lo, hi, lab_ in [(5, 20, "5-20"), (20, 100, "20-100"), (100, 1e9, ">=100")]:
        m = (eg >= lo) & (eg < hi)
        if m.sum() > 10 and len(set(y[m])) == 2:
            print(f"  {lab_}: {auc_rank(y[m], s[m]):.3f}")
    dt = F[:, names.index("d_truth")]
    ok = np.isfinite(dt)
    print(f"\ndiag: wrong clusters with L(truth) < L(best)+depth/2: "
          f"{(dt[ok & wrong] < 0.5*F[ok & wrong, 0]).mean()*100:.0f}% "
          f"(correct: {(dt[ok & ~wrong] < 0.5*F[ok & ~wrong, 0]).mean()*100:.0f}%)")




# ---------------------------------------------------------------------------
# FAST batched implementation (2026-09-03): every cluster is independent, so
# stack ALL support rows of ALL clusters into one matrix -> ONE FFT for the
# whole event -> segment reductions per cluster. Two cores: float32 numpy and
# torch/CUDA (use_gpu=True; in-chain each worker owns a GPU). Adds the
# per-CLUSTER quadratic/waveform block (c_*).
# ---------------------------------------------------------------------------
TPAD32 = np.zeros(3 * NT, np.float32)
TPAD32[NT:2 * NT] = TMPL.astype(np.float32)
NVIS32 = NVIS.astype(np.float32)
TMPL_F32 = np.conj(np.fft.rfft(TMPL.astype(np.float32), NFFT)).astype(np.complex64)
_PC_TT = np.arange(-PC_HW, PC_HW + 1, dtype=np.float64)
_PC_PINV = np.linalg.pinv(np.vander(_PC_TT, 3)).astype(np.float32)   # (3,17)
_C_TT = np.arange(-4, 5, dtype=np.float64)
_C_PINV = np.linalg.pinv(np.vander(_C_TT, 3)).astype(np.float32)     # (3,9)
GAIN_W = 200.0
TARGMAX = int(np.argmax(TMPL))


def _agg_pc(n_ch, wamp, clear, sig_c, d_c, vert):
    """Aggregate per-row PC stats for ONE cluster (tiny arrays)."""
    if n_ch == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 200.0, float(PC_HW)]
    wsum = max(wamp.sum(), 1e-12)
    order = np.argsort(d_c)
    cw = np.cumsum(wamp[order])
    vote_med = (float(np.clip(d_c[order][np.searchsorted(cw, 0.5 * cw[-1])],
                              0, 200)) if cw[-1] > 0 else 200.0)
    if clear.any():
        vw = wamp[clear] / max(wamp[clear].sum(), 1e-12)
        vmean = float((vert[clear] * vw).sum())
        vert_std = float(np.sqrt((vw * np.square(vert[clear] - vmean)).sum()))
    else:
        vert_std = float(PC_HW)
    return [float(n_ch), float(clear.mean()),
            float(wamp[clear].sum() / wsum), float(np.median(sig_c)),
            float(wamp[d_c <= 10].sum() / wsum), vote_med, vert_std]


def _stack(base, act, vm, al, at, ar_raw, lab_t0):
    """Veto + group-share rescale + stack every support row (float32)."""
    ar = ar_raw.astype(np.float64).copy()
    for i in range(ar.shape[0]):
        ar[i][vm[int(at[i])]] = 0.0
    grp = {}
    for i, (L, c) in enumerate(zip(al, at)):
        t0 = lab_t0.get(int(L))
        if t0 is not None:
            grp.setdefault(int(c), []).append((t0, i))
    for c, pairs in grp.items():
        pairs.sort()
        sg, cur = [], [pairs[0]]
        for pr in pairs[1:]:
            if pr[0] - cur[-1][0] <= 10:
                cur.append(pr)
            else:
                sg.append(cur)
                cur = [pr]
        sg.append(cur)
        for members in sg:
            anchor = max(members, key=lambda pr: ar[pr[1]].max())[0]
            col = anchor + TARGMAX
            if col >= NT:
                continue
            pred = np.zeros(120)
            for t0, i in members:
                pred += ar[i] * (TPAD32[NT + col - t0] / TMPL_MAX)
            r = np.clip(base[c][:, col] / np.maximum(pred, 1e-9), 0.0, 1.0)
            for _, i in members:
                ar[i] = ar[i] * r

    D_rows, Apk_rows, a_all, cid, t0_row = [], [], [], [], []
    labs_used, t0_lab, Dmap = [], [], {}
    for L in np.unique(al):
        L = int(L)
        if L not in lab_t0:
            continue
        t0 = lab_t0[L]
        got = False
        for i in np.flatnonzero(al == L):
            c, row = int(at[i]), ar[i]
            sup = np.flatnonzero(row > SUP_THRESH)
            if sup.size == 0:
                continue
            if sup.size > TOP_CH:
                sup = sup[np.argsort(row[sup])[::-1][:TOP_CH]]
            if c not in Dmap:
                Dmap[c] = (act[c] - base[c]).astype(np.float32)
            D_rows.append(Dmap[c][sup])
            Apk_rows.append(act[c][sup, t0 + TARGMAX] if t0 + TARGMAX < NT
                            else np.zeros(sup.size))
            a_all.append(row[sup] / TMPL_MAX)
            cid.append(np.full(sup.size, len(labs_used), np.int64))
            t0_row.append(np.full(sup.size, t0, np.int64))
            got = True
        if got:
            labs_used.append(L)
            t0_lab.append(t0)
    if not labs_used:
        return None
    D = np.vstack(D_rows)
    return (D, np.concatenate(Apk_rows), np.concatenate(a_all),
            np.concatenate(cid), np.concatenate(t0_row),
            labs_used, np.asarray(t0_lab, np.int64))


def _core_numpy(D, a, cid, t0r, M):
    a32 = a.astype(np.float32)
    own = a32[:, None] * TPAD32[np.arange(NT)[None, :] - t0r[:, None] + NT]
    R = D + own
    corr = np.fft.irfft(np.fft.rfft(R, NFFT, axis=1) * TMPL_F32[None, :],
                        NFFT, axis=1)[:, :TMAX]
    Lch = -2.0 * a32[:, None] * corr + np.square(a32)[:, None] * NVIS32[None, :]
    starts = np.r_[0, 1 + np.flatnonzero(np.diff(cid))]
    LCURVE = np.add.reduceat(Lch.astype(np.float64), starts, axis=0)
    i0 = np.argmin(LCURVE, axis=1).astype(np.int64)
    # varhat: rebuild own only for rows whose cluster argmin != placement
    own0 = own
    need = i0[cid] != t0r
    if need.any():
        idx = np.flatnonzero(need)
        own0 = own.copy()
        own0[idx] = a32[idx, None] * TPAD32[
            np.arange(NT)[None, :] - i0[cid[idx]][:, None] + NT]
    rowmax = own0.max(axis=1)
    cmax = np.maximum.reduceat(rowmax, starts)
    thr = (0.01 * np.maximum(cmax[cid], 1e-9)).astype(np.float32)
    act_m = own0 > thr[:, None]
    resid = R - own0
    rs = np.add.reduceat((np.square(resid) * act_m).sum(axis=1), starts)
    rc = np.add.reduceat(act_m.sum(axis=1).astype(np.float64), starts)
    varhat = np.maximum(np.where(rc > 0, rs / np.maximum(rc, 1), 1.0), 1e-9)
    # per-row window quad + votes
    colsw = np.clip(t0r[:, None] + _PC_TT[None, :].astype(np.int64), 0, TMAX - 1)
    W = Lch[np.arange(len(a))[:, None], colsw]
    coef = _PC_PINV @ W.T
    sig_r = np.sqrt(np.maximum(np.square(R).mean(1, dtype=np.float64), 1e-12))
    gv = np.argmin(Lch, axis=1)
    Wmax, Wmin = W.max(1), W.min(1)
    return (LCURVE, i0, varhat, starts, coef[0].astype(np.float64),
            coef[1].astype(np.float64), sig_r, gv.astype(np.int64),
            Wmax.astype(np.float64), Wmin.astype(np.float64))


def _core_gpu(D, a, cid, t0r, M):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        Dt = torch.from_numpy(D).to(dev)
        a32 = torch.from_numpy(a.astype(np.float32)).to(dev)
        cidt = torch.from_numpy(cid).to(dev)
        t0t = torch.from_numpy(t0r).to(dev)
        tpad = torch.from_numpy(TPAD32).to(dev)
        nvis = torch.from_numpy(NVIS32).to(dev)
        tf = torch.from_numpy(TMPL_F32).to(dev)
        ncols = torch.arange(NT, device=dev)[None, :]
        own = a32[:, None] * tpad[ncols - t0t[:, None] + NT]
        R = Dt + own
        corr = torch.fft.irfft(torch.fft.rfft(R, NFFT, dim=1) * tf[None, :],
                               NFFT, dim=1)[:, :TMAX]
        Lch = -2.0 * a32[:, None] * corr + a32[:, None] ** 2 * nvis[None, :]
        LC = torch.zeros((M, TMAX), dtype=torch.float64, device=dev)
        LC.index_add_(0, cidt, Lch.double())
        i0 = LC.argmin(dim=1)
        need = i0[cidt] != t0t
        own0 = own
        if bool(need.any()):
            own0 = own.clone()
            idx = torch.nonzero(need, as_tuple=True)[0]
            own0[idx] = a32[idx, None] * tpad[ncols - i0[cidt[idx]][:, None] + NT]
        rowmax = own0.max(dim=1).values
        cmax = torch.zeros(M, dtype=torch.float32, device=dev)
        cmax.index_reduce_(0, cidt, rowmax, "amax", include_self=False)
        thr = 0.01 * torch.clamp(cmax[cidt], min=1e-9)
        act_m = own0 > thr[:, None]
        resid = R - own0
        rs_row = (resid.square() * act_m).sum(dim=1, dtype=torch.float64)
        rc_row = act_m.sum(dim=1, dtype=torch.float64)
        rs = torch.zeros(M, dtype=torch.float64, device=dev)
        rc = torch.zeros(M, dtype=torch.float64, device=dev)
        rs.index_add_(0, cidt, rs_row)
        rc.index_add_(0, cidt, rc_row)
        varhat = torch.clamp(torch.where(rc > 0, rs / torch.clamp(rc, min=1.0),
                                         torch.ones_like(rs)), min=1e-9)
        pc_tt = torch.from_numpy(_PC_TT.astype(np.int64)).to(dev)
        colsw = torch.clamp(t0t[:, None] + pc_tt[None, :], 0, TMAX - 1)
        W = torch.gather(Lch, 1, colsw)
        pinv = torch.from_numpy(_PC_PINV).to(dev)
        coef = pinv @ W.T
        sig_r = torch.sqrt(torch.clamp(R.double().square().mean(dim=1), min=1e-12))
        gv = Lch.argmin(dim=1)
        Wmax, Wmin = W.max(dim=1).values, W.min(dim=1).values
        starts = np.r_[0, 1 + np.flatnonzero(np.diff(cid))]
        return (LC.cpu().numpy(), i0.cpu().numpy().astype(np.int64),
                varhat.cpu().numpy(), starts,
                coef[0].double().cpu().numpy(), coef[1].double().cpu().numpy(),
                sig_r.cpu().numpy(), gv.cpu().numpy().astype(np.int64),
                Wmax.double().cpu().numpy(), Wmin.double().cpu().numpy())


def event_scan_feats(base, act, vm, al, at, ar_raw, lab_t0, use_gpu=False):
    """Batched scan-landscape + per-channel + per-cluster features.

    Same first-22 contract as event_scan_feats_ref, plus the c_* block.
    One FFT pass per event; use_gpu routes the heavy core through torch."""
    st = _stack(base, act, vm, al, at, ar_raw, lab_t0)
    if st is None:
        return {}
    D, Apk, a, cid, t0r, labs_used, t0_lab = st
    M, N = len(labs_used), a.size
    core = _core_gpu if use_gpu else _core_numpy
    (LCURVE, i0, varhat, starts, c2, c1, sig_r, gv, Wmax, Wmin) = \
        core(D, a, cid, t0r, M)

    tgrid = np.arange(TMAX)[None, :]
    L0 = LCURVE[np.arange(M), i0]
    depth = np.maximum(-L0, 0.0)
    interior = np.zeros((M, TMAX), bool)
    interior[:, 1:-1] = (LCURVE[:, 1:-1] <= LCURVE[:, :-2]) & \
                        (LCURVE[:, 1:-1] <= LCURVE[:, 2:])
    compm = interior & (np.abs(tgrid - i0[:, None]) > 10)
    m2raw = np.where(compm, LCURVE, np.inf).min(axis=1)
    margin2 = np.where(np.isfinite(m2raw), m2raw - L0, -L0)
    n_comp = (compm & (LCURVE - L0[:, None] < 0.5 * depth[:, None])).sum(1)
    Ln = LCURVE / varhat[:, None]
    L0n = Ln[np.arange(M), i0]
    wmat = np.where(compm, -0.5 * Ln, -np.inf)
    wmat[np.arange(M), i0] = -0.5 * L0n
    wmax_ = wmat.max(axis=1)
    lse = wmax_ + np.log(np.exp(wmat - wmax_[:, None]).sum(axis=1))
    post = np.exp(-0.5 * L0n - lse)
    post_null = np.exp(-0.5 * L0n - np.logaddexp(lse, 0.0))
    width = (LCURVE <= (L0 + 0.5 * depth)[:, None]).sum(axis=1)
    d_place = np.abs(i0 - t0_lab)

    # per-row derived
    vert = np.where(np.abs(c2) > 1e-12, -c1 / (2.0 * np.where(c2 == 0, 1, c2)),
                    np.inf)
    nscale = 2.0 * np.abs(a) * sig_r * np.sqrt(NVIS[np.minimum(t0r, TMAX - 1)])
    sig_c = (Wmax - Wmin) / np.maximum(nscale, 1e-12)
    clear = (c2 > 0) & (np.abs(vert) <= PC_HW) & (sig_c > 2.0)
    d_c = np.abs(gv - t0r).astype(np.float64)
    maskb = a * TMPL_MAX >= PC_MIN_ADC

    # per-cluster waveform z at own peak column (model total == base there)
    pkcol = np.minimum(t0r + TARGMAX, NT - 1)
    Dpk = D[np.arange(N), pkcol].astype(np.float64)
    z_pk = -Dpk / np.sqrt(GAIN_W * np.maximum(Apk, 100.0))
    own_pk = a * TMPL_MAX
    res_pk = np.clip(Dpk + own_pk, 0.0, None)
    zn = np.add.reduceat(np.ones(N), starts)
    zsum = np.add.reduceat(np.abs(z_pk), starts)
    z2f = np.add.reduceat((np.abs(z_pk) > 2.0).astype(np.float64), starts)
    dot = np.add.reduceat(own_pk * res_pk, starts)
    na2 = np.add.reduceat(own_pk ** 2, starts)
    nr2 = np.add.reduceat(res_pk ** 2, starts)
    cosres = np.where((na2 > 1e-12) & (nr2 > 1e-12),
                      dot / np.sqrt(np.maximum(na2 * nr2, 1e-12)), 0.0)

    # per-cluster local quadratic on the cluster curve (+-4 around i0)
    ccols = np.clip(i0[:, None] + _C_TT[None, :].astype(np.int64), 0, TMAX - 1)
    CW = Ln[np.arange(M)[:, None], ccols]
    ca2 = (_C_PINV.astype(np.float64) @ CW.T)[0]
    c_sig_t = np.minimum(np.where(ca2 > 1e-9,
                                  1.0 / np.sqrt(np.maximum(ca2, 1e-9)), 10.0),
                         10.0)
    c_contrast = CW.max(1) - CW.min(1)

    out = {}
    bounds = np.r_[starts, N]
    for k, L in enumerate(labs_used):
        rows = slice(bounds[k], bounds[k + 1])
        wamp = np.maximum(a[rows], 0.0)
        mk = maskb[rows]
        pc = _agg_pc(int(mk.sum()), wamp[mk], clear[rows][mk],
                     sig_c[rows][mk], d_c[rows][mk], vert[rows][mk])
        pcu = _agg_pc(int(mk.size), wamp, clear[rows], sig_c[rows],
                      d_c[rows], vert[rows])
        feats = np.array(
            [depth[k] / varhat[k], margin2[k] / varhat[k],
             margin2[k] / max(depth[k], 1e-9), float(n_comp[k]),
             float(post[k]), float(post_null[k]), float(width[k]),
             float(d_place[k])] + pc + pcu +
            [float(c_sig_t[k]), float(c_contrast[k]),
             float(zsum[k] / max(zn[k], 1)), float(z2f[k] / max(zn[k], 1)),
             float(cosres[k]), float(np.log10(varhat[k]))])
        out[L] = (feats, (LCURVE[k], varhat[k], int(i0[k])))
    return out


if __name__ == "__main__":
    out = sys.argv[1]
    if "--eval" in sys.argv:
        if not os.path.exists(os.path.join(out, "scan_margin_feats.npz")):
            extract(out)
        evaluate(out)
    else:
        extract(out)
