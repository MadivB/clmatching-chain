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

SCAN_NAMES = ["depth_n", "margin2_n", "rel_margin", "n_comp", "post",
              "post_null", "width", "d_place"]


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


def event_scan_feats(base, act, vm, al, at, ar_raw, lab_t0):
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
        Lc = cluster_curve(A, O, a)
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
        feats = np.array([depth / varhat, margin2 / varhat,
                          margin2 / max(depth, 1e-9), n_comp,
                          post, post_null, width, float(abs(i0 - t0))])
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


if __name__ == "__main__":
    out = sys.argv[1]
    if "--eval" in sys.argv:
        if not os.path.exists(os.path.join(out, "scan_margin_feats.npz")):
            extract(out)
        evaluate(out)
    else:
        extract(out)
