"""Second pass of the light-based force rule for isolated noise hits: channel-count scan for the residual matched
filter, normalised variants, the cluster-density vote recomputed per candidate, and vote x light combinations.
python v13_force_light2.py <arm_dir> [n_events] [rules: scan|final]"""
import glob
import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from subtick_refine import build_sipm_positions          # noqa: E402

CUT = 10.0; PEAK = 105; D_FLOOR = 5.0; ISOLATE = 25.0; HW = 2
D_MAXES = (10.0, 15.0, 20.0)
LAM = 14.0; BIG_W = 0.5; BIG_MEV = 20.0; BIN = 6.0; R_MAX = 100.0; K_NEIGH = 300


def window_means(X, cands):
    idx = np.clip(np.round(cands).astype(int) + PEAK, HW, X.shape[1] - HW - 1)
    return np.stack([X[:, i - HW:i + HW + 1].mean(1) for i in idx], 1)      # (120, F)


def chan_weights(sipm_t, chan_ok, xyz_h, K):
    d = np.linalg.norm(xyz_h[:, None, :] - sipm_t[chan_ok][None, :, :], axis=2)
    k = min(K, chan_ok.size)
    near = np.argpartition(d, k - 1, axis=1)[:, :k] if k < chan_ok.size else np.tile(np.arange(chan_ok.size), (xyz_h.shape[0], 1))
    w = 1.0 / np.maximum(np.take_along_axis(d, near, 1), D_FLOOR) ** 2
    return chan_ok[near], w / w.sum(1, keepdims=True)


def vote_scores(xh, d1, tree_a, xyz_a, lab_a, ts_a, E_a, cands):
    """cluster-density vote weight per (hit, candidate) as in force_noise_vote (K nearest assigned hits as proxies)."""
    n = xh.shape[0]; V = np.zeros((n, cands.size))
    k = min(K_NEIGH, xyz_a.shape[0])
    d, j = tree_a.query(xh, k=k, distance_upper_bound=R_MAX)
    for i in range(n):
        ok = np.isfinite(d[i]); dd = d[i][ok]; jj = j[i][ok]
        if dd.size == 0:
            continue
        L = lab_a[jj]
        # min distance per cluster and that member's time
        o = np.lexsort((dd, L)); L, dd, jj = L[o], dd[o], jj[o]
        first = np.r_[True, L[1:] != L[:-1]]
        dmin = dd[first]; tm = ts_a[jj[first]]; Ec = E_a[jj[first]]
        w = np.exp(-(dmin - d1[i]) / LAM) * np.where(Ec >= BIG_MEV, BIG_W, 1.0)
        for f, c in enumerate(cands):
            V[i, f] = w[np.abs(tm - c) <= BIN].sum()
    return V


def main(arm, n_max=None, mode="scan"):
    files = sorted(glob.glob(f"{arm}/full_dumps/*.npz"))[: int(n_max) if n_max else None]
    Ks = (3, 6, 12, 30, 120)
    names = ["nearest hit (current)"] + [f"residual 1/d^2, K={K}" for K in Ks] + ["residual/actual, K=12", "locality (actual share at K=12)", "vote (density)",
             "vote x (1 + res12/med)", "vote x (1 + 3 res12/med)", "vote, light tie-break (top-2 within 30 %)", "oracle (best family in TPC)"]
    T = {D: np.zeros((len(names), 2)) for D in D_MAXES}; T_all = np.zeros(2); t_light = 0.0; t_vote = 0.0; sipm = None; n_far = {D: 0 for D in D_MAXES}
    for ff in files:
        z = np.load(ff, allow_pickle=True)
        if sipm is None:
            sipm = build_sipm_positions(str(z["src_file"]))
        e = np.clip(z["e"], 0, None).astype(float); ts = z["ts_final"].astype(float); tru = z["truth_t0"].astype(float)
        labs = z["labels"].astype(int); tpc = z["tpc"].astype(int); xyz = np.stack([z["x"], z["y"], z["z"]], 1).astype(float)
        full = z["full_light"]; base = z["base_image"]; veto = z["veto_mask"]
        ok = np.isfinite(tru) & (tru >= 0) & (tru <= 885)
        cl = (labs >= 0) & np.isfinite(ts) & (ts >= 0)
        d2, _ = cKDTree(xyz).query(xyz, k=2); isolated = d2[:, 1] > ISOLATE
        noise = (labs < 0) & ok & ~isolated
        xyz_a = xyz[cl]; tree_a = cKDTree(xyz_a); d1, j1 = tree_a.query(xyz[noise], k=1)
        ts_a = ts[cl]; lab_a = labs[cl]; labE = {int(L): e[labs == L].sum() for L in np.unique(lab_a)}; E_a = np.array([labE[int(L)] for L in lab_a])
        t_near = ts_a[j1]; xn = xyz[noise]; en = e[noise]; trn = tru[noise]; tpn = tpc[noise]
        T_all += [en.sum(), (en * (np.abs(t_near - trn) <= CUT)).sum()]
        far_any = d1 > min(D_MAXES)
        out = {nm: t_near.copy() for nm in names}
        for t_ in np.unique(tpn[far_any]):
            if int(t_) not in sipm or t_ >= full.shape[0]:
                continue
            mh = np.flatnonzero((tpn == t_) & far_any)
            cands = np.unique(np.round(ts_a[tpc[cl] == t_]))
            chan_ok = np.flatnonzero(~veto[t_])
            if cands.size == 0 or chan_ok.size == 0:
                continue
            tl = time.time()
            A = full[t_].astype(np.float32); R = A - base[t_].astype(np.float32)
            rw = np.maximum(window_means(R, cands), 0.0); aw = np.maximum(window_means(A, cands), 0.0)
            tot_a = aw.sum(0) + 1.0
            S = {}
            for K in Ks:
                ch, w = chan_weights(sipm[int(t_)], chan_ok, xn[mh], K)
                S[K] = (np.einsum("hk,hkf->hf", w, rw[ch]), np.einsum("hk,hkf->hf", w, aw[ch]))
            t_light += time.time() - tl
            tv = time.time()
            V = vote_scores(xn[mh], d1[mh], tree_a, xyz_a, lab_a, ts_a, E_a, cands)
            t_vote += time.time() - tv
            res12, act12 = S[12]
            med = np.median(res12[res12 > 0]) if (res12 > 0).any() else 1.0
            pick = lambda s: cands[np.argmax(s, 1)]
            for K in Ks:
                out[f"residual 1/d^2, K={K}"][mh] = pick(S[K][0])
            out["residual/actual, K=12"][mh] = pick(res12 / (act12 + 50.0))
            out["locality (actual share at K=12)"][mh] = pick(act12 / tot_a[None, :])
            out["vote (density)"][mh] = pick(V)
            out["vote x (1 + res12/med)"][mh] = pick(V * (1 + res12 / med))
            out["vote x (1 + 3 res12/med)"][mh] = pick(V * (1 + 3 * res12 / med))
            # tie-break: among candidates with vote >= 0.7 x best vote, take the largest residual
            top = V >= 0.7 * V.max(1, keepdims=True)
            out["vote, light tie-break (top-2 within 30 %)"][mh] = cands[np.argmax(np.where(top, res12, -1.0), 1)]
            dtr = np.abs(cands[None, :] - trn[mh][:, None]); jo = np.argmin(dtr, 1)
            out["oracle (best family in TPC)"][mh] = np.where(dtr[np.arange(mh.size), jo] <= CUT, cands[jo], t_near[mh])
        for D in D_MAXES:
            far = d1 > D; n_far[D] += int(far.sum())
            for r_i, nm in enumerate(names):
                tt = out[nm]
                T[D][r_i] += [en[far].sum(), (en[far] * (np.abs(tt[far] - trn[far]) <= CUT)).sum()]
    print(f"{len(files)} events; noise E {T_all[0]/1e3:.2f} GeV, nearest-hit correct {T_all[1]/T_all[0]:.3f}; light scoring {1000*t_light/len(files):.0f} ms/event, vote {1000*t_vote/len(files):.0f} ms/event (python loop, unoptimised)")
    for D in D_MAXES:
        print(f"\nfarther than {D:.0f} cm: {n_far[D]/len(files):.0f} hits/event, {T[D][0,0]/1e3:.2f} GeV ({100*T[D][0,0]/T_all[0]:.0f} % of noise E)")
        for r_i, nm in enumerate(names):
            print(f"   {nm:44s} correct {T[D][r_i,1]/T[D][r_i,0]:.3f}   noise class -> {(T_all[1] - T[D][0,1] + T[D][r_i,1])/T_all[0]:.3f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None, sys.argv[3] if len(sys.argv) > 3 else "scan")
