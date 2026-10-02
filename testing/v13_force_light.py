"""Offline test of a LIGHT-based force rule for isolated unclustered hits (user 2026-09-30).
Current stage 4.9: a noise hit takes the time of the nearest assigned hit.  Proposal: beyond a maximum distance
D_max (20 -> 10 cm) spatial information is useless; decide from the light instead: look at the few light channels
nearest to the hit and pick, among the family times of the hit's TPC, the one whose light (residual or actual) is
strongest there.  Also compared: the cluster-density vote (force_noise_vote), and the oracle (best family time).
Exact emulation on certification dumps (force is the last assignment stage; clustered assignments are final).
python v13_force_light.py <arm_dir> [n_events]"""
import glob
import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from subtick_refine import build_sipm_positions          # noqa: E402
from force_noise_vote import _noise_vote                  # noqa: E402

CUT = 10.0; PEAK = 105; K_CH = 6; D_FLOOR = 5.0; ISOLATE = 25.0; HW = 2
D_MAXES = (10.0, 15.0, 20.0)


def light_scores(R, A, veto, sipm_t, xyz_h, cands):
    """per hit x candidate: residual score, actual score (K nearest non-vetoed channels, 1/d^2 weights)."""
    chan_ok = np.flatnonzero(~veto)
    if chan_ok.size == 0 or cands.size == 0:
        return None, None
    d = np.linalg.norm(xyz_h[:, None, :] - sipm_t[chan_ok][None, :, :], axis=2)
    k = min(K_CH, chan_ok.size)
    near = np.argpartition(d, k - 1, axis=1)[:, :k]                      # (n_h, k) indices into chan_ok
    w = 1.0 / np.maximum(np.take_along_axis(d, near, 1), D_FLOOR) ** 2
    w /= w.sum(1, keepdims=True)
    idx = np.clip(np.round(cands).astype(int) + PEAK, HW, R.shape[1] - HW - 1)
    # window-averaged residual / actual at each candidate time for every channel: (120, F)
    rw = np.stack([R[:, i - HW:i + HW + 1].mean(1) for i in idx], 1)
    aw = np.stack([A[:, i - HW:i + HW + 1].mean(1) for i in idx], 1)
    ch = chan_ok[near]                                                   # (n_h, k)
    s_res = np.einsum("hk,hkf->hf", w, np.maximum(rw[ch], 0.0))
    s_act = np.einsum("hk,hkf->hf", w, np.maximum(aw[ch], 0.0))
    return s_res, s_act


def main(arm, n_max=None):
    files = sorted(glob.glob(f"{arm}/full_dumps/*.npz"))[: int(n_max) if n_max else None]
    rules = ["nearest hit (current)", "light residual, K nearest ch", "light actual, K nearest ch", "residual, fallback nearest if < 0", "cluster-density vote", "oracle (best family in TPC)"]
    T = {D: np.zeros((len(rules), 2)) for D in D_MAXES}       # E, E_correct for far hits
    T_all = np.zeros(2); n_far = {D: 0 for D in D_MAXES}; t_light = 0.0
    sipm = None
    for ff in files:
        z = np.load(ff, allow_pickle=True)
        if sipm is None:
            sipm = build_sipm_positions(str(z["src_file"]))
        e = np.clip(z["e"], 0, None).astype(float); ts = z["ts_final"].astype(float); tru = z["truth_t0"].astype(float)
        labs = z["labels"].astype(int); tpc = z["tpc"].astype(int); xyz = np.stack([z["x"], z["y"], z["z"]], 1).astype(float)
        full = z["full_light"]; base = z["base_image"]; veto = z["veto_mask"]
        ok = np.isfinite(tru) & (tru >= 0) & (tru <= 885)
        cl = (labs >= 0) & np.isfinite(ts) & (ts >= 0)
        # isolation veto as in force_simple (nearest OTHER hit over all hits)
        d2, _ = cKDTree(xyz).query(xyz, k=2); isolated = d2[:, 1] > ISOLATE
        noise = (labs < 0) & ok & ~isolated
        tree_a = cKDTree(xyz[cl]); d1, j1 = tree_a.query(xyz[noise], k=1)
        idx_a = np.flatnonzero(cl); t_near = ts[idx_a[j1]]
        xn = xyz[noise]; en = e[noise]; trn = tru[noise]; tpn = tpc[noise]
        T_all += [en.sum(), (en * (np.abs(t_near - trn) <= CUT)).sum()]
        # cluster-density vote on all noise hits (its own contact rule inside)
        t_vote = _noise_vote(xn, xyz[cl], labs[cl], ts[cl], np.array([0.0] * cl.sum()), lam=14.0, big_w=0.5, big_mev=20.0,
                             bin_ticks=6.0, r_max=100.0, r_exact=15.0, contact_cm=3.0)
        # family energies for the vote's big_w: recompute properly
        labE = {int(L): e[labs == L].sum() for L in np.unique(labs[cl])}
        E_a = np.array([labE[int(L)] for L in labs[cl]])
        t_vote = _noise_vote(xn, xyz[cl], labs[cl], ts[cl], E_a, lam=14.0, big_w=0.5, big_mev=20.0, bin_ticks=6.0, r_max=100.0, r_exact=15.0, contact_cm=3.0)
        # light decisions per TPC
        t_res = t_near.copy(); t_act = t_near.copy(); t_resfb = t_near.copy(); t_orc = t_near.copy()
        t0 = time.time()
        for t_ in np.unique(tpn):
            if int(t_) not in sipm or t_ >= full.shape[0]:
                continue
            mh = np.flatnonzero(tpn == t_)
            cands = np.unique(np.round(ts[cl & (tpc == t_)]))
            if cands.size == 0:
                continue
            R = full[t_].astype(np.float32) - base[t_].astype(np.float32); A = full[t_].astype(np.float32)
            s_res, s_act = light_scores(R, A, veto[t_], sipm[int(t_)], xn[mh], cands)
            if s_res is None:
                continue
            t_res[mh] = cands[np.argmax(s_res, 1)]
            t_act[mh] = cands[np.argmax(s_act, 1)]
            best = np.argmax(s_res, 1); has = s_res.max(1) > 0
            t_resfb[mh[has]] = cands[best[has]]
            dtr = np.abs(cands[None, :] - trn[mh][:, None]); jo = np.argmin(dtr, 1)
            t_orc[mh] = np.where(dtr[np.arange(mh.size), jo] <= CUT, cands[jo], t_near[mh])
        t_light += time.time() - t0
        for D in D_MAXES:
            far = d1 > D; n_far[D] += int(far.sum())
            for r_i, tt in enumerate((t_near, t_res, t_act, t_resfb, t_vote, t_orc)):
                T[D][r_i] += [en[far].sum(), (en[far] * (np.abs(tt[far] - trn[far]) <= CUT)).sum()]
    print(f"{len(files)} events; noise hits (not isolated, with truth): E {T_all[0]/1e3:.2f} GeV, nearest-hit correct {T_all[1]/T_all[0]:.3f}; light scoring time {t_light/len(files)*1000:.0f} ms/event")
    for D in D_MAXES:
        print(f"\nhits farther than {D:.0f} cm from any assigned clustered hit: n {n_far[D]} ({n_far[D]/len(files):.0f}/event), E {T[D][0,0]/1e3:.2f} GeV ({100*T[D][0,0]/T_all[0]:.0f} % of noise energy)")
        for r_i, name in enumerate(rules):
            print(f"   {name:36s} correct {T[D][r_i,1]/T[D][r_i,0]:.3f}   (whole noise class would become {(T_all[1] - T[D][0,1] + T[D][r_i,1])/T_all[0]:.3f})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
