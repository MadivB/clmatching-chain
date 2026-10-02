"""Third pass: group-aggregated decisions for far noise hits (single linkage among far hits), energy priors,
long-range vote.  python v13_force_light3.py <arm_dir> [n_events]"""
import glob
import os
import sys
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from subtick_refine import build_sipm_positions          # noqa: E402
from v13_force_light2 import window_means, chan_weights, vote_scores, CUT, ISOLATE  # noqa: E402

D_MAXES = (10.0, 15.0, 20.0); GROUP_CM = (10.0, 20.0, 40.0)


def groups(xh, eps):
    tr = cKDTree(xh); pr = tr.query_pairs(eps, output_type="ndarray"); n = xh.shape[0]
    if pr.size == 0:
        return np.arange(n)
    return connected_components(coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(n, n)), directed=False)[1]


def agg_pick(S, comp, cands):
    """sum scores over the members of each group, assign the group's argmax to all members"""
    n_g = comp.max() + 1
    G = np.zeros((n_g, S.shape[1]))
    np.add.at(G, comp, S)
    return cands[np.argmax(G, 1)][comp]


def main(arm, n_max=None):
    files = sorted(glob.glob(f"{arm}/full_dumps/*.npz"))[: int(n_max) if n_max else None]
    names = (["nearest hit (current)", "residual K=6", "vote", "vote x (1+res6/med)", "max family E in TPC", "E-weighted vote lam=50, no big penalty"]
             + [f"group {g:.0f} cm: residual K=6" for g in GROUP_CM] + [f"group {g:.0f} cm: vote" for g in GROUP_CM]
             + [f"group {g:.0f} cm: vote x (1+res6/med)" for g in GROUP_CM] + [f"group {g:.0f} cm: log-vote + log-res" for g in GROUP_CM]
             + ["oracle (best family in TPC)"])
    T = {D: np.zeros((len(names), 2)) for D in D_MAXES}; T_all = np.zeros(2); sipm = None; gsz = []
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
            sel_a = tpc[cl] == t_
            cands = np.unique(np.round(ts_a[sel_a])); chan_ok = np.flatnonzero(~veto[t_])
            if cands.size == 0 or chan_ok.size == 0:
                continue
            A = full[t_].astype(np.float32); R = A - base[t_].astype(np.float32)
            rw = np.maximum(window_means(R, cands), 0.0)
            ch, w = chan_weights(sipm[int(t_)], chan_ok, xn[mh], 6)
            res6 = np.einsum("hk,hkf->hf", w, rw[ch]); med = np.median(res6[res6 > 0]) if (res6 > 0).any() else 1.0
            V = vote_scores(xn[mh], d1[mh], tree_a, xyz_a, lab_a, ts_a, E_a, cands)
            # family energy per candidate in this TPC
            Ef = np.array([e[cl][sel_a & (np.abs(ts_a - c) <= 6.0)].sum() for c in cands])
            # long-range energy-weighted vote: sum over assigned hits within 100 cm of E_hit * exp(-d/50) per candidate
            k = min(300, xyz_a.shape[0]); dd, jj = tree_a.query(xn[mh], k=k, distance_upper_bound=100.0)
            VE = np.zeros_like(V)
            for i in range(mh.size):
                okk = np.isfinite(dd[i])
                if not okk.any():
                    continue
                tt = ts_a[jj[i][okk]]; ww = e[cl][jj[i][okk]] * np.exp(-dd[i][okk] / 50.0)
                for f, c in enumerate(cands):
                    VE[i, f] = ww[np.abs(tt - c) <= 6.0].sum()
            pick = lambda s: cands[np.argmax(s, 1)]
            out["residual K=6"][mh] = pick(res6); out["vote"][mh] = pick(V); out["vote x (1+res6/med)"][mh] = pick(V * (1 + res6 / med))
            out["max family E in TPC"][mh] = cands[np.argmax(Ef)]; out["E-weighted vote lam=50, no big penalty"][mh] = pick(VE)
            for g in GROUP_CM:
                comp = groups(xn[mh], g); gsz.append(np.bincount(comp).mean())
                out[f"group {g:.0f} cm: residual K=6"][mh] = agg_pick(res6, comp, cands)
                out[f"group {g:.0f} cm: vote"][mh] = agg_pick(V, comp, cands)
                out[f"group {g:.0f} cm: vote x (1+res6/med)"][mh] = agg_pick(V * (1 + res6 / med), comp, cands)
                out[f"group {g:.0f} cm: log-vote + log-res"][mh] = agg_pick(np.log(V + 1e-3) + np.log(1 + res6 / med), comp, cands)
            dtr = np.abs(cands[None, :] - trn[mh][:, None]); jo = np.argmin(dtr, 1)
            out["oracle (best family in TPC)"][mh] = np.where(dtr[np.arange(mh.size), jo] <= CUT, cands[jo], t_near[mh])
        for D in D_MAXES:
            far = d1 > D
            for r_i, nm in enumerate(names):
                tt = out[nm]; T[D][r_i] += [en[far].sum(), (en[far] * (np.abs(tt[far] - trn[far]) <= CUT)).sum()]
    print(f"{len(files)} events; noise E {T_all[0]/1e3:.2f} GeV, nearest-hit correct {T_all[1]/T_all[0]:.3f}; mean group size {np.mean(gsz):.1f}")
    for D in D_MAXES:
        print(f"\nfarther than {D:.0f} cm ({T[D][0,0]/1e3:.2f} GeV):")
        for r_i, nm in enumerate(names):
            print(f"   {nm:44s} correct {T[D][r_i,1]/T[D][r_i,0]:.3f}   noise class -> {(T_all[1] - T[D][0,1] + T[D][r_i,1])/T_all[0]:.3f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
