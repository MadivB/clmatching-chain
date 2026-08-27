"""Offline stage-F combination sweep over the per-channel fit dumps.

The v111 instrumented run fits ALL lit channels; here we re-run only the
combination step (filter + kNN + weighting) per config and score each by
the mechanism metric (corr of per-hit correction vs true within-group
deviation) and the core per-hit sigma.

Usage: python subtick_sweep.py v111CD_out/dumps
"""
import glob
import sys

import numpy as np

sys.path.insert(0, ".")
from subtick_refine import build_sipm_positions

FLOW = ("/global/cfs/cdirs/dunepro/people/abooth/nd-production/output/MiniProdN5/"
        "run-ndlar-flow/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift/FLOW/0000000/"
        "MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.0000001.FLOW.hdf5")
T = 16.0


def load_events(dumps):
    evs = []
    for f in sorted(glob.glob(f"{dumps}/*.npz")):
        d = np.load(f)
        off = np.concatenate([[0], np.cumsum(d["ch_n"])])
        evs.append({k: d[k] for k in d.files} | {"off": off})
    return evs


def per_hit_times(ev, pos, amp_min, knn, weight):
    """Return (hit_idx, t_hit) arrays for this event under the config."""
    tsE = ev["ts_groupE"].astype(np.float64)
    gid, htpc = ev["group_id"], ev["hit_tpc"]
    hxyz = np.stack([ev["hit_x"], ev["hit_y"], ev["hit_z"]], 1).astype(np.float64)
    out_i, out_t = [], []
    for b in range(ev["ch_gi"].size):
        gi, c, i0, i1 = int(ev["ch_gi"][b]), int(ev["ch_tpc"][b]), ev["off"][b], ev["off"][b + 1]
        rows = ev["ch_rows"][i0:i1]
        amp = ev["ch_amp"][i0:i1].astype(np.float64)
        tch = ev["ch_t"][i0:i1].astype(np.float64)
        if c not in pos:
            continue
        sel = amp >= amp_min
        if sel.sum() < min(knn, amp.size):
            sel = np.zeros(amp.size, bool)
            sel[np.argsort(amp)[::-1][:knn]] = True
        if sel.sum() < 2:
            continue
        r, a, t = rows[sel], amp[sel], tch[sel]
        hidx = np.flatnonzero((gid == gi) & (htpc == c))
        if not hidx.size:
            continue
        d = np.linalg.norm(hxyz[hidx][:, None, :] - pos[c][r][None, :, :], axis=2)
        kk = min(knn, r.size)
        near = np.argsort(d, axis=1)[:, :kk]
        dn = np.maximum(np.take_along_axis(d, near, axis=1), 1.0)
        an = a[near]
        tn = t[near]
        if weight == "1/d":
            wn = 1.0 / dn
        elif weight == "1/d2":
            wn = 1.0 / np.square(dn)
        elif weight == "amp/d":
            wn = an / dn
        elif weight == "amp/d2":
            wn = an / np.square(dn)
        elif weight == "flat":
            wn = np.ones_like(dn)
        else:
            raise ValueError(weight)
        out_i.append(hidx)
        out_t.append((wn * tn).sum(axis=1) / wn.sum(axis=1))
    if not out_i:
        return np.array([], int), np.array([])
    return np.concatenate(out_i), np.concatenate(out_t)


def score(evs, pos, amp_min, knn, weight):
    cx, cy, cw, res, ee = [], [], [], [], []
    for ev in evs:
        tr = ev["truth"].astype(np.float64)
        tsb = ev["ts_before"].astype(np.float64)
        tsE = ev["ts_groupE"].astype(np.float64)
        e = ev["e"].astype(np.float64)
        gid = ev["group_id"]
        tsF = tsE.copy()
        hi, ht = per_hit_times(ev, pos, amp_min, knn, weight)
        tsF[hi] = ht
        core = (np.isfinite(tr) & np.isfinite(tsb) & (tsb >= 0) & (e > 0)
                & (np.abs(tsb - tr) <= 5.0))
        res.append((tsF - tr)[core]); ee.append(e[core])
        for g in np.unique(gid[core & (gid >= 0)]):
            gm = core & (gid == g)
            if gm.sum() < 2:
                continue
            tmu = np.average(tr[gm], weights=e[gm])
            cx.append(tr[gm] - tmu); cy.append((tsF - tsE)[gm]); cw.append(e[gm])
    r, w = np.concatenate(res), np.concatenate(ee)
    mu = np.average(r, weights=w)
    sig = np.sqrt(np.average(np.square((r - mu) * T), weights=w))
    cx, cy, cw = map(np.concatenate, (cx, cy, cw))
    mx, my = np.average(cx, weights=cw), np.average(cy, weights=cw)
    cov = np.average((cx - mx) * (cy - my), weights=cw)
    vx = np.average(np.square(cx - mx), weights=cw)
    vy = np.average(np.square(cy - my), weights=cw)
    corr = cov / max(np.sqrt(vx * vy), 1e-12)
    slope = cov / max(vx, 1e-12)
    return sig, corr, slope, np.sqrt(vy) * T


def main():
    evs = load_events(sys.argv[1])
    pos = build_sipm_positions(FLOW)
    print(f"{len(evs)} events loaded")
    print(f"{'amp_min':>8} {'knn':>4} {'weight':>7} {'sigma':>7} {'corr':>6} {'slope':>6} {'corr_std':>8}")
    best = []
    for amp_min in (500, 2000, 8000, 20000, 50000):
        for knn in (3, 6, 12):
            for weight in ("1/d", "1/d2", "amp/d", "flat"):
                sig, corr, slope, cstd = score(evs, pos, amp_min, knn, weight)
                print(f"{amp_min:>8} {knn:>4} {weight:>7} {sig:7.2f} {corr:6.3f} {slope:6.3f} {cstd:8.2f}")
                best.append((sig, corr, amp_min, knn, weight, slope))
    best.sort()
    print("\nbest by sigma:", best[0])
    bc = sorted(best, key=lambda t: -t[1])[0]
    print("best by corr :", bc)


if __name__ == "__main__":
    main()
