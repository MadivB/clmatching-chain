"""v0.8 Task 1 — structural spatial score S(cluster, family) + truth study.

Features (V080_DESIGN.md): contact, continuation (one-piece), crossing
(two-piece contact), vertex pointing, shower-cone membership with
directionality — all in global (x,y,z), hence multi-TPC by construction.

Run the truth study on dumps:

    python spatial_score.py <dumps_dir> [n_events] [cands_per_event]

Outputs per-feature AUCs (midranks — beware tie artifacts), a combined
logistic score, and the OPERATIONAL test: detecting clusters that the chain
actually misplaced.
"""
from __future__ import annotations

import glob
import sys

import numpy as np
from scipy.spatial import cKDTree

RNG = np.random.default_rng(11)
CONTACT_CM = 3.0          # the hit-graph linking radius (contact criterion)
CROSS_ANGLE_DEG = 30.0
TRUTH_GAP = 3.0
CUT = 10.0


# ---------------------------------------------------------------------------
def cluster_shape(xyz, e):
    """Energy-weighted PCA summary of one cluster."""
    w = np.clip(e, 1e-6, None)
    mu = np.average(xyz, axis=0, weights=w)
    d = xyz - mu
    cov = (d * w[:, None]).T @ d / w.sum()
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals, evecs = np.clip(evals[order], 0, None), evecs[:, order]
    v1 = evecs[:, 0]
    u = d @ v1
    ends = xyz[np.argmin(u)], xyz[np.argmax(u)]
    length = float(u.max() - u.min())
    lin = float((evals[0] - evals[1]) / max(evals[0], 1e-9))
    return dict(mu=mu, v1=v1, evals=evals, ends=ends, length=length,
                lin=lin, u=u, xyz=xyz, e=e)


def family_frame(hits_xyz, hits_e):
    """Family axis + vertex estimate with directionality.

    Vertex = the NARROW end (transverse RMS in the first vs last quintile
    along the axis; showers widen with depth).  Returns both ends so callers
    can also test the ambiguous direction."""
    sh = cluster_shape(hits_xyz, hits_e)
    u = sh["u"]
    lo, hi = np.quantile(u, [0.2, 0.8])
    d = hits_xyz - sh["mu"]
    trans = d - np.outer(u, sh["v1"])
    tr = np.linalg.norm(trans, axis=1)
    w = np.clip(hits_e, 1e-6, None)
    rms_lo = float(np.sqrt(np.average(tr[u <= lo] ** 2, weights=w[u <= lo]))) \
        if (u <= lo).any() else 1e9
    rms_hi = float(np.sqrt(np.average(tr[u >= hi] ** 2, weights=w[u >= hi]))) \
        if (u >= hi).any() else 1e9
    vtx = sh["ends"][0] if rms_lo <= rms_hi else sh["ends"][1]
    far = sh["ends"][1] if rms_lo <= rms_hi else sh["ends"][0]
    axis = far - vtx
    axis = axis / max(np.linalg.norm(axis), 1e-9)
    return dict(sh=sh, vtx=vtx, axis=axis)


def _line_point_dca(p0, v, q):
    return float(np.linalg.norm(np.cross(q - p0, v)))


def pair_features(csh, fam, ftree, fam_members):
    """Features of (cluster c, family F)."""
    d_eu, idx = ftree.query(csh["xyz"], k=1)
    d_eu_min = float(d_eu.min())
    contact = d_eu_min <= CONTACT_CM

    # continuation vs nearest track-like member
    cont_gap = cont_cos = cont_lat = np.nan
    cross = 0.0
    best = None
    for msh in fam_members:
        if msh["length"] < 5.0 or msh["lin"] < 0.6:
            continue
        gaps = [np.linalg.norm(ce - me) for ce in csh["ends"] for me in msh["ends"]]
        g = float(min(gaps))
        if best is None or g < best[0]:
            best = (g, msh)
    if best is not None:
        g, msh = best
        cont_gap = g
        cont_cos = float(abs(np.dot(csh["v1"], msh["v1"])))
        cont_lat = _line_point_dca(msh["mu"], msh["v1"], csh["mu"])
        ang = np.degrees(np.arccos(np.clip(cont_cos, 0, 1)))
        if contact and ang > CROSS_ANGLE_DEG:
            # does the member continue on BOTH sides of the closest approach?
            u_m = (csh["mu"] - msh["mu"]) @ msh["v1"]
            if msh["u"].min() < u_m - 3.0 and msh["u"].max() > u_m + 3.0:
                cross = 1.0

    # vertex pointing + cone membership (directional)
    vtx, axis = fam["vtx"], fam["axis"]
    vdca = _line_point_dca(csh["mu"], csh["v1"], vtx)
    r = csh["mu"] - vtx
    rn = float(np.linalg.norm(r))
    cone_cos = float(np.dot(r, axis) / max(rn, 1e-9))
    long_u = float(np.dot(r, axis) / max(fam["sh"]["length"], 1e-9))

    return dict(d_eu=d_eu_min, contact=float(contact),
                cont_gap=cont_gap, cont_cos=cont_cos, cont_lat=cont_lat,
                cross=cross, vdca=vdca, cone_cos=cone_cos, long_u=long_u,
                r_vtx=rn)


FEATURES = ["d_eu", "contact", "cont_gap", "cont_cos", "cont_lat", "cross",
            "vdca", "cone_cos", "long_u", "r_vtx"]


def auc_midrank(y, s):
    """AUC with midranks (tie-safe)."""
    y = np.asarray(y, bool)
    s = np.asarray(s, float)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    if y.sum() == 0 or (~y).sum() == 0:
        return np.nan
    from scipy.stats import rankdata
    r = rankdata(s)
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2)
                 / (y.sum() * (~y).sum()))


# ---------------------------------------------------------------------------
def emedian(t, e):
    o = np.argsort(t)
    cw = np.cumsum(e[o])
    return float(t[o][np.searchsorted(cw, 0.5 * cw[-1])])


def build_dataset(dumps, n_events=30, cands_per_event=120, min_fam_e=50.0):
    rows, labels_out, oper = [], [], []
    files = sorted(glob.glob(f"{dumps}/*.npz"))[: n_events]
    for fp in files:
        d = np.load(fp, allow_pickle=True)
        xyz = np.stack([np.asarray(d[k], np.float64) for k in "xyz"], 1)
        e = np.clip(np.asarray(d["e"], np.float64), 0, None)
        lab = np.asarray(d["labels"])
        tru = np.asarray(d["truth_t0"], np.float64)
        tsf = np.asarray(d["ts_final"], np.float64)

        # cluster shapes + truth times
        shp, ttime, cmask = {}, {}, {}
        for l in np.unique(lab[lab >= 0]):
            m = lab == l
            fin = m & np.isfinite(tru)
            if m.sum() < 3 or not fin.any():
                continue
            shp[int(l)] = cluster_shape(xyz[m], e[m])
            ttime[int(l)] = emedian(tru[fin], e[fin])
            cmask[int(l)] = m

        # truth families over big clusters
        labs = sorted(ttime)
        tt = np.array([ttime[l] for l in labs])
        order = np.argsort(tt)
        fam_id = {}
        gid, last = 0, None
        for i in order:
            if last is not None and tt[i] - last > TRUTH_GAP:
                gid += 1
            fam_id[labs[i]] = gid
            last = tt[i]
        fams = {}
        for l, g in fam_id.items():
            fams.setdefault(g, []).append(l)
        F = {}
        for g, mem in fams.items():
            # families anchored by BIG clusters only: candidates (small) are
            # never members, so no leave-one-out contamination
            mem = [l for l in mem if float(e[cmask[l]].sum()) >= 30.0]
            e_tot = sum(float(e[cmask[l]].sum()) for l in mem)
            if e_tot < min_fam_e or len(mem) < 1:
                continue
            hx = np.concatenate([xyz[cmask[l]] for l in mem])
            he = np.concatenate([e[cmask[l]] for l in mem])
            if hx.shape[0] < 10:
                continue
            F[g] = dict(frame=family_frame(hx, he), tree=cKDTree(hx),
                        members=[shp[l] for l in mem], t0=np.mean([ttime[l] for l in mem]),
                        mem=set(mem))

        if not F:
            d.close()
            continue
        f_ids = sorted(F)
        f_mu = np.stack([F[g]["frame"]["sh"]["mu"] for g in f_ids])

        # candidates: small clusters
        cands = [l for l in shp if float(e[cmask[l]].sum()) < 30.0]
        if len(cands) > cands_per_event:
            cands = list(RNG.choice(cands, size=cands_per_event, replace=False))
        for c in cands:
            csh = shp[c]
            dmu = np.linalg.norm(f_mu - csh["mu"], axis=1)
            near = [f_ids[i] for i in np.argsort(dmu)[:4]]
            tg = fam_id.get(c)
            if tg in F and tg not in near:
                near.append(tg)
            for g in near:
                if c in F[g]["mem"]:
                    continue
                ft = pair_features(csh, F[g]["frame"], F[g]["tree"], F[g]["members"])
                rows.append([ft[k] for k in FEATURES])
                labels_out.append(1 if abs(ttime[c] - F[g]["t0"]) <= TRUTH_GAP else 0)
            # operational: was the chain's own assignment correct?
            m = cmask[c]
            fin = m & np.isfinite(tsf) & np.isfinite(tru)
            if fin.any():
                wrongfrac = float(e[fin & (np.abs(tsf - tru) > CUT)].sum()
                                  / max(e[fin].sum(), 1e-9))
                # score vs the family nearest to the ASSIGNED time
                t_as = emedian(tsf[fin], e[fin])
                gj = min(f_ids, key=lambda g: abs(F[g]["t0"] - t_as))
                if abs(F[gj]["t0"] - t_as) <= 5.0 and c not in F[gj]["mem"]:
                    ft = pair_features(csh, F[gj]["frame"], F[gj]["tree"],
                                       F[gj]["members"])
                    oper.append(([ft[k] for k in FEATURES], 1 if wrongfrac > 0.5 else 0))
        d.close()
    X = np.array(rows, float)
    y = np.array(labels_out, int)
    Xo = np.array([r[0] for r in oper], float) if oper else np.zeros((0, len(FEATURES)))
    yo = np.array([r[1] for r in oper], int) if oper else np.zeros(0, int)
    return X, y, Xo, yo


def main():
    dumps = sys.argv[1] if len(sys.argv) > 1 else "v060dump_out/dumps"
    n_ev = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    npc = int(sys.argv[3]) if len(sys.argv) > 3 else 120
    X, y, Xo, yo = build_dataset(dumps, n_ev, npc)
    print(f"dataset: {len(y)} (cluster,family) pairs, {y.mean()*100:.1f}% same-interaction; "
          f"operational: {len(yo)} assigned pairs, {yo.mean()*100:.1f}% misplaced\n")
    print(f"{'feature':10s} {'AUC(truth pairs)':>16s}   (orientation: higher score = same family)")
    signs = {}
    for i, f in enumerate(FEATURES):
        a = auc_midrank(y, X[:, i])
        signs[f] = 1.0 if (np.isnan(a) or a >= 0.5) else -1.0
        a2 = a if a >= 0.5 or np.isnan(a) else 1 - a
        print(f"{f:10s} {a2:16.3f}   ({'+' if signs[f]>0 else '-'})")
    # combined logistic (median-impute NaN, standardize)
    from sklearn.linear_model import LogisticRegression
    def prep(A, med=None, sd=None, mu=None):
        A = A.copy()
        if med is None:
            med = np.nanmedian(A, axis=0)
        for i in range(A.shape[1]):
            A[np.isnan(A[:, i]), i] = med[i]
        if mu is None:
            mu, sd = A.mean(0), A.std(0) + 1e-9
        return (A - mu) / sd, med, sd, mu
    Xs, med, sd, mu = prep(X)
    n = len(y)
    idx = RNG.permutation(n)
    tr, te = idx[: n // 2], idx[n // 2:]
    clf = LogisticRegression(max_iter=2000).fit(Xs[tr], y[tr])
    s_te = clf.decision_function(Xs[te])
    print(f"\nCOMBINED logistic AUC (held-out half): {auc_midrank(y[te], s_te):.3f}")
    print("coefficients:", {f: round(float(c), 2)
                            for f, c in zip(FEATURES, clf.coef_[0])})
    if len(yo) > 20 and yo.sum() > 3:
        Xos, *_ = prep(Xo, med, sd, mu)
        s_o = clf.decision_function(Xos)
        print(f"\nOPERATIONAL misplacement detection: AUC "
              f"{1 - auc_midrank(yo, s_o):.3f} "
              f"(low family-score -> actually misplaced; n={len(yo)}, "
              f"misplaced {yo.sum()})")


if __name__ == "__main__":
    main()
