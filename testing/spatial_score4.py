"""v0.8 Task 1 v4 — v2 + feature-wise BEST-member pooling (nearest member for
contact, best member for continuation/cone independently) + interface density.

    python spatial_score4.py <dumps_dir> [n_events] [cands_per_event]
"""
from __future__ import annotations

import glob
import sys

import numpy as np
from scipy.spatial import cKDTree

from spatial_score import (CUT, TRUTH_GAP, auc_midrank, cluster_shape, emedian)

RNG = np.random.default_rng(11)
CONTACT_CM = 3.0


def member_type(sh):
    """track / shower / blob from PCA shape."""
    trms = float(np.sqrt(max(sh["evals"][1] + sh["evals"][2], 0.0)))
    if sh["length"] >= 5.0 and sh["lin"] >= 0.6:
        return "track"
    if sh["xyz"].shape[0] >= 30 and trms >= 2.0:
        return "shower"
    return "blob"


def member_direction(sh):
    """Oriented axis: from the NARROW end outward (per-member, not family)."""
    u, xyz, e = sh["u"], sh["xyz"], np.clip(sh["e"], 1e-6, None)
    lo, hi = np.quantile(u, [0.2, 0.8])
    d = xyz - sh["mu"]
    tr = np.linalg.norm(d - np.outer(u, sh["v1"]), axis=1)
    rms_lo = float(np.sqrt(np.average(tr[u <= lo] ** 2, weights=e[u <= lo]))) \
        if (u <= lo).any() else 1e9
    rms_hi = float(np.sqrt(np.average(tr[u >= hi] ** 2, weights=e[u >= hi]))) \
        if (u >= hi).any() else 1e9
    if rms_lo <= rms_hi:
        start, sign = sh["ends"][0], 1.0
    else:
        start, sign = sh["ends"][1], -1.0
    return start, sign * sh["v1"]


def lsq_vertex(track_shapes):
    """Least-squares point closest to all track axis lines."""
    if not track_shapes:
        return None
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for sh in track_shapes:
        v = sh["v1"]
        P = np.eye(3) - np.outer(v, v)
        A += P
        b += P @ sh["mu"]
    A += 1e-6 * np.eye(3)
    return np.linalg.solve(A, b)


def pair_features2(csh, F):
    out = {}
    c_ax_ok = csh["xyz"].shape[0] >= 6 and csh["lin"] >= 0.5
    q = csh["xyz"]
    if q.shape[0] > 40:
        q = q[:: q.shape[0] // 40 + 1]

    # best member by hit distance (limit to the 8 nearest members by centroid)
    mem = F["members2"]
    if len(mem) > 8:
        dc = [float(np.linalg.norm(m[0]["mu"] - csh["mu"])) for m in mem]
        mem = [mem[i] for i in np.argsort(dc)[:8]]
    best = None
    for mi, (msh, mtree, mtype, mstart, mdir) in enumerate(mem):
        dm, _ = mtree.query(q, k=1)
        dmin = float(dm.min())
        if best is None or dmin < best[0]:
            best = (dmin, msh, mtree, mtype, mstart, mdir)
    d_eu, msh, mtree, mtype, mstart, mdir = best
    out["d_eu"] = d_eu
    out["contact"] = float(d_eu <= CONTACT_CM)
    out["m_is_track"] = float(mtype == "track")
    out["m_is_shower"] = float(mtype == "shower")
    out["m_lin"] = msh["lin"]

    # ---- v3: feature-wise BEST member (independent of the nearest one) ----
    best_lat = best_gap = best_cone = best_cont_sc = np.nan
    for msh2, mtree2, mtype2, mstart2, mdir2 in mem:
        u2 = (csh["mu"] - msh2["mu"]) @ msh2["v1"]
        lat2 = float(np.linalg.norm((csh["mu"] - msh2["mu"]) - u2 * msh2["v1"]))
        if not np.isfinite(best_lat) or lat2 < best_lat:
            best_lat = lat2
        g2 = min(float(np.linalg.norm(ce - me2))
                 for ce in csh["ends"] for me2 in msh2["ends"])
        if not np.isfinite(best_gap) or g2 < best_gap:
            best_gap = g2
        if mtype2 == "shower":
            r2 = csh["mu"] - mstart2
            cc = float(np.dot(r2, mdir2) / max(np.linalg.norm(r2), 1e-9))
            if not np.isfinite(best_cone) or cc > best_cone:
                best_cone = cc
        if mtype2 == "track" and c_ax_ok:
            ax2 = float(abs(np.dot(csh["v1"], msh2["v1"])))
            half2 = max(msh2["length"] / 2.0, 1e-9)
            beyond2 = abs(u2) > half2
            sc = ax2 * (1.0 if beyond2 else 0.3) / (1.0 + g2 / 5.0)
            if not np.isfinite(best_cont_sc) or sc > best_cont_sc:
                best_cont_sc = sc
    out["best_lat"] = best_lat
    out["best_gap"] = best_gap
    out["best_cone"] = best_cone
    out["best_cont"] = best_cont_sc

    # ---- v3: interface energy-density continuity at the contact zone ----
    dq, iq = mtree.query(q, k=1)
    near_c = dq <= 2.0 * CONTACT_CM
    if near_c.any():
        e_c_near = float(np.clip(csh["e"], 0, None)[:len(q)][near_c].sum())             if len(csh["e"]) >= len(q) else float(np.clip(csh["e"], 0, None).mean() * near_c.sum())
        e_m_near = float(np.clip(msh["e"], 0, None)[
            np.unique(iq[near_c])].sum())
        out["iface_ratio"] = float(np.log1p(e_c_near) - np.log1p(e_m_near))
    else:
        out["iface_ratio"] = np.nan

    # geometry of the approach to the best member
    u_c = (csh["mu"] - msh["mu"]) @ msh["v1"]
    half = max(msh["length"] / 2.0, 1e-9)
    out["along_frac"] = float(u_c / half)          # |.|>1 => beyond endpoint
    out["beyond_end"] = float(abs(u_c) > half)
    out["lat_m"] = float(np.linalg.norm(
        (csh["mu"] - msh["mu"]) - u_c * msh["v1"]))
    # nearest ends + bridge direction
    gaps = [(float(np.linalg.norm(ce - me)), ce, me)
            for ce in csh["ends"] for me in msh["ends"]]
    g, ce, me = min(gaps, key=lambda x: x[0])
    out["end_gap"] = g
    br = ce - me
    brn = float(np.linalg.norm(br))
    out["bridge_cos"] = float(abs(np.dot(br / max(brn, 1e-9), msh["v1"]))) \
        if brn > 1e-6 else 1.0
    out["axis_cos"] = float(abs(np.dot(csh["v1"], msh["v1"]))) if c_ax_ok else np.nan
    # crossing: contact at MID-track with big angle while member continues both sides
    out["cross"] = 0.0
    if out["contact"] and not out["beyond_end"] and c_ax_ok:
        ang = np.degrees(np.arccos(np.clip(out["axis_cos"], 0, 1)))
        if ang > 30:
            out["cross"] = 1.0
    # one-piece continuation: endpoint contact + collinear + on-axis bridge
    out["continuation"] = float(out["beyond_end"] and g < 6.0
                                and (not c_ax_ok or out["axis_cos"] > 0.8)
                                and out["bridge_cos"] > 0.8)

    # shower-cone membership vs best SHOWER-like member (oriented)
    if mtype == "shower" or out["m_is_shower"]:
        r = csh["mu"] - mstart
        rn = float(np.linalg.norm(r))
        out["cone_cos_m"] = float(np.dot(r, mdir) / max(rn, 1e-9))
        out["cone_s"] = float(np.dot(r, mdir) / max(msh["length"], 1e-9))
    else:
        out["cone_cos_m"] = np.nan
        out["cone_s"] = np.nan

    # vertex features (LSQ vertex of the family's tracks)
    vtx = F["vtx"]
    if vtx is not None:
        rv = csh["mu"] - vtx
        out["r_vtx"] = float(np.linalg.norm(rv))
        out["vdca"] = (float(np.linalg.norm(np.cross(rv, csh["v1"])))
                       if c_ax_ok else np.nan)
        out["vtx_cos"] = float(np.dot(rv / max(out["r_vtx"], 1e-9), F["vdir"])) \
            if F["vdir"] is not None else np.nan
    else:
        out["r_vtx"] = out["vdca"] = out["vtx_cos"] = np.nan

    # how many members' axis lines pass near the candidate
    npoint = 0
    for msh2, *_ in mem:
        if float(np.linalg.norm(np.cross(csh["mu"] - msh2["mu"], msh2["v1"]))) < 5.0:
            npoint += 1
    out["n_pointing"] = float(npoint)
    # v4: candidate reliability + normalized distances + family size
    out["c_nhits"] = float(csh["xyz"].shape[0])
    out["c_len"] = csh["length"]
    out["c_lin"] = csh["lin"]
    m_trms = float(np.sqrt(max(msh["evals"][1] + msh["evals"][2], 1e-9)))
    out["lat_norm"] = out["lat_m"] / max(m_trms, 0.5)
    out["gap_norm"] = out["end_gap"] / max(msh["length"], 1.0)
    out["n_members"] = float(len(mem))
    return out


FEATURES2 = ["d_eu", "contact", "m_is_track", "m_is_shower", "m_lin",
             "along_frac", "beyond_end", "lat_m", "end_gap", "bridge_cos",
             "axis_cos", "cross", "continuation", "cone_cos_m", "cone_s",
             "r_vtx", "vdca", "vtx_cos", "n_pointing",
             "best_lat", "best_gap", "best_cone", "best_cont",
             "c_nhits", "c_len", "c_lin", "lat_norm", "gap_norm", "n_members"]


def build_dataset(dumps, n_events=30, cands_per_event=120, min_fam_e=50.0):
    rows, y, oper = [], [], []
    for fp in sorted(glob.glob(f"{dumps}/*.npz"))[: n_events]:
        d = np.load(fp, allow_pickle=True)
        xyz = np.stack([np.asarray(d[k], np.float64) for k in "xyz"], 1)
        e = np.clip(np.asarray(d["e"], np.float64), 0, None)
        lab = np.asarray(d["labels"])
        tru = np.asarray(d["truth_t0"], np.float64)
        tsf = np.asarray(d["ts_final"], np.float64)
        shp, ttime, cmask = {}, {}, {}
        for l in np.unique(lab[lab >= 0]):
            m = lab == l
            fin = m & np.isfinite(tru)
            if m.sum() < 3 or not fin.any():
                continue
            shp[int(l)] = cluster_shape(xyz[m], e[m])
            ttime[int(l)] = emedian(tru[fin], e[fin])
            cmask[int(l)] = m
        labs = sorted(ttime)
        tt = np.array([ttime[l] for l in labs])
        fam_id, gid, last = {}, 0, None
        for i in np.argsort(tt):
            if last is not None and tt[i] - last > TRUTH_GAP:
                gid += 1
            fam_id[labs[i]] = gid
            last = tt[i]
        fams = {}
        for l, g in fam_id.items():
            fams.setdefault(g, []).append(l)
        F = {}
        for g, mem in fams.items():
            mem = [l for l in mem if float(e[cmask[l]].sum()) >= 30.0]
            if sum(float(e[cmask[l]].sum()) for l in mem) < min_fam_e or not mem:
                continue
            members2 = []
            tracks = []
            for l in mem:
                sh = shp[l]
                mt = member_type(sh)
                ms, md = member_direction(sh)
                members2.append((sh, cKDTree(sh["xyz"]), mt, ms, md))
                if mt == "track":
                    tracks.append(sh)
            vtx = lsq_vertex(tracks)
            vdir = None
            if vtx is not None and tracks:
                ds = [sh["mu"] - vtx for sh in tracks]
                vdir = np.sum(ds, axis=0)
                n = np.linalg.norm(vdir)
                vdir = vdir / n if n > 1e-9 else None
            F[g] = dict(members2=members2, vtx=vtx, vdir=vdir,
                        t0=np.mean([ttime[l] for l in mem]),
                        mu=np.mean([shp[l]["mu"] for l in mem], axis=0))
        if not F:
            d.close()
            continue
        f_ids = sorted(F)
        f_mu = np.stack([F[g]["mu"] for g in f_ids])
        cands = [l for l in shp if float(e[cmask[l]].sum()) < 30.0]
        if len(cands) > cands_per_event:
            cands = list(RNG.choice(cands, size=cands_per_event, replace=False))
        for c in cands:
            csh = shp[c]
            near = [f_ids[i] for i in
                    np.argsort(np.linalg.norm(f_mu - csh["mu"], axis=1))[:4]]
            tg = fam_id.get(c)
            if tg in F and tg not in near:
                near.append(tg)
            for g in near:
                ft = pair_features2(csh, F[g])
                rows.append([ft[k] for k in FEATURES2])
                y.append(1 if abs(ttime[c] - F[g]["t0"]) <= TRUTH_GAP else 0)
            m = cmask[c]
            fin = m & np.isfinite(tsf) & np.isfinite(tru)
            if fin.any():
                wrong = float(e[fin & (np.abs(tsf - tru) > CUT)].sum()
                              / max(e[fin].sum(), 1e-9))
                t_as = emedian(tsf[fin], e[fin])
                gj = min(f_ids, key=lambda g: abs(F[g]["t0"] - t_as))
                if abs(F[gj]["t0"] - t_as) <= 5.0:
                    ft = pair_features2(csh, F[gj])
                    oper.append(([ft[k] for k in FEATURES2], 1 if wrong > 0.5 else 0))
        d.close()
    X = np.array(rows, float)
    yv = np.array(y, int)
    Xo = np.array([r[0] for r in oper], float) if oper else np.zeros((0, len(FEATURES2)))
    yo = np.array([r[1] for r in oper], int) if oper else np.zeros(0, int)
    return X, yv, Xo, yo


def main():
    dumps = sys.argv[1] if len(sys.argv) > 1 else "v060dump_out/dumps"
    n_ev = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    npc = int(sys.argv[3]) if len(sys.argv) > 3 else 120
    X, y, Xo, yo = build_dataset(dumps, n_ev, npc)
    print(f"v4 dataset: {len(y)} pairs ({y.mean()*100:.1f}% same), "
          f"operational {len(yo)} ({yo.mean()*100:.1f}% misplaced)\n")
    for i, f in enumerate(FEATURES2):
        a = auc_midrank(y, X[:, i])
        if not np.isnan(a):
            print(f"{f:12s} {max(a, 1-a):.3f} ({'+' if a>=0.5 else '-'})")
    n = len(y)
    idx = RNG.permutation(n)
    tr, te = idx[: n // 2], idx[n // 2:]
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    # d_eu-only baseline
    a_base = auc_midrank(y[te], -X[te, 0])
    print(f"\nBASELINE d_eu only            : {a_base:.3f}")
    Xl = X.copy()
    med = np.nanmedian(Xl, axis=0)
    for i in range(Xl.shape[1]):
        Xl[np.isnan(Xl[:, i]), i] = med[i]
    mu, sd = Xl.mean(0), Xl.std(0) + 1e-9
    clf = LogisticRegression(max_iter=3000).fit((Xl[tr]-mu)/sd, y[tr])
    print(f"v4 logistic (held-out)        : "
          f"{auc_midrank(y[te], clf.decision_function((Xl[te]-mu)/sd)):.3f}")
    hgb = HistGradientBoostingClassifier(max_iter=600, learning_rate=0.08,
                                         max_depth=6, early_stopping=True,
                                         random_state=0).fit(X[tr], y[tr])
    s_te = hgb.predict_proba(X[te])[:, 1]
    print(f"v4 HGB      (held-out)        : {auc_midrank(y[te], s_te):.3f}")
    if len(yo) > 20 and yo.sum() > 3:
        so = hgb.predict_proba(Xo)[:, 1]
        print(f"\nOPERATIONAL (HGB): misplacement-detection AUC "
              f"{1 - auc_midrank(yo, so):.3f}  (v1 0.724, v2 0.802, v3 0.816; n={len(yo)})")
        sol = clf.decision_function((np.where(np.isnan(Xo), med, Xo)-mu)/sd)
        print(f"OPERATIONAL (logistic)       : {1 - auc_midrank(yo, sol):.3f}")


if __name__ == "__main__":
    main()
