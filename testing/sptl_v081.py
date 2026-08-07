"""v0.8.1 spatial score — faithful NERSC reimplementation of the user's
SPATIAL_SCORE.md design (J_ij pairwise link v0/v1 + S_iF family score v0),
plus the v1.1 upgrades (continuous contact topology, class-pair reach).

Doc targets (files 1-8 train / 9-10 test, event split):
  J0 AUC 0.649 | J1 0.720 (track-track 0.864) | S_iF 0.915

Stages:  python sptl_v081.py [dump_dir]
  1 reproduce doc numbers on the doc split
  2 v1.1 variant (continuous topology + reach) — deltas on hard slices
  3 deploy models (train files 2-10) saved as portable JSON
  4 head-to-head vs spatial_score_v4 HGB on file 0000001 (out-of-sample for
    BOTH): same (cluster, own-reco-family) rows, misplacement AUC +
    trade-decision replay (fixed/broke at PASS-2 gates)
"""
import glob
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

CUT = 10.0
CONTACT = 3.0
PAIR_RCENT = 45.0
PAIR_DMIN = 12.0
REACH_SHOWER = 150.0        # v1.1 class-pair reach for shower-involving pairs
K_MEMBERS = 8


# ---------------------------------------------------------------- summaries
def emedian(t, w):
    i = np.argsort(t)
    c = np.cumsum(w[i])
    return float(t[i][np.searchsorted(c, 0.5 * c[-1])])


def summaries(ev, min_hits=4):
    out = {}
    rng = np.random.default_rng(0)
    labs = ev["labels"]
    for l in np.unique(labs[labs >= 0]):
        m = labs == l
        if m.sum() < min_hits:
            continue
        xyz = np.stack([ev["x"][m], ev["y"][m], ev["z"][m]], 1).astype(np.float64)
        e = np.clip(ev["e"][m].astype(np.float64), 0, None)
        w = np.maximum(e, 1e-6)
        mu = np.average(xyz, 0, weights=w)
        d = (xyz - mu) * np.sqrt(w[:, None] / w.sum())
        try:
            _, sv, vt = np.linalg.svd(d, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        evals = sv ** 2
        v1 = vt[0]
        u = (xyz - mu) @ v1
        q = np.quantile(u, [0.05, 0.1, 0.9, 0.95])
        sub = xyz if len(xyz) <= 300 else xyz[rng.choice(len(xyz), 300, False)]
        tpc_e = {}
        for tc in np.unique(ev["tpc"][m]):
            tpc_e[int(tc)] = float(e[ev["tpc"][m] == tc].sum())
        lin = float(evals[0] / max(evals.sum(), 1e-12))
        length = float(q[3] - q[0])
        perp = np.linalg.norm((xyz - mu) - np.outer(u, v1), axis=1)
        trms = float(np.sqrt(np.average(perp ** 2, weights=w)))
        typ = "track" if (length >= 5.0 and lin >= 0.6) else \
              ("shower" if (m.sum() >= 30 and trms >= 2.0) else "blob")
        tm = m & np.isfinite(ev["truth_t0"]) & (ev["truth_t0"] >= 0)
        out[int(l)] = dict(
            lab=int(l), mu=mu, v1=v1, lin=lin, length=length, trms=trms,
            q10=float(q[1]), q90=float(q[2]), lo=float(q[0]), hi=float(q[3]),
            ends=(mu + q[0] * v1, mu + q[3] * v1),
            sub=sub, tree=cKDTree(sub), e=float(e.sum()), n=int(m.sum()),
            tpc=int(max(tpc_e, key=tpc_e.get)), tpcs=set(tpc_e), typ=typ,
            truth=(emedian(ev["truth_t0"][tm], np.maximum(e[np.isfinite(
                ev["truth_t0"][m]) & (ev["truth_t0"][m] >= 0)], 1e-6))
                if tm.any() else None),
            xyz=xyz, ehits=e, u=u)
    return out


def truth_groups(S):
    """dominant-truth family id per cluster (3-tick grouping)."""
    ts = sorted((s["truth"], l) for l, s in S.items() if s["truth"] is not None)
    gid, groups, cur, t_last = {}, [], [], None
    for t, l in ts:
        if cur and t - t_last > 3.0:
            groups.append(cur)
            cur = []
        cur.append(l)
        t_last = t
    if cur:
        groups.append(cur)
    for g, mem in enumerate(groups):
        for l in mem:
            gid[l] = g
    return gid, groups


# ---------------------------------------------------------------- J features
J0_FEATS = ["d_min", "n_contact", "adot", "lin_min", "lin_max",
            "e_min", "e_max", "n_min", "both_track", "d_cent"]
J1_EXTRA = ["doca_min", "doca_max", "fwd_max", "diff_tpc", "ntpc_max",
            "tri_max", "tri_n90", "tri_cross_max"]


def jfeat0(si, sj):
    d_min = float(si["tree"].query(sj["sub"])[0].min())
    nc = sum(len(v) for v in si["tree"].query_ball_tree(sj["tree"], CONTACT))
    sep = sj["mu"] - si["mu"]
    return dict(
        d_min=d_min, n_contact=float(np.log1p(nc)),
        adot=float(abs(np.dot(si["v1"], sj["v1"]))),
        lin_min=min(si["lin"], sj["lin"]), lin_max=max(si["lin"], sj["lin"]),
        e_min=float(np.log10(max(min(si["e"], sj["e"]), 1e-3))),
        e_max=float(np.log10(max(max(si["e"], sj["e"]), 1e-3))),
        n_min=float(np.log10(min(si["n"], sj["n"]))),
        both_track=float(si["typ"] == "track" and sj["typ"] == "track"),
        d_cent=float(np.linalg.norm(sep)))


def jfeat1(si, sj, ft):
    sep = sj["mu"] - si["mu"]
    sn = float(np.linalg.norm(sep))
    dca_i = float(np.linalg.norm(np.cross(sep, si["v1"])))
    dca_j = float(np.linalg.norm(np.cross(-sep, sj["v1"])))
    ft.update(doca_min=min(dca_i, dca_j), doca_max=max(dca_i, dca_j),
              fwd_max=float(max(abs(np.dot(sep, si["v1"])),
                                abs(np.dot(sep, sj["v1"]))) / max(sn, 1e-9)),
              diff_tpc=float(si["tpc"] != sj["tpc"]),
              ntpc_max=float(max(len(si["tpcs"]), len(sj["tpcs"]))))
    return ft


def add_triangles(pairs, j0_by_pair, S):
    nbrs = {}
    for (a, b) in pairs:
        nbrs.setdefault(a, {})[b] = j0_by_pair[(a, b)]
        nbrs.setdefault(b, {})[a] = j0_by_pair[(a, b)]
    out = {}
    for (a, b) in pairs:
        common = set(nbrs.get(a, {})) & set(nbrs.get(b, {})) - {a, b}
        best, n90, bx = 0.0, 0, 0.0
        for c in common:
            v = min(nbrs[a][c], nbrs[b][c])
            best = max(best, v)
            n90 += v >= 0.9
            if S[c]["tpc"] != S[a]["tpc"] and S[c]["tpc"] != S[b]["tpc"]:
                bx = max(bx, v)
        out[(a, b)] = dict(tri_max=best, tri_n90=float(n90), tri_cross_max=bx)
    return out


# ---------------------------------------------------------------- S features
S_FEATS = ["d_min", "n_contact", "j0_max", "j0_top2", "flow", "r_sigma",
           "ext", "crossing", "endjoin", "thru_shower", "lin_i", "e_i",
           "fam_e", "fam_n", "fam_sigma", "fam_ntpc", "tpc_in", "ang_nn"]
S11_EXTRA = ["end_i", "end_m", "cang"]


def fam_summary(mems, S):
    pts = np.concatenate([S[m]["sub"] for m in mems])
    if len(pts) > 3000:
        pts = pts[:: len(pts) // 3000 + 1]
    mu = pts.mean(0)
    try:
        _, _, vt = np.linalg.svd(pts - mu, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    v1 = vt[0]
    u = (pts - mu) @ v1
    perp = np.linalg.norm((pts - mu) - np.outer(u, v1), axis=1)
    return dict(mems=mems, mu=mu, v1=v1, sigma=float(np.sqrt((perp ** 2).mean())),
                span=(float(np.quantile(u, 0.05)), float(np.quantile(u, 0.95))),
                e=sum(S[m]["e"] for m in mems), n=len(mems),
                tpcs=set().union(*(S[m]["tpcs"] for m in mems)),
                cents=np.stack([S[m]["mu"] for m in mems]),
                has_shower=any(S[m]["typ"] == "shower" for m in mems))


def contact_topology(si, sm):
    """closest hit pair -> fractional axial positions + angle."""
    dq, iq = sm["tree"].query(si["sub"])
    k = int(np.argmin(dq))
    p_i, p_m = si["sub"][k], sm["sub"][int(iq[k])]
    def frac(s, p):
        u = float((p - s["mu"]) @ s["v1"])
        return (u - s["lo"]) / max(s["hi"] - s["lo"], 1e-9)
    fi, fm = frac(si, p_i), frac(sm, p_m)
    ang = float(np.degrees(np.arccos(np.clip(
        abs(np.dot(si["v1"], sm["v1"])), 0, 1))))
    inter_i = si["q10"] < float((p_i - si["mu"]) @ si["v1"]) < si["q90"]
    inter_m = sm["q10"] < float((p_m - sm["mu"]) @ sm["v1"]) < sm["q90"]
    return float(dq[k]), fi, fm, ang, inter_i, inter_m


def sfeat(si, F, S, j0_i, v11=False):
    cd = np.linalg.norm(F["cents"] - si["mu"], axis=1)
    near = [F["mems"][k] for k in np.argsort(cd)[:K_MEMBERS]]
    d_min, nc, sm_best = 1e9, 0, None
    for m in near:
        dm = float(S[m]["tree"].query(si["sub"])[0].min())
        nc += sum(len(v) for v in
                  S[m]["tree"].query_ball_tree(si["tree"], CONTACT))
        if dm < d_min:
            d_min, sm_best = dm, S[m]
    j0s = sorted((j0_i.get(m, 0.0) for m in near), reverse=True)
    flow_w = [(S[m]["e"], abs(np.dot(si["v1"], S[m]["v1"]))) for m in near
              if np.linalg.norm(S[m]["mu"] - si["mu"]) < 30.0]
    flow = (float(np.average([f for _, f in flow_w],
                             weights=[w for w, _ in flow_w]))
            if flow_w else 0.5)
    rv = si["mu"] - F["mu"]
    uu = float(rv @ F["v1"])
    r_sigma = float(np.linalg.norm(rv - uu * F["v1"]) / max(F["sigma"], 0.5))
    lo, hi = F["span"]
    span = max(hi - lo, 1e-9)
    ext = 0.0 if lo <= uu <= hi else min(
        (lo - uu if uu < lo else uu - hi) / span, 3.0)
    dmin_c, fi, fm, ang, inter_i, inter_m = contact_topology(si, sm_best)
    touch = d_min < CONTACT
    ft = dict(
        d_min=d_min, n_contact=float(np.log1p(nc)),
        j0_max=j0s[0] if j0s else 0.0,
        j0_top2=float(np.mean(j0s[:2])) if j0s else 0.0,
        flow=flow, r_sigma=min(r_sigma, 20.0), ext=ext,
        crossing=float(touch and ang > 25 and inter_i and inter_m),
        endjoin=float(touch and ang < 15 and not inter_i and not inter_m),
        thru_shower=float(si["lin"] > 0.9 and F["sigma"] > 4.0 and inter_i),
        lin_i=si["lin"], e_i=float(np.log10(max(si["e"], 1e-3))),
        fam_e=float(np.log10(max(F["e"], 1e-3))), fam_n=float(F["n"]),
        fam_sigma=F["sigma"], fam_ntpc=float(len(F["tpcs"])),
        tpc_in=float(si["tpc"] in F["tpcs"]),
        ang_nn=float(abs(np.dot(si["v1"], sm_best["v1"]))))
    if v11:
        ft.update(end_i=abs(2 * fi - 1.0), end_m=abs(2 * fm - 1.0), cang=ang)
    return ft


# ---------------------------------------------------------------- model
def fit_logistic(X, y):
    from sklearn.linear_model import LogisticRegression
    mu, sd = X.mean(0), X.std(0) + 1e-9
    clf = LogisticRegression(max_iter=2000)
    clf.fit((X - mu) / sd, y)
    return dict(mu=mu, sd=sd, coef=clf.coef_[0], b=float(clf.intercept_[0]))


def predict(md, X):
    z = ((X - md["mu"]) / md["sd"]) @ md["coef"] + md["b"]
    return 1.0 / (1.0 + np.exp(-z))


def auc_midrank(y, s):
    from scipy.stats import rankdata
    r = rankdata(s)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def calib(y, p, edges=(0.5, 0.8, 0.9, 0.95, 0.99, 1.001)):
    rows, lo = [], 0.0
    for hi in edges:
        m = (p >= lo) & (p < hi)
        if m.sum() > 30:
            rows.append(f"[{lo:.2f},{hi:.2f}) pred~{p[m].mean():.3f} "
                        f"actual {y[m].mean():.3f} (n={m.sum()})")
        lo = hi
    return rows


# ---------------------------------------------------------------- harvest
def _proc_event(f):
    d = np.load(f, allow_pickle=True)
    ev = {k: d[k] for k in ("x", "y", "z", "e", "tpc", "labels",
                            "truth_t0", "ts_final")}
    ev["file"] = os.path.basename(f).split("_")[0]
    ev["name"] = os.path.basename(f).replace(".npz", "")
    S = summaries(ev)
    gid, groups = truth_groups(S)
    labs = sorted(S)
    cents = np.stack([S[l]["mu"] for l in labs]) if labs else np.zeros((0, 3))
    pairs = []
    for a_i in range(len(labs)):
        if a_i + 1 >= len(labs):
            break
        dc_row = np.linalg.norm(cents[a_i + 1:] - cents[a_i], axis=1)
        for off in np.flatnonzero(dc_row <= REACH_SHOWER):
            b_i = a_i + 1 + int(off)
            a, b = labs[a_i], labs[b_i]
            dc = float(dc_row[off])
            shower = S[a]["typ"] == "shower" or S[b]["typ"] == "shower"
            if dc > (REACH_SHOWER if shower else PAIR_RCENT):
                continue
            ft = jfeat0(S[a], S[b])
            v0_ok = dc <= PAIR_RCENT and ft["d_min"] <= PAIR_DMIN
            if not v0_ok and not (shower and ft["d_min"] <= 60.0):
                continue
            pairs.append(dict(a=a, b=b, ft=ft, v0=v0_ok,
                              y=float(gid.get(a, -1) == gid.get(b, -2))))
    return dict(ev=ev, S=S, gid=gid, groups=groups, pairs=pairs)


def harvest(dump_dir, nproc=16):
    import multiprocessing as mp
    files = sorted(glob.glob(os.path.join(dump_dir, "*.npz")))
    with mp.Pool(nproc) as pool:
        evs = pool.map(_proc_event, files)
    return evs


def main():
    dump_dir = sys.argv[1] if len(sys.argv) > 1 else "v060dump_out/dumps"
    print("harvesting ...", flush=True)
    evs = harvest(dump_dir)
    tr = [E for E in evs if E["ev"]["file"] <= "0000008"]
    te = [E for E in evs if E["ev"]["file"] >= "0000009"]
    dep = [E for E in evs if E["ev"]["file"] != "0000001"]
    print(f"{len(evs)} events (train {len(tr)} / test {len(te)})", flush=True)

    def pair_mat(Es, feats, v0_only=True):
        rows, ys, meta = [], [], []
        for E in Es:
            for p in E["pairs"]:
                if v0_only and not p["v0"]:
                    continue
                rows.append([p["ft"][k] for k in feats])
                ys.append(p["y"])
                meta.append((E, p))
        return np.array(rows, float), np.array(ys), meta

    # ---------------- stage 1a: J0 on the doc split ----------------
    X, y, _ = pair_mat(tr, J0_FEATS)
    j0 = fit_logistic(X, y)
    Xt, yt, mt = pair_mat(te, J0_FEATS)
    p = predict(j0, Xt)
    print(f"\nJ0: n_train {len(X)} base {y.mean():.3f} | "
          f"test AUC {auc_midrank(yt, p):.3f}  (doc 0.649)")

    # J0 applied everywhere (frozen) for triangles + S features
    for E in evs:
        if E["pairs"]:
            Xa = np.array([[pr["ft"][k] for k in J0_FEATS]
                           for pr in E["pairs"]], float)
            for pr, v in zip(E["pairs"], predict(j0, Xa)):
                pr["j0"] = float(v)
        j0d = {(pr["a"], pr["b"]): pr["j0"] for pr in E["pairs"] if pr["v0"]}
        tri = add_triangles(list(j0d), j0d, E["S"])
        for pr in E["pairs"]:
            pr["ft"] = jfeat1(E["S"][pr["a"]], E["S"][pr["b"]], pr["ft"])
            pr["ft"].update(tri.get((pr["a"], pr["b"]),
                                    dict(tri_max=0.0, tri_n90=0.0,
                                         tri_cross_max=0.0)))

    # ---------------- stage 1b: J1 ----------------
    F1 = J0_FEATS + J1_EXTRA
    X, y, _ = pair_mat(tr, F1)
    j1 = fit_logistic(X, y)
    Xt, yt, mt = pair_mat(te, F1)
    p = predict(j1, Xt)
    tt = np.array([E["S"][pr["a"]]["typ"] == "track"
                   and E["S"][pr["b"]]["typ"] == "track" for E, pr in mt])
    ncon = np.array([pr["ft"]["d_min"] > CONTACT for _, pr in mt])
    xtpc = np.array([pr["ft"]["diff_tpc"] > 0 for _, pr in mt])
    print(f"J1: test AUC {auc_midrank(yt, p):.3f} (doc 0.720) | "
          f"track-track {auc_midrank(yt[tt], p[tt]):.3f} (doc 0.864) | "
          f"cross-TPC {auc_midrank(yt[xtpc], p[xtpc]):.3f} (doc 0.793) | "
          f"non-contact {auc_midrank(yt[ncon], p[ncon]):.3f} (doc 0.717)")

    # ---------------- stage 1c/2: S_iF v0 and v1.1 ----------------
    def s_rows(Es, v11=False):
        rows, ys, meta = [], [], []
        feats = S_FEATS + (S11_EXTRA if v11 else [])
        for E in Es:
            S, gid = E["S"], E["gid"]
            j0_of = {}
            for pr in E["pairs"]:
                j0_of.setdefault(pr["a"], {})[pr["b"]] = pr["j0"]
                j0_of.setdefault(pr["b"], {})[pr["a"]] = pr["j0"]
            fams = {}
            for g, mem in enumerate(E["groups"]):
                mm = [l for l in mem if l in S]
                if len(mm) >= 1:
                    fams[g] = mm
            for l, si in S.items():
                for g, mem in fams.items():
                    mm = [m for m in mem if m != l]
                    if not mm:
                        continue
                    cd = min(np.linalg.norm(S[m]["mu"] - si["mu"]) for m in mm)
                    reach = REACH_SHOWER if (v11 and (
                        si["typ"] == "shower"
                        or any(S[m]["typ"] == "shower" for m in mm))) \
                        else PAIR_RCENT
                    if cd > reach:
                        continue
                    F = fam_summary(mm, S)
                    if F is None:
                        continue
                    ft = sfeat(si, F, S, j0_of.get(l, {}), v11=v11)
                    rows.append([ft[k] for k in feats])
                    ys.append(float(gid.get(l, -1) == g))
                    meta.append((E, l, g, ft))
        return np.array(rows, float), np.array(ys), meta

    for v11, tag, extra_doc in ((False, "S_iF v0", "(doc 0.915)"),
                                (True, "S_iF v1.1 +topo+reach", "")):
        X, y, _ = s_rows(tr, v11)
        smd = fit_logistic(X, y)
        Xt, yt, mt = s_rows(te, v11)
        p = predict(smd, Xt)
        cr = np.array([m[3]["crossing"] > 0 for m in mt])
        th = np.array([m[3]["thru_shower"] > 0 for m in mt])
        nc = np.array([m[3]["d_min"] > CONTACT for m in mt])
        print(f"{tag}: n {len(X)} base {y.mean():.3f} | test AUC "
              f"{auc_midrank(yt, p):.3f} {extra_doc} | crossing "
              f"{auc_midrank(yt[cr], p[cr]):.3f} | thru-shower "
              f"{auc_midrank(yt[th], p[th]):.3f} | non-contact "
              f"{auc_midrank(yt[nc], p[nc]):.3f}")
        for row in calib(yt, p):
            print("   calib", row)
        if v11:
            s11 = smd
        else:
            s0 = smd

    # ---------------- stage 3: deploy models (train != file 0000001) ------
    X, y, _ = pair_mat(dep, J0_FEATS)
    j0_dep = fit_logistic(X, y)
    X, y, _ = s_rows(dep, True)
    s_dep = fit_logistic(X, y)
    art = {"j0": j0, "j1": j1, "s0": s0, "s11": s11,
           "j0_deploy": j0_dep, "s11_deploy": s_dep}
    with open("sptl_v081_models.json", "w") as f:
        json.dump({k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                       for kk, vv in v.items()} for k, v in art.items()},
                  f, indent=1)
    print("\nsaved sptl_v081_models.json "
          f"(deploy S trained on {len(dep)} events, files != 0000001)")
    np.save("sptl_v081_cache.npy",
            np.array([len(evs)], dtype=object), allow_pickle=True)


if __name__ == "__main__":
    main()
