import numpy as np, sys, glob, json
sys.path.insert(0, "/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing")
import v083_prune as VP
from sptl_v081 import auc_midrank

def main():
    import multiprocessing as mp
    d0 = np.load(sorted(glob.glob("v085EVO_out/dumps/*.npz"))[0], allow_pickle=True)
    print("v085 dump has cluster_conf:", "cluster_conf" in d0.files, flush=True)
    z = np.load("v083_prune_rows.npz", allow_pickle=True)
    X, y, fil = z["X"], z["y"], z["fil"]
    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(max_iter=300, max_depth=6, learning_rate=0.08)
    clf.fit(X[fil != "0000001"], y[fil != "0000001"])
    print("trained on v0.6 files 2-10", flush=True)
    with mp.Pool(13) as pool:
        rows = sum(pool.map(VP._proc, sorted(glob.glob("v085EVO_out/dumps/*.npz"))), [])
    Xv = np.array([r["x"] for r in rows], float)
    p = clf.predict_proba(Xv)[:, 1]
    yv = np.array([r["wrong"] for r in rows], float)
    print(f"v0.8 file-1: {len(rows)} clusters, wrong {yv.mean():.3f} "
          f"(energy {sum(r['e'] for r in rows if r['wrong'])/max(sum(r['e'] for r in rows),1e-9):.3f}), "
          f"transfer AUC {auc_midrank(yv, p):.3f}", flush=True)
    order = np.argsort(-p)
    nh = np.array([r["nh_tr"] for r in rows], float); tot = nh.sum()
    def metrics(keep):
        out = {}
        for tag, sel in (("NS", np.array([r["ns"] for r in rows])),
                         ("all", np.ones(len(rows), bool))):
            ge = sum(r["e_good"] for r, k, s in zip(rows, keep, sel) if k and s)
            te = sum(r["e_tr"] for r, k, s in zip(rows, keep, sel) if k and s)
            gh = sum(r["nh_good"] for r, k, s in zip(rows, keep, sel) if k and s)
            th = sum(r["nh_tr"] for r, k, s in zip(rows, keep, sel) if k and s)
            out[tag] = (100*ge/max(te, 1e-9), 100*gh/max(th, 1))
        return out
    keep = np.ones(len(rows), bool); base = metrics(keep)
    print(f"{'pruned':>7} {'NS E-acc':>9} {'NS h-acc':>9} {'all E-acc':>9} {'all h-acc':>9} {'purity':>7}")
    print(f"{'0%':>7} {base['NS'][0]:9.2f} {base['NS'][1]:9.2f} {base['all'][0]:9.2f} {base['all'][1]:9.2f}")
    targets = [0.005, 0.01, 0.02, 0.03]; ti = 0; pn = pe = pw = 0.0
    for i in order:
        keep[i] = False; pn += nh[i]; pe += rows[i]["e_tr"]; pw += rows[i]["e_tr"]-rows[i]["e_good"]
        while ti < len(targets) and pn >= targets[ti]*tot:
            m = metrics(keep)
            print(f"{100*targets[ti]:6.1f}% {m['NS'][0]:9.2f} {m['NS'][1]:9.2f} "
                  f"{m['all'][0]:9.2f} {m['all'][1]:9.2f} {100*pw/max(pe,1e-9):6.1f}%", flush=True)
            ti += 1
        if ti >= len(targets):
            break
    clf_d = HistGradientBoostingClassifier(max_iter=300, max_depth=6, learning_rate=0.08)
    clf_d.fit(X, y)
    p_all = z["p_bad"]; nh_all = z["nh"].astype(float)
    o = np.argsort(-p_all); cn = np.cumsum(nh_all[o])
    thr = {f: float(p_all[o][np.searchsorted(cn, f*nh_all.sum())]) for f in (0.0025, 0.005, 0.01, 0.02)}
    import joblib
    joblib.dump({"model": clf_d, "features": VP.ALL_FEATS, "thresholds": thr,
                 "note": "v0.8.3 pruner: badness p>=thr[frac] -> set hits t=-1"},
                "v083_prune_model.joblib")
    print("thresholds:", {k: round(v, 4) for k, v in thr.items()})
    print("saved v083_prune_model.joblib", flush=True)

if __name__ == "__main__":
    main()
