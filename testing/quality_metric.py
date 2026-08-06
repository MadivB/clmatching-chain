"""Truth-free per-TPC quality metric for the charge-light matcher.

Goal: a number, computable on REAL data (no MC truth), that flags a badly-placed
TPC.  Idea: if placement is wrong, the PREDICTED light (sum of placed cluster
templates) won't reproduce the ACTUAL light.  We quantify predicted-vs-actual at
each flash and test how well each candidate metric separates BAD (>20% energy
wrong, by truth) from GOOD (<5%) TPCs on the v0.3 dumps.
"""
import glob, json, os
import numpy as np

HERE = os.path.abspath(os.path.dirname(__file__))
DUMPS = os.path.join(HERE, "v030v3_out", "dumps")
CLIP, GAIN, PEAK = 60780.0, 200.0, 105
lab = json.load(open(os.path.join(HERE, "quality_labels.json")))
label = {}
for r in lab["bad"]:  label[(r[0], r[1], r[2])] = 1
for r in lab["good"]: label[(r[0], r[1], r[2])] = 0


def flash_peaks(trace, frac=0.06, min_sep=8):
    thr = frac * trace.max()
    pk = []
    for t in range(2, len(trace) - 2):
        if trace[t] > thr and trace[t] >= trace[t-1] and trace[t] > trace[t+1] and trace[t] >= trace[t-2]:
            if pk and t - pk[-1] < min_sep:
                if trace[t] > trace[pk[-1]]: pk[-1] = t
            else: pk.append(t)
    return pk


def metrics_for_tpc(act, pred):
    """act, pred: [120, 1000].  Returns a dict of candidate discrepancy metrics."""
    at = act.sum(0); pt = pred.sum(0)
    peaks = flash_peaks(at)
    if not peaks:
        return None
    W = 4
    A = np.array([at[max(0, t-W):t+W+1].sum() for t in peaks])          # actual amp / flash
    P = np.array([pt[max(0, t-W):t+W+1].sum() for t in peaks])          # predicted amp / flash
    r = P / np.maximum(A, 1.0)                                          # pred/act ratio per flash
    w = A / A.sum()                                                     # energy weight
    # per-flash spatial cosine (shape match), saturation-masked
    coss = []
    for t in peaks:
        av = act[:, max(0, t-W):t+W+1].sum(1); pv = pred[:, max(0, t-W):t+W+1].sum(1)
        sat = act[:, max(0, t-W):t+W+1].max(1) >= 0.98 * CLIP
        a, p = av[~sat], pv[~sat]; na, npd = np.linalg.norm(a), np.linalg.norm(p)
        coss.append(float(a @ p / (na * npd)) if na and npd else 0.0)
    coss = np.array(coss)
    med_r = np.median(r)
    # whole-TPC residual (masked), chi2
    sat_full = act >= 0.98 * CLIP
    resid = np.abs(act - np.minimum(pred, CLIP))
    m = ~sat_full
    return {
        # AMPLITUDE-based (user's intuition: pred vs act amplitude at each flash)
        "ratio_mad":  float(np.sum(w * np.abs(r - med_r))),            # weighted spread of pred/act
        "ratio_std":  float(np.sqrt(np.sum(w * (r - med_r) ** 2))),
        "worst_flash_ampdev": float(np.max(np.abs(A - P) / np.maximum(A, 1.0))),
        "deficit_frac": float(np.sum(np.maximum(A - P, 0)) / A.sum()),  # missing light share
        "excess_frac":  float(np.sum(np.maximum(P - A, 0)) / A.sum()),  # over-predicted share
        # RESIDUAL-based (whole TPC)
        "resid_frac": float(resid[m].sum() / np.maximum(act[m].sum(), 1.0)),
        "chi2_dof": float((((np.minimum(pred, CLIP) - act) ** 2 / (GAIN * np.maximum(act, 100.0)))[m]).mean()),
        # SPATIAL-based (shape mismatch at flashes)
        "min_flash_cos": float(coss.min()),
        "mean_flash_cos_wt": float(np.sum(w * coss)),
        "n_flash": len(peaks),
    }


rows = []
for fp in sorted(glob.glob(os.path.join(DUMPS, "*.npz"))):
    d = np.load(fp, allow_pickle=True)
    ft = os.path.basename(fp).split("_ev")[0]; ev = int(str(d["event"]))
    tpc = np.asarray(d["tpc"]); full = np.asarray(d["full_light"], np.float64); base = np.asarray(d["base_image"], np.float64)
    for c in np.unique(tpc):
        key = (ft, ev, int(c))
        if key not in label:
            continue
        M = metrics_for_tpc(full[int(c)], base[int(c)])
        if M is None: continue
        M["label"] = label[key]; M["key"] = key
        rows.append(M)

y = np.array([r["label"] for r in rows])
print(f"{len(rows)} labeled TPCs: {y.sum()} BAD, {(1-y).sum():.0f} GOOD\n")


def auc(scores, y, higher_bad=True):
    s = np.array(scores)
    if not higher_bad: s = -s
    pos = s[y == 1]; neg = s[y == 0]
    # Mann-Whitney AUC
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty_like(order, dtype=float); ranks[order] = np.arange(1, len(order) + 1)
    rp = ranks[:len(pos)].sum()
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


metrics = ["resid_frac", "chi2_dof", "ratio_mad", "ratio_std", "worst_flash_ampdev",
           "deficit_frac", "excess_frac", "min_flash_cos", "mean_flash_cos_wt"]
higher_bad = {m: True for m in metrics}
higher_bad["min_flash_cos"] = higher_bad["mean_flash_cos_wt"] = False   # higher cos = GOOD
print(f"{'metric':>20} {'AUC':>6}  {'GOOD median':>12}  {'BAD median':>11}")
res = []
for m in metrics:
    sc = [r[m] for r in rows]; a = auc(sc, y, higher_bad[m])
    gm = np.median([r[m] for r in rows if r["label"] == 0]); bm = np.median([r[m] for r in rows if r["label"] == 1])
    res.append((a, m, gm, bm)); print(f"{m:>20} {a:>6.3f}  {gm:>12.3f}  {bm:>11.3f}")
res.sort(reverse=True)
print(f"\nBEST metric: {res[0][1]} (AUC {res[0][0]:.3f})")
json.dump([{k: (list(r[k]) if k == 'key' else r[k]) for k in r} for r in rows], open(os.path.join(HERE, "quality_metrics.json"), "w"))
