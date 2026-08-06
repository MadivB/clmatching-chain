"""Consolidated truth-free per-TPC quality metric.

One pass over the v0.3 dumps.  Per non-shower TPC computes, from PREDICTED vs
ACTUAL light only (no MC truth), a discrepancy score; and, using truth ONLY to
grade the metric, the wrong-energy fraction.  Reports:
  (1) BAD(>20% wrong) vs GOOD(<5%) separation AUC, raw chi2 and amplitude-fit
      chi2_scaled (removes the ~0.78 ML systematic).
  (2) CONTINUOUS Spearman correlation of the metric with the wrong-energy
      fraction across ALL TPCs  -> the on-data efficiency estimator.
Prints per-file progress so it is never a black box.
"""
import glob, json, os, sys, numpy as np

HERE = os.path.abspath(os.path.dirname(__file__))
DUMPS = os.path.join(HERE, "v030v3_out", "dumps")
CLIP, GAIN, PEAK, CUT = 60780.0, 200.0, 105, 10
rows = []
files = sorted(glob.glob(os.path.join(DUMPS, "*.npz")))
for i, fp in enumerate(files):
    d = np.load(fp, allow_pickle=True)
    ft = os.path.basename(fp).split("_ev")[0]; ev = int(str(d["event"]))
    tpc = np.asarray(d["tpc"]); e = np.asarray(d["e"], np.float64)
    tsf = np.asarray(d["ts_final"], np.float64); tr = np.asarray(d["truth_t0"], np.float64)
    full = np.asarray(d["full_light"], np.float32); base = np.asarray(d["base_image"], np.float32)
    for c in np.unique(tpc):
        m = tpc == c; E = float(e[m].sum())
        if E < 100 or E > 1400:            # non-shower energy band (matches labels)
            continue
        ok = m & np.isfinite(tr); den = float(e[ok].sum())
        if den <= 0:
            continue
        wrong = float(e[ok & np.isfinite(tsf) & (np.abs(tsf - tr) > CUT)].sum())
        wf = wrong / den                    # TRUTH wrong-energy fraction (grading only)
        act = full[int(c)].astype(np.float64); pred = np.minimum(base[int(c)].astype(np.float64), CLIP)
        sat = act >= 0.98 * CLIP; msk = ~sat
        a = act[msk]; p = pred[msk]
        v = GAIN * np.maximum(a, 100.0)
        alpha = float((p @ a) / max(p @ p, 1e-9))
        chi2 = float(((p - a) ** 2 / v).mean())
        chi2s = float(((alpha * p - a) ** 2 / v).mean())
        rows.append(dict(file=ft, event=ev, tpc=int(c), E=round(E, 1),
                         wf=round(wf, 4), chi2=chi2, chi2s=chi2s, alpha=round(alpha, 3)))
    d.close(); del full, base
    print(f"  [{i+1}/{len(files)}] {ft} ev{ev}: {len(rows)} TPCs so far", flush=True)

json.dump(rows, open(os.path.join(HERE, "quality_final.json"), "w"))
wf = np.array([r["wf"] for r in rows]); chi2 = np.array([r["chi2"] for r in rows]); chi2s = np.array([r["chi2s"] for r in rows])
E = np.array([r["E"] for r in rows])
print(f"\n{len(rows)} non-shower TPCs.  wrong-frac: BAD(>20%)={int((wf>0.20).sum())}  "
      f"MARGINAL(5-20%)={int(((wf>=0.05)&(wf<=0.20)).sum())}  GOOD(<5%)={int((wf<0.05).sum())}")

def auc(sc, y):
    s = np.asarray(sc); pos = s[y == 1]; neg = s[y == 0]
    o = np.argsort(np.concatenate([pos, neg])); rk = np.empty_like(o, float); rk[o] = np.arange(1, len(o) + 1)
    return (rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])

sel = (wf > 0.20) | (wf < 0.05); y = (wf[sel] > 0.20).astype(int)
print(f"\n== BAD(>20%) vs GOOD(<5%) separation ==")
print(f"  raw    chi2/ch : AUC={auc(chi2[sel], y):.3f}   GOODmed={np.median(chi2[wf<0.05]):.3f}  BADmed={np.median(chi2[wf>0.20]):.3f}")
print(f"  scaled chi2/ch : AUC={auc(chi2s[sel], y):.3f}   GOODmed={np.median(chi2s[wf<0.05]):.3f}  BADmed={np.median(chi2s[wf>0.20]):.3f}")
print(f"\n== CONTINUOUS: metric vs wrong-energy fraction (ALL {len(rows)} TPCs) ==")
print(f"  Spearman  raw chi2   vs wf = {spearman(chi2, wf):+.3f}")
print(f"  Spearman  scaled chi2 vs wf = {spearman(chi2s, wf):+.3f}")

# On-data efficiency estimator: a per-TPC metric threshold predicts "misplaced".
# Report, at a few cut points, the recovered vs flagged energy so it can be quoted on data.
metric = chi2s   # use the systematic-free version
print(f"\n== On-data efficiency estimator (flag TPC if scaled chi2/ch >= cut) ==")
print(f"  {'cut':>6} {'%TPC flagged':>12} {'%E flagged':>11} {'BAD recall':>11} {'GOOD contam':>12} {'flagged purity':>15}")
for cut in [0.05, 0.08, 0.10, 0.15, 0.20, 0.30]:
    fl = metric >= cut
    bad = wf > 0.20; good = wf < 0.05
    recall = (fl & bad).sum() / max(bad.sum(), 1)
    contam = (fl & good).sum() / max(good.sum(), 1)
    purity = (fl & (wf > 0.05)).sum() / max(fl.sum(), 1)   # flagged that are actually >5% wrong
    print(f"  {cut:>6.2f} {100*fl.mean():>11.1f}% {100*E[fl].sum()/E.sum():>10.1f}% "
          f"{100*recall:>10.1f}% {100*contam:>11.1f}% {100*purity:>14.1f}%")

# Data-quality summary number: total-energy-weighted mean metric (a single scalar per run)
print(f"\n== Single run-level scalar (quotable on data) ==")
print(f"  E-weighted mean scaled chi2/ch = {float((metric*E).sum()/E.sum()):.4f}")
print(f"  (truth: E-weighted wrong fraction = {float((wf*E).sum()/E.sum()):.4f})")
print("DONE", flush=True)
