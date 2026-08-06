"""Battery of predicted-vs-actual SIMILARITY metrics, each graded against MC-truth
mis-classification energy.  Reads lightcache.pkl (no cube reload).  For every metric
we report: AUC(BAD>20% vs GOOD<5%), Spearman vs wrong-energy FRACTION, and Spearman
vs ABSOLUTE mis-classified MeV.  Families tried:
  dot/cosine, Pearson correlation, Jensen-Shannon, total-variation, SSIM (structural),
  amplitude-aware normalized distance, per-flash energy-flow (deficit/excess),
  1-D temporal optimal transport (EMD).  Each per-flash score is aggregated with
  weights {equal, amplitude, SNR/significance, saturating, worst-resolved-flash} to
  study the normalization the user flagged (global magnitude -> big peak dominates).
"""
import os, pickle, numpy as np

HERE = os.path.abspath(os.path.dirname(__file__))
GAIN, W = 200.0, 4
NOISE = np.sqrt(GAIN * 100.0)
A_HALF = 20 * NOISE * np.sqrt(120 * (2 * W + 1))
RESOLVED_AMP = 8 * NOISE * np.sqrt(120 * (2 * W + 1))   # flash clearly above window noise
cache = pickle.load(open(os.path.join(HERE, "lightcache.pkl"), "rb"))
print(f"loaded {len(cache)} TPCs")


def _n(v):
    s = v.sum(); return v / s if s > 0 else v


def per_flash_scores(A, P):
    """A,P: (120,) actual & alpha-scaled predicted channel vectors at one flash.
    Return dict of BADNESS scores (higher = worse match)."""
    A = np.maximum(A, 0.0); P = np.maximum(P, 0.0)
    nA = np.linalg.norm(A); nP = np.linalg.norm(P)
    out = {}
    out["cos"] = 1.0 - (A @ P / (nA * nP)) if nA > 0 and nP > 0 else 1.0
    # Pearson (mean-centered cosine) across channels
    Ac = A - A.mean(); Pc = P - P.mean(); dc = np.linalg.norm(Ac) * np.linalg.norm(Pc)
    out["pear"] = 1.0 - (Ac @ Pc / dc) if dc > 0 else 1.0
    # distribution divergences on normalized channel profiles
    a = _n(A); p = _n(P)
    if a.sum() > 0 and p.sum() > 0:
        out["tv"] = 0.5 * np.abs(a - p).sum()
        mmix = 0.5 * (a + p); eps = 1e-12
        kl = lambda x, y: np.sum(np.where(x > 0, x * np.log((x + eps) / (y + eps)), 0.0))
        out["js"] = float(0.5 * kl(a, mmix) + 0.5 * kl(p, mmix)) / np.log(2)
        out["chi2d"] = float(0.5 * np.sum((a - p) ** 2 / (a + p + eps)))
    else:
        out["tv"] = out["js"] = out["chi2d"] = 1.0
    # SSIM (structural) over the channel vector
    muA, muP = A.mean(), P.mean(); vA, vP = A.var(), P.var(); cov = ((A - muA) * (P - muP)).mean()
    Lr = max(A.max(), P.max(), 1.0); c1 = (0.01 * Lr) ** 2; c2 = (0.03 * Lr) ** 2
    ssim = ((2 * muA * muP + c1) * (2 * cov + c2)) / ((muA ** 2 + muP ** 2 + c1) * (vA + vP + c2))
    out["ssim"] = 1.0 - float(ssim)
    # amplitude-aware symmetric normalized distance
    out["dist"] = float(np.linalg.norm(P - A) / (nA + nP)) if (nA + nP) > 0 else 0.0
    return out


def temporal_emd(at, pt):
    """1-D Wasserstein-1 between normalized channel-summed time traces (tick units)."""
    a = _n(np.maximum(at, 0.0)); p = _n(np.maximum(pt, 0.0))
    if a.sum() == 0 or p.sum() == 0:
        return 0.0
    return float(np.abs(np.cumsum(a) - np.cumsum(p)).sum())


SCORES = ["cos", "pear", "tv", "js", "chi2d", "ssim", "dist"]
WEIGHTS = ["eq", "amp", "snr", "sat"]
rows_metric = {f"{s}_{w}": [] for s in SCORES for w in WEIGHTS}
rows_metric.update({f"{s}_worst": [] for s in SCORES})
extra = {"emd_time": [], "misplaced_light": [], "global_cos": [], "global_pear": []}
WF, EE = [], []

for r in cache:
    A_all, P_all, alpha = r["Fa"].astype(np.float64), r["Fp"].astype(np.float64) * r["alpha"], r["alpha"]
    if len(A_all) == 0:
        continue
    amp = A_all.sum(1)                                   # per-flash actual light
    w_eq = np.ones_like(amp); w_amp = amp
    w_snr = np.sqrt(np.maximum(amp, 0.0) / GAIN)
    w_sat = amp / (amp + A_HALF)
    wdict = {"eq": w_eq, "amp": w_amp, "snr": w_snr, "sat": w_sat}
    resolved = amp > RESOLVED_AMP
    sc = {s: [] for s in SCORES}
    for A, P in zip(A_all, P_all):
        d = per_flash_scores(A, P)
        for s in SCORES:
            sc[s].append(d[s])
    for s in SCORES:
        v = np.array(sc[s])
        for wn, wv in wdict.items():
            sm = wv.sum()
            rows_metric[f"{s}_{wn}"].append(float((v * wv).sum() / sm) if sm > 0 else 0.0)
        rows_metric[f"{s}_worst"].append(float(v[resolved].max()) if resolved.any() else float(v.max()))
    # energy-flow: per-flash |actual - predicted| total light / total actual light (light at wrong flash)
    ampP = P_all.sum(1)
    extra["misplaced_light"].append(float(np.abs(amp - ampP).sum() / max(amp.sum(), 1.0)))
    extra["emd_time"].append(temporal_emd(r["at"].astype(np.float64), r["pt"].astype(np.float64)))
    # global (whole-TPC) cosine & pearson on the time trace -> the "big-peak-dominated" baselines
    aT = np.maximum(r["at"].astype(np.float64), 0); pT = np.maximum(r["pt"].astype(np.float64), 0)
    nA, nP = np.linalg.norm(aT), np.linalg.norm(pT)
    extra["global_cos"].append(1.0 - (aT @ pT / (nA * nP)) if nA > 0 and nP > 0 else 1.0)
    ac, pc = aT - aT.mean(), pT - pT.mean(); dc = np.linalg.norm(ac) * np.linalg.norm(pc)
    extra["global_pear"].append(1.0 - (ac @ pc / dc) if dc > 0 else 1.0)
    WF.append(r["wf"]); EE.append(r["E"])

WF = np.array(WF); EE = np.array(EE); WE = WF * EE
bad = WF > 0.20; good = WF < 0.05; sel = bad | good; y = bad[sel].astype(int)


def auc(sc, yy):
    sc = np.asarray(sc); pos = sc[yy == 1]; neg = sc[yy == 0]
    o = np.argsort(np.concatenate([pos, neg])); rk = np.empty_like(o, float); rk[o] = np.arange(1, len(o) + 1)
    return (rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def spr(a, b):
    a = np.asarray(a); return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


allm = {**rows_metric, **extra}
results = []
for name, vals in allm.items():
    v = np.array(vals)
    if v.std() == 0 or not np.isfinite(v).all():
        continue
    results.append((auc(v[sel], y), spr(v, WF), spr(v, WE), name))
results.sort(key=lambda r: -r[2])   # rank by |Spearman vs absolute MeV| (the stated target)
print(f"\n{len(WF)} TPCs  BAD={int(bad.sum())} GOOD={int(good.sum())}  (ref chi2s: AUC0.871 rhoF+0.434 rhoMeV+0.685)")
print(f"{'metric':>16} {'AUC':>7} {'rhoFrac':>9} {'rhoMeV':>8}")
for a, rf, rm, n in results:
    print(f"{n:>16} {a:>7.3f} {rf:>+9.3f} {rm:>+8.3f}")

import json
json.dump({"metrics": {n: list(map(float, allm[n])) for n in allm},
           "wf": list(map(float, WF)), "E": list(map(float, EE))},
          open(os.path.join(HERE, "eval_metrics_out.json"), "w"))
print("\nsaved eval_metrics_out.json")
print("TOP by rhoMeV:", results[0][3], "| TOP by AUC:", max(results, key=lambda r: r[0])[3],
      "| TOP by rhoFrac:", max(results, key=lambda r: r[1])[3])
print("DONE")
