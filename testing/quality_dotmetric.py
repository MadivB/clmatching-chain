"""Per-flash predicted.actual dot-product quality metric, with the normalization
studied carefully (user request 2026-07-09).

Idea: evaluate predicted-vs-actual light at each t0 in the TPC's candidate list
(union of ACTUAL flash peaks and PLACED cluster t0s -> includes empty t0s), using
the FULL predicted base_image in a window (so a ridge fluctuation is explained by
the neighboring flash tail and not falsely penalized).  The single number is a
weighted aggregate of per-flash agreement.  The normalization question: global
magnitude division is dominated by the brightest peak, so instead normalize
LOCALLY (per flash) and weight the aggregate by statistical SIGNIFICANCE, not
brightness.  We compute many (per-flash score x weight) variants and grade each
against MC truth (BAD>20% vs GOOD<5% AUC, and Spearman vs wrong-energy fraction
AND absolute mis-classified MeV).
"""
import glob, json, os, numpy as np

HERE = os.path.abspath(os.path.dirname(__file__))
DUMPS = os.path.join(HERE, "v030v3_out", "dumps")
CLIP, GAIN, PEAK, CUT = 60780.0, 200.0, 105, 10
W = 4                     # +/- ticks per flash window
NOISE = np.sqrt(GAIN * 100.0)   # per-cell baseline sigma ~141 ADC


def peaks(tr, frac=0.06, ms=8):
    thr = frac * tr.max(); pk = []
    for t in range(2, len(tr) - 2):
        if tr[t] > thr and tr[t] >= tr[t - 1] and tr[t] > tr[t + 1] and tr[t] >= tr[t - 2]:
            if pk and t - pk[-1] < ms:
                if tr[t] > tr[pk[-1]]: pk[-1] = t
            else: pk.append(t)
    return pk


def tpc_metrics(act, pred, placed_ticks):
    """act,pred: (120,1000) actual & predicted light.  placed_ticks: reco t0+PEAK ticks."""
    at = act.sum(0)
    # candidate t0 list (ticks): actual flash peaks UNION placed-cluster peaks, dedup within window
    cand = list(peaks(at))
    for pt in placed_ticks:
        if 0 <= pt < act.shape[1] and all(abs(pt - q) >= ms for q, ms in [(c, 8) for c in cand]):
            cand.append(int(pt))
    cand = sorted(set(int(c) for c in cand))
    if not cand:
        return None
    # global amplitude fit alpha (remove 0.78 ML systematic), saturation-masked
    sat_full = act >= 0.98 * CLIP; m = ~sat_full
    a_all = act[m]; p_all = np.minimum(pred, CLIP)[m]
    alpha = float((p_all @ a_all) / max(p_all @ p_all, 1e-9))

    coss, dists, amps = [], [], []
    for t in cand:
        lo, hi = max(0, t - W), min(act.shape[1], t + W + 1)
        Aw = act[:, lo:hi]; Pw = np.minimum(pred[:, lo:hi], CLIP)
        satc = Aw.max(1) >= 0.98 * CLIP
        A = Aw[~satc].sum(1); P = alpha * Pw[~satc].sum(1)     # per-channel vectors at this flash
        nA = np.linalg.norm(A); nP = np.linalg.norm(P)
        amp = float(A.sum())                                    # flash total light (>=0)
        cos = float(A @ P / (nA * nP)) if nA > 0 and nP > 0 else 0.0
        dist = float(np.linalg.norm(P - A) / (nA + nP)) if (nA + nP) > 0 else 0.0
        coss.append(cos); dists.append(dist); amps.append(max(amp, 0.0))
    cos = np.array(coss); dist = np.array(dists); amp = np.array(amps)

    # weights
    w_eq = np.ones_like(amp)
    w_amp = amp.copy()
    w_snr = np.sqrt(np.maximum(amp, 0.0) / GAIN)                       # matched-filter significance
    A_half = 20 * NOISE * np.sqrt(120 * (2 * W + 1))                   # ~ few-sigma window noise
    w_sat = amp / (amp + A_half)                                       # democratic among resolved flashes
    resolved = w_snr > (3 * np.sqrt(120 * (2 * W + 1)))                # flashes clearly above noise

    def wmean(x, w):
        s = w.sum()
        return float((x * w).sum() / s) if s > 0 else 0.0

    out = {}
    # per-flash COSINE agreement -> badness = weighted mean of (1-cos)
    d = 1.0 - cos
    out["cos_eq"] = wmean(d, w_eq)
    out["cos_amp"] = wmean(d, w_amp)          # baseline "divide by amplitude" (linear weight)
    out["cos_snr"] = wmean(d, w_snr)
    out["cos_sat"] = wmean(d, w_sat)
    out["cos_worst"] = float(d[resolved].max()) if resolved.any() else float(d.max())
    # per-flash normalized DISTANCE (catches deficit/excess) -> badness = weighted mean
    out["dist_eq"] = wmean(dist, w_eq)
    out["dist_amp"] = wmean(dist, w_amp)
    out["dist_snr"] = wmean(dist, w_snr)
    out["dist_sat"] = wmean(dist, w_sat)
    out["dist_worst"] = float(dist[resolved].max()) if resolved.any() else float(dist.max())
    # the "bad" GLOBAL-magnitude cosine the user warned about (whole-TPC, dominated by big peak)
    gA = act[m]; gP = alpha * np.minimum(pred, CLIP)[m]
    ngA = np.linalg.norm(gA); ngP = np.linalg.norm(gP)
    out["global_cos"] = float(1.0 - (gA @ gP) / (ngA * ngP)) if ngA > 0 and ngP > 0 else 1.0
    out["n_flash"] = int(len(cand)); out["n_resolved"] = int(resolved.sum())
    return out


rows = []
files = sorted(glob.glob(os.path.join(DUMPS, "*.npz")))
for i, fp in enumerate(files):
    d = np.load(fp, allow_pickle=True)
    ft = os.path.basename(fp).split("_ev")[0]; ev = int(str(d["event"]))
    tpc = np.asarray(d["tpc"]); e = np.asarray(d["e"], np.float64)
    lab = np.asarray(d["labels"]); tsf = np.asarray(d["ts_final"], np.float64); tr = np.asarray(d["truth_t0"], np.float64)
    full = np.asarray(d["full_light"], np.float32); base = np.asarray(d["base_image"], np.float32)
    for c in np.unique(tpc):
        mc = tpc == c; E = float(e[mc].sum())
        if E < 100 or E > 1400:
            continue
        ok = mc & np.isfinite(tr); den = float(e[ok].sum())
        if den <= 0:
            continue
        wf = float(e[ok & np.isfinite(tsf) & (np.abs(tsf - tr) > CUT)].sum() / den)
        # placed cluster ticks in this TPC
        placed = []
        for L in np.unique(lab[mc & (lab >= 0)]):
            mm = mc & (lab == L) & np.isfinite(tsf)
            if mm.sum() and float(e[mm].sum()) >= 10:
                placed.append(float(np.median(tsf[mm])) + PEAK)
        M = tpc_metrics(full[int(c)].astype(np.float64), base[int(c)].astype(np.float64), placed)
        if M is None:
            continue
        M.update(dict(file=ft, event=ev, tpc=int(c), E=round(E, 1), wf=round(wf, 4)))
        rows.append(M)
    d.close(); del full, base
    print(f"  [{i+1}/{len(files)}] {ft} ev{ev}: {len(rows)} TPCs", flush=True)

json.dump(rows, open(os.path.join(HERE, "quality_dotmetric.json"), "w"))
wf = np.array([r["wf"] for r in rows]); E = np.array([r["E"] for r in rows]); we = E * wf
bad = wf > 0.20; good = wf < 0.05; sel = bad | good; y = bad[sel].astype(int)


def auc(sc, yy):
    pos = sc[yy == 1]; neg = sc[yy == 0]
    o = np.argsort(np.concatenate([pos, neg])); rk = np.empty_like(o, float); rk[o] = np.arange(1, len(o) + 1)
    return (rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def spr(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


variants = ["global_cos", "cos_eq", "cos_amp", "cos_snr", "cos_sat", "cos_worst",
            "dist_eq", "dist_amp", "dist_snr", "dist_sat", "dist_worst"]
print(f"\n{len(rows)} TPCs.  BAD={int(bad.sum())} GOOD={int(good.sum())} MARGINAL={int((~bad&~good).sum())}")
print(f"{'variant':>12} {'AUC(BAD/GOOD)':>14} {'rho vs frac':>12} {'rho vs |MeV|':>13}")
res = []
for v in variants:
    s = np.array([r[v] for r in rows])
    a = auc(s[sel], y); rf = spr(s, wf); rm = spr(s, we)
    res.append((a, v, rf, rm))
    print(f"{v:>12} {a:>14.3f} {rf:>+12.3f} {rm:>+13.3f}")
print(f"\n(reference: chi2_scaled AUC=0.871, rho_frac=+0.434, rho_MeV=+0.685)")
best = max(res, key=lambda r: r[0])
print(f"\nBEST by AUC: {best[1]}  (AUC={best[0]:.3f}, rho_frac={best[2]:+.3f}, rho_MeV={best[3]:+.3f})")
print("DONE", flush=True)
