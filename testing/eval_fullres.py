"""Round 8: FULL-resolution displaced-light.  The cleanest conservation form: for each
channel, over the FULL per-tick residual R(c,t)=A-alpha*P, the displaced light =
min(sum_t clear-deficit, sum_t clear-excess).  No t0 windows -> closely-spaced flashes
in busy TPCs (the diagnosed misses) are no longer blurred together.  Compare to the
windowed version; recompute chi2_flash + complexity in the same cube pass.
"""
import glob, os, json, numpy as np

HERE = os.path.abspath(os.path.dirname(__file__))
DUMPS = os.path.join(HERE, "v030v3_out", "dumps")
CLIP, GAIN, PEAK, CUT, W = 60780.0, 200.0, 105, 10, 4


def peaks(tr, frac=0.06, ms=8):
    thr = frac * tr.max(); pk = []
    for t in range(2, len(tr) - 2):
        if tr[t] > thr and tr[t] >= tr[t - 1] and tr[t] > tr[t + 1] and tr[t] >= tr[t - 2]:
            if pk and t - pk[-1] < ms:
                if tr[t] > tr[pk[-1]]: pk[-1] = t
            else: pk.append(t)
    return pk


R = {k: [] for k in ["chi2", "disp_win", "disp_full3", "disp_full2", "disp_full4",
                     "nflash", "E", "wf", "nlit"]}
files = sorted(glob.glob(os.path.join(DUMPS, "*.npz")))
for i, fp in enumerate(files):
    d = np.load(fp, allow_pickle=True); ev = int(str(d["event"]))
    tpc = np.asarray(d["tpc"]); e = np.asarray(d["e"], np.float64)
    lab = np.asarray(d["labels"]); tsf = np.asarray(d["ts_final"], np.float64); tr = np.asarray(d["truth_t0"], np.float64)
    full = np.asarray(d["full_light"], np.float32); base = np.asarray(d["base_image"], np.float32)
    for c in np.unique(tpc):
        mc = tpc == c; E = float(e[mc].sum())
        if E < 100 or E > 1400: continue
        ok = mc & np.isfinite(tr); den = float(e[ok].sum())
        if den <= 0: continue
        wf = float(e[ok & np.isfinite(tsf) & (np.abs(tsf - tr) > CUT)].sum() / den)
        act = full[int(c)].astype(np.float64); pred = np.minimum(base[int(c)].astype(np.float64), CLIP)
        sfull = act >= 0.98 * CLIP; m = ~sfull
        aa = act[m]; pp = pred[m]; alpha = float((pp @ aa) / max(pp @ pp, 1e-9))
        Ps = alpha * pred
        # candidate t0 list
        cand = list(peaks(act.sum(0)))
        for L in np.unique(lab[mc & (lab >= 0)]):
            mm = mc & (lab == L) & np.isfinite(tsf)
            if mm.sum() and float(e[mm].sum()) >= 10:
                pk = int(round(np.median(tsf[mm]))) + PEAK
                if 0 <= pk < 1000 and all(abs(pk - q) >= 8 for q in cand): cand.append(pk)
        cand = sorted(set(int(x) for x in cand))
        # chi2_flash (SNR-weighted) + disp_win from +-4 windows
        red = []; wts = []; Fa = []; Fp = []
        for t in cand:
            lo, hi = max(0, t - W), min(1000, t + W + 1)
            Aw = act[:, lo:hi]; Pw = Ps[:, lo:hi]; satc = Aw.max(1) >= 0.98 * CLIP
            A = Aw.sum(1); P = Pw.sum(1); A[satc] = 0; P[satc] = 0
            v = GAIN * np.maximum(A, 100.0); red.append(float(((P - A) ** 2 / v).sum() / len(A)))
            wts.append(np.sqrt(max(A.sum(), 0) / GAIN)); Fa.append(A); Fp.append(P)
        red = np.array(red); wts = np.array(wts)
        chi2 = float((red * wts).sum() / wts.sum()) if wts.sum() > 0 else 0.0
        Fa = np.array(Fa); Fp = np.array(Fp); tot = max(act.sum(), 1.0)
        if len(Fa):
            Rw = Fa - Fp; sw = np.sqrt(GAIN * np.maximum(Fa, 900.0)); Rwt = np.where(np.abs(Rw) > 3 * sw, Rw, 0.0)
            disp_win = float(np.minimum(np.maximum(Rwt, 0).sum(0), np.maximum(-Rwt, 0).sum(0)).sum() / tot)
        else:
            disp_win = 0.0
        # FULL-resolution per-channel conservation over all ticks
        Rf = act - Ps; sig = np.sqrt(GAIN * np.maximum(act, 100.0))
        def disp_full(k):
            Rt = np.where(np.abs(Rf) > k * sig, Rf, 0.0)
            defc = np.maximum(Rt, 0).sum(1); excc = np.maximum(-Rt, 0).sum(1)   # per channel over ticks
            return float(np.minimum(defc, excc).sum() / tot)
        R["chi2"].append(chi2); R["disp_win"].append(disp_win)
        R["disp_full3"].append(disp_full(3.0)); R["disp_full2"].append(disp_full(2.5)); R["disp_full4"].append(disp_full(4.0))
        R["nflash"].append(len(cand)); R["E"].append(E); R["wf"].append(wf)
        R["nlit"].append(int((act.sum(1) > 5 * np.sqrt(GAIN * 100.0) * np.sqrt(1000)).sum()))
    d.close(); del full, base
    print(f"  [{i+1}/{len(files)}] ev{ev}: {len(R['wf'])} TPCs", flush=True)

for k in R: R[k] = np.array(R[k])
WF = R["wf"]; EE = R["E"]; bad = WF > 0.20; good = WF < 0.05; sel = bad | good; y = bad[sel].astype(int)
logE = np.log(np.maximum(EE, 1))


def rank(x): return np.argsort(np.argsort(np.asarray(x, float))).astype(float) / len(x)
def auc(sc, yy):
    sc = np.asarray(sc); pos = sc[yy == 1]; neg = sc[yy == 0]
    o = np.argsort(np.concatenate([pos, neg])); rk = np.empty_like(o, float); rk[o] = np.arange(1, len(o) + 1)
    return (rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
def spr(a, b): return float(np.corrcoef(rank(a), rank(b))[0, 1])
def presid(metric, ctrls):
    rm = rank(metric); X = np.c_[[np.ones_like(rm)] + [rank(c) for c in ctrls]].T
    return rm - X @ np.linalg.lstsq(X, rm, rcond=None)[0]

print(f"\n{len(WF)} TPCs BAD={int(bad.sum())} GOOD={int(good.sum())}")
print(f"{'metric':>14} {'AUC':>7} {'rhoFrac':>9}")
for k in ["chi2", "disp_win", "disp_full2", "disp_full3", "disp_full4"]:
    print(f"{k:>14} {auc(R[k][sel], y):>7.3f} {spr(R[k], WF):>+9.3f}")
# best combos: chi2 + disp_full(3), raw and complexity-normalized
for dispname in ["disp_win", "disp_full3"]:
    zc, zd = rank(R["chi2"]), rank(R[dispname])
    cm = 0.5 * (zc + zd); cx = np.maximum(zc, zd)
    print(f"combo mean(chi2,{dispname:>10}): AUC={auc(cm[sel], y):.3f} rhoFrac={spr(cm, WF):+.3f}   "
          f"max: AUC={auc(cx[sel], y):.3f} rhoFrac={spr(cx, WF):+.3f}")
    ac = presid(R["chi2"], [R["nflash"], logE, R["nlit"]]); ad = presid(R[dispname], [R["nflash"], logE, R["nlit"]])
    cmr = 0.5 * (rank(ac) + rank(ad))
    print(f"   complexity-normalized combo: AUC={auc(cmr[sel], y):.3f} rhoFrac={spr(cmr, WF):+.3f}")

json.dump({k: list(map(float, R[k])) for k in R}, open(os.path.join(HERE, "eval_fullres_out.json"), "w"))
print("DONE")
