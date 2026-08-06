"""One expensive pass over v0.3 dumps -> compact cache of per-flash light primitives,
so any number of similarity metrics can be evaluated downstream WITHOUT reloading the
33MB light cubes.  Per non-shower TPC we store:
  - at, pt : (1000,) channel-summed actual / predicted(clipped, raw) light traces
  - flashes: for each t0 in the candidate list (actual flash peaks UNION placed t0s),
             the per-channel window-summed actual A(120) and predicted P(120) vectors
             (saturated channels zeroed), plus the flash tick and amplitude
  - alpha  : global per-TPC amplitude fit (removes the ~0.78 ML systematic)
  - wf, E  : MC-truth wrong-energy fraction and TPC energy (grading only)
"""
import glob, os, pickle, numpy as np

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


cache = []
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
        act = full[int(c)].astype(np.float64); pred = np.minimum(base[int(c)].astype(np.float64), CLIP)
        # global alpha (saturation-masked)
        sfull = act >= 0.98 * CLIP; m = ~sfull
        aa = act[m]; pp = pred[m]; alpha = float((pp @ aa) / max(pp @ pp, 1e-9))
        at = act.sum(0).astype(np.float32); pt = pred.sum(0).astype(np.float32)
        # candidate t0 list (ticks)
        cand = list(peaks(act.sum(0)))
        for L in np.unique(lab[mc & (lab >= 0)]):
            mm = mc & (lab == L) & np.isfinite(tsf)
            if mm.sum() and float(e[mm].sum()) >= 10:
                pk = int(round(np.median(tsf[mm]))) + PEAK
                if 0 <= pk < act.shape[1] and all(abs(pk - q) >= 8 for q in cand):
                    cand.append(pk)
        cand = sorted(set(int(x) for x in cand))
        Fa, Fp, Ft = [], [], []
        for t in cand:
            lo, hi = max(0, t - W), min(act.shape[1], t + W + 1)
            Aw = act[:, lo:hi]; Pw = pred[:, lo:hi]
            satc = Aw.max(1) >= 0.98 * CLIP
            A = Aw.sum(1); P = Pw.sum(1); A[satc] = 0.0; P[satc] = 0.0
            Fa.append(A.astype(np.float32)); Fp.append(P.astype(np.float32)); Ft.append(t)
        cache.append(dict(file=ft, event=ev, tpc=int(c), E=float(E), wf=float(wf), alpha=float(alpha),
                          at=at, pt=pt,
                          Fa=np.array(Fa, np.float32), Fp=np.array(Fp, np.float32), Ft=np.array(Ft, np.int32)))
    d.close(); del full, base
    print(f"  [{i+1}/{len(files)}] {ft} ev{ev}: {len(cache)} TPCs", flush=True)

with open(os.path.join(HERE, "lightcache.pkl"), "wb") as f:
    pickle.dump(cache, f)
print(f"WROTE lightcache.pkl : {len(cache)} TPCs", flush=True)
print("DONE", flush=True)
