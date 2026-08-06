"""Confidence metric vGPT: noise-whitened residual DIPOLE score.
Operates on the full time-resolved light image (120 ch x 1000 ticks) per TPC.
Faithful to the spec:
  Z = (P-A)/sqrt(sigma^2)         [Sec 2, sigma^2 = GAIN*(mu+100), mu=(|P|+|A|)/2, clip +-8]
  multi-scale L2-normalized Gaussian smoothing (sig_y in {0,1,2}, sig_t in {2,4,8,16})  [Sec 3]
  excess X=max(Zs-tau,0), deficit D=max(-Zs-tau,0), tau=2
  connected components -> strength m, t-centroid, spatial profile  [Sec 4]
  pair overflow<->deficit: Sy*Sm*St, Hungarian match, Q_dipole=max_s  [Sec 5-6]
  Q_noise (active-mask Huber), Q_patch (top-k of Zs^2)  [Sec 7-8]
  fast shift-corr approximation  [Sec 11]
alpha-corrected P (removes the 0.78 ML systematic so Q_noise~1 for good matches).
Graded vs MC-truth mis-classification energy; combined with v1/v2.
"""
import glob, os, json, numpy as np
from multiprocessing import Pool
from scipy.ndimage import gaussian_filter, label, binary_dilation
from scipy.optimize import linear_sum_assignment

HERE = os.path.abspath(os.path.dirname(__file__))
DUMPS = os.path.join(HERE, "v030v3_out", "dumps")
CLIP, GAIN, PEAK, CUT = 60780.0, 200.0, 105, 10
TAU, TMIN, TRUNC = 2.0, 15.0, 3.0
SCALES = [(sy, st) for sy in (0, 1, 2) for st in (2, 4, 8, 16)]
SIG0 = np.sqrt(GAIN * 100.0)
NPROC = 10

# L2 norm of each scale's effective kernel (for unit-variance preservation)
_delta = np.zeros((41, 141)); _delta[20, 70] = 1.0
KN = {s: np.sqrt((gaussian_filter(_delta, s, mode="constant", truncate=TRUNC) ** 2).sum()) for s in SCALES}


def peaks(tr, frac=0.06, ms=8):
    thr = frac * tr.max(); pk = []
    for t in range(2, len(tr) - 2):
        if tr[t] > thr and tr[t] >= tr[t - 1] and tr[t] > tr[t + 1] and tr[t] >= tr[t - 2]:
            if pk and t - pk[-1] < ms:
                if tr[t] > tr[pk[-1]]: pk[-1] = t
            else: pk.append(t)
    return pk


def components(field, Wsq):
    """connected comps of field>0; return list of (m, tcent, vprofile[nch])."""
    lab, n = label(field > 0)
    out = []
    for c in range(1, n + 1):
        mask = lab == c; W = np.where(mask, Wsq, 0.0)
        m = float(W.sum())
        if m <= 0: continue
        jj = np.arange(W.shape[1]); tcent = float((W.sum(0) * jj).sum() / m)
        v = W.sum(1); nv = np.linalg.norm(v)
        out.append((m, tcent, v / nv if nv > 0 else v))
    return out


def _crop(A, P):
    """crop to the active bounding box (+margin) to cut smoothing cost."""
    cs = A.sum(0)
    at = np.where(cs > 0.02 * cs.max())[0] if cs.max() > 0 else np.array([], int)
    if len(at) == 0:
        return None
    t0 = max(0, at.min() - 24); t1 = min(A.shape[1], at.max() + 24)
    rs = A[:, t0:t1].sum(1) + P[:, t0:t1].sum(1)
    ac = np.where(rs > 0.02 * rs.max())[0] if rs.max() > 0 else np.arange(A.shape[0])
    c0 = max(0, ac.min() - 4); c1 = min(A.shape[0], ac.max() + 5)
    return A[c0:c1, t0:t1], P[c0:c1, t0:t1]


def vgpt_scores(A, P):
    cr = _crop(A, P)
    if cr is None:
        return 0.0, 0.0, 0.0, 0.0
    A, P = cr
    mu = 0.5 * (np.abs(P) + np.abs(A)); sig = np.sqrt(GAIN * (mu + 100.0))
    Z = np.clip((P - A) / sig, -8, 8)
    # active mask (near a real or predicted flash), dilated -> Q_noise
    active = (A > 5 * SIG0) | (P > 5 * SIG0)
    active = binary_dilation(active, structure=np.ones((5, 17)))
    if active.sum() < 10: active = np.ones_like(active, bool)
    hub = np.where(np.abs(Z) < 2.0, 0.5 * Z ** 2, 2.0 * (np.abs(Z) - 1.0))    # Huber
    Q_noise = float(hub[active].mean())
    Q_patch = 0.0; Q_dipole = 0.0; Q_shift = 0.0
    for s in SCALES:
        Zs = gaussian_filter(Z, s, mode="constant", truncate=TRUNC) / KN[s]
        flat = (Zs ** 2).ravel(); k = min(5, flat.size)
        Q_patch = max(Q_patch, float(np.partition(flat, -k)[-k:].mean()))
        X = np.maximum(Zs - TAU, 0.0); D = np.maximum(-Zs - TAU, 0.0)
        if X.max() <= 0 or D.max() <= 0:
            continue
        # dipole via components + Hungarian
        Xc = components(X, X ** 2); Dc = components(D, D ** 2)
        if Xc and Dc:
            Q = np.zeros((len(Xc), len(Dc)))
            for a, (mp, tp, vp) in enumerate(Xc):
                for b, (mn, tn, vn) in enumerate(Dc):
                    Sy = float(vp @ vn); Sm = 2 * mp * mn / (mp + mn)
                    St = 1.0 - np.exp(-((tp - tn) / TMIN) ** 2)
                    Q[a, b] = Sy * Sm * St
            r, c = linear_sum_assignment(-Q)
            Q_dipole = max(Q_dipole, float(Q[r, c].sum()))
        # fast shift-corr approximation (FFT over time, channel-matched, summed)
        nX = np.linalg.norm(X); nD = np.linalg.norm(D)
        if nX > 0 and nD > 0:
            Xf = np.fft.rfft(X, axis=1); Df = np.fft.rfft(D, axis=1)
            cc = np.fft.irfft((Xf * np.conj(Df)).sum(0), n=X.shape[1])     # cross-corr over dt
            dt = np.fft.fftfreq(X.shape[1], d=1) * X.shape[1]
            gate = np.abs(dt) > TMIN
            if gate.any():
                Cmax = (cc[gate] / (nX * nD + 1e-9)).max()
                Q_shift = max(Q_shift, float(Cmax * min(nX, nD)))
    return Q_noise, Q_patch, Q_dipole, Q_shift


R = {k: [] for k in ["Qnoise", "Qpatch", "Qdipole", "Qshift", "wf", "E", "nflash"]}
files = sorted(glob.glob(os.path.join(DUMPS, "*.npz")))
pool = Pool(NPROC)
for i, fp in enumerate(files):
    d = np.load(fp, allow_pickle=True); ev = int(str(d["event"]))
    tpc = np.asarray(d["tpc"]); e = np.asarray(d["e"], np.float64)
    tsf = np.asarray(d["ts_final"], np.float64); tr = np.asarray(d["truth_t0"], np.float64)
    full = np.asarray(d["full_light"], np.float32); base = np.asarray(d["base_image"], np.float32)
    jobs = []; meta = []
    for c in np.unique(tpc):
        mc = tpc == c; E = float(e[mc].sum())
        if E < 100 or E > 1400: continue
        ok = mc & np.isfinite(tr); den = float(e[ok].sum())
        if den <= 0: continue
        wf = float(e[ok & np.isfinite(tsf) & (np.abs(tsf - tr) > CUT)].sum() / den)
        A = full[int(c)].astype(np.float64); pred = np.minimum(base[int(c)].astype(np.float64), CLIP)
        m = A < 0.98 * CLIP; aa = A[m]; pp = pred[m]; alpha = float((pp @ aa) / max(pp @ pp, 1e-9))
        jobs.append((A, alpha * pred)); meta.append((wf, E, len(peaks(A.sum(0)))))
    for (qn, qp, qd, qs), (wf, E, nf) in zip(pool.starmap(vgpt_scores, jobs), meta):
        R["Qnoise"].append(qn); R["Qpatch"].append(qp); R["Qdipole"].append(qd); R["Qshift"].append(qs)
        R["wf"].append(wf); R["E"].append(E); R["nflash"].append(nf)
    d.close(); del full, base
    print(f"  [{i+1}/{len(files)}] ev{ev}: {len(R['wf'])} TPCs", flush=True)
pool.close(); pool.join()

for k in R: R[k] = np.array(R[k])
json.dump({k: list(map(float, R[k])) for k in R}, open(os.path.join(HERE, "vgpt_out.json"), "w"))
WF = R["wf"]; EE = R["E"]; bad = WF > 0.20; good = WF < 0.05; sel = bad | good; y = bad[sel].astype(int)


def rank(x): return np.argsort(np.argsort(np.asarray(x, float))).astype(float) / len(x)
def auc(sc, yy):
    sc = np.asarray(sc); pos = sc[yy == 1]; neg = sc[yy == 0]
    o = np.argsort(np.concatenate([pos, neg])); rk = np.empty_like(o, float); rk[o] = np.arange(1, len(o) + 1)
    return (rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
def spr(a, b): return float(np.corrcoef(rank(a), rank(b))[0, 1])

print(f"\n{len(WF)} TPCs BAD={int(bad.sum())} GOOD={int(good.sum())}  (v2 ref: AUC0.935 rhoFrac0.654)")
print(f"{'metric':>10} {'AUC':>7} {'rhoFrac':>9}")
for k in ["Qnoise", "Qpatch", "Qdipole", "Qshift"]:
    print(f"{k:>10} {auc(R[k][sel], y):>7.3f} {spr(R[k], WF):>+9.3f}")
# calibrated Q_failure (Sec 10): z from GOOD empirical CDF
from math import erfinv, sqrt
def zcal(Q, refmask):
    ref = np.sort(Q[refmask]); F = np.searchsorted(ref, Q, side="right") / (len(ref) + 1)
    F = np.clip(F, 1e-4, 1 - 1e-4); return np.array([sqrt(2) * erfinv(2 * f - 1) for f in F])
zn, zp, zd = zcal(R["Qnoise"], good), zcal(R["Qpatch"], good), zcal(R["Qdipole"], good)
for lam in [1.0, 2.0, 4.0]:
    Qf = zn ** 2 + zp ** 2 + lam * zd ** 2
    print(f"Q_failure(lam={lam}): AUC={auc(Qf[sel], y):.3f} rhoFrac={spr(Qf, WF):+.3f}")

# combine with v1/v2
try:
    V1 = json.load(open(os.path.join(HERE, "eval_final_out.json")))
    VF = json.load(open(os.path.join(HERE, "confidence_vfable_out.json")))
    FR = json.load(open(os.path.join(HERE, "eval_fullres_out.json")))
    if len(V1["wf"]) == len(WF):
        NF = np.array(FR["nflash"], float); NL = np.array(FR["nlit"], float); lE = np.log(np.maximum(EE, 1))
        def presid(mm):
            r = rank(mm); X = np.c_[[np.ones_like(r), rank(NF), rank(lE), rank(NL)]].T
            return r - X @ np.linalg.lstsq(X, r, rcond=None)[0]
        parts = {"chi2": np.array(V1["chi2"]), "disp": np.array(V1["disp"]), "maxk": np.array(VF["maxk"]),
                 "Qdipole": R["Qdipole"], "Qpatch": R["Qpatch"], "Qshift": R["Qshift"]}
        pr = {k: rank(presid(v)) for k, v in parts.items()}
        for nm, ks in [("v2(chi2,disp,maxk)", ["chi2", "disp", "maxk"]),
                       ("v2+Qdipole", ["chi2", "disp", "maxk", "Qdipole"]),
                       ("disp+maxk+Qdipole", ["disp", "maxk", "Qdipole"]),
                       ("disp+maxk+Qshift", ["disp", "maxk", "Qshift"]),
                       ("all6", list(parts.keys()))]:
            c = np.mean([pr[k] for k in ks], axis=0)
            print(f"combo {nm:>22}: AUC={auc(c[sel], y):.3f} rhoFrac={spr(c, WF):+.3f}")
except Exception as ex:
    print("combine:", ex)
print("DONE")
