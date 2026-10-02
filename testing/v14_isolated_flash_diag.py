"""Why does the chain not place small clusters at isolated flashes that a light-only finder sees?
Two outputs on the certification dumps:
 (1) ANATOMY of the missed isolated flashes: truth flashes of 2-20 MeV in a TPC with no >= 3x brighter flash within
     60 ticks in the module, found by the loose summed finder (S10) but with no chain family within 5 ticks.  For the
     clusters of that flash: energy, templated or not, where the chain put them (host family energy, glare), and for
     templated ones the light evidence at the true vs the chosen time: pattern cosine on the support channels,
     amplitude ratio, delta chi2 (own light added back, Poisson weights).
 (2) a per-small-cluster TABLE (all templated 0.5-20 MeV clusters): current time, correct energy at current time,
     cosine / amplitude ratio at the current time and at every RESIDUAL flash of the TPC (S10 peaks of actual - base
     not within 5 ticks of a family) with the correct energy if moved there -> lets a cosine-based mover be evaluated
     offline (v14_isolated_flash_mover.py).
python v14_isolated_flash_diag.py <arm_dir> [n_events]"""
import glob
import json
import os
import sys

import numpy as np
from scipy.signal import find_peaks

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "flash_discovery")
PULSE = os.path.join(HERE, "..", "CLMatching_AlphaRelease", "assets", "avg_pulse.npy")
CUT = 10.0; PEAK = 105; TOL = 5.0; HW = 2; GAIN = 200.0
T = np.load(PULSE).astype(np.float64).ravel(); T = T / T.max(); NT = T.size


def link(times, gap):
    o = np.argsort(times); t = times[o]; cuts = np.flatnonzero(np.diff(t) > gap) + 1
    g = np.zeros(t.size, int); g[o] = np.repeat(np.arange(cuts.size + 1), np.diff(np.r_[0, cuts, t.size])); return g


def render(amp, t, n=1000):
    s = np.arange(n); sh = np.interp(s - t, np.arange(NT), T, left=0.0, right=0.0)
    return amp[:, None] * sh[None, :]


def support(amp, frac=0.9):
    o = np.argsort(-amp); c = np.cumsum(amp[o]); k = int(np.searchsorted(c, frac * c[-1])) + 1
    return o[:max(k, 3)]


def peak_pattern(R, t):
    i = int(np.clip(round(t) + PEAK, HW, R.shape[1] - HW - 1)); return R[:, i - HW:i + HW + 1].mean(1)


def cos_amp(R, amp, sup, t):
    p = peak_pattern(R, t)[sup]; a = amp[sup]
    c = float(a @ p / (np.linalg.norm(a) * np.linalg.norm(p) + 1e-9)); ratio = float(a @ a / max(a @ p, 1e-9))
    return c, ratio


def chi2_at(R, A, amp, sup, t, lo=-10, hi=80):
    i0 = int(np.clip(round(t) + PEAK + lo, 0, R.shape[1])); i1 = int(np.clip(round(t) + PEAK + hi, 0, R.shape[1]))
    if i1 <= i0:
        return np.nan
    own = render(amp[sup], t, R.shape[1])[:, i0:i1]; r = R[sup, i0:i1]; w = 1.0 / (GAIN * np.maximum(A[sup, i0:i1], 100.0))
    return float((w * (r - own) ** 2).sum() - (w * r ** 2).sum())      # chi2 with the cluster here minus without


def main(arm, n_max=None):
    files = sorted(glob.glob(f"{arm}/full_dumps/*.npz"))[: int(n_max) if n_max else None]
    anat = []; table = []; cand_rows = []
    for fi, ff in enumerate(files):
        z = np.load(ff, allow_pickle=True)
        e = np.clip(z["e"], 0, None).astype(float); ts = z["ts_final"].astype(float); tru = z["truth_t0"].astype(float)
        labs = z["labels"].astype(int); tpc = z["tpc"].astype(int); xyz = np.stack([z["x"], z["y"], z["z"]], 1).astype(float)
        full = z["full_light"]; base = z["base_image"]; veto = z["veto_mask"]
        al = z["amp_lab"].astype(int); at = z["amp_tpc"].astype(int); ar = z["amp_row"].astype(float)
        amp = {(int(L), int(c)): ar[i] for i, (L, c) in enumerate(zip(al, at))}
        lr = {r["label"]: r for r in json.loads(str(z["lab_rows"]))}
        ok = np.isfinite(tru) & (tru >= 0) & (tru <= 885)
        g = np.full(tru.size, -1); g[ok] = link(np.round(tru[ok]), 1.5); n_g = g.max() + 1
        gE = np.bincount(g[ok], weights=e[ok], minlength=n_g); gT = np.bincount(g[ok], weights=e[ok] * tru[ok], minlength=n_g) / np.maximum(gE, 1e-9)
        asg = (labs >= 0) & np.isfinite(ts) & (ts >= 0)
        # per-label summary
        labE = {}; labT = {}; labTr = {}
        for L in np.unique(labs[labs >= 0]):
            m = labs == L; labE[int(L)] = e[m].sum()
            v = ts[m & asg]; labT[int(L)] = float(np.median(v)) if v.size else np.nan
            mo = m & ok; labTr[int(L)] = float(np.median(tru[mo])) if mo.any() else np.nan
        Ecg_all = {int(c): np.bincount(g[(tpc == c) & ok], weights=e[(tpc == c) & ok], minlength=n_g) for c in np.unique(tpc)}
        for c in np.unique(tpc):
            c = int(c)
            if c >= full.shape[0]:
                continue
            chans = np.flatnonzero(~veto[c])
            if chans.size == 0:
                continue
            A = full[c].astype(np.float64); B = base[c].astype(np.float64); R = A - B
            sig = max(float(A[chans, :90].sum(0).std()), 1.0)
            pk_act, _ = find_peaks(A[chans].sum(0), prominence=10 * sig, distance=4); t_act = pk_act - PEAK; t_act = t_act[(t_act >= 0) & (t_act <= 885)]
            pk_res, _ = find_peaks(R[chans].sum(0), prominence=10 * sig, distance=4); t_res = pk_res - PEAK; t_res = t_res[(t_res >= 0) & (t_res <= 885)].astype(float)
            mc = tpc == c
            fam_t = np.unique(np.round(ts[mc & asg])); famE = {float(t_): e[mc & asg & (np.round(ts) == t_)].sum() for t_ in fam_t}
            def fam_energy_at(t):
                if fam_t.size == 0:
                    return 0.0
                near = fam_t[np.abs(fam_t - t) <= 3]; return float(sum(famE[float(x)] for x in near))
            t_res_free = np.array([t for t in t_res if fam_t.size == 0 or np.min(np.abs(fam_t - t)) > TOL])
            # ---- (1) missed isolated flashes
            Ecg = Ecg_all[c]; Emod = Ecg + Ecg_all.get(c ^ 1, np.zeros(n_g))
            for gi in np.flatnonzero((Ecg >= 2) & (Ecg < 20)):
                t0 = gT[gi]
                big = np.flatnonzero((Emod >= 3 * Ecg[gi]) & (np.abs(gT - t0) > 0.01))
                if big.size and np.min(np.abs(gT[big] - t0)) <= 60:
                    continue
                found_ff = t_act.size > 0 and np.min(np.abs(t_act - t0)) <= TOL
                found_chain = fam_t.size > 0 and np.min(np.abs(fam_t - t0)) <= TOL
                if not found_ff or found_chain:
                    continue
                mh = mc & ok & (g == gi)
                Ls = np.unique(labs[mh]); rows = []
                for L in Ls:
                    L = int(L)
                    if L < 0:
                        rows.append(dict(L=-1, E=float(e[mh & (labs < 0)].sum()), templ=False)); continue
                    mL = mc & (labs == L); EL = float(e[mL].sum()); Ehere = float(e[mL & mh].sum())
                    key = (L, c); templ = key in amp
                    row = dict(L=L, E=EL, E_at_flash=Ehere, templ=templ, t_reco=labT.get(L, np.nan), host_E=fam_energy_at(labT.get(L, np.nan)) if np.isfinite(labT.get(L, np.nan)) else np.nan)
                    if templ and np.isfinite(row["t_reco"]):
                        a = amp[key]; sup = support(a); tr = lr.get(L, {}).get("reco_t0", row["t_reco"])
                        Rf = R + render(a, tr, R.shape[1])          # own light added back
                        row["cos_true"], row["ratio_true"] = cos_amp(Rf, a, sup, t0)
                        row["cos_reco"], row["ratio_reco"] = cos_amp(Rf, a, sup, tr)
                        row["dchi2_true"] = chi2_at(Rf, A, a, sup, t0); row["dchi2_reco"] = chi2_at(Rf, A, a, sup, tr)
                        others = np.maximum(peak_pattern(B - render(a, tr, R.shape[1]), tr)[sup], 0).sum(); own = a[sup].sum()
                        row["glare"] = float(own / max(others, 1e-9))
                        row["in_resid_peaks"] = bool(t_res.size and np.min(np.abs(t_res - t0)) <= TOL)
                    rows.append(row)
                anat.append(dict(ev=fi, tpc=c, t=float(t0), E=float(Ecg[gi]), n_fam=int(fam_t.size), rows=rows))
            # ---- (2) table of templated small clusters in this TPC vs residual flashes
            for L in np.unique(labs[mc & (labs >= 0)]):
                L = int(L); key = (L, c)
                if key not in amp or not (0.5 <= labE[L] < 20) or not np.isfinite(labT.get(L, np.nan)):
                    continue
                mL = mc & (labs == L); a = amp[key]; sup = support(a); tr = lr.get(L, {}).get("reco_t0", labT[L])
                Rf = R + render(a, tr, R.shape[1])
                cc, rc = cos_amp(Rf, a, sup, tr)
                okL = mL & ok; EL = float(e[mL].sum()); Eok = float(e[okL].sum())
                Ecorr_cur = float((e[okL] * (np.abs(ts[okL] - tru[okL]) <= CUT)).sum())
                others = np.maximum(peak_pattern(B - render(a, tr, R.shape[1]), tr)[sup], 0).sum(); glare = float(a[sup].sum() / max(others, 1e-9))
                # nearest placed hit of another cluster in the TPC at the current time (host distance)
                host = mc & asg & (labs != L) & (np.abs(ts - tr) <= 3)
                dh = float(np.min(np.linalg.norm(xyz[host][:, None, :] - xyz[mL][None, :, :], axis=2))) if host.any() else 999.0
                base_row = dict(ev=fi, tpc=c, L=L, E=EL, E_ok=Eok, t_cur=float(tr), truth=labTr.get(L, np.nan), Ecorr_cur=Ecorr_cur, cos_cur=cc, ratio_cur=rc, glare=glare, host_E=fam_energy_at(tr), host_d=dh)
                best = None
                for t_r in t_res_free:
                    if abs(t_r - tr) <= TOL:
                        continue
                    cr, rr = cos_amp(Rf, a, sup, t_r)
                    Ecorr_new = float((e[okL] * (np.abs(t_r - tru[okL]) <= CUT)).sum())
                    if best is None or cr > best[0]:
                        best = (cr, rr, float(t_r), Ecorr_new)
                base_row.update(dict(n_res=int(t_res_free.size), best_cos=best[0] if best else np.nan, best_ratio=best[1] if best else np.nan, best_t=best[2] if best else np.nan, best_Ecorr=best[3] if best else np.nan))
                table.append(base_row)
        print(f"event {fi} done: missed isolated flashes so far {len(anat)}, small clusters {len(table)}", flush=True)
    # ---------------- report (1)
    print(f"\n{len(files)} events; MISSED ISOLATED 2-20 MeV FLASHES (S10 sees them, chain has no family within {TOL:.0f} ticks): {len(anat)}")
    E_tot = sum(a["E"] for a in anat); n_cl = [sum(1 for r in a["rows"] if r["L"] >= 0) for a in anat]
    E_tm = sum(r.get("E_at_flash", 0) for a in anat for r in a["rows"] if r.get("templ")); E_untm = sum(r.get("E_at_flash", 0) for a in anat for r in a["rows"] if r["L"] >= 0 and not r.get("templ"))
    E_noise = sum(r["E"] for a in anat for r in a["rows"] if r["L"] < 0)
    print(f"  energy at them {E_tot/1e3:.2f} GeV: in templated clusters {E_tm/1e3:.2f}, untemplated clusters {E_untm/1e3:.2f}, noise hits {E_noise/1e3:.2f}; clusters per flash median {np.median(n_cl):.0f}; families already in the TPC median {np.median([a['n_fam'] for a in anat]):.0f}")
    R_ = [r for a in anat for r in a["rows"] if r.get("templ") and "cos_true" in r]
    if R_:
        ct = np.array([r["cos_true"] for r in R_]); cr = np.array([r["cos_reco"] for r in R_]); dt = np.array([r["dchi2_true"] for r in R_]); dr = np.array([r["dchi2_reco"] for r in R_])
        hE = np.array([r["host_E"] for r in R_]); gl = np.array([r["glare"] for r in R_]); rt = np.array([r["ratio_true"] for r in R_]); E_ = np.array([r["E"] for r in R_]); inr = np.array([r["in_resid_peaks"] for r in R_])
        print(f"  templated clusters of these flashes: n {len(R_)}, E p10/50/90 {np.percentile(E_,[10,50,90]).round(1)} MeV")
        print(f"    host family at the chosen time: E p10/50/90 {np.percentile(hE,[10,50,90]).round(0)} MeV; own light / others at the chosen slot p10/50/90 {np.percentile(gl,[10,50,90]).round(3)}")
        print(f"    loss (own light added back, support channels): chi2 prefers the TRUE time in {np.mean(dt < dr):.2f} of cases; median dchi2(true) {np.median(dt):.0f}, dchi2(reco) {np.median(dr):.0f}")
        print(f"    pattern cosine on support channels: at TRUE time p10/50/90 {np.percentile(ct,[10,50,90]).round(2)}; at CHOSEN time {np.percentile(cr,[10,50,90]).round(2)}; cos_true > cos_reco + 0.1 in {np.mean(ct > cr + 0.1):.2f}; cos_true >= 0.9 in {np.mean(ct >= 0.9):.2f}")
        print(f"    amplitude ratio template/residual at the true time p10/50/90 {np.percentile(rt,[10,50,90]).round(2)}; true time is a residual-light peak in {inr.mean():.2f}")
    np.save(os.path.join(OUT, "isolated_anatomy.npy"), np.array(anat, dtype=object), allow_pickle=True)
    import pandas as pd
    pd.DataFrame(table).to_csv(os.path.join(OUT, "small_cluster_table.csv"), index=False)
    tb = pd.DataFrame(table)
    print(f"\nTABLE: templated 0.5-20 MeV clusters {len(tb)}, E {tb.E.sum()/1e3:.1f} GeV, correct energy now {tb.Ecorr_cur.sum()/tb.E_ok.sum():.3f}; clusters with >= 1 residual flash in their TPC {np.mean(tb.n_res > 0):.2f}")
    wrong = tb.Ecorr_cur < 0.5 * tb.E_ok
    print(f"  wrong clusters {wrong.sum()} ({tb.E_ok[wrong].sum()/1e3:.1f} GeV); of them best residual-flash cos >= 0.9 and > cos_cur + 0.1: {np.mean((tb.best_cos >= 0.9) & (tb.best_cos > tb.cos_cur + 0.1) & wrong)/max(wrong.mean(),1e-9):.2f}, and that move would be right (>= 50 % E): {(( tb.best_Ecorr >= 0.5*tb.E_ok) & wrong).sum()/max(wrong.sum(),1):.2f}")
    print(f"  right clusters: best residual-flash cos >= 0.9 and > cos_cur + 0.1 (would be broken): {np.mean((tb.best_cos >= 0.9) & (tb.best_cos > tb.cos_cur + 0.1) & ~wrong)/max((~wrong).mean(),1e-9):.3f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
