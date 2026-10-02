"""What truth energy sits at the RESIDUAL flashes (unexplained light) and in what form, and how pure the residual-flash
finder is vs its gates.  Per light TPC: residual flashes = S10 peaks of (actual - base) not within 5 ticks of a family
time.  For each: summed residual amplitude, residual / model ratio at the peak, truth energy in the TPC within 5 ticks
of the flash (0 = fake), split by the form of that energy: noise hits; untemplated clusters; templated clusters
< 20 MeV whose majority truth is this flash (movable whole); clusters 20-100 / >= 100 MeV whole; SUB-pieces of
clusters whose majority truth is elsewhere (delayed daughters / mixed clusters); and whether that energy is currently
correct.   python v14_resid_flash_census.py <arm_dir> [n_events]"""
import glob
import sys

import numpy as np
from scipy.signal import find_peaks

PEAK = 105; TOL = 5.0; CUT = 10.0
FORMS = ["noise hits", "untemplated clusters (< 1.5 MeV)", "templated < 20 MeV, whole", "20-100 MeV, whole", ">= 100 MeV, whole", "sub-piece of a cluster timed elsewhere"]


def link(times, gap):
    o = np.argsort(times); t = times[o]; cuts = np.flatnonzero(np.diff(t) > gap) + 1
    g = np.zeros(t.size, int); g[o] = np.repeat(np.arange(cuts.size + 1), np.diff(np.r_[0, cuts, t.size])); return g


def main(arm, n_max=None):
    files = sorted(glob.glob(f"{arm}/full_dumps/*.npz"))[: int(n_max) if n_max else None]
    F = np.zeros((len(FORMS), 2))          # energy at residual flashes by form: total, currently correct
    fl = []                                # per residual flash: amp, ratio_to_model, truth E in TPC, truth E in module
    n_tpc = 0
    for ff in files:
        z = np.load(ff, allow_pickle=True)
        e = np.clip(z["e"], 0, None).astype(float); ts = z["ts_final"].astype(float); tru = z["truth_t0"].astype(float)
        labs = z["labels"].astype(int); tpc = z["tpc"].astype(int); full = z["full_light"]; base = z["base_image"]; veto = z["veto_mask"]
        templ = np.isin(labs, np.unique(z["amp_lab"].astype(int)))
        ok = np.isfinite(tru) & (tru >= 0) & (tru <= 885); asg = np.isfinite(ts) & (ts >= 0)
        good = ok & asg & (np.abs(ts - tru) <= CUT)
        labE = np.bincount(labs[labs >= 0], weights=e[labs >= 0], minlength=int(labs.max()) + 1) if (labs >= 0).any() else np.zeros(1)
        # majority truth time per label
        labTr = {}
        for L in np.unique(labs[(labs >= 0) & ok]):
            m = (labs == L) & ok; labTr[int(L)] = float(np.median(tru[m]))
        for c in np.unique(tpc):
            if c >= full.shape[0]:
                continue
            chans = np.flatnonzero(~veto[c])
            if chans.size == 0:
                continue
            A = full[c].astype(np.float64); B = base[c].astype(np.float64); rs = (A - B)[chans].sum(0); bs = B[chans].sum(0)
            sig = max(float(A[chans, :90].sum(0).std()), 1.0)
            p, _ = find_peaks(rs, prominence=10 * sig, distance=4); t = (p - PEAK).astype(float); k = (t >= 0) & (t <= 885); p, t = p[k], t[k]
            mc = tpc == c; fam = np.unique(np.round(ts[mc & asg & (labs >= 0)]))
            n_tpc += 1
            for pi, tf in zip(p, t):
                if fam.size and np.min(np.abs(fam - tf)) <= TOL:
                    continue
                mh = mc & ok & (np.abs(tru - tf) <= TOL)
                mh_mod = (tpc == (c ^ 1)) & ok & (np.abs(tru - tf) <= TOL)
                fl.append((float(rs[pi]), float(rs[pi] / max(bs[pi], 1.0)), float(e[mh].sum()), float(e[mh].sum() + e[mh_mod].sum())))
                if not mh.any():
                    continue
                for L in np.unique(labs[mh]):
                    m = mh & (labs == L)
                    if L < 0:
                        f = 0
                    elif not templ[m].any():
                        f = 1
                    else:
                        whole = abs(labTr.get(int(L), 1e9) - tf) <= TOL
                        EL = labE[L]
                        f = 5 if not whole else (2 if EL < 20 else 3 if EL < 100 else 4)
                    F[f] += [e[m].sum(), e[m & good].sum()]
    fl = np.array(fl); n_ev = len(files)
    print(f"{n_ev} events, {n_tpc} TPC-events; residual flashes (S10 on actual - base, not within 5 ticks of a family): {len(fl)/n_ev:.0f} per event")
    real = fl[:, 2] > 0.5; real_mod = fl[:, 3] > 0.5
    print(f"  purity: truth >= 0.5 MeV in the TPC within 5 ticks {real.mean():.3f}; in the module {real_mod.mean():.3f}")
    for lo, hi in ((0, 1000), (1000, 2000), (2000, 5000), (5000, 20000), (20000, 1e9)):
        m = (fl[:, 0] >= lo) & (fl[:, 0] < hi)
        if m.any():
            print(f"    summed residual amplitude {lo:6.0f}-{min(hi,99999):6.0f} ADC ({lo/300:4.1f}-{min(hi,99999)/300:5.1f} PE): n/ev {m.sum()/n_ev:6.1f}  pure(TPC) {real[m].mean():.3f}  pure(module) {real_mod[m].mean():.3f}  truth E/ev at them {fl[m,2].sum()/n_ev:.0f} MeV")
    for lo, hi in ((0, 0.2), (0.2, 0.5), (0.5, 1), (1, 3), (3, 1e9)):
        m = (fl[:, 1] >= lo) & (fl[:, 1] < hi)
        if m.any():
            print(f"    residual / model at the peak {lo:4.1f}-{min(hi,99):4.1f}: n/ev {m.sum()/n_ev:6.1f}  pure(TPC) {real[m].mean():.3f}  pure(module) {real_mod[m].mean():.3f}  truth E/ev {fl[m,2].sum()/n_ev:.0f} MeV")
    print(f"\n  truth energy at residual flashes (TPC-level), per event: {F[:,0].sum()/n_ev:.0f} MeV, of which currently correct {F[:,1].sum()/max(F[:,0].sum(),1e-9):.2f}")
    for i, nm in enumerate(FORMS):
        print(f"    {nm:42s} {F[i,0]/n_ev:7.1f} MeV/ev ({100*F[i,0]/F[:,0].sum():4.1f} %)  currently correct {F[i,1]/max(F[i,0],1e-9):.2f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
