"""Does the chain discover more flashes (t0s) than a conventional flash finder, and are the extra ones real?
Per light TPC and event, three sets of times (ticks, 16 ns):
  TRUTH flashes  : truth interaction-time groups (1-tick linking of truth t0 over the event) with >= 0.5 MeV deposited
                   in the TPC; time = E-weighted mean truth t0 of the group.  "Module" truth = the TPC or its optical
                   partner (c ^ 1: the two drift volumes of a module share the light).
  FLASH FINDER   : peak finding on the ACTUAL light alone -
                   S(k): summed waveform of the non-vetoed channels, scipy find_peaks, prominence k x pre-trigger sigma;
                   C(2PE,n): per-channel hits (prominence >= 600 ADC = 2 PE), >= n distinct channels within +-1 tick
                   (channel-count histogram, local maxima, 4-tick separation).
  CHAIN t0s      : the family times of the final assignment (rounded final times of assigned clustered hits in the
                   TPC, linked within 1.5 ticks) = what the residual-based scans + assignment discovered, per TPC.
Matching tolerance TOL ticks.  ACCIDENTAL matches (a finder with many peaks matches a random time) are measured with
truth shifted by 300 ticks (mod 886) and removed: eff_corr = (eff - acc) / (1 - acc).
Reports: candidate purity (TPC / module / event level); corrected discovery efficiency vs energy-in-TPC and vs how
buried the flash is (gap to the nearest >= 3x brighter truth flash in the module); chain t0s with no finder peak
("new") and whether they are real and how correct the energy placed there is.   python v13_flash_discovery.py <arm> [n]"""
import glob
import os
import sys

import numpy as np
from scipy.signal import find_peaks

CUT = 10.0; PEAK = 105; TOL = 5.0; E_MIN = 0.5; SHIFT = 300.0; WIN = 886.0
E_BINS = [0.5, 2, 5, 20, 100, 500, 1e9]; E_LAB = ["0.5-2", "2-5", "5-20", "20-100", "100-500", ">=500"]
B_BINS = [0, 5, 20, 60, 1e9]; B_LAB = ["bigger flash within 5 ticks", "5-20 ticks", "20-60 ticks", "none within 60 (isolated)"]
FF_S = [10, 20, 40, 60]; FF_C = [(600, 3), (600, 6), (600, 12)]   # C: per-channel prominence in ADC (300 ADC = 1 PE)
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "flash_discovery")


def link(times, gap):
    o = np.argsort(times); t = times[o]
    cuts = np.flatnonzero(np.diff(t) > gap) + 1
    g = np.zeros(t.size, int); g[o] = np.repeat(np.arange(cuts.size + 1), np.diff(np.r_[0, cuts, t.size]))
    return g


def ff_summed(act, chans, k):
    s = act[chans].sum(0); sig = max(float(s[:90].std()), 1.0)
    p, _ = find_peaks(s, prominence=k * sig, distance=4)
    t = p - PEAK; return t[(t >= 0) & (t <= 885)].astype(float)


def chan_peaks(act, chans, adc):
    out = []
    for c in chans:
        p, _ = find_peaks(act[c], prominence=adc, distance=4)
        out.append(p)
    return out


def ff_coinc(peaks, n):
    h = np.zeros(1000)
    for p in peaks:
        if p.size:
            u = np.unique(p); h[u] += 1; h[np.clip(u - 1, 0, 999)] += 1; h[np.clip(u + 1, 0, 999)] += 1
    # a channel contributes to 3 adjacent bins, so a coincidence of n channels at the same tick gives n; local maxima
    p, _ = find_peaks(h, height=n - 0.5, distance=4)
    t = p - PEAK; return t[(t >= 0) & (t <= 885)].astype(float)


def match(a, b, tol):
    if a.size == 0:
        return np.zeros(0, int)
    if b.size == 0:
        return np.full(a.size, -1)
    d = np.abs(a[:, None] - b[None, :]); j = np.argmin(d, 1)
    return np.where(d[np.arange(a.size), j] <= tol, j, -1)


def main(arm, n_max=None):
    files = sorted(glob.glob(f"{arm}/full_dumps/*.npz"))[: int(n_max) if n_max else None]
    FF = [("S%d" % k, ("S", k)) for k in FF_S] + [("C2PE/%d" % kn[1], ("C",) + kn) for kn in FF_C]
    names = [f[0] for f in FF] + ["chain"]; nF = len(names)
    T_n = np.zeros((len(E_LAB), len(B_LAB))); T_f = np.zeros((len(E_LAB), len(B_LAB), nF)); T_a = np.zeros((len(E_LAB), len(B_LAB), nF))
    C_tot = np.zeros((nF, 4))            # n, matched TPC, matched module, matched event
    C_acc = np.zeros((nF, 4))            # same against shifted truth
    N = np.zeros((len(FF), 7)); NE_BINS = [0, 2, 5, 20, 100, 1e9]; NE = np.zeros((len(FF), len(NE_BINS) - 1, 4))
    n_tpc = 0; per_tpc = np.zeros(nF + 1)
    for ff in files:
        z = np.load(ff, allow_pickle=True)
        e = np.clip(z["e"], 0, None).astype(float); ts = z["ts_final"].astype(float); tru = z["truth_t0"].astype(float)
        labs = z["labels"].astype(int); tpc = z["tpc"].astype(int); full = z["full_light"]; veto = z["veto_mask"]
        ok = np.isfinite(tru) & (tru >= 0) & (tru <= 885)
        g = np.full(tru.size, -1); g[ok] = link(np.round(tru[ok]), 1.5); n_g = g.max() + 1
        gE = np.bincount(g[ok], weights=e[ok], minlength=n_g); gT = np.bincount(g[ok], weights=e[ok] * tru[ok], minlength=n_g) / np.maximum(gE, 1e-9)
        t_event = gT[gE > 0]
        asg = (labs >= 0) & np.isfinite(ts) & (ts >= 0)
        Ecg_all = {}
        for c in np.unique(tpc):
            mt = (tpc == c) & ok; Ecg_all[int(c)] = np.bincount(g[mt], weights=e[mt], minlength=n_g)
        for c in np.unique(tpc):
            c = int(c)
            if c >= full.shape[0]:
                continue
            mc = tpc == c
            Ecg = Ecg_all[c]; gid = np.flatnonzero(Ecg >= E_MIN); t_true = gT[gid]; E_true = Ecg[gid]
            Emod = Ecg + Ecg_all.get(c ^ 1, np.zeros(n_g)); gm = np.flatnonzero(Emod >= E_MIN); t_mod = gT[gm]; E_mod = Emod[gm]
            t_shift = (t_true + SHIFT) % WIN; t_mod_shift = (t_mod + SHIFT) % WIN; t_ev_shift = (t_event + SHIFT) % WIN
            chans = np.flatnonzero(~veto[c]); act = full[c].astype(np.float32)
            cand = []; pk = chan_peaks(act, chans, FF_C[0][0]) if chans.size else []
            for nm, spec in FF:
                if chans.size == 0:
                    cand.append(np.zeros(0)); continue
                cand.append(ff_summed(act, chans, spec[1]) if spec[0] == "S" else ff_coinc(pk, spec[2]))
            ma = mc & asg
            if ma.any():
                tt = ts[ma]; gg = link(np.round(tt), 1.5); n_c = gg.max() + 1
                cE = np.bincount(gg, weights=e[ma], minlength=n_c); cT = np.bincount(gg, weights=e[ma] * tt, minlength=n_c) / np.maximum(cE, 1e-9)
                good = np.abs(tt - tru[ma]) <= CUT; okh = ok[ma]
                cEc = np.bincount(gg, weights=e[ma] * (good & okh), minlength=n_c)
            else:
                cT = np.zeros(0); cE = np.zeros(0); cEc = np.zeros(0)
            cand.append(cT)
            n_tpc += 1; per_tpc += [x.size for x in cand] + [t_true.size]
            if t_true.size:
                gap = np.full(t_true.size, 1e9)
                for i in range(t_true.size):
                    big = E_mod >= 3 * E_true[i]
                    big &= np.abs(t_mod - t_true[i]) > 0.01
                    if big.any():
                        gap[i] = np.min(np.abs(t_mod[big] - t_true[i]))
                eb = np.clip(np.searchsorted(E_BINS, E_true, "right") - 1, 0, len(E_LAB) - 1)
                bb = np.clip(np.searchsorted(B_BINS, gap, "right") - 1, 0, len(B_LAB) - 1)
                np.add.at(T_n, (eb, bb), 1)
                for fi, x in enumerate(cand):
                    f = match(t_true, x, TOL) >= 0; a = match(t_shift, x, TOL) >= 0
                    np.add.at(T_f, (eb[f], bb[f], fi), 1); np.add.at(T_a, (eb[a], bb[a], fi), 1)
            for fi, x in enumerate(cand):
                if x.size == 0:
                    continue
                C_tot[fi] += [x.size, (match(x, t_true, TOL) >= 0).sum(), (match(x, t_mod, TOL) >= 0).sum(), (match(x, t_event, TOL) >= 0).sum()]
                C_acc[fi] += [x.size, (match(x, t_shift, TOL) >= 0).sum(), (match(x, t_mod_shift, TOL) >= 0).sum(), (match(x, t_ev_shift, TOL) >= 0).sum()]
            if cT.size:
                m_tpc = match(cT, t_true, TOL) >= 0; m_mod = match(cT, t_mod, TOL) >= 0
                neb = np.clip(np.searchsorted(NE_BINS, cE, "right") - 1, 0, len(NE_BINS) - 2)
                for fi in range(len(FF)):
                    new = match(cT, cand[fi], TOL) < 0
                    N[fi] += [new.sum(), (new & m_tpc).sum(), (new & m_mod).sum(), cE[new].sum(), cEc[new].sum(), 0, 0]
                    np.add.at(NE[fi], (neb[new],), np.stack([np.ones(new.sum()), m_tpc[new], cE[new], cEc[new]], 1))
                    if cand[fi].size:
                        only = match(cand[fi], cT, TOL) < 0; jt = match(cand[fi][only], t_true, TOL)
                        N[fi, 5] += only.sum(); N[fi, 6] += E_true[jt[jt >= 0]].sum()
            else:
                for fi in range(len(FF)):
                    if cand[fi].size:
                        jt = match(cand[fi], t_true, TOL); N[fi, 5] += cand[fi].size; N[fi, 6] += E_true[jt[jt >= 0]].sum()
    n_ev = len(files)
    print(f"{n_ev} events, {n_tpc} TPC-events; per TPC-event: truth flashes (>= {E_MIN} MeV in TPC) {per_tpc[-1]/n_tpc:.1f}; candidates: " + ", ".join(f"{nm} {per_tpc[i]/n_tpc:.1f}" for i, nm in enumerate(names)))
    print(f"matching tolerance {TOL:.0f} ticks; accidental-match control = truth shifted by {SHIFT:.0f} ticks; summed-waveform sigma ~133 ADC so S10 ~ 4 PE, S20 ~ 9 PE, S40 ~ 18 PE, S60 ~ 27 PE summed over the TPC\n")
    print("CANDIDATE PURITY: fraction of a finder's t0s with a truth flash within TOL (raw [accidental]) and fakes per TPC-event at module level:")
    print("  finder     n/TPC-ev   TPC level        module level     event level      fakes/TPC-ev (module, accidental-corrected)")
    for i, nm in enumerate(names):
        n = max(C_tot[i, 0], 1); pt, pm, pe = C_tot[i, 1:] / n; at, am, ae = C_acc[i, 1:] / n
        pm_c = (pm - am) / max(1 - am, 1e-9)
        print(f"  {nm:8s} {C_tot[i,0]/n_tpc:8.2f}   {pt:.3f} [{at:.3f}]   {pm:.3f} [{am:.3f}]   {pe:.3f} [{ae:.3f}]   {C_tot[i,0]/n_tpc*(1-pm_c):.2f}")
    print("\nTRUTH-FLASH DISCOVERY EFFICIENCY vs energy deposited in the TPC, accidental-corrected (raw in brackets):")
    print("  E in TPC [MeV]   n     " + "".join(f"{nm:>14s}" for nm in names))
    for a in range(len(E_LAB)):
        n = T_n[a].sum()
        if n:
            row = ""
            for i in range(nF):
                f = T_f[a, :, i].sum() / n; ac = T_a[a, :, i].sum() / n
                row += f"{(f-ac)/max(1-ac,1e-9):7.3f}[{f:.2f}]  "
            print(f"  {E_LAB[a]:12s} {int(n):6d}  " + row)
    print("\nBURIED FLASHES: corrected efficiency vs gap to the nearest >= 3x brighter truth flash in the same MODULE:")
    for lo, hi, lab in ((1, 3, "2-20 MeV"), (3, 4, "20-100 MeV"), (0, 1, "0.5-2 MeV")):
        print(f"  {lab}:")
        for b in range(len(B_LAB)):
            n = T_n[lo:hi, b].sum()
            if n:
                row = ""
                for i, nm in enumerate(names):
                    f = T_f[lo:hi, b, i].sum() / n; ac = T_a[lo:hi, b, i].sum() / n
                    row += f"{nm} {(f-ac)/max(1-ac,1e-9):.3f}  "
                print(f"    {B_LAB[b]:30s} n {int(n):6d}  " + row)
    print("\nCHAIN t0s NOT FOUND BY THE FLASH FINDER ('new' t0s), per finder:")
    print("  finder   n_new  /TPC-ev  real(TPC)  real(module)  E placed [GeV]  E correct   | finder-only peaks /TPC-ev, truth E at them [GeV]")
    for fi, (nm, _) in enumerate(FF):
        r = N[fi]
        print(f"  {nm:8s} {int(r[0]):6d} {r[0]/n_tpc:7.2f}   {r[1]/max(r[0],1):.3f}      {r[2]/max(r[0],1):.3f}       {r[3]/1e3:7.2f}      {r[4]/max(r[3],1e-9):.3f}      | {r[5]/n_tpc:6.2f}  {r[6]/1e3:.2f}")
    print("\n  new t0s by the energy the chain placed at them:")
    for fi in range(len(FF)):
        print(f"   {FF[fi][0]}: " + "  ".join(f"{NE_BINS[b]:.0f}-{min(NE_BINS[b+1],1e5):.0f} MeV: n {int(NE[fi,b,0])} real {NE[fi,b,1]/max(NE[fi,b,0],1):.2f} Ecorr {NE[fi,b,3]/max(NE[fi,b,2],1e-9):.2f}" for b in range(len(NE_BINS) - 1)))
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, "flash_discovery_data.npz"), T_n=T_n, T_f=T_f, T_a=T_a, C_tot=C_tot, C_acc=C_acc, N=N, NE=NE, names=np.array(names), E_LAB=np.array(E_LAB), B_LAB=np.array(B_LAB), n_tpc=n_tpc, n_ev=n_ev)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
