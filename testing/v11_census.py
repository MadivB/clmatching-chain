"""v1.1 residual-honesty census on an arm's full_dumps:  python v11_census.py <arm_dir>

(a) correctly placed families (>=95% right, >=20 MeV): actual/predicted at the
    peak tick on own-dominated support channels -> under-prediction bias/spread
(b) piggyback: wrong clusters >=5 MeV sitting at a correctly placed bigger
    cluster's time: the family's actual/pred WITHOUT the wrong cluster
(c) eye-view: significant unexplained residual peaks (>=1 ch z>3): how many per
    TPC-event, how many at existing families, wrong E under them
(d) wrong E by hit category (noise / no-template / imaged bands)
"""
import glob, sys
import numpy as np

PEAK, CLIP, CUT, GAIN = 105, 60780.0, 10.0, 200.0


def main(d):
    fam_rows, pig_rows, peak_rows = [], [], []
    cats = ["noise", "<1.5 no-tmpl", "img 1.5-5", "img 5-20", "img 20-100", "img>=100"]
    T = np.zeros((len(cats), 2)); n_tpc = 0
    files = sorted(glob.glob(f"{d}/full_dumps/*.npz"))
    for ff in files:
        z = np.load(ff, allow_pickle=True)
        e = np.clip(z["e"], 0, None).astype(np.float64)
        ts = z["ts_final"].astype(np.float64); tru = z["truth_t0"].astype(np.float64)
        tpc = z["tpc"].astype(int); labs = z["labels"].astype(int)
        ok_t = np.isfinite(tru) & (tru >= 0); assigned = np.isfinite(ts) & (ts >= 0)
        wrong = ok_t & (~assigned | (np.abs(ts - tru) > CUT))
        full = z["full_light"].astype(np.float32); base = np.minimum(z["base_image"].astype(np.float32), CLIP)
        veto = z["veto_mask"]
        al, at, ar = z["amp_lab"].astype(int), z["amp_tpc"].astype(int), z["amp_row"].astype(np.float64)
        amp = {(int(al[i]), int(at[i])): ar[i] for i in range(len(al))}
        imaged = set(int(l) for l in al)
        labset = np.unique(labs[labs >= 0])
        lab_E = {int(L): e[labs == L].sum() for L in labset}
        lab_t = {int(L): np.median(ts[labs == L]) for L in labset}
        lab_wr = {int(L): e[(labs == L) & wrong].sum() / max(lab_E[int(L)], 1e-9) for L in labset}
        cat = np.zeros(labs.size, int)
        for L in labset:
            m = labs == L; E = lab_E[int(L)]
            cat[m] = 1 if int(L) not in imaged else 2 if E < 5 else 3 if E < 20 else 4 if E < 100 else 5
        for k in range(len(cats)):
            mk = (cat == k) & ok_t
            T[k, 0] += e[mk].sum(); T[k, 1] += e[mk & wrong].sum()
        for c in range(full.shape[0]):
            m = tpc == c
            if not m.any():
                continue
            n_tpc += 1
            ma = m & assigned
            fam_t = np.unique(np.round(ts[ma]))
            # (a) family ratios
            for t0 in fam_t:
                mf = ma & (np.abs(ts - t0) <= 2)
                Ef = e[mf].sum()
                if Ef < 20:
                    continue
                members = [int(L) for L in np.unique(labs[mf & (labs >= 0)]) if (int(L), c) in amp]
                if not members:
                    continue
                col = int(round(t0)) + PEAK
                if not (0 <= col < full.shape[2]):
                    continue
                a = full[c][:, col].astype(np.float64); p = base[c][:, col].astype(np.float64)
                pat = np.zeros(120)
                for L in members:
                    pat += amp[(L, c)]
                pat[veto[c]] = 0
                if pat.max() <= 0:
                    continue
                own = (pat > 0.1 * pat.max()) & (pat >= 0.5 * np.maximum(p, 1e-9))
                if own.sum() < 2:
                    continue
                if e[mf & wrong].sum() / Ef < 0.05:
                    fam_rows.append((Ef, a[own].sum() / max(p[own].sum(), 1e-9)))
            # (b) piggyback
            for L in labset:
                L = int(L)
                if lab_wr[L] < 0.5 or lab_E[L] < 5 or (L, c) not in amp or not np.isfinite(lab_t[L]) or lab_t[L] < 0:
                    continue
                tw = lab_t[L]; col = int(round(tw)) + PEAK
                if not (0 <= col < full.shape[2]):
                    continue
                others = [int(K) for K in np.unique(labs[m & (np.abs(ts - tw) <= CUT) & (labs >= 0)])
                          if int(K) != L and lab_wr[int(K)] < 0.5 and (int(K), c) in amp and lab_E[int(K)] >= lab_E[L]]
                if not others:
                    continue
                a = full[c][:, col].astype(np.float64); p = base[c][:, col].astype(np.float64)
                mine = amp[(L, c)].copy(); mine[veto[c]] = 0
                if mine.max() <= 0:
                    continue
                sup = mine > 0.1 * mine.max()
                if sup.sum() < 2:
                    continue
                p_wo = np.clip(p - mine, 0, None)
                pig_rows.append((lab_E[L], a[sup].sum() / max(p_wo[sup].sum(), 1e-9)))
            # (c) residual peaks
            R = full[c] - base[c]; Z = R / np.sqrt(GAIN * np.maximum(full[c], 100.0)); Z[veto[c]] = 0
            cnt = (Z > 3).sum(axis=0); trace = np.where(cnt >= 1, np.clip(R, 0, None).sum(axis=0), 0.0)
            peaks = []
            for t in range(PEAK, full.shape[2] - 2):
                if trace[t] > 0 and trace[t] >= trace[t - 1] and trace[t] > trace[t + 1] and trace[t] >= trace[t - 2]:
                    if peaks and t - peaks[-1][0] < 12:
                        if trace[t] > peaks[-1][1]:
                            peaks[-1] = (t, trace[t])
                    else:
                        peaks.append((t, trace[t]))
            for (t, a_) in peaks:
                t0 = t - PEAK
                near = m & ok_t & (np.abs(tru - t0) <= CUT)
                has_fam = fam_t.size > 0 and np.min(np.abs(fam_t - t0)) <= CUT
                peak_rows.append((a_, e[near].sum(), e[near & wrong].sum(), has_fam))
    fr = np.array(fam_rows); pr = np.array(pig_rows); pk = np.array(peak_rows, float)
    print(f"{d}: {len(files)} events, {n_tpc} TPC-events")
    print("(a) correct families actual/pred at peak (own-dominated channels):")
    for lo, hi in [(20, 100), (100, 300), (300, 1000), (1000, 1e9)]:
        s = (fr[:, 0] >= lo) & (fr[:, 0] < hi)
        if s.any():
            r = fr[s, 1]
            print(f"    E [{lo},{hi}): n={s.sum():5d} median {np.median(r):.3f} p16-p84 {np.percentile(r,16):.2f}-{np.percentile(r,84):.2f} frac>1.2 {np.mean(r>1.2):.2f}")
    print(f"(b) piggyback wrong clusters >=5 MeV at a bigger right cluster's time: n={len(pr)}, E {pr[:,0].sum()/1e3:.2f} GeV; family ratio without them median {np.median(pr[:,1]):.2f}, frac>1.1 {np.mean(pr[:,1]>1.1):.2f}")
    wt = pk[:, 1] > 0.5
    print(f"(c) residual peaks (>=1 ch z>3): {len(pk)/n_tpc:.2f}/TPC-event; with truth {wt.mean():.2f}; at existing family {pk[wt,3].mean():.2f}; wrong E under family peaks {pk[wt & (pk[:,3]==1), 2].sum()/1e3:.2f} GeV; sum +resid median {np.median(pk[:,0]):.0f}")
    print("(d) wrong E by category (GeV): " + " | ".join(f"{c} {T[k,1]/1e3:.2f}/{T[k,0]/1e3:.0f}" for k, c in enumerate(cats)) + f" | TOTAL {T[:,1].sum()/1e3:.2f} = {100*T[:,1].sum()/T[:,0].sum():.2f}%")


if __name__ == "__main__":
    main(sys.argv[1])
