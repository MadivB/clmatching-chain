"""v1.2 phase 0b: per-CHANNEL leading-edge timing of clean families on the
v1.1 outputs.  For each clean family (>= 20 MeV, >= 90 % correct) and each of
its support channels with negligible pile-up, measure the 20 %-of-peak
crossing of the ACTUAL waveform and compare with (a) the family's fitted t0
(chain reference) and (b) the truth apex time + geometric onset delay.
Reports: per-channel residual spread, its within-family consistency, and its
dependence on channel amplitude (slew) and on pile-up.
python v12_channel_edges.py <dump_dir> <flow_file> [max_events]"""
import glob, os, sys
import numpy as np
HERE = os.path.abspath(os.path.dirname(__file__))
BASE = os.path.join(HERE, "..", "CLMatching_AlphaRelease")
for p in (BASE, os.path.join(BASE, "M5p1"), os.path.join(BASE, "NewMLSection"), HERE):
    if p not in sys.path:
        sys.path.insert(0, os.path.abspath(p))
from subtick_refine import build_sipm_positions

C, N_LAR, NS, PEAK, CUT = 29.9792458, 2.237, 16.0, 105, 10.0
TMPL = np.load(os.path.join(BASE, "assets", "avg_pulse.npy")).astype(np.float64)
TMPL = TMPL / TMPL.max()
FRAC = 0.2


def cf_cross(w, ipk, frac=FRAC):
    """fractional sample of the frac*peak crossing on the rising edge before ipk"""
    thr = frac * w[ipk]
    i = ipk
    while i > 0 and w[i] > thr:
        i -= 1
    if i == ipk or w[i + 1] == w[i]:
        return np.nan
    return i + (thr - w[i]) / (w[i + 1] - w[i])


T_CF_TMPL = cf_cross(TMPL, int(np.argmax(TMPL)))


def wstd(v, w=None):
    w = np.ones_like(v) if w is None else w
    m = np.average(v, weights=w)
    return float(np.sqrt(np.average((v - m) ** 2, weights=w)))


def robust(v):
    return 1.4826 * float(np.median(np.abs(v - np.median(v))))


def main(dump_dir, flow, max_events=39):
    sipm = build_sipm_positions(flow)
    rec = []       # per (family, channel): fam_id, A_peak, pile, d_chain(ns), d_truth(ns), onset(ns), fam_E, n_sup
    fam_id = 0
    files = sorted(glob.glob(f"{dump_dir}/full_dumps/*.npz"))[:max_events]
    for ff in files:
        z = np.load(ff, allow_pickle=True)
        e = np.clip(z["e"], 0, None).astype(np.float64)
        ts = z["ts_final"].astype(np.float64); tru = z["truth_t0"].astype(np.float64)
        labs = z["labels"].astype(int); tpc = z["tpc"].astype(int)
        xyz = np.stack([z["x"], z["y"], z["z"]], 1).astype(np.float64)
        ok = np.isfinite(ts) & (ts >= 0) & np.isfinite(tru) & (tru >= 0)
        full = z["full_light"].astype(np.float64); base = z["base_image"].astype(np.float64)
        veto = z["veto_mask"]
        al, at, ar = z["amp_lab"].astype(int), z["amp_tpc"].astype(int), z["amp_row"].astype(np.float64)
        amp = {(int(al[i]), int(at[i])): ar[i] for i in range(len(al))}
        for c in np.unique(tpc[ok]):
            if c not in sipm:
                continue
            mc = ok & (tpc == c)
            for tv in np.unique(ts[mc]):
                m = mc & (ts == tv)
                E = e[m].sum()
                if E < 20 or m.sum() < 20:
                    continue
                r = (ts[m] - tru[m]) * NS
                good = np.abs(r) <= CUT * NS
                if e[m][good].sum() < 0.9 * E:
                    continue
                idx = np.flatnonzero(m)[good]; w = e[idx]
                tt = tru[idx] * NS
                o = np.argsort(tt); n2 = max(3, int(0.02 * o.size))
                apex = np.average(xyz[idx][o[:n2]], axis=0, weights=w[o[:n2]])
                t_apex = float(np.average(tt[o[:n2]], weights=w[o[:n2]]))
                pat = np.zeros(120)
                for L in np.unique(labs[idx]):
                    if (int(L), int(c)) in amp:
                        pat += amp[(int(L), int(c))]
                pat[veto[c]] = 0
                if pat.max() <= 0:
                    continue
                order = np.argsort(pat)[::-1]; cs = np.cumsum(pat[order])
                sup = order[: int(np.searchsorted(cs, 0.9 * cs[-1])) + 1]
                hs = idx if idx.size <= 3000 else idx[np.linspace(0, idx.size - 1, 3000).astype(int)]
                dprop = np.linalg.norm(xyz[hs] - apex, axis=1) / C
                col = int(round(tv)) + PEAK
                if not (60 <= col < 985):
                    continue
                fam_id += 1
                for ch in sup:
                    P = sipm[c][ch]
                    onset = float((dprop + np.linalg.norm(xyz[hs] - P, axis=1) * N_LAR / C).min())
                    wf = full[c][ch]
                    bl = np.median(wf[col - 60: col - 30])
                    seg = wf[col - 15: col + 16] - bl
                    ipk = int(np.argmax(seg))
                    A = float(seg[ipk])
                    if A < 300:
                        continue
                    other = float(base[c][ch][col] - pat[ch])
                    pile = max(other, 0.0) / max(A, 1e-9)
                    x = cf_cross(seg, ipk)
                    if not np.isfinite(x):
                        continue
                    t_cf = (col - 15) + x                       # sample of the 20 % crossing
                    d_chain = (t_cf - (tv + T_CF_TMPL)) * NS    # ns, actual edge relative to the chain's template edge
                    d_truth = t_cf * NS - (t_apex + onset)      # ns, absolute edge minus (truth apex + geometric onset)
                    rec.append((fam_id, A, pile, d_chain, d_truth, onset, E, sup.size))
    R = np.array(rec)
    print(f"{fam_id} clean families, {len(R)} (family, channel) edges with peak >= 300 ADC; template 20% edge at sample {T_CF_TMPL:.2f}")
    clean = R[:, 2] < 0.1
    for name, s in (("all", np.ones(len(R), bool)), ("pile-up < 10 %", clean), ("pile-up >= 10 %", ~clean)):
        if s.sum() < 10:
            continue
        print(f"  {name:16s} n={s.sum():6d} | edge vs chain t0: std {wstd(R[s,3]):.2f} robust {robust(R[s,3]):.2f} ns | edge vs truth apex + onset: std {wstd(R[s,4]):.2f} robust {robust(R[s,4]):.2f} ns (mean {np.mean(R[s,4]):+.1f})")
    # within-family consistency (clean channels): std across channels within a family vs spread of family means
    fams = np.unique(R[clean, 0]); within, fmean, fE = [], [], []
    for f in fams:
        s = clean & (R[:, 0] == f)
        if s.sum() >= 4:
            within.append(np.std(R[s, 4])); fmean.append(np.mean(R[s, 4])); fE.append(R[s, 6][0])
    within, fmean, fE = map(np.array, (within, fmean, fE))
    print(f"  per-family (>=4 clean channels, n={len(fams)}): within-family channel scatter median {np.median(within):.2f} ns | spread of family-mean edge residual: std {wstd(fmean):.2f} robust {robust(fmean):.2f} ns (E-wt std {wstd(fmean, fE):.2f})")
    # slew: edge residual vs log10 amplitude (clean channels)
    A = np.log10(R[clean, 1]); y = R[clean, 4]
    coef = np.polyfit(A, y, 1)
    print(f"  slew (clean): d(edge)/d(log10 A) = {coef[0]:+.2f} ns/decade; residual robust after removing it {robust(y - np.polyval(coef, A)):.2f} ns")
    for lo, hi in ((300, 1000), (1000, 3000), (3000, 10000), (10000, 30000), (30000, 70000)):
        s = clean & (R[:, 1] >= lo) & (R[:, 1] < hi)
        if s.sum() > 20:
            print(f"    A [{lo},{hi}): n={s.sum():6d} edge-vs-truth mean {np.mean(R[s,4]):+.2f} robust {robust(R[s,4]):.2f} ns | edge-vs-chain mean {np.mean(R[s,3]):+.2f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 39)
