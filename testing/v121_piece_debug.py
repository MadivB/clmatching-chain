"""Debug genuinely delayed pieces: for pieces with true delay > 20 ns that pass the own-dominated
rule, print per channel the residual peak position vs the true delay, the piece's own predicted
amplitude vs others at the chain tick, and the fitted shift.  python v121_piece_debug.py <dump_dir> <flow> <calib> <lo> <hi>"""
import glob, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v120_precision as vp
C, NS, PEAK = 29.9792458, 16.0, 105


def main(d, flow, calib, lo, hi):
    from subtick_refine import build_sipm_positions
    files = sorted(glob.glob(f"{d}/full_dumps/*.npz"))[lo:hi]
    shown = 0
    for f in files:
        new_t, sig, st, z = vp.run_on_dump(f, flow, calib, rounds=1, pieces=True, collect=True)
        tru = z["truth_t0"].astype(float); ts = z["ts_final"].astype(float); e = np.clip(z["e"], 0, None); labs = z["labels"].astype(int); tpc = z["tpc"]
        full = z["full_light"].astype(float); base = z["base_image"].astype(float)
        al, at, ar = z["amp_lab"].astype(int), z["amp_tpc"].astype(int), z["amp_row"].astype(float)
        amp = {(int(al[i]), int(at[i])): ar[i] for i in range(len(al))}
        tsr = np.round(np.where(np.isfinite(ts), ts, -1))
        for pc in st["pieces"]:
            m = np.isin(labs, pc["labels"]) & (tpc == pc["tpc"]) & np.isfinite(tru) & (tru >= 0)
            mc = (tpc == pc["tpc"]) & (tsr == np.round(pc["t_chain"])) & ~np.isin(labs, pc["labels"]) & np.isfinite(tru) & (tru >= 0)
            if not m.any() or mc.sum() < 10:
                continue
            xyz_ = np.stack([z["x"], z["y"], z["z"]], 1).astype(float)
            o = np.argsort(tru[mc]); n2 = max(3, int(0.02 * o.size)); apex_ = np.average(xyz_[mc][o[:n2]], axis=0, weights=e[mc][o[:n2]])
            t_apex = float(np.median(tru[mc] * NS - np.linalg.norm(xyz_[mc] - apex_, axis=1) / C))
            dt_true = float(np.average(tru[m], weights=e[m])) * NS - t_apex
            if not (30 < dt_true < 160):
                continue
            print("   core truth-time spread (robust, ns): %.1f; piece hits truth spread (robust, ns): %.1f; piece E-wt truth rel. chain tick: %.2f ticks" % (1.4826*np.median(np.abs(tru[mc]*NS - np.median(tru[mc]*NS))), 1.4826*np.median(np.abs(tru[m]*NS - np.median(tru[m]*NS))), np.average(tru[m], weights=e[m]) - pc["t_chain"]))
            c = pc["tpc"]; tv = pc["t_chain"]; col = int(round(tv)) + PEAK
            pat = np.zeros(120)
            for L in pc["labels"]:
                if (L, c) in amp: pat += amp[(L, c)]
            order = np.argsort(pat)[::-1][:6]
            print(f"\n{os.path.basename(f)} tpc {c} chain tick {tv:.1f} piece E {pc['E']:.1f} MeV labels {pc['labels']} | true delay {dt_true:.1f} ns ({dt_true/NS:.1f} ticks), fitted delay {pc['dt_ns']:.1f} ns, sigma {pc['sig_ns']:.2f}, n_ch {pc['n_ch']}, dist {pc['dist_cm']:.0f} cm")
            print("   truth times of the piece hits (ticks rel. chain): median %.2f, p10 %.2f, p90 %.2f" % tuple(np.percentile(tru[m] - tv, [50, 10, 90])))
            for ch in order:
                w = full[c][ch]; bl = np.median(w[col - 60: col - 30]); seg = w[col - 5: col + 25] - bl
                ipk = int(np.argmax(seg))
                oth = base[c][ch][col] - pat[ch]
                print(f"   ch {ch:3d}: own amp {pat[ch]:7.0f} others@tick {oth:7.0f} | actual peak in [-5,+25] ticks at {ipk-5:+d} ticks, height {seg[ipk]:7.0f} | actual at chain tick {w[col]-bl:7.0f}, at +{int(round(dt_true/NS))} ticks {w[min(col+int(round(dt_true/NS)),999)]-bl:7.0f}")
            shown += 1
            if shown >= 6:
                return


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
