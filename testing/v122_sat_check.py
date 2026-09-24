"""v1.2.2 saturation-ceiling checks on chain dumps.

1. Every dump carries sat_ceiling (n_tpc,120); on channels that touch the rail the actual
   waveform's maximum equals the ceiling to within 1 count (the definition is right).
2. The final predicted light, clipped at the ceiling, never exceeds the actual plateau on
   vetoed channels by more than 1 count (the display and the loss agree).
3. Paired against a reference arm: per-event assignment differences (ts_final) and the
   energy-weighted efficiency of both arms.
python v122_sat_check.py <arm_dir> [<ref_dir>]"""
import glob, os, sys
import numpy as np
CUT = 10.0


def main(arm, ref=None):
    print(f"{'event':>16} {'ceil?':>5} {'rail ch':>7} {'|max-ceil|':>10} {'pred>rail':>9} {'eff':>7}" + ("  ref eff   changed hits  changedE" if ref else ""))
    for ff in sorted(glob.glob(f"{arm}/full_dumps/*.npz")):
        z = np.load(ff, allow_pickle=True)
        tag = os.path.basename(ff).replace(".npz", "")
        has = "sat_ceiling" in z.files
        fl = z["full_light"].astype(np.float32); bi = z["base_image"].astype(np.float32)
        e = np.clip(z["e"], 0, None).astype(float); tru = z["truth_t0"].astype(float); ts = z["ts_final"].astype(float)
        ok = np.isfinite(tru) & np.isfinite(ts) & (ts >= 0)
        eff = e[ok & (np.abs(ts - tru) <= CUT)].sum() / max(e[ok].sum(), 1e-9)
        if has:
            ceil = z["sat_ceiling"].astype(np.float32)
            mx = fl.max(axis=-1)
            rail = mx >= ceil - 1.0
            dmax = float(np.abs(mx[rail] - ceil[rail]).max()) if rail.any() else 0.0
            # over: channels where the clipped prediction exceeds the actual plateau on the
            # plateau samples themselves (must be 0 if the ceiling is the real rail)
            plateau = (fl >= ceil[..., None] - 1.0)
            over = int(np.sum(np.any((np.minimum(bi, ceil[..., None]) - fl > 1.0) & plateau, axis=-1)))
            line = f"{tag:>16} {'yes':>5} {int(rail.sum()):7d} {dmax:10.2f} {over:9d} {eff:7.4f}"
        else:
            line = f"{tag:>16} {'no':>5} {'-':>7} {'-':>10} {'-':>9} {eff:7.4f}"
        if ref:
            rf = os.path.join(ref, "full_dumps", os.path.basename(ff))
            if os.path.exists(rf):
                r = np.load(rf, allow_pickle=True)
                tsr = r["ts_final"].astype(float)
                okr = np.isfinite(tru) & np.isfinite(tsr) & (tsr >= 0)
                effr = e[okr & (np.abs(tsr - tru) <= CUT)].sum() / max(e[okr].sum(), 1e-9)
                a = np.where(np.isfinite(ts), ts, -1.0); b = np.where(np.isfinite(tsr), tsr, -1.0)
                ch = np.abs(a - b) > 0.5
                line += f"  {effr:7.4f}  {int(ch.sum()):12d}  {e[ch].sum():8.1f}"
            else:
                line += "  (no ref dump)"
        print(line)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
