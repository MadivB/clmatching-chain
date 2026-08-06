"""Feasibility ceiling for the residual-peak new-flash rescue (user idea 2026-07-09).
On the v0.4 dumps, find mis-placed SMALL BLOBS (compact clusters, wrong t0 by truth) and
check whether their TRUE flash shows a residual DEFICIT (actual > predicted) that a
residual-peak pass could discover -> how much wrong energy is recoverable this way."""
import glob, os, numpy as np

HERE = os.path.abspath(os.path.dirname(__file__))
DUMPS = os.path.join(HERE, "v030v3_out", "dumps")
CLIP, GAIN, PEAK, CUT = 60780.0, 200.0, 105, 10
BLOB_EXTENT = 30.0     # cm: a "blob" (not a track) spans less than this
FWIN = 4

tot_wrong_blob = 0.0; tot_recoverable = 0.0; n_blob = 0; n_recov = 0
tot_wrong_all = 0.0
for fp in sorted(glob.glob(os.path.join(DUMPS, "*.npz"))):
    d = np.load(fp, allow_pickle=True)
    x = np.asarray(d["x"], float); y = np.asarray(d["y"], float); z = np.asarray(d["z"], float)
    e = np.asarray(d["e"], float); lab = np.asarray(d["labels"]); tpc = np.asarray(d["tpc"])
    tsf = np.asarray(d["ts_final"], float); tr = np.asarray(d["truth_t0"], float)
    full = np.asarray(d["full_light"], np.float32); base = np.asarray(d["base_image"], np.float32)
    for L in np.unique(lab[lab >= 0]):
        m = lab == L
        ht = m & np.isfinite(tr)
        if ht.sum() < 5:
            continue
        # spatial extent (bounding-box diagonal) -> blob vs track
        ext = float(np.sqrt(np.ptp(x[m]) ** 2 + np.ptp(y[m]) ** 2 + np.ptp(z[m]) ** 2))
        wrong = ht & np.isfinite(tsf) & (np.abs(tsf - tr) > CUT)
        we = float(e[wrong].sum())
        tot_wrong_all += we
        if ext > BLOB_EXTENT or we <= 0:
            continue
        # this is a mis-placed BLOB
        n_blob += 1; tot_wrong_blob += we
        # its dominant TPC + true t0
        c = int(np.bincount(tpc[wrong]).argmax())
        true_t0 = float(np.median(tr[wrong & (tpc == c)]))
        tick = int(round(true_t0)) + PEAK
        if not (0 <= tick < full.shape[2]):
            continue
        # residual deficit at the true flash in this TPC (actual > predicted)?
        lo, hi = max(0, tick - FWIN), min(full.shape[2], tick + FWIN + 1)
        A = full[c, :, lo:hi].sum()
        P = np.minimum(base[c, :, lo:hi], CLIP).sum()
        deficit = A - P
        sig = np.sqrt(GAIN * max(A, 100.0))
        if deficit > 3 * sig and deficit > 0.15 * A:      # a clear unexplained flash
            n_recov += 1; tot_recoverable += we
    d.close(); del full, base

print(f"total wrong energy (all clusters):        {tot_wrong_all/1000:.1f} GeV")
print(f"mis-placed BLOBS (extent<{BLOB_EXTENT:.0f}cm): n={n_blob}  wrong E={tot_wrong_blob/1000:.2f} GeV "
      f"({100*tot_wrong_blob/max(tot_wrong_all,1):.1f}% of all wrong)")
print(f"  of those, TRUE flash shows clear residual DEFICIT (recoverable): n={n_recov} "
      f"wrong E={tot_recoverable/1000:.2f} GeV ({100*tot_recoverable/max(tot_wrong_blob,1):.1f}% of blob-wrong)")
print(f"CEILING: residual-peak blob rescue could recover up to "
      f"{100*tot_recoverable/max(tot_wrong_all,1):.2f}% of ALL wrong energy")
