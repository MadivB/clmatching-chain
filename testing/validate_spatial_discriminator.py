"""Decisive test for the swap fix: does a SATURATION-MASKED, per-channel SPATIAL
light pattern prefer the CORRECT t0 over the (wrong) placed t0?

For every dominant-wrong cluster that has a dumped per-channel template img_{L}_{c}:
  cluster_profile = template summed over ticks           (120-ch spatial pattern)
  flash_profile(t0) = actual light summed over ticks in [t0+PEAK +/- W]
Compare cosine(cluster_profile, flash_profile@placed) vs @true, with and without
masking channels that saturate at that flash.  If cos@true > cos@placed, the
spatial discriminator points at truth -> the swap fix is viable.
Also, for the one TPC with BOTH swap partners (b0), solve the 2x2 assignment and
check it recovers truth.
"""
import json, os, sys
import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = os.path.abspath(os.path.dirname(__file__))
DUMPS = os.path.join(HERE, "v030D50_out", "dumps")
PEAK, CLIP, W = 105, 60780.0, 4

def prof(arr):  # channel spatial profile
    return np.asarray(arr, np.float64).sum(1) if arr.ndim == 2 else np.asarray(arr, np.float64)

def flash_prof(actual, t0):
    t = int(round(t0)) + PEAK
    lo, hi = max(0, t - W), min(actual.shape[1], t + W + 1)
    col = actual[:, lo:hi]
    sat = col.max(1) >= 0.98 * CLIP          # channels that saturate at this flash
    return col.sum(1), sat

def cos(a, b, mask=None):
    a = a.copy(); b = b.copy()
    if mask is not None:
        a = a[~mask]; b = b[~mask]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0

ctx = json.load(open(os.path.join(HERE, "bad_tpc_gallery", "_all_context.json")))
print(f"{'case':>4} {'TPC':>4} {'L':>6} {'E':>6} {'placed':>7} {'true':>6} "
      f"{'cosP_mask':>9} {'cosT_mask':>9} {'->true?':>8} | {'cosP_raw':>8} {'cosT_raw':>8} {'raw->true?':>10}")
n_mask_ok = n_mask = n_raw_ok = n_raw = 0
b0 = None
for c in ctx:
    d = np.load(os.path.join(DUMPS, f"{c['file']}_ev{c['event']}.npz"), allow_pickle=True)
    keys = set(d.files); tpc = c['tpc']
    actual = np.asarray(d['full_light'], np.float64)[tpc]
    pair = {}
    for cc in c['dominant_wrong_clusters'][:4]:
        L = cc['label']; key = f'img_{L}_{tpc}'
        if key not in keys or cc['true_t0'] is None:
            continue
        cp = prof(d[key])
        fp_p, sat_p = flash_prof(actual, cc['placed_t0'])
        fp_t, sat_t = flash_prof(actual, cc['true_t0'])
        cP_m = cos(cp, fp_p, sat_p); cT_m = cos(cp, fp_t, sat_t)
        cP_r = cos(cp, fp_p); cT_r = cos(cp, fp_t)
        ok_m = cT_m > cP_m; ok_r = cT_r > cP_r
        n_mask += 1; n_mask_ok += ok_m; n_raw += 1; n_raw_ok += ok_r
        print(f"b{c['idx']:>2} {tpc:>4} {L:>6} {cc['cluster_E']:>6.0f} "
              f"{cc['placed_t0']:>7.1f} {cc['true_t0']:>6.1f} "
              f"{cP_m:>9.3f} {cT_m:>9.3f} {str(ok_m):>8} | {cP_r:>8.3f} {cT_r:>8.3f} {str(ok_r):>10}")
        pair[L] = dict(cp=cp, placed=cc['placed_t0'], true=cc['true_t0'], E=cc['cluster_E'])
    if c['idx'] == 0 and len(pair) >= 2:
        b0 = (tpc, actual, pair)

print(f"\nMASKED: {n_mask_ok}/{n_mask} clusters have cos@true > cos@placed "
      f"({100*n_mask_ok/max(n_mask,1):.0f}%)")
print(f"RAW   : {n_raw_ok}/{n_raw} ({100*n_raw_ok/max(n_raw,1):.0f}%)")

# ---- full 2x2 assignment test on b0 (both partners available) ----
if b0:
    tpc, actual, pair = b0
    Ls = list(pair)[:2]
    flashes = sorted({round(pair[L]['placed']) for L in Ls} | {round(pair[L]['true']) for L in Ls})
    print(f"\n[b0 TPC{tpc}] 2x2 assignment over clusters {Ls} vs flash-t0s {flashes}")
    C = np.zeros((len(Ls), len(flashes)))
    for i, L in enumerate(Ls):
        for j, f in enumerate(flashes):
            fp, sat = flash_prof(actual, f)
            C[i, j] = 1.0 - cos(pair[L]['cp'], fp, sat)   # cost = 1 - masked cosine
    ri, ci = linear_sum_assignment(C)
    print("  cost matrix (1 - masked cosine):")
    print("      " + "  ".join(f"t0={f:>4}" for f in flashes))
    for i, L in enumerate(Ls):
        print(f"  L{L:<5} " + "  ".join(f"{C[i,j]:>7.3f}" for j in range(len(flashes))))
    assign = {Ls[i]: flashes[ci[i]] for i in range(len(Ls))}
    print("  optimal assignment:", {f"L{L}": t for L, t in assign.items()})
    print("  truth             :", {f"L{L}": round(pair[L]['true']) for L in Ls})
    ok = all(abs(assign[L] - pair[L]['true']) <= 10 for L in Ls)
    print("  ==> RECOVERS TRUTH" if ok else "  ==> does NOT recover truth")
