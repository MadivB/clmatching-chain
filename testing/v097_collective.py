"""Superposition test (user question): per-cluster ML renders SUMMED vs one
COLLECTIVE render of the whole family — is the summation the overflow source?

Family: ev1 TPC16, t0 ~ 588 (the family whose defer of cluster 446 we
diagnosed).  Members and final t0s from the v0.8.6 dump; hits from the dump;
model = the canonical small perceiver.
"""
import sys, os, json
import numpy as np

BASE = "/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/CLMatching_AlphaRelease"
sys.path.insert(0, BASE); sys.path.insert(1, os.path.join(BASE, "NewMLSection")); sys.path.insert(2, os.path.join(BASE, "M5p1"))
from ML_NDfull_perceiver import load_perceiver_model, process_clusters_to_imageMaps
from pulse_shapes import timeinterpolation

CKPT = "/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/NewMLSection/runs/ndfull_small4x/best_model_arch.pt"
TPC, T_FAM, TICK = 16, 588.0, 588 + 105

d = np.load("v096EVO_out/dumps/0000001_ev1.npz", allow_pickle=True)
labs = d["labels"]; tpc = d["tpc"]
x, y, z = d["x"].astype(np.float64), d["y"].astype(np.float64), d["z"].astype(np.float64)
e = np.clip(d["e"].astype(np.float64), 0, None)
tsf = d["ts_final"].astype(np.float64)
actual = d["full_light"].astype(np.float64)

# family = labels assigned within +-10 of 588 with hits in TPC16
fam, t_of = [], {}
for l in np.unique(labs[(labs >= 0)]):
    m = (labs == l) & np.isfinite(tsf) & (tsf >= 0)
    if not m.any():
        continue
    t0 = float(np.median(tsf[m]))
    if abs(t0 - T_FAM) <= 10 and ((labs == l) & (tpc == TPC)).any():
        fam.append(int(l)); t_of[int(l)] = t0
print(f"family members in TPC{TPC} @~{T_FAM:.0f}: {fam}")

model, meta = load_perceiver_model(CKPT)
template = np.load(os.path.join(BASE, "assets", "avg_pulse.npy")).astype(np.float32) / np.float32(999.0)

msel = np.isin(labs, fam)
# A) SEPARATE: render each member alone (what the chain does), stack at t0s
imgA, _ = process_clusters_to_imageMaps(x[msel], y[msel], z[msel], e[msel],
                                        tpc[msel], labs[msel],
                                        model=model, template=template)
n_ticks = actual.shape[2]
A = np.zeros_like(actual[TPC])
for l in fam:
    if (l, TPC) in imgA:
        A += timeinterpolation(np.asarray(imgA[(l, TPC)], np.float64),
                               shift=t_of[l], baseline=0)
# B) COLLECTIVE: all members as ONE cluster, one forward pass
one = np.zeros(msel.sum(), dtype=labs.dtype)
imgB, _ = process_clusters_to_imageMaps(x[msel], y[msel], z[msel], e[msel],
                                        tpc[msel], one,
                                        model=model, template=template)
B = timeinterpolation(np.asarray(imgB[(0, TPC)], np.float64),
                      shift=T_FAM, baseline=0)

a = actual[TPC][:, TICK]; pa = A[:, TICK]; pb = B[:, TICK]
print(f"\nAT THE FAMILY TICK ({TICK}), TPC {TPC}:")
print(f"  total actual {a.sum():9.0f}   sum-of-separate {pa.sum():9.0f} "
      f"({100*(pa.sum()/max(a.sum(),1e-9)-1):+.1f}%)   collective {pb.sum():9.0f} "
      f"({100*(pb.sum()/max(a.sum(),1e-9)-1):+.1f}%)")
for ch in (27, 56, 84):
    print(f"  ch {ch:3d}: actual {a[ch]:7.0f}  separate {pa[ch]:7.0f} "
          f"({100*(pa[ch]/max(a[ch],1e-9)-1):+.0f}%)  collective {pb[ch]:7.0f} "
          f"({100*(pb[ch]/max(a[ch],1e-9)-1):+.0f}%)")
# overflow-branch counting with each prediction (family-only base)
for name, p in (("separate", pa), ("collective", pb)):
    ex = np.minimum(p, 60780.0) - a
    zsc = ex / np.sqrt(200 * np.maximum(a, 100.0))
    n = int(((ex >= 500) & (zsc > 2)).sum())
    print(f"  overflow channels (ex>=500 & z>2) with {name:10s}: {n}")
# channel-wise ratio summary where the family is bright
bright = a > 1000
r = pa[bright] / np.maximum(pb[bright], 1e-9)
print(f"\nseparate/collective ratio on bright channels (a>1000, n={bright.sum()}): "
      f"median {np.median(r):.3f}, p90 {np.percentile(r,90):.3f}")
print("DONE")
