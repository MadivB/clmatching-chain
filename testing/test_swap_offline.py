"""Offline unit test of run_swap_correction on b0 (TPC5), the one dump with BOTH
swap partners' templates. Build a minimal ns and confirm the pass swaps L257<->L16
back toward truth."""
import json, os, numpy as np
HERE = os.path.abspath(os.path.dirname(__file__))
from swap_fix import run_swap_correction

d = np.load(os.path.join(HERE, "v030D50_out/dumps/0000002_ev1.npz"), allow_pickle=True)
tpc = np.asarray(d["tpc"]); lab = np.asarray(d["labels"]); e = np.asarray(d["e"], np.float64)
tsf = np.asarray(d["ts_final"], np.float64); tr = np.asarray(d["truth_t0"], np.float64)
A, B, C = 257, 16, 5    # cluster A, cluster B, TPC

ns = {
    "imageMaps": {(A, C): np.asarray(d[f"img_{A}_{C}"], np.float64),
                  (B, C): np.asarray(d[f"img_{B}_{C}"], np.float64)},
    "fullLightWaveform": np.asarray(d["full_light"], np.float64),
    "baseImage": np.asarray(d["base_image"], np.float64).copy(),
    "labels_global": lab, "hit_timestamps": tsf.astype(np.float32).copy(), "Eset": e,
}
def truth_fn(L):
    m = (lab == L) & np.isfinite(tr)
    return float(np.median(tr[m])) if m.any() else None

for L in (A, B):
    m = lab == L
    print(f"L{L}: placed t0={np.median(tsf[m]):.1f}  true t0={truth_fn(L):.1f}  E={e[m].sum():.0f}")

st = run_swap_correction(ns, hit_tpc=tpc, cluster_to_tpcs={A: [C], B: [C]},
                         min_energy=25.0, min_cos_gain=0.03, truth_fn=truth_fn, verbose=True)
print("\nstats:", {k: v for k, v in st.items() if k != "audit"})
for r in st["audit"]:
    print("  swap:", r)

ts2 = np.asarray(ns["hit_timestamps"], np.float64)
for L in (A, B):
    m = lab == L
    tv = truth_fn(L)
    err = abs(float(np.median(ts2[m])) - tv)
    print(f"AFTER L{L}: t0={np.median(ts2[m]):.1f}  true={tv:.1f}  |err|={err:.1f}  "
          f"{'CORRECT' if err <= 10 else 'still wrong'}")
