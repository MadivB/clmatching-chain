"""Global failure taxonomy of v0.3.0 non-shower charge-light mistakes.

Over all D50 dumps, for every wrong real-cluster hit (labels>=0, truth-valid,
|ts_final-truth|>CUT, in a non-shower TPC), classify the WRONG cluster's mistake:

  SWAP        another cluster in the SAME TPC truly lives at this cluster's
              placed_t0 (|other.true - this.placed| <= CUT) AND this cluster
              truly lives elsewhere -> the two interactions' flashes were crossed.
  LANDED_ON_FLASH  placed_t0 sits on a real light-flash peak in this TPC
              (<=SWAP_TOL ticks) but no same-TPC cluster truly lives there
              -> grabbed a real neighbouring flash (often another TPC's, or a
                 flash from a cluster now placed elsewhere).
  OFF_FLASH   placed_t0 is NOT near any flash peak -> drift/geometry/forced guess.

Aggregates wrong energy by category (energy-weighted, per cluster's median t0)
and prints the split.  This is computable globally without images.
"""
import glob, json, os, sys
import numpy as np

HERE = os.path.abspath(os.path.dirname(__file__))
DUMPS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "v030D50_out", "dumps")
CUT, PEAK, SWAP_TOL = 10.0, 105, 6.0
SHOWER_E, MIN_TPC_E = 1400.0, 50.0


def flash_peaks(trace, thresh_frac=0.06, min_sep=8):
    tr = np.asarray(trace, float)
    if tr.max() <= 0:
        return []
    thr = thresh_frac * tr.max()
    pk = []
    for t in range(2, tr.size - 2):
        if tr[t] > thr and tr[t] >= tr[t-1] and tr[t] > tr[t+1] and tr[t] >= tr[t-2] and tr[t] >= tr[t+2]:
            if pk and t - pk[-1] < min_sep:
                continue
            pk.append(t)
    return [p - PEAK for p in pk]   # t0 units


cat_E = {"SWAP": 0.0, "LANDED_ON_FLASH": 0.0, "OFF_FLASH": 0.0}
cat_N = {"SWAP": 0, "LANDED_ON_FLASH": 0, "OFF_FLASH": 0}
tot_wrong_real = 0.0
tot_ok = 0.0
per_tpc_swapfrac = []
files = sorted(glob.glob(os.path.join(DUMPS, "*.npz")))
print(f"{len(files)} dumps")
for fp in files:
    d = np.load(fp, allow_pickle=True)
    tpc = np.asarray(d["tpc"]); es = np.asarray(d["e"], np.float64)
    tsf = np.asarray(d["ts_final"], np.float64); tr = np.asarray(d["truth_t0"], np.float64)
    lab = np.asarray(d["labels"])
    fll = np.asarray(d["full_light"], np.float64)
    for c in np.unique(tpc):
        mt = tpc == c
        if float(es[mt].sum()) < MIN_TPC_E or float(es[mt].sum()) > SHOWER_E:
            continue
        ok = mt & np.isfinite(tr)
        tot_ok += float(es[ok].sum())
        # per real cluster in this TPC: placed & true t0 (energy-weighted median)
        cl = {}
        for L in np.unique(lab[ok & (lab >= 0)]):
            ml = ok & (lab == L)
            trv = tr[ml]
            cl[int(L)] = {
                "placed": float(np.median(tsf[ml])),
                "true": float(np.median(trv[np.isfinite(trv)])) if np.isfinite(trv).any() else None,
                "E": float(es[ml].sum()),
                "wrongE": float(es[ml & (np.abs(tsf - tr) > CUT)].sum()),
            }
        peaks = flash_peaks(fll[c].sum(0)) if c < fll.shape[0] else []
        true_t0s = [v["true"] for v in cl.values() if v["true"] is not None]
        tpc_swap_E = 0.0; tpc_wrong_E = 0.0
        for L, v in cl.items():
            if v["wrongE"] <= 0 or v["true"] is None:
                continue
            tot_wrong_real += v["wrongE"]; tpc_wrong_E += v["wrongE"]
            placed = v["placed"]
            # SWAP: some OTHER cluster truly lives at this placed_t0
            is_swap = any(abs(v2["true"] - placed) <= CUT
                          for L2, v2 in cl.items()
                          if L2 != L and v2["true"] is not None)
            on_flash = any(abs(placed - p) <= SWAP_TOL for p in peaks)
            if is_swap:
                cat = "SWAP"; tpc_swap_E += v["wrongE"]
            elif on_flash:
                cat = "LANDED_ON_FLASH"
            else:
                cat = "OFF_FLASH"
            cat_E[cat] += v["wrongE"]; cat_N[cat] += 1
        if tpc_wrong_E > 30:
            per_tpc_swapfrac.append(tpc_swap_E / tpc_wrong_E)

print(f"\nnon-shower truth energy: {tot_ok:.0f} MeV")
print(f"real-cluster WRONG energy: {tot_wrong_real:.0f} MeV "
      f"({100*tot_wrong_real/tot_ok:.2f}% of non-shower truth E)\n")
print(f"{'category':>16} {'wrongE(MeV)':>12} {'frac':>7} {'nClusters':>10}")
for k in ["SWAP", "LANDED_ON_FLASH", "OFF_FLASH"]:
    print(f"{k:>16} {cat_E[k]:>12.0f} {100*cat_E[k]/max(tot_wrong_real,1e-9):>6.1f}% {cat_N[k]:>10}")
if per_tpc_swapfrac:
    a = np.array(per_tpc_swapfrac)
    print(f"\nper-bad-TPC swap-energy fraction: median {np.median(a):.2f}, "
          f"mean {a.mean():.2f} over {a.size} TPCs with wrongE>30MeV")
print("\n[done]")
