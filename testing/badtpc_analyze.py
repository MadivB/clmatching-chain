"""Rank per-(event,TPC) charge-light matching quality from v0.3.0 --dump-dir npz.

For every (file,event,tpc) with hits, compute energy-weighted efficiency among
truth-valid hits (|ts_final - truth_t0| <= CUT ticks = correct).  Focus on the
NON-SHOWER cohort (tpc total E <= 1.4 GeV, the target definition) and rank the
worst by wrong energy.  Splits mistakes into:
  - real-cluster wrong E   (labels_global >= 0)  -> actionable matcher mistakes
  - noise/forced wrong E    (labels_global <  0)  -> force-coverage guesses
Writes badtpc_ranked.json and prints a table.
"""
import glob, json, os, sys
import numpy as np

HERE = os.path.abspath(os.path.dirname(__file__))
DUMPS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "v030D50_out", "dumps")
CUT = 10.0
PEAK = 105
SWAP_TOL = 6.0
SHOWER_E = 1400.0
MIN_TPC_E = 50.0   # ignore near-empty TPCs


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

rows = []
files = sorted(glob.glob(os.path.join(DUMPS, "*.npz")))
print(f"{len(files)} dumps in {DUMPS}")
for fp in files:
    try:
        d = np.load(fp, allow_pickle=True)
    except Exception as ex:
        print("skip", fp, ex); continue
    tpc = np.asarray(d["tpc"]); e = np.asarray(d["e"], np.float64)
    tsf = np.asarray(d["ts_final"], np.float64); tr = np.asarray(d["truth_t0"], np.float64)
    lab = np.asarray(d["labels"])
    fll = np.asarray(d["full_light"], np.float64)
    ftag = os.path.basename(fp).split("_ev")[0]
    ev = int(str(d["event"]))
    # per-tpc total energy (matches pipeline mask_ns definition)
    for c in np.unique(tpc):
        mt = tpc == c
        tpc_E = float(e[mt].sum())
        if tpc_E < MIN_TPC_E:
            continue
        ok = mt & np.isfinite(tr)                      # truth-valid hits in this tpc
        e_ok = float(e[ok].sum())
        if e_ok <= 0:
            continue
        asg = ok & np.isfinite(tsf) & (tsf >= 0)
        wrong = asg & (np.abs(tsf - tr) > CUT)
        unasg = ok & ~asg
        e_wrong = float(e[wrong].sum())
        e_unasg = float(e[unasg].sum())
        eff = 1.0 - (e_wrong + e_unasg) / e_ok         # correct fraction of truth energy
        # split wrong energy by real cluster vs noise/forced
        e_wrong_real = float(e[wrong & (lab >= 0)].sum())
        e_wrong_noise = float(e[wrong & (lab < 0)].sum())
        # how many distinct real clusters carry the wrong energy
        wl = lab[wrong & (lab >= 0)]
        we = e[wrong & (lab >= 0)]
        clu = {}
        for L, ee in zip(wl, we):
            clu[int(L)] = clu.get(int(L), 0.0) + float(ee)
        top_clu = sorted(clu.items(), key=lambda kv: -kv[1])[:4]
        # ---- categorize wrong energy: SWAP / LANDED_ON_FLASH / OFF_FLASH ----
        cl_t0 = {}
        for L in np.unique(lab[ok & (lab >= 0)]):
            ml = ok & (lab == L)
            trv = tr[ml]
            cl_t0[int(L)] = (float(np.median(tsf[ml])),
                             float(np.median(trv[np.isfinite(trv)])) if np.isfinite(trv).any() else None)
        peaks = flash_peaks(fll[c].sum(0)) if c < fll.shape[0] else []
        catE = {"SWAP": 0.0, "LANDED": 0.0, "OFF": 0.0}
        for L, wE in clu.items():
            placed = cl_t0.get(L, (None, None))[0]
            if placed is None:
                catE["OFF"] += wE; continue
            is_swap = any(L2 != L and t2[1] is not None and abs(t2[1] - placed) <= CUT
                          for L2, t2 in cl_t0.items())
            on_flash = any(abs(placed - p) <= SWAP_TOL for p in peaks)
            catE["SWAP" if is_swap else ("LANDED" if on_flash else "OFF")] += wE
        dom_cat = max(catE, key=catE.get) if e_wrong_real > 0 else "NONE"
        rows.append({
            "file": ftag, "event": ev, "tpc": int(c),
            "tpc_E": tpc_E, "shower": tpc_E > SHOWER_E,
            "e_ok": e_ok, "e_wrong": e_wrong, "e_unasg": e_unasg,
            "eff": eff, "e_wrong_real": e_wrong_real, "e_wrong_noise": e_wrong_noise,
            "n_wrong_clusters": len(clu),
            "top_wrong_clusters": [[L, round(v, 1)] for L, v in top_clu],
            "n_flashes": len(peaks),
            "catE": {k: round(v, 1) for k, v in catE.items()}, "dom_cat": dom_cat,
        })

json.dump(rows, open(os.path.join(HERE, "badtpc_ranked.json"), "w"))

ns = [r for r in rows if not r["shower"]]
sh = [r for r in rows if r["shower"]]
print(f"\n{len(rows)} (event,tpc) cells | {len(ns)} non-shower | {len(sh)} shower(>1.4GeV)")

# ---- global non-shower efficiency (sanity vs ns_total_eff) ----
tot_ok = sum(r["e_ok"] for r in ns)
tot_bad = sum(r["e_wrong"] + r["e_unasg"] for r in ns)
print(f"non-shower energy-weighted eff (all cells): {1 - tot_bad/tot_ok:.4f} "
      f"(ok={tot_ok:.0f}, wrong={sum(r['e_wrong'] for r in ns):.0f}, unasg={sum(r['e_unasg'] for r in ns):.0f} MeV)")

def show(title, lst, key, n=25):
    print(f"\n===== {title} =====")
    print(f"{'file':>8} {'ev':>3} {'tpc':>3} {'tpcE':>7} {'eff':>6} {'wRONG':>7} "
          f"{'wReal':>6} {'wNoise':>6} {'unasg':>6} {'nClu':>4}  topClusters(E)")
    for r in sorted(lst, key=key)[:n]:
        print(f"{r['file']:>8} {r['event']:>3} {r['tpc']:>3} {r['tpc_E']:>7.0f} "
              f"{r['eff']:>6.3f} {r['e_wrong']:>7.0f} {r['e_wrong_real']:>6.0f} "
              f"{r['e_wrong_noise']:>6.0f} {r['e_unasg']:>6.0f} {r['n_wrong_clusters']:>4}  "
              f"{r['top_wrong_clusters']}")

# worst non-shower by REAL-cluster wrong energy (actionable matcher mistakes)
show("WORST non-shower by REAL-cluster wrong energy", ns, lambda r: -r["e_wrong_real"])
# worst non-shower by total (wrong+unassigned) energy fraction, require some substance
show("WORST non-shower by (1-eff), tpcE>=200MeV", [r for r in ns if r["tpc_E"] >= 200],
     lambda r: (r["eff"], -r["tpc_E"]))
print("\n[done]")
