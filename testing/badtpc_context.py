"""Build per-target diagnostic context for the worst non-shower TPC cells.

For each target (file,event,tpc): fuse the D50 dump (per-hit reco/truth t0,
energy, labels) with the matcher's own JSONL diagnostics (trig_rows = flash
trigger per backbone label; group_diags = grouping loss_stay/loss_best) and a
flash-peak census of the actual light in that TPC.  Writes one JSON per target
to bad_tpc_gallery/b{i}_context.json and prints a compact summary.
"""
import glob, json, os, sys
import numpy as np

HERE = os.path.abspath(os.path.dirname(__file__))
DUMPS = os.environ.get("BADTPC_DUMPS") or os.path.join(HERE, "v030D50_out", "dumps")
OUT = os.path.join(HERE, "bad_tpc_gallery")
os.makedirs(OUT, exist_ok=True)
CUT, PEAK = 10.0, 105

# ---- load D50 JSONL diagnostics, keyed by (file_ftag, event) ----
diag = {}
for fp in glob.glob(os.path.join(HERE, "v030D50_out", "groups_worker*.jsonl")):
    for line in open(fp):
        if not line.strip():
            continue
        r = json.loads(line)
        ftag = r["file"].split(".")[-3]   # ...0000001.FLOW.hdf5 -> 0000001
        diag[(ftag, int(r["event"]))] = r

rows = json.load(open(os.path.join(HERE, "badtpc_ranked.json")))
by_key = {(r["file"], r["event"], r["tpc"]): r for r in rows}
ns = [r for r in rows if not r["shower"]]
ns.sort(key=lambda r: -r["e_wrong_real"])

# targets: "file:event:tpc,..." tokens (same order as renderer -> matching b{i}),
# else top-N of the ranked non-shower list.
if len(sys.argv) > 1 and ":" in sys.argv[1]:
    targets = []
    for tok in sys.argv[1].split(","):
        f, e, c = tok.split(":")
        targets.append(by_key[(f, int(e), int(c))])
else:
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    targets = ns[:N]


def flash_peaks(trace, thresh_frac=0.08, min_sep=8):
    """Return [(tick, height)] local maxima above thresh_frac*max, merged."""
    tr = np.asarray(trace, float)
    thr = thresh_frac * tr.max()
    peaks = []
    for t in range(2, tr.size - 2):
        if tr[t] > thr and tr[t] >= tr[t-1] and tr[t] > tr[t+1] \
           and tr[t] >= tr[t-2] and tr[t] >= tr[t+2]:
            if peaks and t - peaks[-1][0] < min_sep:
                if tr[t] > peaks[-1][1]:
                    peaks[-1] = (t, float(tr[t]))
            else:
                peaks.append((t, float(tr[t])))
    return peaks


summ = []
for i, r in enumerate(targets):
    ftag, ev, c = r["file"], r["event"], r["tpc"]
    d = np.load(os.path.join(DUMPS, f"{ftag}_ev{ev}.npz"), allow_pickle=True)
    tpc = np.asarray(d["tpc"]); es = np.asarray(d["e"], np.float64)
    tsf = np.asarray(d["ts_final"], np.float64); tr = np.asarray(d["truth_t0"], np.float64)
    lab = np.asarray(d["labels"])
    lab_rows = {x["label"]: x for x in json.loads(str(d["lab_rows"]))}
    dg = diag.get((ftag, ev), {})
    trig = {x["label"]: x for x in dg.get("trig_rows", [])}
    gdiags = dg.get("group_diags", [])
    gd_by_lab = {}
    for g in gdiags:
        for L in g["mem"]:
            gd_by_lab[int(L)] = g

    m = tpc == c
    fl = np.asarray(d["full_light"], np.float64)[c].sum(0)     # actual light trace
    bi = np.asarray(d["base_image"], np.float64)[c].sum(0)     # predicted light trace
    peaks = flash_peaks(fl)
    peaks_t0 = [(round(t - PEAK), round(h)) for t, h in peaks]  # in t0 units

    ok = m & np.isfinite(tr)
    wrong = ok & np.isfinite(tsf) & (tsf >= 0) & (np.abs(tsf - tr) > CUT)
    wl = lab[wrong & (lab >= 0)]; we = es[wrong & (lab >= 0)]
    clu = {}
    for L, ee in zip(wl, we):
        clu[int(L)] = clu.get(int(L), 0.0) + float(ee)
    dom = sorted(clu.items(), key=lambda kv: -kv[1])[:5]

    cl_ctx = []
    for L, weng in dom:
        ml = m & (lab == L)
        placed = float(np.median(tsf[ml]))
        trv = tr[ml][np.isfinite(tr[ml])]
        true_t0 = float(np.median(trv)) if trv.size else None
        # does placed t0 land on a real flash peak? nearest peak within 6 ticks
        near_placed = min([abs(placed - pt) for pt, _ in peaks_t0], default=None)
        near_true = (min([abs(true_t0 - pt) for pt, _ in peaks_t0], default=None)
                     if true_t0 is not None else None)
        lr = lab_rows.get(L, {})
        tg = trig.get(L)
        gd = gd_by_lab.get(L)
        gd_small = None
        if gd:
            gd_small = {"mem": gd["mem"], "loss_stay": round(gd["loss_stay"], 1),
                        "loss_best": round(gd["loss_best"], 1),
                        "would_move": gd.get("would_move", {}),
                        "sib_truths": gd.get("truths", {}), "sib_t0s": gd.get("t0s", {})}
        cl_ctx.append({
            "label": L, "wrong_E": round(weng, 1),
            "cluster_E": round(float(es[ml].sum()), 1),
            "placed_t0": round(placed, 1), "true_t0": round(true_t0, 1) if true_t0 else None,
            "dt": round(placed - true_t0, 1) if true_t0 else None,
            "type": lr.get("type"), "tpcs_spanned": lr.get("tpcs"),
            "placed_on_real_flash(dt_ticks)": round(near_placed, 1) if near_placed is not None else None,
            "true_on_real_flash(dt_ticks)": round(near_true, 1) if near_true is not None else None,
            "trig": ({"t0": tg["t0"], "fired": tg["fired"], "cmin": round(tg["cmin"], 3),
                      "z3": tg["z3"], "wrong": tg["wrong"]} if tg else None),
            "group": gd_small,
        })

    ctx = {
        "idx": i, "file": ftag, "event": ev, "tpc": c,
        "tpc_E_MeV": round(r["tpc_E"], 1), "eff": round(r["eff"], 3),
        "wrong_E_MeV": round(r["e_wrong"], 1),
        "wrong_E_real_cluster": round(r["e_wrong_real"], 1),
        "wrong_E_noise_forced": round(r["e_wrong_noise"], 1),
        "n_wrong_clusters": r["n_wrong_clusters"],
        "flash_peaks_t0_height": peaks_t0,
        "n_flashes_in_tpc": len(peaks_t0),
        "dominant_wrong_clusters": cl_ctx,
        "pngs": [f"b{i}_{ftag}_ev{ev}_tpc{c}_{k}.png"
                 for k in ("wrong3d", "reco3d", "truth3d", "light_charge")],
        "htmls": [f"b{i}_{ftag}_ev{ev}_tpc{c}_{k}.html"
                  for k in ("wrong3d", "reco3d", "truth3d")],
    }
    json.dump(ctx, open(os.path.join(OUT, f"b{i}_context.json"), "w"), indent=1)
    summ.append(ctx)
    print(f"b{i}: {ftag} ev{ev} TPC{c} E={ctx['tpc_E_MeV']:.0f} eff={ctx['eff']:.2f} "
          f"wReal={ctx['wrong_E_real_cluster']:.0f} nFlash={ctx['n_flashes_in_tpc']} "
          f"| dom: " + "; ".join(
              f"L{cc['label']}({cc['cluster_E']:.0f}MeV {cc['type']}) placed {cc['placed_t0']}"
              f"->true {cc['true_t0']} [placeOnFlash{cc['placed_on_real_flash(dt_ticks)']}]"
              for cc in cl_ctx[:3]))

json.dump(summ, open(os.path.join(OUT, "_all_context.json"), "w"), indent=1)
print(f"\n[done] {len(summ)} contexts -> {OUT}")
