"""Render event displays for the worst non-shower (event,TPC) cells of the
v0.3.0 charge-light matcher, from the --dump-dir npz.

Targets come from badtpc_ranked.json (top-N non-shower by real-cluster wrong E)
or an explicit list "file:event:tpc,..." as argv[1].

Per target -> bad_tpc_gallery/b{i}_{file}_ev{ev}_tpc{c}_*:
  _wrong3d   RED = |ts_final - truth_t0| > 10 ticks, gray = correct   [headline]
  _reco3d    hits colored by reco (ts_final) clustered t0
  _truth3d   hits colored by truth_t0 clustered t0
  _light     actual vs predicted light (base_image) time trace for the TPC,
             with placed-t0 / true-t0 markers of the dominant wrong cluster
             + charge z-y scatter colored by drift-x

Run under:  module load pytorch/2.8.0   (kaleido lives in that user-site).
"""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.abspath(os.path.dirname(__file__))
BASE = os.path.abspath(os.path.join(_HERE, "..", "CLMatching_AlphaRelease"))
sys.path.insert(0, BASE)
sys.path.insert(1, os.path.join(BASE, "NewMLSection"))
sys.path.insert(2, os.path.join(BASE, "M5p1"))

import plotly.graph_objects as go
OUT = os.path.join(_HERE, "bad_tpc_gallery")
os.makedirs(OUT, exist_ok=True)
_png = {"path": None}
_orig_show = go.Figure.show
def _show_to_png(self, *a, **k):
    if _png["path"]:
        self.write_image(_png["path"], width=1150, height=880, scale=2)
go.Figure.show = _show_to_png

from M5p1 import plottingTools as pt
from M5p1 import truth_plotting as tp

DUMPS = os.environ.get("BADTPC_DUMPS", os.path.join(_HERE, "v030v3_out", "dumps"))
CUT, PEAK, TW = 10.0, 105, 5

# ---- resolve targets ----
if len(sys.argv) > 1 and ":" in sys.argv[1]:
    tgts = []
    for tok in sys.argv[1].split(","):
        f, e, c = tok.split(":")
        tgts.append({"file": f, "event": int(e), "tpc": int(c)})
else:
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    rows = json.load(open(os.path.join(_HERE, "badtpc_ranked.json")))
    ns = [r for r in rows if not r["shower"]]
    ns.sort(key=lambda r: -r["e_wrong_real"])
    tgts = ns[:N]

_dcache = {}
def load(ftag, ev):
    k = (ftag, ev)
    if k not in _dcache:
        _dcache[k] = np.load(os.path.join(DUMPS, f"{ftag}_ev{ev}.npz"), allow_pickle=True)
    return _dcache[k]

for i, t in enumerate(tgts):
    ftag, ev, c = t["file"], t["event"], t["tpc"]
    dp = os.path.join(DUMPS, f"{ftag}_ev{ev}.npz")
    if not os.path.exists(dp):
        print(f"b{i}: MISSING dump {dp}"); continue
    d = load(ftag, ev)
    xs = np.asarray(d["x"], np.float64); ys = np.asarray(d["y"], np.float64)
    zs = np.asarray(d["z"], np.float64); es = np.asarray(d["e"], np.float64)
    tpc = np.asarray(d["tpc"]); tsf = np.asarray(d["ts_final"], np.float64)
    tr = np.asarray(d["truth_t0"], np.float64); lab = np.asarray(d["labels"])
    m = tpc == c
    tpcE = float(es[m].sum())
    ok = m & np.isfinite(tr)
    wrong = ok & np.isfinite(tsf) & (tsf >= 0) & (np.abs(tsf - tr) > CUT)
    eff = 1.0 - float(es[wrong].sum()) / max(float(es[ok].sum()), 1e-9)
    # dominant wrong cluster
    wl = lab[wrong & (lab >= 0)]; we = es[wrong & (lab >= 0)]
    clu = {}
    for L, ee in zip(wl, we): clu[int(L)] = clu.get(int(L), 0.0) + float(ee)
    dom = sorted(clu.items(), key=lambda kv: -kv[1])
    domL = dom[0][0] if dom else None
    tag = f"b{i}_{ftag}_ev{ev}_tpc{c}"
    ttl = (f"{ftag} ev{ev} TPC{c} | E={tpcE:.0f}MeV eff={eff:.2f} "
           f"wrongE={float(es[wrong].sum()):.0f}MeV | dom cluster {domL}")

    # ---------- 3D: reco / truth / wrong ----------
    tt_safe = np.where(np.isfinite(tr), tr, -1.0)
    rt_safe = np.where(np.isfinite(tsf), tsf, -1.0)
    _png["path"] = os.path.join(OUT, f"{tag}_truth3d.png")
    pt.plot_3d_clusters_with_t0(x=xs[m], y=ys[m], z=zs[m],
                                labels=pt.group_hits_by_time(tt_safe[m], time_window=TW),
                                t0s=tt_safe[m], energies=es[m],
                                title=f"TRUTH t0 | {ttl}",
                                save_path=os.path.join(OUT, f"{tag}_truth3d.html"))
    _png["path"] = os.path.join(OUT, f"{tag}_reco3d.png")
    pt.plot_3d_clusters_with_t0(x=xs[m], y=ys[m], z=zs[m],
                                labels=pt.group_hits_by_time(rt_safe[m], time_window=TW),
                                t0s=rt_safe[m], energies=es[m],
                                title=f"RECO t0 (v0.3.0) | {ttl}",
                                save_path=os.path.join(OUT, f"{tag}_reco3d.html"))
    _png["path"] = os.path.join(OUT, f"{tag}_wrong3d.png")
    tp.plot_wrong_hits_3d(int(c), tr, hit_timestamps=tsf, hitTPCid=tpc,
                          xset=xs, yset=ys, zset=zs, Eset=es, residual_cut_ticks=CUT,
                          save_path=os.path.join(OUT, f"{tag}_wrong3d.html"),
                          marker_size=3.5, wrong_color="red",
                          correct_color="lightgray", show=True)
    _png["path"] = None

    # ---------- light + charge 2D ----------
    fl = np.asarray(d["full_light"], np.float64)[c]     # (120,1000) actual
    bi = np.asarray(d["base_image"], np.float64)[c]     # (120,1000) predicted
    fig, (axL, axC) = plt.subplots(2, 1, figsize=(12, 9),
                                   gridspec_kw={"height_ratios": [1.0, 1.2]})
    fig.suptitle(ttl, fontsize=11)
    act = fl.sum(0); pred = bi.sum(0)
    axL.plot(act, color="tab:blue", lw=1.3, label="actual light (sum ch)")
    axL.plot(pred, color="tab:orange", lw=1.3, label="predicted light (base_image, sum ch)")
    # markers: placed & true t0 of dominant wrong cluster (light peak = t0 + PEAK)
    def vlines(labset, color, ls, name):
        seen = set()
        for L in labset:
            ml = m & (lab == L)
            if not ml.any(): continue
            rp = float(np.median(tsf[ml])); tpv = tr[ml]
            tv = float(np.median(tpv[np.isfinite(tpv)])) if np.isfinite(tpv).any() else None
            if name == "placed" and round(rp) not in seen:
                axL.axvline(rp + PEAK, color=color, ls=ls, lw=1.4,
                            label=f"placed t0={rp:.0f}" if not seen else None); seen.add(round(rp))
            if name == "true" and tv is not None and round(tv) not in seen:
                axL.axvline(tv + PEAK, color=color, ls=ls, lw=1.6,
                            label=f"TRUE t0={tv:.0f}" if not seen else None); seen.add(round(tv))
    wrong_labs = [L for L, _ in dom[:3]]
    if domL is not None:
        vlines(wrong_labs, "red", "--", "placed")
        vlines(wrong_labs, "green", ":", "true")
    axL.set_xlabel("tick"); axL.set_ylabel("ADC (sum over channels)")
    axL.legend(fontsize=8, loc="upper right"); axL.set_xlim(0, 999)
    axL.set_title("light: does predicted (placed) match actual? red=placed peak, green=true peak", fontsize=9)
    # charge z-y colored by drift x, wrong hits ringed
    sc = axC.scatter(zs[m], ys[m], c=xs[m], s=6 + 30 * es[m] / max(es[m].max(), 1e-9),
                     cmap="viridis", alpha=0.8)
    axC.scatter(zs[wrong], ys[wrong], s=42, facecolors="none", edgecolors="red", lw=0.7,
                label="wrong hits")
    plt.colorbar(sc, ax=axC, label="drift x (cm)")
    axC.set_xlabel("z (cm)"); axC.set_ylabel("y (cm)")
    axC.legend(fontsize=8, loc="upper right")
    axC.set_title("charge (this TPC): red-ringed = |reco-truth|>10 ticks", fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUT, f"{tag}_light_charge.png"), dpi=125)
    plt.close(fig)

    print(f"b{i}: {ftag} ev{ev} TPC{c} eff={eff:.3f} wrongE={float(es[wrong].sum()):.0f} "
          f"dom={domL}({dom[0][1]:.0f}MeV) clusters={[L for L,_ in dom[:4]]}", flush=True)

print("[done] gallery:", OUT)
