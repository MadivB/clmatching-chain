"""Per-TPC event displays for the demo_html presentation event.

For each target (file, event, TPC) — default: the 4 TPCs shown in
testing/demo_html (file 0000001, ev5, TPCs 57/59/66/68), dumps from the
v1.0 canonical run (v138FS_out) — produce the demo_iter-style set:

  demo_{file}_ev{ev}_tpc{c}_light_charge.png   per-channel actual (blue) vs
        predicted/baseImage (orange) light + center charge z-y scatter
        (M5p1/visualizationTools.plot_tpc_light_and_charge)
  demo_..._reco3d.html    hits colored by reco (ts_final) t0 family
  demo_..._truth3d.html   hits colored by truth_t0 family
  demo_..._wrong3d.html (+.png)   RED = |ts_final - truth_t0| > 10 ticks

Usage:  python demo_tpc_displays.py ["file:event:tpc,..."]
Run under:  module load pytorch/2.8.0   (plotly + kaleido live in that user-site).
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.abspath(os.path.dirname(__file__))
BASE = os.path.abspath(os.path.join(_HERE, "..", "CLMatching_AlphaRelease"))
sys.path.insert(0, BASE)
sys.path.insert(1, os.path.join(BASE, "M5p1"))

import plotly.graph_objects as go
_png = {"path": None}
def _show_to_png(self, *a, **k):
    if _png["path"]:
        self.write_image(_png["path"], width=1150, height=880, scale=2)
go.Figure.show = _show_to_png

from M5p1 import plottingTools as pt
from M5p1 import truth_plotting as tp
from M5p1 import visualizationTools as vt

DUMPS = os.environ.get("DEMO_DUMPS", os.path.join(_HERE, "v138FS_out", "full_dumps"))
OUT = os.environ.get("DEMO_OUT", os.path.join(_HERE, "demo_html"))
os.makedirs(OUT, exist_ok=True)
CUT, TW = 10.0, 5

if len(sys.argv) > 1 and ":" in sys.argv[1]:
    TARGETS = []
    for tok in sys.argv[1].split(","):
        f, e, c = tok.split(":")
        TARGETS.append((f, int(e), int(c)))
else:
    TARGETS = [("0000001", 5, c) for c in (57, 59, 66, 68)]

_dcache = {}
def load(ftag, ev):
    k = (ftag, ev)
    if k not in _dcache:
        _dcache[k] = np.load(os.path.join(DUMPS, f"{ftag}_ev{ev}.npz"), allow_pickle=True)
    return _dcache[k]

for ftag, ev, c in TARGETS:
    d = load(ftag, ev)
    xs = np.asarray(d["x"], np.float64); ys = np.asarray(d["y"], np.float64)
    zs = np.asarray(d["z"], np.float64); es = np.asarray(d["e"], np.float64)
    tpc = np.asarray(d["tpc"]); tsf = np.asarray(d["ts_final"], np.float64)
    tr = np.asarray(d["truth_t0"], np.float64); iog = np.asarray(d["io_group"])
    m = tpc == c
    tpcE = float(es[m].sum())
    ok = m & np.isfinite(tr)
    wrong = ok & np.isfinite(tsf) & (tsf >= 0) & (np.abs(tsf - tr) > CUT)
    wrongE = float(es[wrong].sum())
    eff = 1.0 - wrongE / max(float(es[ok].sum()), 1e-9)
    tag = f"demo_{ftag}_ev{ev}_tpc{c}"
    ttl = f"{ftag} ev{ev} TPC{c} | E={tpcE:.0f} MeV eff={eff:.3f} wrongE={wrongE:.0f} MeV"
    print(f"[{tag}] hits {int(m.sum())}  {ttl}", flush=True)

    # ---------- type 1: per-channel light + charge (actual vs baseImage) ----------
    fl = np.asarray(d["full_light"], np.float64)[c]     # (120,1000) actual
    bi = np.asarray(d["base_image"], np.float64)[c]     # (120,1000) predicted
    # v1.2.2: draw the prediction the way the loss sees it, clipped at each channel's rail
    # (int16 2**15-1 minus the dynamic baseline, stored in the dump as sat_ceiling).  Older
    # dumps: channels that touch the rail reveal it as their maximum; the rest keep 60780.
    if "sat_ceiling" in d.files:
        ceil = np.asarray(d["sat_ceiling"], np.float64)[c]
    else:
        mx = fl.max(axis=1)
        ceil = np.where(mx >= 60700.0, mx, 60780.0)
    bi = np.minimum(bi, ceil[:, None])
    hits = np.zeros(xs.size, dtype=[("io_group", "i4"), ("x", "f4"), ("y", "f4"),
                                    ("z", "f4"), ("E", "f4")])
    hits["io_group"] = iog; hits["x"] = xs; hits["y"] = ys; hits["z"] = zs; hits["E"] = es
    fn1 = os.path.join(OUT, f"{tag}_light_charge.png")
    plt.show = lambda *a, **k: (plt.savefig(fn1, dpi=95, bbox_inches="tight"),
                                plt.close("all"))
    vt.plot_tpc_light_and_charge(ev_id=ev, TPCid=c, hits_full_ev=hits,
                                 actual=fl, predicted=bi)
    print("  [light_charge]", fn1, flush=True)

    # ---------- types 2+3: reco / truth 3D ----------
    tt_safe = np.where(np.isfinite(tr), tr, -1.0)
    rt_safe = np.where(np.isfinite(tsf), tsf, -1.0)
    _png["path"] = os.path.join(OUT, f"{tag}_reco3d.png")
    pt.plot_3d_clusters_with_t0(x=xs[m], y=ys[m], z=zs[m],
                                labels=pt.group_hits_by_time(rt_safe[m], time_window=TW),
                                t0s=rt_safe[m], energies=es[m],
                                title=f"RECO t0 (v1.2.1) | {ttl}",
                                save_path=os.path.join(OUT, f"{tag}_reco3d.html"))
    print("  [reco3d]", f"{tag}_reco3d.html", flush=True)
    _png["path"] = os.path.join(OUT, f"{tag}_truth3d.png")
    pt.plot_3d_clusters_with_t0(x=xs[m], y=ys[m], z=zs[m],
                                labels=pt.group_hits_by_time(tt_safe[m], time_window=TW),
                                t0s=tt_safe[m], energies=es[m],
                                title=f"TRUTH t0 | {ttl}",
                                save_path=os.path.join(OUT, f"{tag}_truth3d.html"))
    print("  [truth3d]", f"{tag}_truth3d.html", flush=True)

    # ---------- type 4: wrong-hits 3D ----------
    _png["path"] = os.path.join(OUT, f"{tag}_wrong3d.png")
    tp.plot_wrong_hits_3d(int(c), tr, hit_timestamps=tsf, hitTPCid=tpc,
                          xset=xs, yset=ys, zset=zs, Eset=es, residual_cut_ticks=CUT,
                          save_path=os.path.join(OUT, f"{tag}_wrong3d.html"),
                          marker_size=3.5, wrong_color="red",
                          correct_color="lightgray", show=True)
    _png["path"] = None
    print("  [wrong3d]", f"{tag}_wrong3d.html", flush=True)

print("[done] ->", OUT, flush=True)
