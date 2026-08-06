"""Non-shower failure displays: TPCs whose wrong energy is carried by TRACK
labels (policed final pipeline). Renders nt{i}_type1/2/3/4 into v02_displays/."""
import os
import sys

_HERE = os.path.abspath(os.path.dirname(__file__))
BASE = os.path.abspath(os.path.join(_HERE, "..", "CLMatching_AlphaRelease"))
sys.path.insert(0, BASE)
sys.path.insert(1, os.path.join(BASE, "NewMLSection"))
sys.path.insert(2, os.path.join(BASE, "M5p1"))

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plotly.graph_objects as go

_png_target = {"path": None}


def _patched_show(self, *a, **k):
    if _png_target["path"]:
        self.write_image(_png_target["path"], width=1100, height=850, scale=1)


go.Figure.show = _patched_show

from ML_NDfull_perceiver import load_tpc_geometries, DEFAULT_TPC_YAML
from visualizationTools import plot_tpc_light_and_charge
import plottingTools as pt
from M5p1 import truth_plotting as tpmod

DUMPS = os.path.join(_HERE, "v02dump2_out", "dumps")
OUT = os.path.join(_HERE, "v02_displays")
CUT = 10.0
geom_map = load_tpc_geometries(DEFAULT_TPC_YAML)

EXAMPLES = [
    ("0000002_ev3", 50, "pure TRACK failure (546/547 MeV wrong is type=track)"),
    ("0000001_ev7", 27, "track-dominated failure (505/565 MeV track)"),
    ("0000001_ev3", 7, "track-dominated failure (491/759 MeV track)"),
]

hits_cache = {}
for i, (dtag, tpc, note) in enumerate(EXAMPLES):
    d = np.load(os.path.join(DUMPS, f"{dtag}.npz"), allow_pickle=True)
    ev = int(d["event"])
    ftag = dtag.split("_ev")[0]
    xs = np.asarray(d["x"], np.float64)
    ys = np.asarray(d["y"], np.float64)
    zs = np.asarray(d["z"], np.float64)
    es = np.asarray(d["e"], np.float64)
    tpcs_h = np.asarray(d["tpc"], int)
    ts = np.asarray(d["ts_final"], np.float64)
    tt = np.asarray(d["truth_t0"], np.float64)
    m = tpcs_h == tpc
    title = f"nt{i} [NON-SHOWER] {ftag} ev{ev} TPC {tpc} | {note}"

    fp_src = str(d["src_file"])
    if fp_src not in hits_cache:
        with h5py.File(fp_src, "r") as h5:
            hits_cache[fp_src] = h5["charge/calib_prompt_hits/data"][...]
    hits_evt = hits_cache[fp_src][d["hit_refs"]]
    plot_tpc_light_and_charge(ev_id=ev, TPCid=tpc, hits_full_ev=hits_evt,
                              actual=np.asarray(d["full_light"], np.float32)[tpc],
                              predicted=np.asarray(d["base_image"], np.float32)[tpc],
                              geom_map=geom_map, individualScaling=False)
    for num in plt.get_fignums():
        f2 = plt.figure(num)
        f2.suptitle(title, y=0.995, fontsize=11)
        f2.savefig(os.path.join(OUT, f"nt{i}_type1_light_charge.png"),
                   dpi=110, bbox_inches="tight")
        plt.close(f2)

    rt_safe = np.where(np.isfinite(ts) & (ts >= 0), ts, -1.0)
    _png_target["path"] = os.path.join(OUT, f"nt{i}_type2_reco3d.png")
    pt.plot_3d_clusters_with_t0(
        x=xs[m], y=ys[m], z=zs[m], labels=pt.group_hits_by_time(rt_safe[m], time_window=10),
        t0s=rt_safe[m], energies=es[m], title=f"{title} | FINAL reco t0",
        save_path=os.path.join(OUT, f"nt{i}_type2_reco3d.html"))
    tt_safe = np.where(np.isfinite(tt), tt, -1.0)
    _png_target["path"] = os.path.join(OUT, f"nt{i}_type3_truth3d.png")
    pt.plot_3d_clusters_with_t0(
        x=xs[m], y=ys[m], z=zs[m], labels=pt.group_hits_by_time(tt_safe[m], time_window=10),
        t0s=tt_safe[m], energies=es[m], title=f"{title} | TRUTH t0",
        save_path=os.path.join(OUT, f"nt{i}_type3_truth3d.html"))
    _png_target["path"] = os.path.join(OUT, f"nt{i}_type4_wrong3d.png")
    tpmod.plot_wrong_hits_3d(tpc, tt, hit_timestamps=ts, hitTPCid=tpcs_h,
                             xset=xs, yset=ys, zset=zs, Eset=es,
                             residual_cut_ticks=CUT,
                             save_path=os.path.join(OUT, f"nt{i}_type4_wrong3d.html"),
                             marker_size=3.5, wrong_color="red",
                             correct_color="lightgray", show=True)
    _png_target["path"] = None
    print(f"nt{i}: {ftag} ev{ev} TPC {tpc} done", flush=True)

print("non-shower displays complete")
