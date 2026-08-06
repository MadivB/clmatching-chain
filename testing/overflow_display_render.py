"""Render the 3 skill-format displays per overflow example (CPU, from dumps)."""
import json
import os
import sys

_HERE = os.path.abspath(os.path.dirname(__file__))
BASE_DIR = os.path.abspath(os.path.join(_HERE, "..", "CLMatching_AlphaRelease"))
sys.path.insert(0, BASE_DIR)
sys.path.insert(1, os.path.join(BASE_DIR, "NewMLSection"))
sys.path.insert(2, os.path.join(BASE_DIR, "M5p1"))

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plotly.graph_objects as go

_orig_write_html = go.Figure.write_html
_png_target = {"path": None}


def _patched_show(self, *a, **k):
    if _png_target["path"]:
        try:
            self.write_image(_png_target["path"], width=1100, height=850, scale=1)
            print(f"  wrote {_png_target['path']}")
        except Exception as exc:
            print(f"  (no PNG for 3D plot: {exc}; use the HTML)")
    return None


go.Figure.show = _patched_show

from ML_NDfull_perceiver import load_tpc_geometries, DEFAULT_TPC_YAML
from visualizationTools import plot_tpc_light_and_charge
import plottingTools as pt

geom_map = load_tpc_geometries(DEFAULT_TPC_YAML)
DUMP = os.path.join(_HERE, "display_dump")
OUT = os.path.join(_HERE, "overflow_displays")
os.makedirs(OUT, exist_ok=True)

EXAMPLES = [   # (tag, label, max_ov)
    ("0000002_ev9", 105, 211),
    ("0000001_ev11", 102, 98),
    ("0000002_ev9", 99, 50),
    ("0000003_ev5", 124, 25),
]

hits_cache = {}
for i, (tag, label, ov) in enumerate(EXAMPLES):
    d = np.load(os.path.join(DUMP, f"{tag}.npz"), allow_pickle=True)
    label_tpcs = json.loads(str(d["label_tpcs"]))
    tpc = int(label_tpcs[str(label)][0])
    ev = int(d["event"])
    fp = str(d["src_file"])

    if fp not in hits_cache:
        with h5py.File(fp, "r") as h5:
            hits_cache[fp] = h5["charge/calib_prompt_hits/data"][...]
    hits_evt = hits_cache[fp][d["hit_refs"]]

    reco = np.asarray(d["hit_t0"], dtype=np.float64)
    truth = np.asarray(d["truth_t0"], dtype=np.float64)
    tmask = np.asarray(d["tpc"]) == tpc
    reco_safe = np.where(np.isfinite(reco) & (reco >= 0), reco, -1.0)
    truth_safe = np.where(np.isfinite(truth), truth, -1.0)
    lab_t0 = float(np.median(reco[np.asarray(d["labels"]) == label]))

    print(f"[{i}] {tag} label {label} TPC {tpc} t0={lab_t0:.1f} max_ov={ov} "
          f"({int(tmask.sum())} hits in TPC)")

    # type1: 2D actual vs predicted light + charge
    plot_tpc_light_and_charge(
        ev_id=ev, TPCid=tpc, hits_full_ev=hits_evt,
        actual=d["full_light"][tpc], predicted=d["base_image"][tpc],
        geom_map=geom_map, individualScaling=False)
    for num in plt.get_fignums():
        f = plt.figure(num)
        f.suptitle(f"ex{i}: CORRECT placement label {label} at t0={lab_t0:.0f} "
                   f"(truth-correct) but {ov} sigma overflow | {tag} TPC {tpc}", y=0.995, fontsize=11)
        f.savefig(os.path.join(OUT, f"ex{i}_type1_light_charge.png"), dpi=110, bbox_inches="tight")
        plt.close(f)
    print(f"  wrote ex{i}_type1_light_charge.png")

    # type2: reco 3D (hits grouped by phase-1 reco t0)
    _png_target["path"] = os.path.join(OUT, f"ex{i}_type2_reco3d.png")
    g = pt.group_hits_by_time(reco_safe[tmask], time_window=10)
    pt.plot_3d_clusters_with_t0(
        x=np.asarray(d["x"])[tmask], y=np.asarray(d["y"])[tmask], z=np.asarray(d["z"])[tmask],
        labels=g, t0s=reco_safe[tmask], energies=np.asarray(d["e"], dtype=np.float64)[tmask],
        title=f"ex{i} reco t0 | {tag} TPC {tpc} (label {label} at {lab_t0:.0f})",
        save_path=os.path.join(OUT, f"ex{i}_type2_reco3d.html"))

    # type3: truth 3D
    _png_target["path"] = os.path.join(OUT, f"ex{i}_type3_truth3d.png")
    gt = pt.group_hits_by_time(truth_safe[tmask], time_window=10)
    pt.plot_3d_clusters_with_t0(
        x=np.asarray(d["x"])[tmask], y=np.asarray(d["y"])[tmask], z=np.asarray(d["z"])[tmask],
        labels=gt, t0s=truth_safe[tmask], energies=np.asarray(d["e"], dtype=np.float64)[tmask],
        title=f"ex{i} TRUTH t0 | {tag} TPC {tpc}",
        save_path=os.path.join(OUT, f"ex{i}_type3_truth3d.html"))
    _png_target["path"] = None

print("done")
