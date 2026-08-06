"""Render failed-TPC displays with the CANONICAL M5p1.visualizationTools.plot_tpc_light_and_charge
(per-channel actual-vs-predicted light L0..59 / R0..59 + center charge z-y scatter), from the v0.4 dumps."""
import os, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.abspath(os.path.dirname(__file__))
BASE = os.path.abspath(os.path.join(HERE, "..", "CLMatching_AlphaRelease"))
sys.path.insert(0, BASE); sys.path.insert(1, os.path.join(BASE, "NewMLSection")); sys.path.insert(2, os.path.join(BASE, "M5p1"))
from M5p1 import visualizationTools as vt

CLIP = 60780.0
DUMPS = os.path.join(HERE, "v030v3_out", "dumps")
OUT = os.path.join(HERE, "bad_tpc_gallery"); os.makedirs(OUT, exist_ok=True)
TARGETS = [("0000001", 12, 38), ("0000002", 8, 61), ("0000002", 5, 26)]

# capture the figure that plot_tpc_light_and_charge draws (it ends in plt.show())
_save = {"path": None}
def _patched_show(*a, **k):
    if _save["path"]:
        plt.gcf().savefig(_save["path"], dpi=110, bbox_inches="tight")
plt.show = _patched_show

dt = np.dtype([('io_group', 'i4'), ('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('E', 'f8')])
for ftag, ev, tpc in TARGETS:
    d = np.load(os.path.join(DUMPS, f"{ftag}_ev{ev}.npz"), allow_pickle=True)
    x = np.asarray(d['x'], float); y = np.asarray(d['y'], float); z = np.asarray(d['z'], float)
    e = np.asarray(d['e'], float); ct = np.asarray(d['tpc'])
    h = np.zeros(len(x), dtype=dt)
    h['io_group'] = 2 * ct + 1; h['x'] = x; h['y'] = y; h['z'] = z; h['E'] = e     # io_group encodes TPC
    actual = np.asarray(d['full_light'], float)[tpc]
    predicted = np.minimum(np.asarray(d['base_image'], float)[tpc], CLIP)
    _save["path"] = os.path.join(OUT, f"vt_{ftag}_ev{ev}_tpc{tpc}_light_charge.png")
    vt.plot_tpc_light_and_charge(ev, tpc, h, actual, predicted=predicted)
    plt.close('all')
    print("saved", _save["path"], flush=True)
print("[done]")
