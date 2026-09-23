"""Synthetic self-test for v110_components / v110_redecide (no GPU, no flow)."""
import os, sys, numpy as np
HERE = os.path.abspath(os.path.dirname(__file__))
BASE = os.path.join(HERE, "..", "CLMatching_AlphaRelease")
for p in (BASE, os.path.join(BASE, "M5p1"), HERE):
    sys.path.insert(0, os.path.abspath(p))
from pulse_shapes import timeinterpolation
import v02_ordering_pipeline as vop
import v110_components as v110
from v110_redecide import run_redecide

rng = np.random.default_rng(11)
TMPL = np.load(os.path.join(BASE, "assets", "avg_pulse.npy")).astype(np.float64) / 999.0
NT, NCH, NL, GAIN = 1000, 120, 2, 200.0
def ti(arr, t): return timeinterpolation(arr, shift=float(t), baseline=0).astype(np.float64)
def pattern(center, width, peak):
    ch = np.arange(NCH); return peak * np.exp(-0.5 * ((ch - center) / width) ** 2)

def build(clusters, truth, true_gain=None):
    actual = np.full((NL, NCH, NT), 100.0); image_maps, info = {}, {}
    for lab, tpc, amps, E in clusters:
        img = np.outer(amps, TMPL); image_maps[(lab, tpc)] = img.astype(np.float32)
        actual[tpc] += (true_gain or {}).get(lab, 1.0) * ti(img, truth[lab])
        info[lab] = {"t0": float("nan"), "tpcs": [tpc], "sup": {tpc: np.flatnonzero(amps > 1.0)}, "type": "cluster", "energy": E}
    actual = np.clip(rng.normal(actual, np.sqrt(GAIN * np.maximum(actual, 100.0))), 0, None)
    return actual, 1.0 / (GAIN * np.maximum(actual, 100.0)), image_maps, info, np.zeros((NL, NCH), bool)

# --- pool: recovery + exactness ---
cl = [(1, 0, pattern(40, 6, 20000), 500.0), (2, 0, pattern(42, 6, 3000), 60.0), (3, 0, pattern(95, 5, 8000), 200.0), (4, 1, pattern(20, 6, 6000), 150.0)]
truth = {1: 300, 2: 520, 3: 150, 4: 700}
actual, w, im, info, veto = build(cl, truth)
tmap, st = v110.place_pool_v110([1, 2, 3, 4], np.zeros_like(actual), info, im, actual, w, ti, veto)
ok_pool = all(abs(tmap[l] - truth[l]) <= 1 for l in truth)
print("pool recovery:", ok_pool, "| components", st["n_components"], "edges", st["n_edges"])

# --- search_chunk incumbent mode ---
base = np.zeros_like(actual)
ts, diag = v110.search_chunk([1, 2], base, info, im, actual, w, ti, None, extra_cands={1: [300], 2: [300]}, incumbent={1: [300], 2: [300]})
print("incumbent mode: best", ts, "loss_best", round(diag["loss_best"], 1), "loss_incumbent(B piggybacked)", round(diag["loss_incumbent"], 1), "improves:", diag["loss_best"] < diag["loss_incumbent"])

# --- run_redecide on a fake namespace where B is parked at A's time ---
nh = 400
labels = np.repeat([1, 2, 3, 4], nh // 4); hit_tpc = np.where(labels == 4, 1, 0)
E = np.where(labels == 1, 5.0, np.where(labels == 2, 0.6, np.where(labels == 3, 2.0, 1.5)))
ts_hits = np.array([{1: 300.0, 2: 300.0, 3: 150.0, 4: 700.0}[l] for l in labels])   # B wrong (piggyback)
base_wrong = np.zeros_like(actual)
for l, t in {1: 300, 2: 300, 3: 150, 4: 700}.items():
    base_wrong[info[l]["tpcs"][0]] += ti(np.asarray(im[(l, info[l]["tpcs"][0])], np.float64), t)
ns = {"imageMaps": im, "baseImage": base_wrong.astype(np.float32), "fullLightWaveform": actual,
      "labels_global": labels, "hit_timestamps": ts_hits.astype(np.float32), "Eset": E, "hitTPCid": hit_tpc}
support_cache = {}; label_info = {l: {"type": "cluster"} for l in (1, 2, 3, 4)}
st_rd = run_redecide(ns, veto_mask=veto, support_cache=support_cache, label_info=label_info, ti=ti, pick=None,
                     e_min=5.0, truth_fn=lambda l: float(truth[l]))
new_t = {l: float(np.median(ns["hit_timestamps"][labels == l])) for l in (1, 2, 3, 4)}
print("redecide: participants", st_rd["n_participants"], "components", st_rd["n_components"], "moved", st_rd["n_moved"], "fixed/broke", st_rd["n_fixed"], st_rd["n_broke"], "| times", new_t)
# base consistency after redecide
reb = np.zeros_like(actual)
for l, t in new_t.items():
    reb[info[l]["tpcs"][0]] += ti(np.asarray(im[(l, info[l]["tpcs"][0])], np.float64), t)
print("base consistent after redecide:", np.allclose(np.asarray(ns["baseImage"], np.float64), reb, atol=1e-2))
print("ALL OK" if ok_pool and st_rd["n_fixed"] == 1 and st_rd["n_broke"] == 0 else "CHECK FAILED")
