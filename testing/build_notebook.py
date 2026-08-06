"""Assemble swap_playground.ipynb — a working sandbox over the v0.3.0 bad-TPC
failure examples with copy-pastable data-extraction helpers."""
import json, os
HERE = os.path.abspath(os.path.dirname(__file__))

def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(lines)}

def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": _src(lines)}

def _src(lines):
    text = "\n".join(lines)
    out = text.split("\n")
    return [l + "\n" for l in out[:-1]] + [out[-1]]

cells = []

cells.append(md(
"# Charge–light matcher: t0-swap playground",
"",
"A sandbox over the **v0.3.0** matcher's worst charge-TPCs (the D50 `--phase3 both",
"--force-coverage` run), pre-loaded with the pairwise-t0-**swap** failure examples so you",
"can look at event displays and prototype your own swap-detection algorithms.",
"",
"**Run cells 1–6 once** to load the data + helpers. After that, drop your own snippets in the",
"scratch cells at the bottom — everything you need is in scope: the loaded event `EV`, the",
"current target `T`, and the `Event` helper methods.",
"",
"### Environment",
"Use the NERSC module torch kernel so plotly + the M5p1 plotting tools import:",
"```bash",
"module load pytorch/2.8.0",
"cd /global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing",
"jupyter lab   # or attach this .ipynb to that kernel",
"```",
"(Data extraction + matplotlib work under any numpy env; only the 3D plotly displays need the",
"M5p1 tools.)",
"",
"### What's a 'swap'",
"When two interactions share one charge-TPC, the matcher sometimes gives each the *other's*",
"scintillation flash (t0). 91% of the non-shower wrong energy is this. A hit is **correct** iff",
"`|reco_t0 - truth_t0| <= 10 ticks (160 ns)`. The light peak of a cluster at t0 sits at tick",
"`t0 + 105`.",
))

cells.append(md("## 1 · Imports, paths, constants"))
cells.append(code(
"import os, sys, json, glob",
"import numpy as np",
"import matplotlib.pyplot as plt",
"",
"HERE = '/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing'",
"# fixed dumps (correct FINAL base_image); falls back to the old dir if absent",
"DUMPS = os.path.join(HERE, 'v030D50fix_out', 'dumps')",
"if not os.path.isdir(DUMPS): DUMPS = os.path.join(HERE, 'v030D50_out', 'dumps')",
"GALLERY = os.path.join(HERE, 'bad_tpc_gallery')          # pre-rendered displays + context",
"",
"# make the CLMatching light utilities importable (timeinterpolation, M5p1 plotting)",
"BASE = os.path.join(HERE, '..', 'CLMatching_AlphaRelease')",
"for p in (HERE, BASE, os.path.join(BASE, 'NewMLSection'), os.path.join(BASE, 'M5p1')):",
"    p = os.path.abspath(p)",
"    if p not in sys.path: sys.path.append(p)",
"from pulse_shapes import timeinterpolation           # shift a light template to a t0",
"",
"PEAK   = 105       # light peak sits at tick  t0 + PEAK",
"CUT    = 10.0      # correctness threshold (ticks)",
"CLIP   = 60780.0   # ADC saturation level",
"GAIN   = 200.0     # for Poisson variance:  var = GAIN * max(actual, 100)",
"print('paths set; dumps present:', len(glob.glob(os.path.join(DUMPS, '*.npz'))))",
))

cells.append(md(
"## 2 · The `Event` object — your data API",
"",
"`load_event(file, event)` returns an `Event` holding every per-hit array plus the full",
"actual/predicted light, and exposes the extraction helpers you'll build algorithms on.",
"",
"| method | returns |",
"|---|---|",
"| `EV.actual_light(tpc)` | `[120, 1000]` **actual** scintillation light for that charge-TPC |",
"| `EV.predicted_light(tpc)` | `[120, 1000]` matcher's **predicted** light (sum of placed clusters) |",
"| `EV.template(L, tpc)` | `[120, 1000]` cluster `L`'s per-channel light image (ML prediction), or `None` |",
"| `EV.place(L, tpc, t0)` | that template time-shifted to `t0` (peak at `t0+105`) |",
"| `EV.t0(L)` / `EV.truth(L)` | cluster `L`'s reco / true t0 |",
"| `EV.energy(L)` / `EV.hits(L)` | cluster energy / boolean hit mask |",
"| `EV.clusters_in_tpc(tpc)` | labels present in that TPC (energy-sorted) |",
"| `EV.flash_profile(tpc, t0)` | `(profile[120], sat_mask[120])` spatial pattern of the actual light at a flash |",
"| `EV.spatial_cos(L, tpc, t0)` | saturation-masked cosine of `L`'s template vs the actual flash at `t0` |",
"",
"Note: the dumps keep per-cluster templates only for the ~8 worst clusters per event (to save",
"disk), so `EV.template(L, tpc)` is `None` for most small clusters — but it **is** present for the",
"dominant swap clusters in the targets below. `actual_light`/`predicted_light` are always full.",
))
cells.append(code(
"class Event:",
"    def __init__(self, npz, file, event):",
"        self.file, self.event = file, event",
"        for k in ('x','y','z','e','tpc','labels','ts_final','ts_p1','truth_t0'):",
"            setattr(self, k, np.asarray(npz[k]))",
"        self.full_light = np.asarray(npz['full_light'], np.float64)   # [70,120,1000] actual",
"        self.base_image = np.asarray(npz['base_image'], np.float64)   # [70,120,1000] predicted",
"        self.templates = {}",
"        for k in npz.files:",
"            if k.startswith('img_'):",
"                _, L, c = k.split('_'); self.templates[(int(L), int(c))] = np.asarray(npz[k], np.float64)",
"        self.lab_rows = {r['label']: r for r in json.loads(str(npz['lab_rows']))}",
"    # --- light ---",
"    def actual_light(self, tpc):    return self.full_light[tpc]         # [120,1000]",
"    def predicted_light(self, tpc): return self.base_image[tpc]         # [120,1000]",
"    def template(self, L, tpc):     return self.templates.get((int(L), int(tpc)))",
"    def place(self, L, tpc, t0):",
"        P = self.template(L, tpc)",
"        return None if P is None else timeinterpolation(P, shift=float(t0), baseline=0)",
"    # --- clusters ---",
"    def hits(self, L):   return self.labels == L",
"    def energy(self, L): return float(self.e[self.labels == L].sum())",
"    def t0(self, L):",
"        m = (self.labels == L) & np.isfinite(self.ts_final)",
"        return float(np.median(self.ts_final[m])) if m.any() else None",
"    def truth(self, L):",
"        m = (self.labels == L) & np.isfinite(self.truth_t0)",
"        return float(np.median(self.truth_t0[m])) if m.any() else None",
"    def clusters_in_tpc(self, tpc):",
"        ls = np.unique(self.labels[(self.tpc == tpc) & (self.labels >= 0)])",
"        return sorted((int(L) for L in ls), key=lambda L: -self.energy(L))",
"    # --- light spatial pattern + discriminator ---",
"    def flash_profile(self, tpc, t0, win=4):",
"        a = self.full_light[tpc]; t = int(round(t0)) + PEAK",
"        lo, hi = max(0, t - win), min(a.shape[1], t + win + 1)",
"        col = a[:, lo:hi]",
"        return col.sum(1), col.max(1) >= 0.98 * CLIP        # profile[120], saturated mask",
"    def spatial_cos(self, L, tpc, t0, win=4):",
"        P = self.template(L, tpc)",
"        if P is None: return None",
"        tmpl = P.sum(1)                                     # cluster spatial template",
"        prof, sat = self.flash_profile(tpc, t0, win)",
"        a, b = tmpl[~sat], prof[~sat]",
"        na, nb = np.linalg.norm(a), np.linalg.norm(b)",
"        return float(a @ b / (na * nb)) if na and nb else 0.0",
"",
"def load_event(file, event):",
"    npz = np.load(os.path.join(DUMPS, f'{file}_ev{event}.npz'), allow_pickle=True)",
"    return Event(npz, file, event)",
"print('Event class ready')",
))

cells.append(md(
"## 3 · The failure targets (pairwise-swap examples)",
"",
"Ten of the worst non-shower TPCs, each with its dominant wrong cluster(s). `swap_pair` names two",
"clusters that (should) exchange t0s. `b0` (TPC 5) has **both** partners' templates — the cleanest",
"worked example. Pre-rendered 3D displays live in `bad_tpc_gallery/b{i}_*`.",
))
# pull the targets from _all_context.json to keep them exact
ctx = json.load(open(os.path.join(HERE, "bad_tpc_gallery", "_all_context.json")))
tg = []
for c in ctx:
    doms = c["dominant_wrong_clusters"][:2]
    pair = [d["label"] for d in doms]
    tg.append({"i": c["idx"], "file": c["file"], "event": c["event"], "tpc": c["tpc"],
               "eff": c["eff"], "wrongE": c["wrong_E_real_cluster"],
               "swap_pair": pair,
               "clusters": [{"L": d["label"], "E": d["cluster_E"], "placed": d["placed_t0"],
                             "true": d["true_t0"], "type": d["type"]} for d in doms]})
cells.append(code(
"TARGETS = " + json.dumps(tg, indent=1),
"",
"print(f'{len(TARGETS)} targets:')",
"for t in TARGETS:",
"    cs = ' / '.join(f\"L{c['L']}({c['E']:.0f}MeV placed {c['placed']}->true {c['true']})\" for c in t['clusters'])",
"    print(f\"  b{t['i']}: {t['file']} ev{t['event']} TPC{t['tpc']:<3} eff={t['eff']:.2f} wrongE={t['wrongE']:.0f}  | {cs}\")",
))

cells.append(md(
"## 4 · Pick a target and load it",
"",
"Change `PICK` to any index above. `b0` (TPC 5) is the recommended starting point (both swap",
"templates available).",
))
cells.append(code(
"PICK = 0",
"T   = TARGETS[PICK]",
"EV  = load_event(T['file'], T['event'])",
"TPC = T['tpc']",
"print(f\"loaded {T['file']} ev{T['event']}  TPC {TPC}  (eff={T['eff']:.2f})\")",
"print('clusters in this TPC (energy-sorted):', EV.clusters_in_tpc(TPC)[:12])",
"for c in T['clusters']:",
"    L = c['L']",
"    print(f\"  L{L}: E={EV.energy(L):.0f}MeV  reco_t0={EV.t0(L):.1f}  true_t0={EV.truth(L):.1f}  \"",
"          f\"template={'yes' if EV.template(L, TPC) is not None else 'NO'}\")",
))

cells.append(md(
"## 5 · Extract the actual & predicted light (the building blocks)",
"",
"This is the kind of snippet you asked for — grab the current actual vs predicted light for the",
"TPC, and any cluster's own template.",
))
cells.append(code(
"act  = EV.actual_light(TPC)      # [120, 1000]  what the light detectors saw",
"pred = EV.predicted_light(TPC)   # [120, 1000]  what the matcher's assignment predicts",
"print('actual/predicted light shape:', act.shape, pred.shape)",
"",
"# the two swap clusters and their own light templates (unshifted ML images)",
"A, B = T['swap_pair']",
"tA, tB = EV.t0(A), EV.t0(B)",
"PA = EV.place(A, TPC, tA)   # cluster A's template placed at its current reco t0  -> [120,1000]",
"PB = EV.place(B, TPC, tB)",
"print(f'cluster {A}: E={EV.energy(A):.0f}MeV  reco {tA:.0f} / true {EV.truth(A):.0f}')",
"print(f'cluster {B}: E={EV.energy(B):.0f}MeV  reco {tB:.0f} / true {EV.truth(B):.0f}')",
"",
"# quick look: light summed over channels vs tick, with the two flashes marked",
"fig, ax = plt.subplots(figsize=(11,3.2))",
"ax.plot(act.sum(0),  label='actual (sum ch)',    lw=1.2)",
"ax.plot(pred.sum(0), label='predicted (sum ch)', lw=1.2)",
"for t0, lab, col in [(tA, f'{A}@{tA:.0f}', 'C3'), (tB, f'{B}@{tB:.0f}', 'C2')]:",
"    ax.axvline(t0 + PEAK, color=col, ls='--', lw=1.2, label=f'reco {lab}')",
"for L, col in [(A, 'C3'), (B, 'C2')]:",
"    tr = EV.truth(L)",
"    if tr is not None: ax.axvline(tr + PEAK, color=col, ls=':', lw=1.6, label=f'TRUE {L}@{tr:.0f}')",
"ax.set_xlabel('tick'); ax.set_ylabel('ADC'); ax.legend(fontsize=8, ncol=2); ax.set_xlim(0, 999)",
"plt.tight_layout(); plt.show()",
))

cells.append(md(
"## 6 · Worked example — the masked spatial-cosine discriminator",
"",
"The physical signal that separates the correct from the swapped assignment: each flash's",
"**per-channel spatial pattern** is set by the interaction's *position*, so a cluster's template",
"matches the actual light at its **true** flash better than at its **placed** (wrong) flash —",
"with **saturated channels masked** (they carry no spatial info). This is what `swap_fix.py` uses.",
))
cells.append(code(
"for L in (A, B):",
"    cp = EV.spatial_cos(L, TPC, EV.t0(L))      # cosine at its current (placed) flash",
"    ct = EV.spatial_cos(L, TPC, EV.truth(L))   # cosine at its true flash",
"    print(f'cluster {L}:  cos@placed({EV.t0(L):.0f}) = {cp:.3f}   cos@true({EV.truth(L):.0f}) = {ct:.3f}'",
"          f'   -> prefers {\"TRUE\" if ct > cp else \"placed\"}')",
"",
"# generalize: score cluster A against EVERY flash occupied in this TPC",
"occupied = sorted({round(EV.t0(L)) for L in EV.clusters_in_tpc(TPC) if EV.t0(L) is not None})",
"print(f'\\ncluster {A} spatial-cos vs each occupied flash t0 in TPC {TPC}:')",
"for t0 in occupied:",
"    print(f'   t0={t0:>4}: cos = {EV.spatial_cos(A, TPC, t0):.3f}')",
))

cells.append(md(
"## 7 · Event displays",
"",
"Pre-rendered interactive 3D + light/charge for this target are in `bad_tpc_gallery/`:",
"open the `.html` files (rotatable) or view the `.png`. To (re)render inline for any TPC, run the",
"cell below — it uses the same M5p1 tools as the gallery (needs the pytorch/2.8.0 kernel).",
))
cells.append(code(
"stem = f\"b{T['i']}_{T['file']}_ev{T['event']}_tpc{TPC}\"",
"print('pre-rendered files:')",
"for k in ('wrong3d', 'reco3d', 'truth3d', 'light_charge'):",
"    for ext in ('html', 'png'):",
"        p = os.path.join(GALLERY, f'{stem}_{k}.{ext}')",
"        if os.path.exists(p): print('  ', p)",
"",
"# --- inline 3D (RED = wrong hits) ---",
"try:",
"    from M5p1 import plottingTools as pt",
"    from M5p1 import truth_plotting as tp",
"    m = EV.tpc == TPC",
"    tp.plot_wrong_hits_3d(int(TPC), EV.truth_t0, hit_timestamps=EV.ts_final, hitTPCid=EV.tpc,",
"                          xset=EV.x, yset=EV.y, zset=EV.z, Eset=EV.e, residual_cut_ticks=CUT,",
"                          marker_size=3.5, wrong_color='red', correct_color='lightgray', show=True)",
"except Exception as ex:",
"    print('inline 3D needs the pytorch/2.8.0 kernel + M5p1 tools; open the gallery .html instead.', ex)",
))

cells.append(md(
"## 8 · Run / experiment with the swap-fix algorithm",
"",
"`swap_fix.run_swap_correction` implements the mutual-confirmation pairwise swap (with margin",
"whitening). Here it runs on just this target's two clusters so you can watch it fire and inspect",
"the audit. Tune `min_cos_gain`, `whiten`, `min_energy`, or swap in your **own** `pair_gains`.",
))
cells.append(code(
"import importlib, swap_fix; importlib.reload(swap_fix)",
"",
"def truth_fn(L): return EV.truth(L)",
"",
"# build a minimal namespace with just the swap pair (so the run is easy to reason about)",
"A, B = T['swap_pair']",
"ns = dict(",
"    imageMaps    = {(A, TPC): EV.template(A, TPC), (B, TPC): EV.template(B, TPC)},",
"    fullLightWaveform = EV.full_light,",
"    baseImage    = EV.base_image.copy(),",
"    labels_global = EV.labels,",
"    hit_timestamps = EV.ts_final.astype(np.float32).copy(),",
"    Eset         = EV.e,",
")",
"# NOTE: whiten=True subtracts the TPC's MEAN flash; with only 2 clusters in `ns` that mean is",
"# degenerate, so the isolated demo uses whiten=False (raw cosine, which fires on a clean pair).",
"# In the real pipeline the TPC has many clusters, so the mean flash is proper and whiten=True is used.",
"st = swap_fix.run_swap_correction(ns, hit_tpc=EV.tpc, cluster_to_tpcs={A: [TPC], B: [TPC]},",
"                                  min_energy=25.0, min_cos_gain=0.03, whiten=False,",
"                                  truth_fn=truth_fn, verbose=True)",
"print('stats:', {k: v for k, v in st.items() if k != 'audit'})",
"for r in st['audit']: print('  ', r)",
"# after the swap, ns['hit_timestamps'] is updated in place; check per-hit correctness recovered:",
"ts2 = np.asarray(ns['hit_timestamps'], np.float64)",
"for L in (A, B):",
"    m = EV.hits(L); tr = EV.truth(L)",
"    print(f\"  L{L}: t0 {EV.t0(L):.0f} -> {np.median(ts2[m]):.0f}  (true {tr:.0f})  \"",
"          f\"{'CORRECT' if abs(np.median(ts2[m]) - tr) <= CUT else 'still wrong'}\")",
))

cells.append(md(
"## 9 · Your scratch space",
"",
"Everything is in scope: `EV`, `T`, `TPC`, `A`, `B`, and all the `Event` helpers. Some snippets to",
"start from (uncomment / adapt):",
))
cells.append(code(
"# --- current actual vs predicted light for this TPC ---",
"# act, pred = EV.actual_light(TPC), EV.predicted_light(TPC)",
"",
"# --- residual: what light is NOT explained by the current assignment ---",
"# residual = EV.actual_light(TPC) - EV.predicted_light(TPC)",
"",
"# --- your own discriminator: score a cluster at an arbitrary t0 ---",
"# def my_score(L, t0):",
"#     tmpl = EV.template(L, TPC).sum(1)",
"#     prof, sat = EV.flash_profile(TPC, t0)",
"#     ...  # e.g. whiten by the mean flash, use only channels where A and B templates differ, etc.",
"",
"# --- try a candidate swap and see if it improves per-hit correctness ---",
"# def correctness(ts):",
"#     m = EV.tpc == TPC",
"#     ok = m & np.isfinite(EV.truth_t0) & (np.abs(ts - EV.truth_t0) <= CUT)",
"#     return float(EV.e[ok].sum() / EV.e[m & np.isfinite(EV.truth_t0)].sum())",
"# ts2 = EV.ts_final.copy(); ts2[EV.hits(A)], ts2[EV.hits(B)] = EV.t0(B), EV.t0(A)",
"# print('correct fraction  now:', correctness(EV.ts_final), ' swapped:', correctness(ts2))",
"",
"",
))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(HERE, "swap_playground.ipynb")
json.dump(nb, open(out, "w"), indent=1)
print("wrote", out, f"({len(cells)} cells)")
