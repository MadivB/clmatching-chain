"""Phase-1 failure gallery (post-rescue): where are we still failing?

For each failure example: fail{i}_channels.png (Poisson sigma),
fail{i}_channels_varmodel.png (ML variance model sigma), and the
type1/2/3 event displays.
"""
import glob
import json
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
from matplotlib.ticker import MultipleLocator

import plotly.graph_objects as go
_png_target = {"path": None}


def _patched_show(self, *a, **k):
    if _png_target["path"]:
        try:
            self.write_image(_png_target["path"], width=1100, height=850, scale=1)
        except Exception as exc:
            print(f"  (no 3D PNG: {exc})")
    return None


go.Figure.show = _patched_show

from ML_NDfull_perceiver import load_tpc_geometries, DEFAULT_TPC_YAML
from visualizationTools import plot_tpc_light_and_charge
import plottingTools as pt
from var_prediction.inference_ndfl_v2 import load_model as load_var_model, predict as predict_var

VAR_CKPT = ("/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/NewMLSection/var_prediction/"
            "runs/var_run_perceiver_aligned_0001000_v2/best_model.pt")
OUT = os.path.join(_HERE, "overflow_displays")
DUMP = os.path.join(_HERE, "display_dump")
PEAK = 105
GAIN = 200.0
EXCESS_GATE = 500.0

geom_map = load_tpc_geometries(DEFAULT_TPC_YAML)
var_model, _ = load_var_model(VAR_CKPT, device="cpu", waveform_len=1000, num_channels=120)
var_cache = {}


def get_var(dtag, tpc, full_light):
    key = (dtag, tpc)
    if key not in var_cache:
        v = predict_var(var_model, full_light[tpc][None, :, :], batch_size=8,
                        input_scale=1e-3, target_scale=1e-3, device="cpu",
                        return_variance=True, min_sigma=1.0).astype(np.float64)[0]
        var_cache[key] = np.maximum(v, 1.0)
    return var_cache[key]


rows = []
for fp in sorted(glob.glob(os.path.join(_HERE, "overflow_study_out", "rows_worker*.jsonl"))):
    with open(fp) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))


def get_row(ftag, ev, lab):
    return next(r for r in rows if ftag in r["file"] and r["event"] == ev and r["label"] == lab)


FAILS = [
    (0, "0000002", 6, 141, "0000002_ev6", "NEVER FLAGGED by trigger"),
    (1, "0000001", 11, 96, "0000001_ev11", "NEVER FLAGGED by trigger"),
    (2, "0000001", 12, 201, "0000001_ev12", "NEVER FLAGGED by trigger"),
    (3, "0000002", 7, 50, "0000002_ev7", "flagged, Stage A FAILED to fix"),
    (4, "0000001", 12, 210, "0000001_ev12", "Stage B detached a part, parent STILL WRONG"),
    (5, "0000002", 6, 24, "0000002_ev6", "Stage-B all-stay, still wrong"),
    (6, "0000002", 7, 95, "0000002_ev7", "Stage-B all-stay, still wrong"),
]


def ch_key(c):
    return f"L{c}" if c < 60 else f"R{c - 60}"


def channels_figure(i, r, d, dtag, note, mode):
    """mode: 'pois' or 'model'."""
    t0 = float(r["t0"])
    truth = r["truth_t0"]
    tick = int(round(t0)) + PEAK
    full_light = np.asarray(d["full_light"], np.float32)

    best = None
    for td in r["tpc_channels"]:
        o = np.asarray(td["own"], float)
        b = np.asarray(td["base"], float)
        a = np.asarray(td["actual"], float)
        if mode == "pois":
            var_tick = GAIN * np.maximum(a, 100.0)
        else:
            var_tick = get_var(dtag, int(td["tpc"]), full_light)[:, tick]
        ov = (o + b - a) / np.sqrt(var_tick)
        sup = np.asarray(td["support"], int)
        if sup.size == 0:
            continue
        m = float(np.max(ov[sup]))
        if best is None or m > best[0]:
            best = (m, td, o, b, a, ov, sup, var_tick)
    max_ov, td, o, b, a, ov, sup, var_tick = best
    tpc = int(td["tpc"])
    k_ch = int(sup[np.argmax(ov[sup])])
    keys = [ch_key(c) for c in sup]
    x = np.arange(sup.size)
    gate_ok = (o + b - a)[sup] >= EXCESS_GATE
    ymax, ymin, loc = (4.0, -2.0, 0.5) if mode == "pois" else (10.0, -5.0, 1.0)
    sig_name = "sigma_Poisson (g=200)" if mode == "pois" else "sigma_MODEL"

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13, 11),
                                        gridspec_kw={"height_ratios": [1.1, 1.0, 1.1]})
    fig.suptitle(f"fail{i} [{sig_name}]: label {r['label']} ({dtag}) TPC {tpc} — {note}\n"
                 f"placed t0={t0:.1f} vs TRUTH {truth:.1f} | max overflow {max_ov:.1f} at "
                 f"{ch_key(k_ch)} (probe tick {tick})", fontsize=10.5)

    sig1 = np.sqrt(var_tick[sup])
    ax1.fill_between(np.arange(sup.size + 1) - 0.5,
                     np.append(a[sup] - sig1, (a[sup] - sig1)[-1]),
                     np.append(a[sup] + sig1, (a[sup] + sig1)[-1]),
                     step="post", color="green", alpha=0.25, label=f"+-1 {sig_name}")
    ax1.bar(x, b[sup], color="lightgray", label="base (other placed clusters)")
    ax1.bar(x, o[sup], bottom=b[sup], color="mediumpurple", label="this cluster (own)")
    ax1.step(np.arange(sup.size + 1) - 0.5, np.append(a[sup], a[sup][-1]), where="post",
             color="k", lw=1.4, label="actual light")
    ax1.set_ylabel("ADC at probe tick")
    ax1.set_xticks(x)
    ax1.set_xticklabels(keys, rotation=90, fontsize=7.5)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.set_title(f"support channels of (label {r['label']}, TPC {tpc}) only", fontsize=9)

    sig_clip = np.clip(ov[sup], ymin, ymax)
    colors = ["crimson" if int(c) == k_ch else ("steelblue" if g else "silver")
              for c, g in zip(sup, gate_ok)]
    ax2.bar(x, sig_clip, color=colors)
    for j, sv in enumerate(ov[sup]):
        if sv > ymax or sv < ymin:
            ax2.annotate(f"{sv:.1f}", (j, sig_clip[j]), ha="center",
                         va="bottom" if sv > 0 else "top", fontsize=6.5, rotation=90)
    ax2.set_ylim(ymin - 2, ymax + 2)
    ax2.yaxis.set_major_locator(MultipleLocator(loc))
    ax2.grid(axis="y", alpha=0.45, lw=0.6)
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_ylabel(f"(pred - actual)/{sig_name}")
    ax2.set_xticks(x)
    ax2.set_xticklabels(keys, rotation=90, fontsize=7.5)
    ax2.set_title(f"silver bars fail the excess >= {EXCESS_GATE:.0f} ADC gate", fontsize=9)

    act_w = full_light[tpc, k_ch].astype(np.float64)
    pred_w = np.asarray(d["base_image"], np.float32)[tpc, k_ch]
    sig_w = (np.sqrt(GAIN * np.maximum(act_w, 100.0)) if mode == "pois"
             else np.sqrt(get_var(dtag, tpc, full_light)[k_ch]))
    ax3.fill_between(np.arange(act_w.size), act_w - sig_w, act_w + sig_w,
                     color="green", alpha=0.25, lw=0, label=f"+-1 {sig_name}")
    ax3.plot(np.arange(act_w.size), act_w, color="tab:blue", lw=1.1, label="actual light")
    ax3.plot(np.arange(act_w.size), pred_w, color="tab:orange", lw=1.1,
             label="predicted (final phase-1 baseImage)")
    ax3.axvline(t0 + PEAK, color="red", ls="-", lw=1.3,
                label=f"placed pulse: t0={t0:.0f} -> peak at {t0 + PEAK:.0f}")
    if truth is not None and abs(truth - t0) > 10:
        ax3.axvline(truth + PEAK, color="green", ls=":", lw=1.6,
                    label=f"TRUE pulse: t0={truth:.0f} -> peak at {truth + PEAK:.0f}")
    ax3.set_xlabel("tick")
    ax3.set_ylabel("ADC")
    ax3.set_title(f"channel {ch_key(k_ch)} (TPC {tpc}) waveforms", fontsize=9)
    ax3.legend(fontsize=8, loc="upper right")
    ax3.set_xlim(0, act_w.size - 1)
    plt.tight_layout(rect=[0, 0, 1, 0.955])
    suffix = "" if mode == "pois" else "_varmodel"
    fig.savefig(os.path.join(OUT, f"fail{i}_channels{suffix}.png"), dpi=130)
    plt.close(fig)
    return tpc, max_ov


hits_cache = {}
for i, ftag, ev, lab, dtag, note in FAILS:
    r = get_row(ftag, ev, lab)
    d = np.load(os.path.join(DUMP, f"{dtag}.npz"), allow_pickle=True)
    tpc, mo_p = channels_figure(i, r, d, dtag, note, "pois")
    _, mo_m = channels_figure(i, r, d, dtag, note, "model")

    fp_src = str(d["src_file"])
    if fp_src not in hits_cache:
        with h5py.File(fp_src, "r") as h5:
            hits_cache[fp_src] = h5["charge/calib_prompt_hits/data"][...]
    hits_evt = hits_cache[fp_src][d["hit_refs"]]
    plot_tpc_light_and_charge(
        ev_id=int(ev), TPCid=tpc, hits_full_ev=hits_evt,
        actual=d["full_light"][tpc], predicted=d["base_image"][tpc],
        geom_map=geom_map, individualScaling=False)
    for num in plt.get_fignums():
        f2 = plt.figure(num)
        f2.suptitle(f"fail{i}: {note} | label {lab} t0={r['t0']:.0f} truth={r['truth_t0']:.0f} | "
                    f"{ftag} ev{ev} TPC {tpc}", y=0.995, fontsize=11)
        f2.savefig(os.path.join(OUT, f"fail{i}_type1_light_charge.png"), dpi=110, bbox_inches="tight")
        plt.close(f2)

    tmask = np.asarray(d["tpc"]) == tpc
    reco = np.asarray(d["hit_t0"], np.float64)
    reco_safe = np.where(np.isfinite(reco) & (reco >= 0), reco, -1.0)
    truth_h = np.asarray(d["truth_t0"], np.float64)
    truth_safe = np.where(np.isfinite(truth_h), truth_h, -1.0)
    _png_target["path"] = os.path.join(OUT, f"fail{i}_type2_reco3d.png")
    g = pt.group_hits_by_time(reco_safe[tmask], time_window=10)
    pt.plot_3d_clusters_with_t0(
        x=np.asarray(d["x"])[tmask], y=np.asarray(d["y"])[tmask], z=np.asarray(d["z"])[tmask],
        labels=g, t0s=reco_safe[tmask], energies=np.asarray(d["e"], np.float64)[tmask],
        title=f"fail{i} reco t0 | {ftag} ev{ev} TPC {tpc} (label {lab} at {r['t0']:.0f})",
        save_path=os.path.join(OUT, f"fail{i}_type2_reco3d.html"))
    _png_target["path"] = os.path.join(OUT, f"fail{i}_type3_truth3d.png")
    gt = pt.group_hits_by_time(truth_safe[tmask], time_window=10)
    pt.plot_3d_clusters_with_t0(
        x=np.asarray(d["x"])[tmask], y=np.asarray(d["y"])[tmask], z=np.asarray(d["z"])[tmask],
        labels=gt, t0s=truth_safe[tmask], energies=np.asarray(d["e"], np.float64)[tmask],
        title=f"fail{i} TRUTH t0 | {ftag} ev{ev} TPC {tpc}",
        save_path=os.path.join(OUT, f"fail{i}_type3_truth3d.html"))
    _png_target["path"] = None
    print(f"fail{i}: label {lab} TPC {tpc} | max {mo_p:.1f} sig_pois / {mo_m:.1f} sig_model — done", flush=True)

print("gallery complete")
