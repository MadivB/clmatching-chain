"""Redraw all example channel figures + full event-display sets.

Per example (11 total: 4 correct-flagged, 3 wrong, 4 clean-but-triggered):
  ex{i}_channels.png       3 panels: support-only bars of the max-sigma TPC
                           (ADC composition; sigma bars w/ 2.5-sigma grid and
                           excess>=500 gate shown; waveform of max channel with
                           pulse-position markers at t0+105 / truth+105)
  ex{i}_type1_light_charge.png   2D actual-vs-predicted light + charge
  ex{i}_type2_reco3d.html/.png   reco-t0-grouped 3D
  ex{i}_type3_truth3d.html/.png  truth-t0-grouped 3D
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
from M5p1 import truth_plotting as tp  # noqa: F401  (namespace parity)

OUT = os.path.join(_HERE, "overflow_displays")
DUMP = os.path.join(_HERE, "display_dump")
PEAK = 105
EXCESS_GATE = 500.0
GAIN = 200.0            # ADC per photon -> sigma^2 = GAIN * A (Poisson)
SIG_YMAX, SIG_YMIN = 4.0, -2.0

geom_map = load_tpc_geometries(DEFAULT_TPC_YAML)

rows = []
for fp in sorted(glob.glob(os.path.join(_HERE, "overflow_study_out", "rows_worker*.jsonl"))):
    with open(fp) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))


def get_row(ftag, ev, lab):
    return next(r for r in rows if ftag in r["file"] and r["event"] == ev and r["label"] == lab)


EXAMPLES = [
    (0, "0000002", 9, 105, "0000002_ev9", "CORRECT placement (slot polluted by label 68)"),
    (1, "0000001", 11, 102, "0000001_ev11", "CORRECT placement (merged cluster, 16% foreign charge)"),
    (2, "0000002", 9, 99, "0000002_ev9", "CORRECT + clean slot -> FALSE ALARM"),
    (3, "0000003", 5, 124, "0000003_ev5", "CORRECT placement (slot polluted by label 122 tail)"),
    (4, "0000002", 9, 68, "0000002_ev9", "WRONG placement"),
    (5, "0000002", 9, 101, "0000002_ev9", "WRONG placement"),
    (6, "0000003", 5, 122, "0000003_ev5", "WRONG placement"),
    (7, "0000003", 5, 114, "0000003_ev5", "OK placement, clean slot -> FALSE ALARM"),
    (8, "0000001", 11, 140, "0000001_ev11", "OK placement, clean slot -> FALSE ALARM"),
    (9, "0000002", 9, 25, "0000002_ev9", "OK placement, clean slot -> FALSE ALARM"),
    (10, "0000001", 11, 15, "0000001_ev11", "OK placement, clean slot -> FALSE ALARM (bright multi-TPC)"),
]


def ch_key(c):
    return f"L{c}" if c < 60 else f"R{c - 60}"


hits_cache = {}
for i, ftag, ev, lab, dtag, note in EXAMPLES:
    r = get_row(ftag, ev, lab)
    t0 = float(r["t0"])
    truth = r["truth_t0"]
    tick = int(round(t0)) + PEAK

    # pick the TPC holding the max support-channel sigma
    best = None
    for td in r["tpc_channels"]:
        o = np.asarray(td["own"], float)
        b = np.asarray(td["base"], float)
        a = np.asarray(td["actual"], float)
        ov = (o + b - a) / np.sqrt(GAIN * np.maximum(a, 100.0))
        sup = np.asarray(td["support"], int)
        if sup.size == 0:
            continue
        m = float(np.max(ov[sup]))
        if best is None or m > best[0]:
            best = (m, td, o, b, a, ov, sup)
    max_ov, td, o, b, a, ov, sup = best
    tpc = int(td["tpc"])
    k_ch = int(sup[np.argmax(ov[sup])])
    other_tpcs = [int(t["tpc"]) for t in r["tpc_channels"] if int(t["tpc"]) != tpc]

    keys = [ch_key(c) for c in sup]
    x = np.arange(sup.size)
    excess = (o + b - a)[sup]
    gate_ok = excess >= EXCESS_GATE

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13, 11),
                                        gridspec_kw={"height_ratios": [1.1, 1.0, 1.1]})
    status = f"{note} | t0={t0:.1f} vs truth {truth:.1f}" if truth is not None else note
    extra = f" | other TPCs of this label: {other_tpcs}" if other_tpcs else ""
    fig.suptitle(f"ex{i}: label {lab} ({ftag} ev{ev}) TPC {tpc} — {status}\n"
                 f"max overflow {max_ov:.1f} Poisson sigma at {ch_key(k_ch)} (probe tick {tick}){extra}",
                 fontsize=10.5)

    ax1.bar(x, b[sup], color="lightgray", label="base (other placed clusters)")
    ax1.bar(x, o[sup], bottom=b[sup], color="mediumpurple", label="this cluster (own)")
    _sig1 = np.sqrt(GAIN * np.maximum(a[sup], 100.0))
    ax1.fill_between(np.arange(sup.size + 1) - 0.5,
                     np.append(a[sup] - _sig1, (a[sup] - _sig1)[-1]),
                     np.append(a[sup] + _sig1, (a[sup] + _sig1)[-1]),
                     step="post", color="green", alpha=0.25,
                     label="predicted variance band (+-1 sigma_Poisson)")
    ax1.step(np.arange(sup.size + 1) - 0.5, np.append(a[sup], a[sup][-1]), where="post",
             color="k", lw=1.4, label="actual light")
    ax1.set_ylabel("ADC at probe tick")
    ax1.set_xticks(x)
    ax1.set_xticklabels(keys, rotation=90, fontsize=7.5)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.set_title(f"support channels of (label {lab}, TPC {tpc}) only", fontsize=9)

    sig_clip = np.clip(ov[sup], SIG_YMIN, SIG_YMAX)
    colors = ["crimson" if int(c) == k_ch else ("steelblue" if g else "silver")
              for c, g in zip(sup, gate_ok)]
    ax2.bar(x, sig_clip, color=colors)
    for j, sv in enumerate(ov[sup]):
        if sv > SIG_YMAX or sv < SIG_YMIN:
            ax2.annotate(f"{sv:.1f}", (j, sig_clip[j]), ha="center",
                         va="bottom" if sv > 0 else "top", fontsize=6.5, rotation=90)
    ax2.set_ylim(SIG_YMIN - 3, SIG_YMAX + 3)
    ax2.yaxis.set_major_locator(MultipleLocator(0.5))
    ax2.grid(axis="y", which="major", alpha=0.45, lw=0.6)
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_ylabel("(pred - actual)/sigma_Poisson  (g=200)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(keys, rotation=90, fontsize=7.5)
    ax2.tick_params(axis="y", labelsize=7)
    ax2.set_title(f"silver bars fail the excess >= {EXCESS_GATE:.0f} ADC gate "
                  "(blue/crimson pass)", fontsize=9)

    d = np.load(os.path.join(DUMP, f"{dtag}.npz"), allow_pickle=True)
    act_w = np.asarray(d["full_light"], np.float32)[tpc, k_ch]
    pred_w = np.asarray(d["base_image"], np.float32)[tpc, k_ch]
    _sig3 = np.sqrt(GAIN * np.maximum(act_w.astype(np.float64), 100.0))
    ax3.fill_between(np.arange(act_w.size), act_w - _sig3, act_w + _sig3,
                     color="green", alpha=0.25, lw=0,
                     label="predicted variance band (+-1 sigma_Poisson)")
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
    fig.savefig(os.path.join(OUT, f"ex{i}_channels.png"), dpi=130)
    plt.close(fig)

    # ---- event displays ----
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
        f2.suptitle(f"ex{i}: {note} | label {lab} t0={t0:.0f} | {ftag} ev{ev} TPC {tpc}",
                    y=0.995, fontsize=11)
        f2.savefig(os.path.join(OUT, f"ex{i}_type1_light_charge.png"), dpi=110, bbox_inches="tight")
        plt.close(f2)

    tmask = np.asarray(d["tpc"]) == tpc
    reco = np.asarray(d["hit_t0"], np.float64)
    reco_safe = np.where(np.isfinite(reco) & (reco >= 0), reco, -1.0)
    truth_h = np.asarray(d["truth_t0"], np.float64)
    truth_safe = np.where(np.isfinite(truth_h), truth_h, -1.0)

    _png_target["path"] = os.path.join(OUT, f"ex{i}_type2_reco3d.png")
    g = pt.group_hits_by_time(reco_safe[tmask], time_window=10)
    pt.plot_3d_clusters_with_t0(
        x=np.asarray(d["x"])[tmask], y=np.asarray(d["y"])[tmask], z=np.asarray(d["z"])[tmask],
        labels=g, t0s=reco_safe[tmask], energies=np.asarray(d["e"], np.float64)[tmask],
        title=f"ex{i} reco t0 | {ftag} ev{ev} TPC {tpc} (label {lab} at {t0:.0f})",
        save_path=os.path.join(OUT, f"ex{i}_type2_reco3d.html"))
    _png_target["path"] = os.path.join(OUT, f"ex{i}_type3_truth3d.png")
    gt = pt.group_hits_by_time(truth_safe[tmask], time_window=10)
    pt.plot_3d_clusters_with_t0(
        x=np.asarray(d["x"])[tmask], y=np.asarray(d["y"])[tmask], z=np.asarray(d["z"])[tmask],
        labels=gt, t0s=truth_safe[tmask], energies=np.asarray(d["e"], np.float64)[tmask],
        title=f"ex{i} TRUTH t0 | {ftag} ev{ev} TPC {tpc}",
        save_path=os.path.join(OUT, f"ex{i}_type3_truth3d.html"))
    _png_target["path"] = None
    print(f"ex{i}: label {lab} TPC {tpc} max {max_ov:.1f} Poisson sigma at {ch_key(k_ch)} — channels + displays done", flush=True)

print("all examples rendered")
