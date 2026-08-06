"""Variance-MODEL version of the example channel figures.

Same 11 examples, but the green band and the sigma panel use the ML variance
prediction (var_run_perceiver_aligned_0001000_v2/best_model.pt) instead of
the Poisson g=200 form.  Output: ex{i}_channels_varmodel.png.
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from var_prediction.inference_ndfl_v2 import load_model as load_var_model, predict as predict_var

VAR_CKPT = ("/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/NewMLSection/var_prediction/"
            "runs/var_run_perceiver_aligned_0001000_v2/best_model.pt")
OUT = os.path.join(_HERE, "overflow_displays")
DUMP = os.path.join(_HERE, "display_dump")
PEAK = 105
EXCESS_GATE = 500.0
SIG_YMAX, SIG_YMIN = 10.0, -5.0

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


print("loading variance model (CPU) ...", flush=True)
var_model, var_meta = load_var_model(VAR_CKPT, device="cpu", waveform_len=1000, num_channels=120)

var_cache = {}


def get_var(dtag, tpc, full_light):
    key = (dtag, tpc)
    if key not in var_cache:
        v = predict_var(var_model, full_light[tpc][None, :, :], batch_size=8,
                        input_scale=1e-3, target_scale=1e-3, device="cpu",
                        return_variance=True, min_sigma=1.0).astype(np.float64)[0]
        var_cache[key] = np.maximum(v, 1.0)
    return var_cache[key]


for i, ftag, ev, lab, dtag, note in EXAMPLES:
    r = get_row(ftag, ev, lab)
    t0 = float(r["t0"])
    truth = r["truth_t0"]
    tick = int(round(t0)) + PEAK
    d = np.load(os.path.join(DUMP, f"{dtag}.npz"), allow_pickle=True)
    full_light = np.asarray(d["full_light"], np.float32)

    # pick TPC by max variance-model sigma on support channels
    best = None
    for td in r["tpc_channels"]:
        o = np.asarray(td["own"], float)
        b = np.asarray(td["base"], float)
        a = np.asarray(td["actual"], float)
        var = get_var(dtag, int(td["tpc"]), full_light)[:, tick]
        ov = (o + b - a) / np.sqrt(var)
        sup = np.asarray(td["support"], int)
        if sup.size == 0:
            continue
        m = float(np.max(ov[sup]))
        if best is None or m > best[0]:
            best = (m, td, o, b, a, ov, sup, var)
    max_ov, td, o, b, a, ov, sup, var_t = best
    tpc = int(td["tpc"])
    k_ch = int(sup[np.argmax(ov[sup])])
    other_tpcs = [int(t["tpc"]) for t in r["tpc_channels"] if int(t["tpc"]) != tpc]

    keys = [ch_key(c) for c in sup]
    x = np.arange(sup.size)
    gate_ok = (o + b - a)[sup] >= EXCESS_GATE

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13, 11),
                                        gridspec_kw={"height_ratios": [1.1, 1.0, 1.1]})
    status = f"{note} | t0={t0:.1f} vs truth {truth:.1f}" if truth is not None else note
    extra = f" | other TPCs: {other_tpcs}" if other_tpcs else ""
    fig.suptitle(f"ex{i} [VARIANCE MODEL]: label {lab} ({ftag} ev{ev}) TPC {tpc} — {status}\n"
                 f"max overflow {max_ov:.1f} model-sigma at {ch_key(k_ch)} (probe tick {tick}){extra}",
                 fontsize=10.5)

    sig1 = np.sqrt(var_t[sup])
    ax1.fill_between(np.arange(sup.size + 1) - 0.5,
                     np.append(a[sup] - sig1, (a[sup] - sig1)[-1]),
                     np.append(a[sup] + sig1, (a[sup] + sig1)[-1]),
                     step="post", color="green", alpha=0.25,
                     label="variance-model band (+-1 sigma_model)")
    ax1.bar(x, b[sup], color="lightgray", label="base (other placed clusters)")
    ax1.bar(x, o[sup], bottom=b[sup], color="mediumpurple", label="this cluster (own)")
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
    ax2.set_ylim(SIG_YMIN - 2, SIG_YMAX + 2)
    ax2.yaxis.set_major_locator(MultipleLocator(1.0))
    ax2.grid(axis="y", which="major", alpha=0.45, lw=0.6)
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_ylabel("(pred - actual)/sigma_MODEL")
    ax2.set_xticks(x)
    ax2.set_xticklabels(keys, rotation=90, fontsize=7.5)
    ax2.tick_params(axis="y", labelsize=7)
    ax2.set_title(f"silver bars fail the excess >= {EXCESS_GATE:.0f} ADC gate", fontsize=9)

    act_w = full_light[tpc, k_ch].astype(np.float64)
    pred_w = np.asarray(d["base_image"], np.float32)[tpc, k_ch]
    sig_w = np.sqrt(get_var(dtag, tpc, full_light)[k_ch])
    ax3.fill_between(np.arange(act_w.size), act_w - sig_w, act_w + sig_w,
                     color="green", alpha=0.25, lw=0, label="variance-model band (+-1 sigma_model)")
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
    fig.savefig(os.path.join(OUT, f"ex{i}_channels_varmodel.png"), dpi=130)
    plt.close(fig)
    print(f"ex{i}: label {lab} TPC {tpc} max {max_ov:.1f} model-sigma at {ch_key(k_ch)}", flush=True)

print("done")
