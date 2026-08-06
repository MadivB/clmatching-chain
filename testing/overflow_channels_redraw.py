"""Redraw ex*_channels.png with channel keys, 2.5-sigma grid, and a waveform
panel for the max-sigma channel (actual vs predicted, vlines at t0 / probe /
truth).  Adds three known-WRONG placements as new examples."""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

HERE = os.path.abspath(os.path.dirname(__file__))
OUT = os.path.join(HERE, "overflow_displays")
DUMP = os.path.join(HERE, "display_dump")
PEAK = 105
SIG_YMAX, SIG_YMIN = 30.0, -15.0   # sigma panel clip (out-of-range bars annotated)

rows = []
for fp in sorted(glob.glob(os.path.join(HERE, "overflow_study_out", "rows_worker*.jsonl"))):
    with open(fp) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

def get_row(ftag, ev, lab):
    return next(r for r in rows if ftag in r["file"] and r["event"] == ev and r["label"] == lab)

EXAMPLES = [
    # (index, file-tag, event, label, dump-tag, note)
    (0, "0000002", 9, 105, "0000002_ev9", "CORRECT placement"),
    (1, "0000001", 11, 102, "0000001_ev11", "CORRECT placement (merged cluster, 16% foreign charge)"),
    (2, "0000002", 9, 99, "0000002_ev9", "CORRECT placement"),
    (3, "0000003", 5, 124, "0000003_ev5", "CORRECT placement"),
    (4, "0000002", 9, 68, "0000002_ev9", "WRONG placement"),
    (5, "0000002", 9, 101, "0000002_ev9", "WRONG placement"),
    (6, "0000003", 5, 122, "0000003_ev5", "WRONG placement"),
]

def ch_key(c, tpc=None, multi=False):
    base = f"L{c}" if c < 60 else f"R{c - 60}"
    return f"T{tpc}:{base}" if multi else base

# back up the original figures once
for i in range(4):
    src = os.path.join(OUT, f"ex{i}_channels.png")
    bak = os.path.join(OUT, f"ex{i}_channels_orig.png")
    if os.path.exists(src) and not os.path.exists(bak):
        os.rename(src, bak)

for i, ftag, ev, lab, dtag, note in EXAMPLES:
    r = get_row(ftag, ev, lab)
    t0 = float(r["t0"])
    truth = r["truth_t0"]
    tick = int(round(t0)) + PEAK
    multi = len(r["tpc_channels"]) > 1

    keys, owns, bases, actuals, sigs, tpcs_of = [], [], [], [], [], []
    for td in r["tpc_channels"]:
        o = np.asarray(td["own"], float)
        b = np.asarray(td["base"], float)
        a = np.asarray(td["actual"], float)
        sig = (o + b - a) / np.sqrt(np.maximum(a, 100.0))
        for c in td["support"]:
            keys.append(ch_key(int(c), td["tpc"], multi))
            owns.append(o[c]); bases.append(b[c]); actuals.append(a[c])
            sigs.append(sig[c]); tpcs_of.append((int(td["tpc"]), int(c)))
    owns, bases, actuals, sigs = map(np.asarray, (owns, bases, actuals, sigs))
    k = int(np.argmax(sigs))
    max_tpc, max_ch = tpcs_of[k]
    x = np.arange(len(keys))

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13, 11),
                                        gridspec_kw={"height_ratios": [1.1, 1.0, 1.1]})
    status = f"{note} | t0={t0:.1f} vs truth {truth:.1f}" if truth is not None else note
    fig.suptitle(f"ex{i}: label {lab} ({ftag} ev{ev}) — {status} | "
                 f"max overflow {sigs[k]:.0f} sigma at {keys[k]} (probe tick {tick})", fontsize=11)

    # panel 1: ADC composition at the probe tick
    ax1.bar(x, bases, color="lightgray", label="base (other placed clusters)")
    ax1.bar(x, owns, bottom=bases, color="mediumpurple", label="this cluster (own)")
    ax1.step(np.arange(len(x) + 1) - 0.5, np.append(actuals, actuals[-1]), where="post",
             color="k", lw=1.4, label="actual light")
    ax1.set_ylabel("ADC at probe tick")
    ax1.set_xticks(x)
    ax1.set_xticklabels(keys, rotation=90, fontsize=7)
    ax1.legend(fontsize=8, loc="upper right")

    # panel 2: sigma bars with 2.5-sigma grid
    sig_clip = np.clip(sigs, SIG_YMIN, SIG_YMAX)
    colors = ["crimson" if j == k else "steelblue" for j in range(len(x))]
    ax2.bar(x, sig_clip, color=colors)
    for j, (sv, sc) in enumerate(zip(sigs, sig_clip)):
        if sv > SIG_YMAX or sv < SIG_YMIN:
            ax2.annotate(f"{sv:.0f}", (j, sc), ha="center",
                         va="bottom" if sv > 0 else "top", fontsize=6.5, rotation=90)
    ax2.set_ylim(SIG_YMIN - 3, SIG_YMAX + 3)
    ax2.yaxis.set_major_locator(MultipleLocator(2.5))
    ax2.grid(axis="y", which="major", alpha=0.45, lw=0.6)
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_ylabel("(pred - actual)/sigma")
    ax2.set_xticks(x)
    ax2.set_xticklabels(keys, rotation=90, fontsize=7)
    ax2.tick_params(axis="y", labelsize=7)

    # panel 3: waveforms of the max-sigma channel (final phase-1 prediction)
    d = np.load(os.path.join(DUMP, f"{dtag}.npz"), allow_pickle=True)
    act_w = np.asarray(d["full_light"], np.float32)[max_tpc, max_ch]
    pred_w = np.asarray(d["base_image"], np.float32)[max_tpc, max_ch]
    ticks = np.arange(act_w.size)
    ax3.plot(ticks, act_w, color="tab:blue", lw=1.1, label="actual light")
    ax3.plot(ticks, pred_w, color="tab:orange", lw=1.1,
             label="predicted (final phase-1 baseImage)")
    ax3.axvline(t0, color="red", ls="--", lw=1.2, label=f"placement t0 = {t0:.0f}")
    ax3.axvline(tick, color="red", ls="-", lw=0.9, alpha=0.7, label=f"probe tick = t0+{PEAK}")
    if truth is not None and abs(truth - t0) > 10:
        ax3.axvline(truth, color="green", ls=":", lw=1.4, label=f"TRUTH t0 = {truth:.0f}")
    ax3.set_xlabel("tick")
    ax3.set_ylabel("ADC")
    ax3.set_title(f"channel {keys[k]} (TPC {max_tpc}) waveforms", fontsize=9)
    ax3.legend(fontsize=8, loc="upper right")
    ax3.set_xlim(0, act_w.size - 1)

    plt.tight_layout(rect=[0, 0, 1, 0.965])
    out = os.path.join(OUT, f"ex{i}_channels.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}  (max {sigs[k]:.0f} sigma at {keys[k]}, TPC {max_tpc})")
