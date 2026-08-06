"""Render the still-wrong gallery from still_wrong_out/ (census rows + npz dumps).

Per top-N example i:
  sw{i}_channels.png   3-panel support-channel decomposition at the RESCUED state
  sw{i}_truth3d.png    truth clustered-t0 3D (main TPC)      [+ .html]
  sw{i}_reco3d.png     rescued clustered-t0 3D (main TPC)    [+ .html]
  sw{i}_wrong3d.png    wrong-hits view (red = |reco-truth|>10) [+ .html]
Run under:  module load pytorch/2.8.0  (kaleido lives in that user-site).
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

import plotly.graph_objects as go

OUT = os.path.join(_HERE, "still_wrong_gallery")
os.makedirs(OUT, exist_ok=True)
_orig_show = go.Figure.show
_png_target = {"path": None}


def _show_to_png(self, *a, **k):
    if _png_target["path"]:
        self.write_image(_png_target["path"], width=1100, height=850, scale=2)


go.Figure.show = _show_to_png

from M5p1 import plottingTools as pt
from M5p1 import truth_plotting as tp
from pulse_shapes import timeinterpolation

PEAK, GAIN, CLIP, CUT = 105, 200.0, 60780.0, 10.0
SIG_YMAX, SIG_YMIN = 10.0, -5.0
TOP_N = int(sys.argv[1]) if len(sys.argv) > 1 else 8

rows = []
for fp in sorted(glob.glob(os.path.join(_HERE, "still_wrong_out", "rows_worker*.jsonl"))):
    for line in open(fp):
        if line.strip():
            r = json.loads(line)
            for x in r["rows"]:
                rows.append(x)
rows.sort(key=lambda r: -r["wrong_e"])
print(f"{len(rows)} still-wrong labels total; rendering top {TOP_N} with dumps")


def ch_key(c):
    return f"L{c}" if c < 60 else f"R{c - 60}"


made = 0
for r in rows:
    if made >= TOP_N:
        break
    ftag, ev, lab = r["file"], r["event"], r["label"]
    dpath = os.path.join(_HERE, "still_wrong_out", "dumps", f"{ftag}_ev{ev}.npz")
    if not os.path.exists(dpath):
        continue
    d = np.load(dpath)
    keys = set(d.files)
    t0_now, truth = r["t0_now"], r["truth"]
    tick = int(round(t0_now)) + PEAK

    # ---- pick display TPC: largest gated z among the label's dumped TPCs ----
    best = None
    for c in r["tpcs"]:
        if f"L{lab}_T{c}_own" not in keys:
            continue
        P = np.asarray(d[f"L{lab}_T{c}_own"], np.float64)
        sup = np.asarray(d[f"L{lab}_T{c}_sup"], int)
        a = np.asarray(d[f"T{c}_actual"], np.float64)[:, tick]
        b = np.asarray(d[f"T{c}_base"], np.float64)[:, tick]
        own_t = timeinterpolation(P, shift=t0_now, baseline=0)[:, tick]
        b_wo = np.clip(b - own_t, 0.0, None)
        p = np.minimum(b_wo + own_t, CLIP)
        z = (p - a) / np.sqrt(GAIN * np.maximum(a, 100.0))
        m = float(np.max(z[sup])) if sup.size else -99.0
        if best is None or m > best[0]:
            best = (m, int(c), own_t, b_wo, a, z, sup, P)
    if best is None:
        continue
    max_z, tpc, own_t, b_wo_t, a_t, z_t, sup, P = best
    k_ch = int(sup[np.argmax(z_t[sup])])
    i = made
    sav = r.get("sav") or {}
    sav_s = (f"cos truth/now {sav.get('cos_truth', float('nan')):.2f}/{sav.get('cos_now', float('nan')):.2f}"
             if sav else "no sav diag")

    # ---------- channel figure ----------
    x = np.arange(sup.size)
    klabels = [ch_key(int(c)) for c in sup]
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13, 11),
                                        gridspec_kw={"height_ratios": [1.1, 1.0, 1.1]})
    fig.suptitle(f"sw{i}: {ftag} ev{ev} label {lab} ({r['type']}, {r['e']:.0f} MeV, "
                 f"wrong {r['wrong_e']:.0f} MeV) | {r['cat']} | t0={t0_now:.1f} vs truth {truth:.1f} "
                 f"| TPC {tpc} | {sav_s}", fontsize=10)
    sig1 = np.sqrt(GAIN * np.maximum(a_t[sup], 100.0))
    ax1.fill_between(np.arange(sup.size + 1) - 0.5,
                     np.append(a_t[sup] - sig1, (a_t[sup] - sig1)[-1]),
                     np.append(a_t[sup] + sig1, (a_t[sup] + sig1)[-1]),
                     step="post", color="green", alpha=0.25, label="+-1 sigma band")
    ax1.bar(x, b_wo_t[sup], color="lightgray", label="base (others, rescued)")
    ax1.bar(x, own_t[sup], bottom=b_wo_t[sup], color="mediumpurple", label="this cluster (own)")
    ax1.step(np.arange(sup.size + 1) - 0.5, np.append(a_t[sup], a_t[sup][-1]), where="post",
             color="k", lw=1.4, label="actual light")
    ax1.set_ylabel("ADC at probe tick")
    ax1.set_xticks(x); ax1.set_xticklabels(klabels, rotation=90, fontsize=7.5)
    ax1.legend(fontsize=8, loc="upper right")
    sig_clip = np.clip(z_t[sup], SIG_YMIN, SIG_YMAX)
    colors = ["crimson" if int(c) == k_ch else "steelblue" for c in sup]
    ax2.bar(x, sig_clip, color=colors)
    ax2.set_ylim(SIG_YMIN - 2, SIG_YMAX + 2)
    ax2.yaxis.set_major_locator(MultipleLocator(1.0))
    ax2.grid(axis="y", alpha=0.45, lw=0.6)
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_ylabel("z_pois")
    ax2.set_xticks(x); ax2.set_xticklabels(klabels, rotation=90, fontsize=7.5)
    act_w = np.asarray(d[f"T{tpc}_actual"], np.float64)[k_ch]
    pred_w = np.asarray(d[f"T{tpc}_base"], np.float64)[k_ch]
    sig_w = np.sqrt(GAIN * np.maximum(act_w, 100.0))
    ax3.fill_between(np.arange(act_w.size), act_w - sig_w, act_w + sig_w, color="green", alpha=0.25, lw=0)
    ax3.plot(act_w, color="tab:blue", lw=1.1, label="actual light")
    ax3.plot(pred_w, color="tab:orange", lw=1.1, label="predicted (rescued baseImage)")
    ax3.axvline(t0_now + PEAK, color="red", lw=1.3, label=f"placed: t0={t0_now:.0f} -> peak {t0_now+PEAK:.0f}")
    if truth is not None and abs(truth - t0_now) > CUT:
        ax3.axvline(truth + PEAK, color="green", ls=":", lw=1.6,
                    label=f"TRUE: t0={truth:.0f} -> peak {truth+PEAK:.0f}")
    ax3.set_xlabel("tick"); ax3.set_ylabel("ADC")
    ax3.set_title(f"channel {ch_key(k_ch)} (TPC {tpc}) waveforms", fontsize=9)
    ax3.legend(fontsize=8, loc="upper right")
    ax3.set_xlim(0, act_w.size - 1)
    plt.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(os.path.join(OUT, f"sw{i}_channels.png"), dpi=130)
    plt.close(fig)

    # ---------- 3D displays (main TPC) ----------
    xs = np.asarray(d["x"], np.float64); ys = np.asarray(d["y"], np.float64)
    zs = np.asarray(d["z"], np.float64); es = np.asarray(d["e"], np.float64)
    tpcs_h = np.asarray(d["tpc"], int)
    rts = np.asarray(d["hit_t0_rescued"], np.float64)
    tts = np.asarray(d["truth_t0"], np.float64)
    m = tpcs_h == tpc

    tt_safe = np.where(np.isfinite(tts), tts, -1.0)
    tl = pt.group_hits_by_time(tt_safe[m], time_window=5)
    _png_target["path"] = os.path.join(OUT, f"sw{i}_truth3d.png")
    pt.plot_3d_clusters_with_t0(x=xs[m], y=ys[m], z=zs[m], labels=tl, t0s=tt_safe[m], energies=es[m],
                                title=f"sw{i} TRUTH t0 | {ftag} ev{ev} TPC {tpc} (label {lab} true t0 {truth:.0f})",
                                save_path=os.path.join(OUT, f"sw{i}_truth3d.html"))
    rt_safe = np.where(np.isfinite(rts), rts, -1.0)
    rl = pt.group_hits_by_time(rt_safe[m], time_window=5)
    _png_target["path"] = os.path.join(OUT, f"sw{i}_reco3d.png")
    pt.plot_3d_clusters_with_t0(x=xs[m], y=ys[m], z=zs[m], labels=rl, t0s=rt_safe[m], energies=es[m],
                                title=f"sw{i} RESCUED t0 | {ftag} ev{ev} TPC {tpc} (label {lab} placed {t0_now:.0f})",
                                save_path=os.path.join(OUT, f"sw{i}_reco3d.html"))
    _png_target["path"] = os.path.join(OUT, f"sw{i}_wrong3d.png")
    tp.plot_wrong_hits_3d(tpc, tts, hit_timestamps=rts, hitTPCid=tpcs_h,
                          xset=xs, yset=ys, zset=zs, Eset=es,
                          residual_cut_ticks=CUT,
                          save_path=os.path.join(OUT, f"sw{i}_wrong3d.html"),
                          marker_size=3.5, wrong_color="red", correct_color="lightgray", show=True)
    _png_target["path"] = None
    print(f"sw{i}: {ftag} ev{ev} label {lab} ({r['cat']}, wrong {r['wrong_e']:.0f} MeV) "
          f"TPC {tpc} | {sav_s}", flush=True)
    made += 1

print("gallery done:", OUT)
