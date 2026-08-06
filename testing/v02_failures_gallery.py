"""Failure gallery for the FINAL pipeline (stack2 backbone + A/B/N=1 rescue + Phase2 + V2).

Reads v02dump_out/dumps/*.npz (final-state dumps), picks the highest-energy
still-wrong labels dataset-wide (cap 2 per event), classifies each failure by
stage, and renders per failure i into v02_failures/:
  vfail{i}_channels.png          own/base/actual bars at the probe tick +
                                 z_pois panel + brightest-channel waveforms
  vfail{i}_type1_light_charge.png  2D light+charge comparison for the TPC
  vfail{i}_type2_reco3d.png/.html  3D hits grouped by FINAL reco t0
  vfail{i}_type3_truth3d.png/.html 3D hits grouped by truth t0
Run with the module pytorch/2.8.0 python (kaleido lives in the py3.12 user-site).
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
        self.write_image(_png_target["path"], width=1100, height=850, scale=1)


go.Figure.show = _patched_show

from ML_NDfull_perceiver import load_tpc_geometries, DEFAULT_TPC_YAML
from visualizationTools import plot_tpc_light_and_charge
import plottingTools as pt
from pulse_shapes import timeinterpolation

DUMPS = os.path.join(_HERE, "v02dump_out", "dumps")
OUT = os.path.join(_HERE, "v02_failures")
os.makedirs(OUT, exist_ok=True)
PEAK = 105
GAIN = 200.0
CLIP = 60780.0
CUT = 10.0
MAX_PER_EVENT = 2
N_FAILS = 7

geom_map = load_tpc_geometries(DEFAULT_TPC_YAML)


def ch_key(c):
    return f"L{c}" if c < 60 else f"R{c - 60}"


def ti(arr, t):
    return timeinterpolation(np.asarray(arr, np.float64), shift=float(t), baseline=0)


# ---------- collect failures across all dumped events ----------
cands = []
for fp in sorted(glob.glob(os.path.join(DUMPS, "*.npz"))):
    d = np.load(fp, allow_pickle=True)
    ftag = os.path.basename(fp).split("_ev")[0]
    ev = int(d["event"])
    per_event = []
    for r in json.loads(str(d["lab_rows"])):
        if r["truth"] is None or r["e_wrong"] < 200.0 or r["e_wrong"] < 0.5 * r["e_ok"]:
            continue
        has_img = any(f"img_{r['label']}_{c}" in d for c in r["tpcs"])
        per_event.append({**r, "ftag": ftag, "ev": ev, "fp": fp, "has_img": has_img})
    per_event.sort(key=lambda r: -r["e_wrong"])
    cands.extend(per_event[:MAX_PER_EVENT])
cands.sort(key=lambda r: -r["e_wrong"])
fails = cands[:N_FAILS]
print(f"{len(cands)} candidate failures; rendering top {len(fails)}")


def classify(r):
    if r["p1_t0"] is None:
        return "PLACED BY PHASE 2 (never in backbone)"
    if abs(r["reco_t0"] - r["p1_t0"]) > CUT:
        if abs(r["p1_t0"] - r["truth"]) <= CUT:
            return "backbone was RIGHT — P2/V2 moved it WRONG"
        return "backbone wrong, P2/V2 moved it (still wrong)"
    if abs(r["p1_t0"] - r["truth"]) > CUT:
        return "backbone placement stood WRONG (trigger/rescue missed or failed)"
    return "backbone right, final wrong (?)"


hits_cache = {}
for i, r in enumerate(fails):
    d = np.load(r["fp"], allow_pickle=True)
    lab, ev, ftag = r["label"], r["ev"], r["ftag"]
    note = classify(r)
    t0, truth = r["reco_t0"], r["truth"]
    tick = int(round(t0)) + PEAK
    full_light = np.asarray(d["full_light"], np.float64)
    base_image = np.asarray(d["base_image"], np.float64)
    veto = np.asarray(d["veto_mask"], bool)
    n_ticks = full_light.shape[2]

    # ---------- channel + waveform figure ----------
    tpc = None
    if r["has_img"] and 0 <= tick < n_ticks:
        best = None
        for c in r["tpcs"]:
            key = f"img_{lab}_{c}"
            if key not in d or c >= full_light.shape[0]:
                continue
            img = np.asarray(d[key], np.float64)
            own = ti(img, t0)[:, tick]
            score = float(np.max(own))
            if best is None or score > best[0]:
                best = (score, int(c), img, own)
        if best is not None:
            _, tpc, img, own = best
            a = full_light[tpc, :, tick]
            b_all = np.minimum(base_image[tpc, :, tick], CLIP)
            in_base = abs(r["reco_t0"] - (r["p1_t0"] if r["p1_t0"] is not None else -1e9)) <= CUT \
                and r["p1_t0"] is not None
            b_wo = np.clip(b_all - own, 0, None) if in_base else b_all
            sup = np.flatnonzero((img.max(axis=1) > 1.0) & ~veto[tpc])
            if sup.size:
                x_ = np.arange(sup.size)
                keys = [ch_key(c) for c in sup]
                z = (np.minimum(b_wo + own, CLIP) - a) / np.sqrt(GAIN * np.maximum(a, 100.0))
                sig1 = np.sqrt(GAIN * np.maximum(a[sup], 100.0)) / 1.0

                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13, 11),
                                                    gridspec_kw={"height_ratios": [1.1, 1.0, 1.1]})
                fig.suptitle(f"vfail{i}: label {lab} ({ftag} ev{ev}) TPC {tpc} — {note}\n"
                             f"final t0={t0:.0f} vs truth {truth:.0f} | wrong E "
                             f"{r['e_wrong']:.0f} MeV | type {r['type']}", fontsize=10.5)
                ax1.fill_between(np.arange(sup.size + 1) - 0.5,
                                 np.append(a[sup] - sig1, (a[sup] - sig1)[-1]),
                                 np.append(a[sup] + sig1, (a[sup] + sig1)[-1]),
                                 step="post", color="green", alpha=0.25,
                                 label="+-1 sigma_pois band")
                ax1.bar(x_, b_wo[sup], color="lightgray", label="base (other clusters)")
                ax1.bar(x_, own[sup], bottom=b_wo[sup], color="mediumpurple",
                        label="this cluster (own)")
                ax1.step(np.arange(sup.size + 1) - 0.5, np.append(a[sup], a[sup][-1]),
                         where="post", color="k", lw=1.4, label="actual light")
                ax1.set_ylabel("ADC at probe tick")
                ax1.set_xticks(x_)
                ax1.set_xticklabels(keys, rotation=90, fontsize=7.5)
                ax1.legend(fontsize=8, loc="upper right")
                ax1.set_title(f"active channels of (label {lab}, TPC {tpc}) at tick {tick}",
                              fontsize=9)

                zc = np.clip(z[sup], -5, 10)
                ax2.bar(x_, zc, color="steelblue")
                ax2.set_ylim(-7, 12)
                ax2.yaxis.set_major_locator(MultipleLocator(1.0))
                ax2.grid(axis="y", which="major", alpha=0.45, lw=0.6)
                ax2.axhline(0, color="k", lw=0.8)
                ax2.set_ylabel("(pred - actual)/sigma_pois")
                ax2.set_xticks(x_)
                ax2.set_xticklabels(keys, rotation=90, fontsize=7.5)
                ax2.tick_params(axis="y", labelsize=7)

                k_ch = int(sup[np.argmax(own[sup])])
                act_w = full_light[tpc, k_ch]
                pred_w = base_image[tpc, k_ch]
                sig_w = np.sqrt(GAIN * np.maximum(act_w, 100.0))
                ax3.fill_between(np.arange(n_ticks), act_w - sig_w, act_w + sig_w,
                                 color="green", alpha=0.25, lw=0, label="+-1 sigma_pois")
                ax3.plot(act_w, color="tab:blue", lw=1.1, label="actual light")
                ax3.plot(pred_w, color="tab:orange", lw=1.1, label="backbone base image")
                ax3.axvline(t0 + PEAK, color="red", lw=1.3,
                            label=f"FINAL t0={t0:.0f} -> peak {t0 + PEAK:.0f}")
                if r["p1_t0"] is not None and abs(r["p1_t0"] - t0) > CUT:
                    ax3.axvline(r["p1_t0"] + PEAK, color="purple", ls="--", lw=1.2,
                                label=f"backbone t0={r['p1_t0']:.0f}")
                ax3.axvline(truth + PEAK, color="green", ls=":", lw=1.6,
                            label=f"TRUE t0={truth:.0f} -> peak {truth + PEAK:.0f}")
                ax3.set_xlabel("tick")
                ax3.set_ylabel("ADC")
                ax3.set_title(f"channel {ch_key(k_ch)} (TPC {tpc}) waveforms", fontsize=9)
                ax3.legend(fontsize=8, loc="upper right")
                ax3.set_xlim(0, n_ticks - 1)
                plt.tight_layout(rect=[0, 0, 1, 0.94])
                fig.savefig(os.path.join(OUT, f"vfail{i}_channels.png"), dpi=130)
                plt.close(fig)
    if tpc is None:
        tpc = int(r["tpcs"][0]) if r["tpcs"] else int(np.asarray(d["tpc"])[
            np.asarray(d["labels"]) == lab][0])

    # ---------- type1 light + charge ----------
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
        f2.suptitle(f"vfail{i}: {note} | label {lab} final t0={t0:.0f} truth={truth:.0f} | "
                    f"{ftag} ev{ev} TPC {tpc}", y=0.995, fontsize=11)
        f2.savefig(os.path.join(OUT, f"vfail{i}_type1_light_charge.png"),
                   dpi=110, bbox_inches="tight")
        plt.close(f2)

    # ---------- 3D reco vs truth ----------
    tmask = np.asarray(d["tpc"]) == tpc
    reco = np.asarray(d["ts_final"], np.float64)
    reco_safe = np.where(np.isfinite(reco) & (reco >= 0), reco, -1.0)
    truth_h = np.asarray(d["truth_t0"], np.float64)
    truth_safe = np.where(np.isfinite(truth_h), truth_h, -1.0)
    _png_target["path"] = os.path.join(OUT, f"vfail{i}_type2_reco3d.png")
    g = pt.group_hits_by_time(reco_safe[tmask], time_window=10)
    pt.plot_3d_clusters_with_t0(
        x=np.asarray(d["x"])[tmask], y=np.asarray(d["y"])[tmask], z=np.asarray(d["z"])[tmask],
        labels=g, t0s=reco_safe[tmask], energies=np.asarray(d["e"], np.float64)[tmask],
        title=f"vfail{i} FINAL reco t0 | {ftag} ev{ev} TPC {tpc} (label {lab} at {t0:.0f})",
        save_path=os.path.join(OUT, f"vfail{i}_type2_reco3d.html"))
    _png_target["path"] = os.path.join(OUT, f"vfail{i}_type3_truth3d.png")
    gt = pt.group_hits_by_time(truth_safe[tmask], time_window=10)
    pt.plot_3d_clusters_with_t0(
        x=np.asarray(d["x"])[tmask], y=np.asarray(d["y"])[tmask], z=np.asarray(d["z"])[tmask],
        labels=gt, t0s=truth_safe[tmask], energies=np.asarray(d["e"], np.float64)[tmask],
        title=f"vfail{i} TRUTH t0 | {ftag} ev{ev} TPC {tpc} (true {truth:.0f})",
        save_path=os.path.join(OUT, f"vfail{i}_type3_truth3d.html"))
    _png_target["path"] = None
    print(f"vfail{i}: label {lab} ({ftag} ev{ev}) TPC {tpc} E_wrong {r['e_wrong']:.0f} MeV "
          f"| {note}", flush=True)

print("gallery complete")
