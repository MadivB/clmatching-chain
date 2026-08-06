"""Hit-level, per-TPC ground truth for the overflow trigger (user's definition).

For each placement k probing TPC c at tick t0_k+105: go over ALL hits in TPC c
already assigned (labels placed at or before k), and ask per hit whether it is
correctly placed (|reco_t0 - truth_t0| <= 10 ticks).  The pollution of the
probe is the energy of MISPLACED hits weighted by their placed pulse's
template amplitude at the probe tick.  slot-bad = pollution > THRESH in any
probed TPC.

Scores both probes: support-channel max overflow and all-channel max overflow.
"""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.abspath(os.path.dirname(__file__))
BASE = os.path.abspath(os.path.join(HERE, "..", "CLMatching_AlphaRelease"))
OUT_DIR = os.path.join(HERE, "overflow_study_out")
HITS_DIR = os.path.join(OUT_DIR, "hits")
PEAK = 105
HIT_CUT_TICKS = 10.0
THRESH_MEV = 5.0
CUT_MARK = 20.0
W_MIN = 0.02

tmpl = np.load(os.path.join(BASE, "assets", "avg_pulse.npy")).astype(np.float64)
tmpl = np.maximum(tmpl / tmpl[PEAK], 0.0)

rows = []
for fp in sorted(glob.glob(os.path.join(OUT_DIR, "rows_worker*.jsonl"))):
    with open(fp) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
by_event = {}
for r in rows:
    by_event.setdefault((r["file"], r["event"]), []).append(r)
for ev in by_event.values():
    ev.sort(key=lambda r: r["label"])
n_events = len(by_event)

s_sup, s_all, pol_arr, slot_pol_arr = [], [], [], []
for (fname, ev_id), ev_rows in by_event.items():
    ftag = fname.split(".")[-3]
    hf = os.path.join(HITS_DIR, f"{ftag}_ev{ev_id}.npz")
    if not os.path.exists(hf):
        continue
    d = np.load(hf)
    h_lab = d["labels"].astype(np.int64)
    h_tpc = d["tpc"].astype(np.int64)
    h_e = np.clip(d["e"].astype(np.float64), 0, None)
    h_reco = d["reco_t0"].astype(np.float64)
    h_truth = d["truth_t0"].astype(np.float64)

    assigned = (h_lab >= 0) & np.isfinite(h_reco) & (h_reco >= 0)
    truthok = np.isfinite(h_truth)
    misplaced = assigned & truthok & (np.abs(h_reco - h_truth) > HIT_CUT_TICKS)

    for r in ev_rows:
        if r["type"] != "track" or r["wrong"] is None or r.get("max_overflow_allch_sigma") is None:
            continue
        L, t0k = int(r["label"]), float(r["t0"])
        pol = 0.0
        slot_pol = 0.0
        for c in r["tpcs"]:
            m = misplaced & (h_tpc == int(c)) & (h_lab <= L)
            if not m.any():
                continue
            idx = np.clip(np.round(PEAK + t0k - h_reco[m]).astype(int), 0, tmpl.size - 1)
            w = tmpl[idx]
            w[np.round(PEAK + t0k - h_reco[m]).astype(int) < 0] = 0.0
            w[w < W_MIN] = 0.0
            pol = max(pol, float(np.sum(h_e[m] * w)))
            slot_pol = max(slot_pol, float(np.sum(h_e[m][np.abs(h_reco[m] - t0k) <= 10.0])))
        s_sup.append(r["max_overflow_sigma"])
        s_all.append(r["max_overflow_allch_sigma"])
        pol_arr.append(pol)
        slot_pol_arr.append(slot_pol)

s_sup = np.asarray(s_sup)
s_all = np.asarray(s_all)
pol = np.asarray(pol_arr)
slot_pol = np.asarray(slot_pol_arr)
bad = pol > THRESH_MEV
print(f"scored track placements: {s_sup.size} over {n_events} events")
print(f"polluted probes (hit-level, template-weighted > {THRESH_MEV} MeV): {int(bad.sum())} ({bad.mean():.1%})")
print(f"  of which with misplaced energy AT the slot itself (+-10): {int((slot_pol[bad] > THRESH_MEV).sum())}")

for name, s in (("support-ch probe", s_sup), ("all-ch probe", s_all)):
    print(f"\n{name}:")
    print(f"  polluted: median={np.median(s[bad]):.1f} p75={np.percentile(s[bad], 75):.1f}")
    print(f"  clean   : median={np.median(s[~bad]):.1f} p75={np.percentile(s[~bad], 75):.1f}")
    print(f"  P(polluted | s>20) = {np.mean(bad[s > 20]):.1%} | P(polluted | s<=20) = {np.mean(bad[s <= 20]):.1%}")
    print(f"  at 20 sigma: catch {np.mean(s[bad] > 20)*100:.1f}% / false-alarm {np.mean(s[~bad] > 20)*100:.2f}%")

# ---- histograms (all-channel probe, the fair slot-state test) ----
pos_max = max(float(s_all.max()), 50.0)
log_bins = np.logspace(np.log10(1.0), np.log10(pos_max * 1.1), 36)
for vals, color, tag, fname in (
    (s_all[bad], "crimson", f"POLLUTED probe (>{THRESH_MEV:.0f} MeV misplaced hit-energy lighting the probe tick)", "overflow_hist_hitlevel_bad.png"),
    (s_all[~bad], "steelblue", "clean probe (all hits in these TPCs correctly placed)", "overflow_hist_hitlevel_good.png"),
):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    h, edges = np.histogram(vals[vals > 1.0], bins=log_bins)
    ax.bar(np.sqrt(edges[:-1] * edges[1:]), h, width=np.diff(edges), color=color, alpha=0.85)
    n_le = int(np.sum(vals <= 1.0))
    ax.bar([0.55], [n_le], width=0.35, color="dimgray", alpha=0.8)
    ax.text(0.55, n_le, f" <= 1 sigma\n n={n_le}", ha="center", va="bottom", fontsize=8)
    ax.axvline(CUT_MARK, color="k", ls="--", lw=1.4, label=f"cut = {CUT_MARK:.0f} sigma")
    ax.set_xscale("log")
    ax.set_xlabel("max (pred - actual)/sigma over ALL unsaturated channels at placed t0 peak")
    ax.set_ylabel("placements")
    ax.set_title(f"{tag}  ({n_events} events, n={vals.size}; hit-level per-TPC truth)", fontsize=10)
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(HERE, fname), dpi=140)
    plt.close(fig)
    print(f"wrote {fname}")

# ---- efficiency curves: both probes ----
cuts = np.logspace(np.log10(0.5), np.log10(max(pos_max, 100.0)), 200)
fig, ax = plt.subplots(figsize=(10.5, 5.5))
for s, ls, tag in ((s_all, "-", "all-channel probe"), (s_sup, "--", "support-channel probe")):
    ax.plot(cuts, [np.mean(s[bad] > c) * 100 for c in cuts], color="crimson", ls=ls, lw=2,
            label=f"polluted probes caught ({tag})")
    ax.plot(cuts, [np.mean(s[~bad] > c) * 100 for c in cuts], color="steelblue", ls=ls, lw=2,
            label=f"clean-probe false alarms ({tag})")
ax.axvline(CUT_MARK, color="k", ls=":", lw=1.4)
ax.set_xscale("log")
ax.set_xlabel("overflow cut [sigma]")
ax.set_ylabel("fraction above cut [%]")
ax.set_title(f"Hit-level per-TPC ground truth (n_bad={int(bad.sum())}, n_clean={int((~bad).sum())}, {n_events} events)")
ax.grid(alpha=0.3)
ax.legend(loc="upper right", fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(HERE, "overflow_hitlevel_efficiency.png"), dpi=140)
plt.close(fig)
print("wrote overflow_hitlevel_efficiency.png")

print(f"\ncut table (all-channel probe, hit-level truth, thresh {THRESH_MEV:.0f} MeV):")
print(f"{'cut':>6} | {'polluted caught':>15} | {'clean false alarm':>17}")
for c in (5, 10, 15, 20, 30, 50, 75, 100):
    print(f"{c:>6} | {np.mean(s_all[bad] > c)*100:>14.1f}% | {np.mean(s_all[~bad] > c)*100:>16.2f}%")

for th in (2.0, 10.0, 20.0):
    b = pol > th
    print(f"thresh {th:>4.0f} MeV: polluted={b.mean():.1%} | all-ch at 20 sigma: "
          f"catch {np.mean(s_all[b] > 20)*100:.1f}% / false {np.mean(s_all[~b] > 20)*100:.2f}%")
