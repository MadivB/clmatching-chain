"""Corrected group-level scoring of the overflow trigger.

Fixes two ground-truth bugs of the first attempt:
1. contributors at the probe tick are weighted by the actual pulse-template
   amplitude (tails included), not a +-10 tick window;
2. cluster impurity counts: a "correctly placed" but merged cluster carries
   misplaced energy E*(1-coherence) whose light is at the wrong t0.

slot-bad(k) := sum over same-TPC contributors j (placed at or before k, pulse
covering k's probe tick) of misplacedE_j * w_j  >  MISPLACED_MEV_THRESH,
  misplacedE_j = E_j (truth-wrong) or E_j*(1-coherence_j) (otherwise),
  w_j = template(t0_k + 105 - t0_j) / template(105).
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
PEAK = 105
CUT_MARK = 20.0
MISPLACED_MEV_THRESH = 5.0

tmpl = np.load(os.path.join(BASE, "assets", "avg_pulse.npy")).astype(np.float64)
tmpl = tmpl / tmpl[PEAK]

def tmpl_w(dt):
    """relative pulse amplitude at (probe tick - t0_j) = PEAK + dt, dt = t0_k - t0_j"""
    idx = PEAK + int(round(dt))
    if idx < 0 or idx >= tmpl.size:
        return 0.0
    return float(max(tmpl[idx], 0.0))

rows = []
for fp in sorted(glob.glob(os.path.join(OUT_DIR, "rows_worker*.jsonl"))):
    with open(fp) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
n_events = len({(r["file"], r["event"]) for r in rows})
by_event = {}
for r in rows:
    by_event.setdefault((r["file"], r["event"]), []).append(r)
for ev in by_event.values():
    ev.sort(key=lambda r: r["label"])


def misplaced_e(q):
    if q["wrong"] is None:
        return q["energy_mev"] * (1.0 - q["coherence"])  # best available estimate
    if q["wrong"]:
        return q["energy_mev"]
    return q["energy_mev"] * (1.0 - q["coherence"])


s_arr, bad_arr, mis_arr = [], [], []
for ev_rows in by_event.values():
    for i, r in enumerate(ev_rows):
        if r["type"] != "track" or r["wrong"] is None:
            continue
        tp = set(r["tpcs"])
        pol = 0.0
        for q in ev_rows[: i + 1]:
            if not (tp & set(q["tpcs"])):
                continue
            w = tmpl_w(r["t0"] - q["t0"])
            if w <= 0.02:
                continue
            pol += misplaced_e(q) * w
        s_arr.append(r["max_overflow_sigma"])
        mis_arr.append(pol)
        bad_arr.append(pol > MISPLACED_MEV_THRESH)

s = np.asarray(s_arr)
bad = np.asarray(bad_arr)
mis = np.asarray(mis_arr)
print(f"scored track placements: {s.size} over {n_events} events")
print(f"slot-bad (template-weighted misplaced energy > {MISPLACED_MEV_THRESH} MeV): "
      f"{int(bad.sum())} ({bad.mean():.1%})")
s_bad, s_good = s[bad], s[~bad]
print(f"slot-bad : median={np.median(s_bad):.1f} p75={np.percentile(s_bad, 75):.1f}")
print(f"slot-good: median={np.median(s_good):.1f} p75={np.percentile(s_good, 75):.1f}")

# crux conditional, corrected truth
print(f"P(slot-bad | s>20) = {np.mean(bad[s > 20]):.1%} | P(slot-bad | s<=20) = {np.mean(bad[s <= 20]):.1%}")

# ---- histograms ----
pos_max = max(float(s.max()), 50.0)
log_bins = np.logspace(np.log10(1.0), np.log10(pos_max * 1.1), 36)
for vals, color, tag, fname in (
    (s_bad, "crimson", f"POLLUTED t0 slot (misplaced light > {MISPLACED_MEV_THRESH:.0f} MeV at probe)", "overflow_hist_slotbad_v2.png"),
    (s_good, "steelblue", "clean t0 slot", "overflow_hist_slotgood_v2.png"),
):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    h, edges = np.histogram(vals[vals > 1.0], bins=log_bins)
    ax.bar(np.sqrt(edges[:-1] * edges[1:]), h, width=np.diff(edges), color=color, alpha=0.85)
    n_le = int(np.sum(vals <= 1.0))
    ax.bar([0.55], [n_le], width=0.35, color="dimgray", alpha=0.8)
    ax.text(0.55, n_le, f" <= 1 sigma\n n={n_le}", ha="center", va="bottom", fontsize=8)
    ax.axvline(CUT_MARK, color="k", ls="--", lw=1.4, label=f"cut = {CUT_MARK:.0f} sigma")
    ax.set_xscale("log")
    ax.set_xlabel("max (pred - actual)/sigma over fit channels at placed t0 peak")
    ax.set_ylabel("placements")
    ax.set_title(f"{tag}  ({n_events} events, n={vals.size}; pulse-template-weighted contributors)", fontsize=10)
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(HERE, fname), dpi=140)
    plt.close(fig)
    print(f"wrote {fname}")

# ---- efficiency ----
cuts = np.logspace(np.log10(0.5), np.log10(max(pos_max, 100.0)), 200)
tpr = np.array([np.mean(s_bad > c) for c in cuts])
fpr = np.array([np.mean(s_good > c) for c in cuts])
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(cuts, tpr * 100, color="crimson", lw=2, label=f"alarm on POLLUTED slot (n={s_bad.size})")
ax.plot(cuts, fpr * 100, color="steelblue", lw=2, label=f"alarm on clean slot (n={s_good.size})")
ax.axvline(CUT_MARK, color="k", ls="--", lw=1.4)
i20 = int(np.argmin(np.abs(cuts - CUT_MARK)))
ax.annotate(f"at {CUT_MARK:.0f} sigma:\ncatch {tpr[i20]*100:.0f}% of polluted slots\nfalse-alarm {fpr[i20]*100:.1f}%",
            xy=(CUT_MARK, tpr[i20] * 100), xytext=(CUT_MARK * 3, min(tpr[i20] * 100 + 12, 92)),
            arrowprops=dict(arrowstyle="->", lw=1), fontsize=10)
ax.set_xscale("log")
ax.set_xlabel("overflow cut [sigma]")
ax.set_ylabel("fraction above cut [%]")
ax.set_title("Corrected trigger efficiency: polluted-slot ground truth (tails + impurity)")
ax.grid(alpha=0.3)
ax.legend(loc="center left")
plt.tight_layout()
fig.savefig(os.path.join(HERE, "overflow_slot_efficiency_v2.png"), dpi=140)
plt.close(fig)
print("wrote overflow_slot_efficiency_v2.png")

print(f"\ncut table (corrected ground truth, threshold {MISPLACED_MEV_THRESH:.0f} MeV):")
print(f"{'cut':>6} | {'polluted caught':>15} | {'clean false alarm':>17}")
for c in (5, 10, 15, 20, 30, 50, 75, 100):
    print(f"{c:>6} | {np.mean(s_bad > c)*100:>14.1f}% | {np.mean(s_good > c)*100:>16.2f}%")

for th in (2.0, 10.0, 20.0):
    b = mis > th
    print(f"threshold {th:>4.0f} MeV: slot-bad={b.mean():.1%} | at 20 sigma: "
          f"catch {np.mean(s[b] > 20)*100:.1f}% / false {np.mean(s[~b] > 20)*100:.2f}%")
