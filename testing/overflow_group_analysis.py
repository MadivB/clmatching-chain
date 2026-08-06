"""Group-level re-scoring of the overflow trigger.

New definition (user's): the probe at placement k inspects the STATE of the
t0 slot, not just cluster k --
  s > cut  is a SUCCESS iff at least one cluster already placed at this t0
           (same event, |t0_j - t0_k| <= T0_TOL, sharing a TPC with k,
           placed at or before k) is truth-wrong;
  s < cut  is correct iff every such cluster (incl. k) is truth-good.
"""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.abspath(os.path.dirname(__file__))
OUT_DIR = os.path.join(HERE, "overflow_study_out")
COHERENCE_MIN = 0.7
T0_TOL = 10.0
CUT_MARK = 20.0

rows = []
for fp in sorted(glob.glob(os.path.join(OUT_DIR, "rows_worker*.jsonl"))):
    with open(fp) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
n_events = len({(r["file"], r["event"]) for r in rows})
print(f"{len(rows)} placements over {n_events} events")

# rows are written in placement order per event (label ascending); group them
by_event = {}
for r in rows:
    by_event.setdefault((r["file"], r["event"]), []).append(r)
for ev_rows in by_event.values():
    ev_rows.sort(key=lambda r: r["label"])   # placement order

group_bad_arr, group_unknown_arr, s_arr, self_wrong_arr = [], [], [], []
group_sizes = []
for ev_rows in by_event.values():
    for i, r in enumerate(ev_rows):
        if r["wrong"] is None or r["coherence"] < COHERENCE_MIN or r["type"] != "track":
            continue
        tpcs_k = set(r["tpcs"])
        members = [q for q in ev_rows[: i + 1]
                   if abs(q["t0"] - r["t0"]) <= T0_TOL and tpcs_k & set(q["tpcs"])]
        defined = [q for q in members if q["wrong"] is not None and q["coherence"] >= COHERENCE_MIN]
        unknown = len(defined) < len(members)
        bad = any(q["wrong"] for q in defined)
        group_bad_arr.append(bad)
        group_unknown_arr.append(unknown)
        s_arr.append(r["max_overflow_sigma"])
        self_wrong_arr.append(bool(r["wrong"]))
        group_sizes.append(len(members))

s = np.asarray(s_arr)
gbad = np.asarray(group_bad_arr)
gunk = np.asarray(group_unknown_arr)
selfw = np.asarray(self_wrong_arr)
print(f"scored track placements: {s.size} | group sizes: median={int(np.median(group_sizes))} "
      f"max={max(group_sizes)} | groups with undefined-truth members: {int(gunk.sum())}")
print(f"group-bad (something at this t0 is misplaced): {int(gbad.sum())} "
      f"({gbad.mean():.1%}) vs self-wrong only: {int(selfw.sum())} ({selfw.mean():.1%})")
print(f"placements where SELF is good but the t0 group is bad: {int((gbad & ~selfw).sum())}")

s_bad = s[gbad]
s_good = s[~gbad]
print(f"\ngroup-bad  : n={s_bad.size} median={np.median(s_bad):.1f} p25={np.percentile(s_bad,25):.1f} p75={np.percentile(s_bad,75):.1f}")
print(f"group-good : n={s_good.size} median={np.median(s_good):.1f} p25={np.percentile(s_good,25):.1f} p75={np.percentile(s_good,75):.1f}")

# ---- histograms ----
pos_max = max(float(s.max()), 50.0)
log_bins = np.logspace(np.log10(1.0), np.log10(pos_max * 1.1), 36)
for vals, color, tag, fname in (
    (s_bad, "crimson", "t0 slot CONTAINS a misplaced cluster (group-bad)", "overflow_hist_groupbad.png"),
    (s_good, "steelblue", "every cluster at this t0 is correct (group-good)", "overflow_hist_groupgood.png"),
):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    h, edges = np.histogram(vals[vals > 1.0], bins=log_bins)
    ax.bar(np.sqrt(edges[:-1] * edges[1:]), h, width=np.diff(edges), color=color, alpha=0.85)
    n_le = int(np.sum(vals <= 1.0))
    ax.bar([0.55], [n_le], width=0.35, color="dimgray", alpha=0.8)
    ax.text(0.55, n_le, f" <= 1 sigma\n (incl. <= 0)\n n={n_le}", ha="center", va="bottom", fontsize=8)
    ax.axvline(CUT_MARK, color="k", ls="--", lw=1.4, label=f"cut = {CUT_MARK:.0f} sigma")
    ax.set_xscale("log")
    ax.set_xlabel("max (pred - actual)/sigma over fit channels at placed t0 peak")
    ax.set_ylabel("placements")
    ax.set_title(f"{tag}  ({n_events} events, n={vals.size}; |dt0| <= {T0_TOL:.0f} ticks, shared TPC)", fontsize=10)
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(HERE, fname), dpi=140)
    plt.close(fig)
    print(f"wrote {fname}")

# ---- efficiency curve ----
cuts = np.logspace(np.log10(0.5), np.log10(max(pos_max, 100.0)), 200)
tpr = np.array([np.mean(s_bad > c) for c in cuts])
fpr = np.array([np.mean(s_good > c) for c in cuts])
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(cuts, tpr * 100, color="crimson", lw=2, label=f"alarm fired & t0 slot IS bad (n={s_bad.size})")
ax.plot(cuts, fpr * 100, color="steelblue", lw=2, label=f"alarm fired but t0 slot is clean (n={s_good.size})")
ax.axvline(CUT_MARK, color="k", ls="--", lw=1.4)
i20 = int(np.argmin(np.abs(cuts - CUT_MARK)))
ax.annotate(f"at {CUT_MARK:.0f} sigma:\ncatch {tpr[i20]*100:.0f}% of bad t0 slots\nfalse-alarm {fpr[i20]*100:.1f}%",
            xy=(CUT_MARK, tpr[i20] * 100), xytext=(CUT_MARK * 3, min(tpr[i20] * 100 + 12, 92)),
            arrowprops=dict(arrowstyle="->", lw=1), fontsize=10)
ax.set_xscale("log")
ax.set_xlabel("overflow cut [sigma]")
ax.set_ylabel("fraction of placements above cut [%]")
ax.set_title('"Something at this t0 went wrong" trigger efficiency (group-level definition)')
ax.grid(alpha=0.3)
ax.legend(loc="center left")
plt.tight_layout()
fig.savefig(os.path.join(HERE, "overflow_group_efficiency.png"), dpi=140)
plt.close(fig)
print("wrote overflow_group_efficiency.png")

print(f"\ncut table (group-level definition, |dt0| <= {T0_TOL:.0f} ticks):")
print(f"{'cut':>6} | {'bad-slot caught':>15} | {'clean-slot false alarm':>21}")
for c in (5, 10, 15, 20, 30, 50, 75, 100):
    print(f"{c:>6} | {np.mean(s_bad > c)*100:>14.1f}% | {np.mean(s_good > c)*100:>20.2f}%")

# sensitivity to the grouping window
for tol in (5.0, 20.0):
    gb = []
    for ev_rows in by_event.values():
        for i, r in enumerate(ev_rows):
            if r["wrong"] is None or r["coherence"] < COHERENCE_MIN or r["type"] != "track":
                continue
            tp = set(r["tpcs"])
            mem = [q for q in ev_rows[: i + 1]
                   if abs(q["t0"] - r["t0"]) <= tol and tp & set(q["tpcs"])
                   and q["wrong"] is not None and q["coherence"] >= COHERENCE_MIN]
            gb.append(any(q["wrong"] for q in mem))
    gb = np.asarray(gb)
    t20 = np.mean(s[gb] > CUT_MARK) * 100
    f20 = np.mean(s[~gb] > CUT_MARK) * 100
    print(f"T0_TOL={tol:.0f}: group-bad={gb.mean():.1%} | at 20 sigma: catch {t20:.1f}% / false {f20:.2f}%")
