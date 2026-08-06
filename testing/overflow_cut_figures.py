"""Figures for the overflow-cut study: wrong/correct histograms + cut-efficiency."""
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.abspath(os.path.dirname(__file__))
OUT_DIR = os.path.join(HERE, "overflow_study_out")
COHERENCE_MIN = 0.7      # label truth must be this coherent to count as a clean verdict
CUT_MARK = 20.0

rows = []
for fp in sorted(glob.glob(os.path.join(OUT_DIR, "rows_worker*.jsonl"))):
    with open(fp) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
print(f"loaded {len(rows)} placement rows from {OUT_DIR}")

n_events = len({(r["file"], r["event"]) for r in rows})
print(f"events covered: {n_events}")

usable = [r for r in rows if r["wrong"] is not None and r["coherence"] >= COHERENCE_MIN]
tracks = [r for r in usable if r["type"] == "track"]
others = [r for r in usable if r["type"] != "track"]
print(f"usable labels (truth coherent >= {COHERENCE_MIN}): {len(usable)} "
      f"of {len(rows)} ({len(rows) - len(usable)} ambiguous/no-truth dropped)")
print(f"  track-type: {len(tracks)} | shower/vertex: {len(others)}")

ov_wrong = np.array([r["max_overflow_sigma"] for r in tracks if r["wrong"]])
ov_right = np.array([r["max_overflow_sigma"] for r in tracks if not r["wrong"]])
print(f"track placements: correct={ov_right.size} wrong={ov_wrong.size} "
      f"(accuracy {ov_right.size / max(ov_right.size + ov_wrong.size, 1):.1%})")
if ov_wrong.size:
    print(f"  wrong  : median={np.median(ov_wrong):.1f} p25={np.percentile(ov_wrong, 25):.1f} "
          f"p75={np.percentile(ov_wrong, 75):.1f} min={ov_wrong.min():.1f} max={ov_wrong.max():.1f}")
if ov_right.size:
    print(f"  correct: median={np.median(ov_right):.1f} p25={np.percentile(ov_right, 25):.1f} "
          f"p75={np.percentile(ov_right, 75):.1f} min={ov_right.min():.1f} max={ov_right.max():.1f}")

# ---- histograms (shared log-x binning over the positive range) ----
_all = np.concatenate([ov_wrong, ov_right]) if ov_wrong.size else ov_right
pos_max = max(float(np.max(_all)), 50.0)
bins = np.concatenate([[-np.inf, 0.0], np.logspace(np.log10(0.3), np.log10(pos_max * 1.1), 40)])


def _hist_fig(vals, color, title, path):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    h, edges = np.histogram(vals, bins=bins)
    centers = [(0.2 if not np.isfinite(edges[i]) else np.sqrt(max(edges[i], 0.1) * edges[i + 1]))
               for i in range(len(h))]
    widths = [(0.15 if not np.isfinite(edges[i]) else edges[i + 1] - edges[i]) for i in range(len(h))]
    ax.bar(centers, h, width=widths, color=color, alpha=0.85, align="center")
    ax.axvline(CUT_MARK, color="k", ls="--", lw=1.4, label=f"proposed cut = {CUT_MARK:.0f} sigma")
    ax.set_xscale("log")
    ax.set_xlabel("max (pred - actual)/sigma over fit channels at placed t0 peak")
    ax.set_ylabel("placements")
    n_le0 = int(np.sum(vals <= 0))
    ax.set_title(f"{title}  (n={vals.size}; {n_le0} placements with overflow <= 0 shown in leftmost bin)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}")


_hist_fig(ov_wrong, "crimson",
          f"WRONGLY placed track clusters ({n_events} events)",
          os.path.join(HERE, "overflow_hist_wrong.png"))
_hist_fig(ov_right, "steelblue",
          f"CORRECTLY placed track clusters ({n_events} events)",
          os.path.join(HERE, "overflow_hist_correct.png"))

# ---- efficiency vs cut ----
cuts = np.logspace(np.log10(0.5), np.log10(max(pos_max, 100.0)), 200)
tpr = np.array([np.mean(ov_wrong > c) for c in cuts]) if ov_wrong.size else np.zeros_like(cuts)
fpr = np.array([np.mean(ov_right > c) for c in cuts]) if ov_right.size else np.zeros_like(cuts)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(cuts, tpr * 100, color="crimson", lw=2,
        label=f"bad clusters successfully hindered (n={ov_wrong.size})")
ax.plot(cuts, fpr * 100, color="steelblue", lw=2,
        label=f"good clusters wrongly blocked (n={ov_right.size})")
ax.axvline(CUT_MARK, color="k", ls="--", lw=1.4)
i20 = int(np.argmin(np.abs(cuts - CUT_MARK)))
ax.annotate(f"at {CUT_MARK:.0f} sigma:\ncatch {tpr[i20] * 100:.0f}% of bad\nblock {fpr[i20] * 100:.1f}% of good",
            xy=(CUT_MARK, tpr[i20] * 100), xytext=(CUT_MARK * 2.2, min(tpr[i20] * 100 + 15, 95)),
            arrowprops=dict(arrowstyle="->", lw=1), fontsize=10)
ax.set_xscale("log")
ax.set_xlabel("overflow cut [sigma]")
ax.set_ylabel("fraction of placements above cut [%]")
ax.set_title("Pause-on-overflow cut efficiency (phase-1 track placements)")
ax.grid(alpha=0.3)
ax.legend(loc="center left")
plt.tight_layout()
fig.savefig(os.path.join(HERE, "overflow_cut_efficiency.png"), dpi=140)
plt.close(fig)
print(f"wrote {os.path.join(HERE, 'overflow_cut_efficiency.png')}")

print("\ncut table (track placements):")
print(f"{'cut':>6} | {'bad caught':>11} | {'good blocked':>12}")
for c in (5, 10, 15, 20, 30, 50, 75, 100):
    t = float(np.mean(ov_wrong > c)) if ov_wrong.size else 0.0
    f_ = float(np.mean(ov_right > c)) if ov_right.size else 0.0
    print(f"{c:>6} | {t * 100:>10.1f}% | {f_ * 100:>11.2f}%")

# where do the blocked-good placements come from?
if ov_right.size:
    blocked = [r for r in tracks if (not r["wrong"]) and r["max_overflow_sigma"] > CUT_MARK]
    blocked.sort(key=lambda r: -r["max_overflow_sigma"])
    print(f"\ncorrect-but-blocked at {CUT_MARK:.0f} sigma: {len(blocked)}")
    for r in blocked[:10]:
        print(f"  {r['file']} ev{r['event']} label {r['label']:4d} E={r['energy_mev']:7.1f} MeV "
              f"t0={r['t0']:6.1f} ov={r['max_overflow_sigma']:7.1f}")
