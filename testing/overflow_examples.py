"""Diagnose correct-but-high-overflow placements + fix the histogram underflow bin.

Uses the per-channel [own, base_before, actual] dumps at each placement's
peak tick, so everything runs offline from overflow_study_out/*.jsonl.
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
SIGMA_FLOOR = 100.0
ADC_CLIP_NEAR = 60000.0
CUT = 20.0

rows = []
for fp in sorted(glob.glob(os.path.join(OUT_DIR, "rows_worker*.jsonl"))):
    with open(fp) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
usable = [r for r in rows if r["wrong"] is not None and r["coherence"] >= 0.7
          and r["type"] == "track" and r.get("fit_channels_oba")]
correct = [r for r in usable if not r["wrong"]]
wrong = [r for r in usable if r["wrong"]]
n_events = len({(r["file"], r["event"]) for r in rows})


def decompose(r):
    oba = np.asarray(r["fit_channels_oba"], dtype=np.float64)
    o, b, a = oba[:, 0], oba[:, 1], oba[:, 2]
    sig = np.sqrt(np.maximum(a, SIGMA_FLOOR))
    ov = (o + b - a) / sig
    k = int(np.argmax(ov))
    return {
        "ov": ov, "o": o, "b": b, "a": a, "sig": sig, "k": k,
        "max_ov": float(ov[k]),
        "base_only_ov_at_k": float((b[k] - a[k]) / sig[k]),
        "own_frac_at_k": float(o[k] / max(o[k] + b[k], 1e-9)),
        "a_near_clip_at_k": bool(a[k] > ADC_CLIP_NEAR),
        "base_only_max_ov": float(np.max((b - a) / sig)),
    }


# ---------- population diagnosis: correct placements above the 20-sigma cut ----------
blocked = [r for r in correct if r["max_overflow_sigma"] > CUT]
n_inherited = n_own = n_clip = 0
for r in blocked:
    d = decompose(r)
    if d["a_near_clip_at_k"]:
        n_clip += 1
    if d["base_only_ov_at_k"] > CUT:
        n_inherited += 1          # base alone already overflowed before this cluster
    elif d["own_frac_at_k"] >= 0.5:
        n_own += 1                # this cluster's own prediction dominates the excess
print(f"correct placements above {CUT:.0f} sigma: {len(blocked)} of {len(correct)}")
print(f"  inherited (base alone already > {CUT:.0f} sigma at the max channel): "
      f"{n_inherited} ({n_inherited / max(len(blocked), 1):.0%})")
print(f"  own-prediction-dominated (own >= 50% of pred at max channel, base not guilty): "
      f"{n_own} ({n_own / max(len(blocked), 1):.0%})")
print(f"  max channel near ADC clip (actual > {ADC_CLIP_NEAR:.0f}): {n_clip} ({n_clip / max(len(blocked), 1):.0%})")

# same for wrong placements (context)
w_inherited = sum(1 for r in wrong if r["max_overflow_sigma"] > CUT
                  and decompose(r)["base_only_ov_at_k"] > CUT)
w_above = sum(1 for r in wrong if r["max_overflow_sigma"] > CUT)
print(f"wrong placements above cut with inherited base overflow: {w_inherited}/{w_above}")

# ---------- example gallery ----------
blocked.sort(key=lambda r: -r["max_overflow_sigma"])
picks = []
for target in (None, None, 100.0, 50.0, 30.0, 22.0):   # 2 highest + spread
    if target is None:
        cand = [r for r in blocked if r not in picks]
    else:
        cand = sorted((r for r in blocked if r not in picks),
                      key=lambda r: abs(r["max_overflow_sigma"] - target))
    if cand:
        picks.append(cand[0])

for i, r in enumerate(picks):
    d = decompose(r)
    ov, o, b, a, k = d["ov"], d["o"], d["b"], d["a"], d["k"]
    n = ov.size
    x = np.arange(n)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    ax1.bar(x, b, color="lightgray", label="base before this cluster")
    ax1.bar(x, o, bottom=b, color="mediumpurple", label="this cluster (own)")
    ax1.step(np.arange(n + 1) - 0.5, np.append(a, a[-1]), where="post",
             color="k", lw=1.4, label="actual light")
    ax1.axhline(60780, color="red", ls=":", lw=0.8, label="ADC clip")
    ax1.set_ylabel("ADC at peak tick")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.set_title(
        f"CORRECT placement, max overflow {d['max_ov']:.0f} sigma | {r['file'].split('.')[-3]} "
        f"ev{r['event']} label {r['label']} | E={r['energy_mev']:.0f} MeV, {r['n_tpcs']} TPC(s) | "
        f"t0={r['t0']:.1f} vs truth {r['truth_t0']:.1f} (|d|={abs(r['t0'] - r['truth_t0']):.1f} ticks)\n"
        f"at max channel: own={o[k]:.0f} base={b[k]:.0f} actual={a[k]:.0f} "
        f"(own fraction {d['own_frac_at_k']:.0%}; base-only overflow {d['base_only_ov_at_k']:.0f} sigma"
        f"{'; NEAR CLIP' if d['a_near_clip_at_k'] else ''})", fontsize=9)

    colors = ["crimson" if j == k else "steelblue" for j in range(n)]
    ax2.bar(x, ov, color=colors)
    ax2.axhline(CUT, color="k", ls="--", lw=1, label=f"{CUT:.0f} sigma")
    ax2.axhline(0, color="k", lw=0.6)
    ax2.set_ylabel("(pred - actual)/sigma")
    ax2.set_xlabel("fit channel (stacked over the label's TPCs; crimson = max channel)")
    ax2.legend(fontsize=8)
    plt.tight_layout()
    out = os.path.join(HERE, f"overflow_example_correct_{i}.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}  (max_ov={d['max_ov']:.0f})")

# ---------- histograms with an honest underflow bin ----------
ov_w = np.array([r["max_overflow_sigma"] for r in wrong])
ov_c = np.array([r["max_overflow_sigma"] for r in correct])
pos_max = max(float(max(ov_w.max(), ov_c.max())), 50.0)
log_bins = np.logspace(np.log10(1.0), np.log10(pos_max * 1.1), 36)

for vals, color, tag, fname in (
    (ov_w, "crimson", "WRONGLY placed", "overflow_hist_wrong.png"),
    (ov_c, "steelblue", "CORRECTLY placed", "overflow_hist_correct.png"),
):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    h, edges = np.histogram(vals[vals > 1.0], bins=log_bins)
    centers = np.sqrt(edges[:-1] * edges[1:])
    ax.bar(centers, h, width=np.diff(edges), color=color, alpha=0.85)
    n_le = int(np.sum(vals <= 1.0))
    ax.bar([0.55], [n_le], width=0.35, color="dimgray", alpha=0.8)
    ax.text(0.55, n_le, f" overflow <= 1 sigma\n (incl. all <= 0)\n n={n_le}",
            ha="center", va="bottom", fontsize=8)
    ax.axvline(CUT, color="k", ls="--", lw=1.4, label=f"proposed cut = {CUT:.0f} sigma")
    ax.set_xscale("log")
    ax.set_xlabel("max (pred - actual)/sigma over fit channels at placed t0 peak")
    ax.set_ylabel("placements")
    ax.set_title(f"{tag} track clusters ({n_events} events, n={vals.size}) — gray bar is the underflow bin, "
                 f"drawn off-scale (log axis cannot show <= 0)", fontsize=10)
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(HERE, fname), dpi=140)
    plt.close(fig)
    print(f"rewrote {os.path.join(HERE, fname)}")
