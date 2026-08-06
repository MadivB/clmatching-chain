"""Multi-channel coincidence triggers for the pause-on-overflow check.

User designs:
  A) 3-adjacent: top support channel and its two physical neighbours (same
     wall, not saturation-vetoed) ALL exceed t sigma.
  B) top support channel > 20 sigma AND at least k other channels > 5 sigma.

Scored against the hit-level +-10-tick ground truth (a hit is wrongly placed
iff |reco_t0 - truth_t0| > 10): probe is POLLUTED if template-weighted
misplaced-hit energy lighting the probe tick exceeds 5 MeV in any probed TPC.
Baseline for comparison: single-channel max-over-support cut (full ROC).
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
SIGMA_FLOOR = 100.0
THRESH_MEV = 5.0
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
n_events = len(by_event)

# ---- ground truth per row (hit-level, +-10 ticks wrongness) ----
placements = []
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
    mis = ((h_lab >= 0) & np.isfinite(h_reco) & (h_reco >= 0)
           & np.isfinite(h_truth) & (np.abs(h_reco - h_truth) > 10.0))
    for r in sorted(ev_rows, key=lambda q: q["label"]):
        if r["type"] != "track" or r["wrong"] is None or not r.get("tpc_channels"):
            continue
        pol = 0.0
        for c in r["tpcs"]:
            m = mis & (h_tpc == int(c)) & (h_lab <= int(r["label"]))
            if not m.any():
                continue
            idx = np.round(PEAK + float(r["t0"]) - h_reco[m]).astype(int)
            w = np.where((idx >= 0) & (idx < tmpl.size), tmpl[np.clip(idx, 0, tmpl.size - 1)], 0.0)
            w[w < W_MIN] = 0.0
            pol = max(pol, float(np.sum(h_e[m] * w)))
        placements.append({"r": r, "polluted": pol > THRESH_MEV})

n_bad = sum(1 for p in placements if p["polluted"])
print(f"{len(placements)} placements ({n_events} events): polluted={n_bad} ({n_bad/len(placements):.1%})")


def per_tpc_ov(td):
    o = np.asarray(td["own"], dtype=np.float64)
    b = np.asarray(td["base"], dtype=np.float64)
    a = np.asarray(td["actual"], dtype=np.float64)
    ov = (o + b - a) / np.sqrt(np.maximum(a, SIGMA_FLOOR))
    ov[np.asarray(td["veto"], dtype=int)] = -np.inf if len(td["veto"]) else ov[[]]
    if td["veto"]:
        ov[np.asarray(td["veto"], dtype=int)] = -np.inf
    return ov, np.asarray(td["support"], dtype=int)


def trig_A(p, t):
    """top support channel and both same-wall neighbours all > t (any probed TPC)."""
    for td in p["r"]["tpc_channels"]:
        ov, sup = per_tpc_ov(td)
        if sup.size == 0:
            continue
        c = int(sup[np.argmax(ov[sup])])
        wall_lo, wall_hi = (0, 59) if c < 60 else (60, 119)
        group = [c] + [n for n in (c - 1, c + 1) if wall_lo <= n <= wall_hi and np.isfinite(ov[n])]
        if len(group) >= 3 and all(ov[g] > t for g in group):
            return True
    return False


def trig_B(p, top, k, oth, scope="all"):
    """top support channel > top AND >= k OTHER channels > oth (any probed TPC)."""
    for td in p["r"]["tpc_channels"]:
        ov, sup = per_tpc_ov(td)
        if sup.size == 0:
            continue
        c = int(sup[np.argmax(ov[sup])])
        if ov[c] <= top:
            continue
        pool = sup if scope == "support" else np.flatnonzero(np.isfinite(ov))
        n_other = int(np.sum(ov[pool] > oth)) - (1 if ov[c] > oth else 0)
        if n_other >= k:
            return True
    return False


def trig_base(p, cut):
    return p["r"]["max_overflow_sigma"] > cut


def score(fn, **kw):
    hits = np.array([fn(p, **kw) for p in placements])
    bad = np.array([p["polluted"] for p in placements])
    return float(np.mean(hits[bad])), float(np.mean(hits[~bad]))


# baseline ROC
cuts = np.logspace(np.log10(2), np.log10(500), 60)
base_tpr = []
base_fpr = []
for c in cuts:
    t, f = score(trig_base, cut=c)
    base_tpr.append(t)
    base_fpr.append(f)

print(f"\n{'trigger':<46} | {'catch':>7} | {'false':>7}")
results = []
for t in (5, 10, 15):
    tp, fp = score(trig_A, t=t)
    results.append((f"A: 3-adjacent all > {t}", tp, fp))
for k in (2, 4, 6):
    for oth in (3, 5, 10):
        tp, fp = score(trig_B, top=20, k=k, oth=oth, scope="all")
        results.append((f"B: top>20 & >={k} others > {oth} (all ch)", tp, fp))
tp, fp = score(trig_B, top=20, k=4, oth=5, scope="support")
results.append(("B: top>20 & >=4 others > 5 (support only)", tp, fp))
for top in (15, 30, 50):
    tp, fp = score(trig_B, top=top, k=4, oth=5, scope="all")
    results.append((f"B: top>{top} & >=4 others > 5 (all ch)", tp, fp))
for name, tp, fp in results:
    print(f"{name:<46} | {tp*100:>6.1f}% | {fp*100:>6.2f}%")

# reference points from the baseline
for c in (20, 30, 50):
    t, f = score(trig_base, cut=c)
    print(f"{f'baseline: max support ch > {c}':<46} | {t*100:>6.1f}% | {f*100:>6.2f}%")

# ---- ROC comparison figure ----
fig, ax = plt.subplots(figsize=(9.5, 7))
ax.plot(np.array(base_fpr) * 100, np.array(base_tpr) * 100, color="gray", lw=2,
        label="baseline: single-channel max cut (scanned)")
markers = ["o", "s", "^", "D", "v", "P", "*", "X", "<", ">", "h", "8", "p"]
for i, (name, tp, fp) in enumerate(results):
    ax.scatter(fp * 100, tp * 100, s=70, marker=markers[i % len(markers)], label=name, zorder=5)
ax.set_xlabel("false alarm on clean probes [%]")
ax.set_ylabel("polluted probes caught [%]")
ax.set_xscale("symlog", linthresh=1.0)
ax.set_title(f"Multi-channel coincidence triggers vs single-channel baseline\n"
             f"(hit-level +-10-tick truth, {n_bad} polluted / {len(placements) - n_bad} clean, {n_events} events)")
ax.grid(alpha=0.3)
ax.legend(fontsize=7.5, loc="lower right")
plt.tight_layout()
fig.savefig(os.path.join(HERE, "overflow_multich_roc.png"), dpi=140)
plt.close(fig)
print("\nwrote overflow_multich_roc.png")

# ---- continuous statistics for full ROC comparison ----
def stat_adj3(p):
    """max over support channels c of min(ov[c-1], ov[c], ov[c+1]) same-wall."""
    best = -np.inf
    for td in p["r"]["tpc_channels"]:
        ov, sup = per_tpc_ov(td)
        for c in sup:
            wall_lo, wall_hi = (0, 59) if c < 60 else (60, 119)
            if c - 1 < wall_lo or c + 1 > wall_hi:
                continue
            trio = ov[[c - 1, c, c + 1]]
            if np.all(np.isfinite(trio)):
                best = max(best, float(np.min(trio)))
    return best


def stat_pair(p):
    """same but 2-adjacent (min of best pair)."""
    best = -np.inf
    for td in p["r"]["tpc_channels"]:
        ov, sup = per_tpc_ov(td)
        for c in sup:
            for n in (c - 1, c + 1):
                if 0 <= n < 120 and (n // 60 == c // 60) and np.isfinite(ov[n]) and np.isfinite(ov[c]):
                    best = max(best, float(min(ov[c], ov[n])))
    return best


bad_mask = np.array([p["polluted"] for p in placements])


def roc(stat_vals):
    v = np.asarray(stat_vals, dtype=np.float64)
    cs = np.unique(v[np.isfinite(v)])
    tpr = np.array([np.mean(v[bad_mask] > c) for c in cs])
    fpr = np.array([np.mean(v[~bad_mask] > c) for c in cs])
    return cs, tpr, fpr


def best_at_budget(cs, tpr, fpr, budget):
    ok = fpr <= budget
    if not ok.any():
        return None
    i = int(np.argmax(tpr[ok]))
    return float(tpr[ok][i]), float(np.asarray(cs)[ok][i]), float(fpr[ok][i])


v_max = [p["r"]["max_overflow_sigma"] for p in placements]
v_adj3 = [stat_adj3(p) for p in placements]
v_pair = [stat_pair(p) for p in placements]
v_n5 = []
v_n5_given_top20 = []
for p in placements:
    nbest = 0
    for td in p["r"]["tpc_channels"]:
        ov, _ = per_tpc_ov(td)
        nbest = max(nbest, int(np.sum(ov[np.isfinite(ov)] > 5.0)))
    v_n5.append(nbest)
    v_n5_given_top20.append(nbest if p["r"]["max_overflow_sigma"] > 20 else -1)

print("\n=== optimal operating points at fixed false-alarm budgets (continuous ROC) ===")
print(f"{'statistic':<38} | {'TPR@FPR<=2%':>11} | {'TPR@FPR<=5%':>11} | {'TPR@FPR<=10%':>12}")
for name, vals in (("baseline: max support channel", v_max),
                   ("A-continuous: best 3-adjacent min", v_adj3),
                   ("A-continuous: best 2-adjacent min", v_pair),
                   ("B-continuous: N(ch>5s) given top>20", v_n5_given_top20),
                   ("multiplicity alone: N(ch>5s)", v_n5)):
    cs, tpr, fpr = roc(vals)
    cells = []
    for b in (0.02, 0.05, 0.10):
        r = best_at_budget(cs, tpr, fpr, b)
        cells.append(f"{r[0]*100:5.1f}% @>{r[1]:.3g}" if r else "   --")
    print(f"{name:<38} | {cells[0]:>11} | {cells[1]:>11} | {cells[2]:>12}")

# 2D grid for B: top-threshold x k(others>5)
print("\nB grid: catch% / false% for top>T & >=k others > 5 sigma (all channels)")
header = "k\\T   " + "  ".join(f"{t:>13}" for t in (10, 15, 20, 30, 50))
print(header)
for k in (1, 2, 3, 4, 6, 8):
    cells = []
    for t in (10, 15, 20, 30, 50):
        tp, fp = score(trig_B, top=t, k=k, oth=5, scope="all")
        cells.append(f"{tp*100:5.1f}/{fp*100:5.2f}")
    print(f"k={k:<3} " + "  ".join(f"{c:>13}" for c in cells))


# ---- multiplicity histogram: n channels > 5 sigma, polluted vs clean ----
def n_over(p, oth=5.0):
    best = 0
    for td in p["r"]["tpc_channels"]:
        ov, _ = per_tpc_ov(td)
        best = max(best, int(np.sum(ov[np.isfinite(ov)] > oth)))
    return best

n5_bad = [n_over(p) for p in placements if p["polluted"]]
n5_good = [n_over(p) for p in placements if not p["polluted"]]
fig, ax = plt.subplots(figsize=(10, 4.5))
bins = np.arange(0, 42) - 0.5
ax.hist(n5_good, bins=bins, color="steelblue", alpha=0.6, density=True, label=f"clean (n={len(n5_good)})")
ax.hist(n5_bad, bins=bins, color="crimson", alpha=0.6, density=True, label=f"polluted (n={len(n5_bad)})")
ax.set_xlabel("number of channels with (pred-actual)/sigma > 5 at the probe tick (max over probed TPCs)")
ax.set_ylabel("fraction of placements")
ax.set_title("Channel multiplicity above 5 sigma: polluted vs clean probes")
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(HERE, "overflow_multich_multiplicity.png"), dpi=140)
plt.close(fig)
print("wrote overflow_multich_multiplicity.png")
