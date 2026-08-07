"""Assignment-evolution gallery: how each cluster's t0 evolved through the
chain stages, focused on final failures.  Reads v085EVO_out (rows with
row['t_evo'] + npz dumps with t_evo/assign_stage/worst-label images).

Per example (worst wrong clusters): 4 panels —
  A hits (z-y) in the subject TPC colored correct/wrong, subject outlined
  B per-stage t0 timeline for the subject + biggest same-TPC clusters,
    truth times dashed
  C light closure on the subject's loudest channel (actual vs predicted)
  D the subject's own loss curve a_c(t) (itself removed from the base),
    with each stage's chosen t marked

Aggregate: classify every label's story (placed-wrong-never-fixed / broken-by
-stage-X / fixed-by-stage-Y) and tabulate wrong energy by story.

Usage: python evo_gallery.py [out_dir=v085EVO_out] [n_examples=10]
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v030_expansion import GAIN, _shift, weighted_scan

CUT = 10.0
ORDER = ["2+3.1_backbone_pool", "3.2_rescue_swap", "3.25_early_expand",
         "3.3+3.5_p2_v2", "3.9_joint", "3.9_expansion", "4.1_matrix",
         "4.2_matrix_relax", "3.95_pass2_trade", "3.97_pass3_family",
         "4.5_swap_fix", "4.9_force"]
SHORT = {"2+3.1_backbone_pool": "backbone\n+pool", "3.2_rescue_swap": "rescue",
         "3.25_early_expand": "early\nexpand", "3.3+3.5_p2_v2": "P2+V2",
         "3.9_joint": "joint", "3.9_expansion": "expand", "4.1_matrix": "matrix",
         "4.2_matrix_relax": "mtx\nrelax", "3.95_pass2_trade": "PASS2\ntrade",
         "3.97_pass3_family": "PASS3", "4.5_swap_fix": "swap\nfix",
         "4.9_force": "force"}


def load_events(out_dir):
    evs = []
    for f in sorted(glob.glob(os.path.join(out_dir, "dumps", "*.npz"))):
        d = np.load(f, allow_pickle=True)
        if "t_evo" not in d.files:
            continue
        ev = {k: d[k] for k in ("x", "y", "z", "e", "tpc", "labels",
                                "ts_final", "truth_t0", "base_image",
                                "full_light")}
        ev["t_evo"] = {s: {int(l): t for l, t in m.items()}
                       for s, m in json.loads(str(d["t_evo"])).items()}
        ev["assign_stage"] = (json.loads(str(d["assign_stage"]))
                              if "assign_stage" in d.files else {})
        ev["lab_rows"] = json.loads(str(d["lab_rows"]))
        ev["imgs"] = {tuple(int(v) for v in k.split("_")[1:]):
                      d[k].astype(np.float64)
                      for k in d.files if k.startswith("img_")}
        ev["name"] = os.path.basename(f).replace(".npz", "")
        evs.append(ev)
    return evs


def label_story(ev, lab, tr):
    """(first_stage, history[(stage, t)], story string)."""
    hist = [(s, ev["t_evo"][s][lab]) for s in ORDER
            if s in ev["t_evo"] and lab in ev["t_evo"][s]]
    if not hist or tr is None:
        return hist, "unassigned-or-no-truth"
    okv = [abs(t - tr) <= CUT for _, t in hist]
    first = hist[0][0]
    moved = any(abs(hist[i][1] - hist[i - 1][1]) > 2.0
                for i in range(1, len(hist)))
    if okv[-1]:
        if all(okv):
            return hist, f"correct from {SHORT[first]}".replace("\n", " ")
        i_fix = len(okv) - 1
        while i_fix > 0 and okv[i_fix - 1]:
            i_fix -= 1
        return hist, f"FIXED by {SHORT[hist[i_fix][0]]}".replace("\n", " ")
    if not any(okv):
        extra = " (later moves never found truth)" if moved else " (never touched again)"
        return hist, f"WRONG from placement [{SHORT[first]}]".replace("\n", " ") + extra
    i_br = next(i for i in range(len(okv) - 1, -1, -1) if okv[i]) + 1
    return hist, f"was correct, BROKEN by {SHORT[hist[i_br][0]]}".replace("\n", " ")


def own_curve(ev, lab, c):
    img = ev["imgs"].get((lab, c))
    if img is None:
        return None
    actual = ev["full_light"][c].astype(np.float64)
    base = ev["base_image"][c].astype(np.float64)
    n_ticks = actual.shape[1]
    m = (ev["labels"] == lab) & np.isfinite(ev["ts_final"]) & (ev["ts_final"] >= 0)
    t_fin = float(np.median(ev["ts_final"][m])) if m.any() else 0.0
    ch = np.argsort(img.max(axis=1))[::-1][:12]
    ch = ch[img[ch].max(axis=1) > 5.0]
    if not ch.size:
        return None
    own = _shift(img, t_fin, n_ticks)
    w = 1.0 / (GAIN * np.maximum(actual, 100.0))
    return weighted_scan(img[ch], base[ch] - own[ch] - actual[ch], w[ch])


def render_example(ev, r, out_png):
    lab, tr = r["label"], r["truth"]
    hist, story = label_story(ev, lab, tr)
    c_main = max([c for (l, c) in ev["imgs"] if l == lab],
                 key=lambda c: ev["imgs"][(lab, c)].max(), default=None)
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

    # ---- A: hits in the subject cluster's TPCs ----
    ax = fig.add_subplot(gs[0, 0])
    tpcs = set(r["tpcs"]) if r["tpcs"] else set(
        np.unique(ev["tpc"][ev["labels"] == lab]).tolist())
    sel = np.isin(ev["tpc"], sorted(tpcs))
    okh = (np.isfinite(ev["truth_t0"]) & np.isfinite(ev["ts_final"])
           & (ev["ts_final"] >= 0))
    good = okh & (np.abs(ev["ts_final"] - ev["truth_t0"]) <= CUT)
    sub = ev["labels"] == lab
    ax.scatter(ev["z"][sel & good & ~sub], ev["y"][sel & good & ~sub],
               s=2, c="seagreen", alpha=0.4, label="other, correct")
    ax.scatter(ev["z"][sel & okh & ~good & ~sub], ev["y"][sel & okh & ~good & ~sub],
               s=4, c="orange", alpha=0.6, label="other, wrong")
    ax.scatter(ev["z"][sub & good], ev["y"][sub & good], s=10, c="green",
               edgecolors="k", linewidths=0.2, label="subject, correct")
    ax.scatter(ev["z"][sub & ~good], ev["y"][sub & ~good], s=10, c="red",
               edgecolors="k", linewidths=0.2, label="subject, wrong")
    ax.set_xlabel("z [cm]"); ax.set_ylabel("y [cm]")
    ax.set_title(f"{ev['name']}  label {lab}  ({r['e_ok']:.0f} MeV, "
                 f"{r['e_wrong']:.0f} wrong)  TPC {sorted(tpcs)}")
    ax.legend(fontsize=7, loc="best")

    # ---- B: per-stage t0 timeline ----
    ax = fig.add_subplot(gs[0, 1])
    stages = [s for s in ORDER if s in ev["t_evo"]]
    xi = {s: i for i, s in enumerate(stages)}
    peers = sorted({rr["label"]: rr for rr in ev["lab_rows"]
                    if set(rr["tpcs"]) & tpcs and rr["label"] != lab}.values(),
                   key=lambda rr: -rr["e_ok"])[:6]
    for rr in peers:
        lp = rr["label"]
        hp = [(xi[s], ev["t_evo"][s][lp]) for s in stages
              if lp in ev["t_evo"][s]]
        if not hp:
            continue
        ln, = ax.plot(*zip(*hp), "-o", ms=3, lw=1, alpha=0.55,
                      label=f"{lp} ({rr['e_ok']:.0f} MeV)")
        if rr["truth"] is not None:
            ax.axhline(rr["truth"], color=ln.get_color(), ls=":", lw=0.8, alpha=0.5)
    hs = [(xi[s], t) for s, t in hist]
    if hs:
        ax.plot(*zip(*hs), "-o", color="red", ms=6, lw=2.5,
                label=f"subject {lab}")
    if tr is not None:
        ax.axhline(tr, color="red", ls="--", lw=1.5, alpha=0.8)
        ax.annotate("truth", (0.01, tr), xycoords=("axes fraction", "data"),
                    color="red", fontsize=8, va="bottom")
    stg = {k: v for k, v in ev["assign_stage"].items()
           if k.startswith(f"{lab}_")}
    tag = next(iter(stg.values()), {}).get("stage", "?") if stg else "?"
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels([SHORT[s] for s in stages], fontsize=7)
    ax.set_ylabel("t0 [ticks]")
    ax.set_title(f"t0 evolution — {story}   (assign tag: {tag})", fontsize=10)
    ax.legend(fontsize=6, loc="best")

    # ---- C: light closure on the loudest channel ----
    ax = fig.add_subplot(gs[1, 0])
    if c_main is not None:
        img = ev["imgs"][(lab, c_main)]
        ch = int(np.argmax(img.max(axis=1)))
        act = ev["full_light"][c_main][ch]
        pred = ev["base_image"][c_main][ch]
        n_ticks = act.shape[0]
        t_fin = hist[-1][1] if hist else 0.0
        own = _shift(img, t_fin, n_ticks)[ch]
        ts_ref = [t for t in [tr, t_fin] if t is not None]
        lo = int(max(0, min(ts_ref) - 60)); hi = int(min(n_ticks, max(ts_ref) + 200))
        xs = np.arange(lo, hi)
        ax.plot(xs, act[lo:hi], "b-", lw=1.2, label="actual")
        ax.plot(xs, pred[lo:hi], color="darkorange", lw=1.2, label="predicted (all)")
        ax.fill_between(xs, pred[lo:hi] - own[lo:hi], pred[lo:hi],
                        color="red", alpha=0.3, label="subject's share")
        if tr is not None:
            ax.axvline(tr, color="red", ls="--", lw=1, label="truth t")
        ax.axvline(t_fin, color="k", ls="-", lw=1, label="final t")
        ax.set_xlabel("tick"); ax.set_ylabel("ADC")
        ax.set_title(f"light closure  TPC {c_main} ch {ch}", fontsize=10)
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "no dumped image", ha="center")

    # ---- D: the subject's own loss curve ----
    ax = fig.add_subplot(gs[1, 1])
    curve = own_curve(ev, lab, c_main) if c_main is not None else None
    if curve is not None:
        ax.plot(curve, "k-", lw=0.8)
        for i, (s, t) in enumerate(hist):
            j = int(round(min(max(t, 0), len(curve) - 1)))
            ax.plot(j, curve[j], "o", ms=7,
                    color=plt.cm.viridis(i / max(len(hist) - 1, 1)),
                    label=f"{SHORT[s]} t={t:.0f}".replace("\n", " "))
        if tr is not None:
            j = int(round(min(max(tr, 0), len(curve) - 1)))
            ax.plot(j, curve[j], "r*", ms=16, label=f"truth t={tr:.0f}")
        ax.set_xlabel("candidate t0 [tick]"); ax.set_ylabel("weighted light loss")
        ax.set_title("subject's own loss curve a_c(t) (self removed from base)",
                     fontsize=10)
        ax.legend(fontsize=6, loc="best")
        ax.set_yscale("log")
    else:
        ax.text(0.5, 0.5, "no dumped image", ha="center")
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return story


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "v085EVO_out"
    n_ex = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    gal = os.path.join(out_dir, "gallery")
    os.makedirs(gal, exist_ok=True)
    evs = load_events(out_dir)
    print(f"{len(evs)} events with evolution dumps")

    # ---- aggregate story table over ALL labels ----
    from collections import defaultdict
    agg_wrong, agg_fixed = defaultdict(float), defaultdict(float)
    for ev in evs:
        for r in ev["lab_rows"]:
            _, story = label_story(ev, r["label"], r["truth"])
            key = story.split(" (")[0]
            if "WRONG" in story or "BROKEN" in story:
                agg_wrong[story if "placement" in story else key] += r["e_wrong"]
            elif "FIXED" in story:
                agg_fixed[key] += r["e_ok"]
    lines = ["# Assignment-evolution gallery (v0.8 canonical, file 0000001)\n",
             "## Where final wrong energy comes from (all 13 events)\n",
             "| story | wrong energy [MeV] |", "|---|---|"]
    tot = sum(agg_wrong.values())
    for k, v in sorted(agg_wrong.items(), key=lambda x: -x[1]):
        lines.append(f"| {k} | {v:.0f} ({100*v/max(tot,1e-9):.0f}%) |")
    lines += ["", "## Energy rescued during evolution (was wrong, ended correct)\n",
              "| story | energy [MeV] |", "|---|---|"]
    for k, v in sorted(agg_fixed.items(), key=lambda x: -x[1]):
        lines.append(f"| {k} | {v:.0f} |")

    # ---- per-example figures: worst wrong labels with dumped images ----
    cands = []
    for ev in evs:
        for r in ev["lab_rows"]:
            if r["e_wrong"] > 50.0 and any(l == r["label"] for l, _ in ev["imgs"]):
                cands.append((r["e_wrong"], ev, r))
    cands.sort(key=lambda x: -x[0])
    lines += ["", "## Examples (worst final failures)\n"]
    for i, (_, ev, r) in enumerate(cands[:n_ex]):
        png = os.path.join(gal, f"ex{i:02d}_{ev['name']}_lab{r['label']}.png")
        story = render_example(ev, r, png)
        print(f"ex{i:02d}: {ev['name']} label {r['label']} "
              f"({r['e_ok']:.0f} MeV, {r['e_wrong']:.0f} wrong) — {story}")
        lines.append(f"- **ex{i:02d}** `{os.path.abspath(png)}` — {ev['name']} "
                     f"label {r['label']}, {r['e_ok']:.0f} MeV "
                     f"({r['e_wrong']:.0f} wrong): {story}")
    with open(os.path.join(gal, "README.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("gallery:", os.path.abspath(gal))


if __name__ == "__main__":
    main()
