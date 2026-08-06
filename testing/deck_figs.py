"""Generate figures for the v0.3.0 algorithm + failure deck. Writes to deck_figs/."""
import os, sys, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.abspath(os.path.dirname(__file__))
for p in (HERE, os.path.join(HERE, "..", "CLMatching_AlphaRelease"),
          os.path.join(HERE, "..", "CLMatching_AlphaRelease", "M5p1")):
    sys.path.append(os.path.abspath(p))
from pulse_shapes import timeinterpolation
OUT = os.path.join(HERE, "deck_figs"); os.makedirs(OUT, exist_ok=True)
DUMPS = os.path.join(HERE, "v030D50fix_out", "dumps")     # FIXED base_image
PEAK, CLIP, CUT = 105, 60780.0, 10.0
BLUE, ORANGE, RED, GREEN = "#1f77b4", "#ff7f0e", "#d83a3f", "#2ca02c"

# ---------------- 1. pipeline flowchart ----------------
def flowchart():
    steps = [
        ("1 · Clustering", "group hits into\ncandidate interactions\n/ tracks per TPC\n(+ v0.2.5 vertex merge)"),
        ("2 · Charge->Light", "predict each cluster's\nper-channel light\nwaveform (Charge2Light)\n= match template"),
        ("3 · Backbone", "place confident tracks\n/ showers first:\nscan t0, soft-cosine\n+ overflow rescue"),
        ("4 · P2 place", "energy-banded\npermutation of the\n1-2 TPC small-\ncluster pool"),
        ("5 · V2 + police", "re-validate every\nplacement; police\nkeeps / rewinds /\nrescans at commit"),
        ("6 · Phase-3 expand", "absorb the <50 MeV\ntail into placed\ninteractions (geodesic\n+ light regret)"),
        ("7 · Matrix+force", "error-matrix sweeps\nthe tail; force-cover\nfills the rest ->\n100% coverage"),
    ]
    fig, ax = plt.subplots(figsize=(13.3, 2.7)); ax.axis("off")
    ax.set_xlim(0, len(steps)); ax.set_ylim(0, 1)
    for i, (t, d) in enumerate(steps):
        x = i + 0.5
        box = FancyBboxPatch((i + 0.05, 0.22), 0.90, 0.72, boxstyle="round,pad=0.015,rounding_size=0.025",
                             linewidth=1.4, edgecolor="#2b6cb0", facecolor="#eaf2fb")
        ax.add_patch(box)
        ax.text(x, 0.84, t, ha="center", va="center", fontsize=9.0, fontweight="bold", color="#1a365d")
        ax.text(x, 0.50, d, ha="center", va="center", fontsize=6.9, color="#2d3748", linespacing=1.3)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((i + 0.955, 0.58), (i + 1.045, 0.58), arrowstyle="-|>",
                                         mutation_scale=12, color="#2b6cb0", lw=1.6))
    ax.text(len(steps)/2, 0.05, "actual scintillation light  +  per-cluster predicted templates  ->  assign each charge hit its drift-time t0",
            ha="center", fontsize=8.4, style="italic", color="#4a5568")
    plt.tight_layout(); fig.savefig(os.path.join(OUT, "fig_pipeline.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------------- 2. failure taxonomy ----------------
def taxonomy():
    cats = ["SWAP", "LANDED\non flash", "OFF\nflash"]; vals = [91.0, 5.5, 3.5]
    cols = ["#c76a00", "#8a5cf6", "#4a90c2"]
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    b = ax.bar(cats, vals, color=cols, width=0.62, edgecolor="k", linewidth=0.6)
    for r, v in zip(b, vals): ax.text(r.get_x()+r.get_width()/2, v+1.5, f"{v:.0f}%", ha="center", fontsize=13, fontweight="bold")
    ax.set_ylabel("share of non-shower WRONG energy", fontsize=11)
    ax.set_ylim(0, 100); ax.set_title("Residual mistakes: 91% are pairwise t0 SWAPS", fontsize=12.5, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); fig.savefig(os.path.join(OUT, "fig_taxonomy.png"), dpi=200); plt.close(fig)

# ---------------- 3. results summary ----------------
def results():
    coh = ["ALL TPCs", "non-shower\n(<=1.4 GeV)", "shower\n(>1.4 GeV)"]
    hits = [94.87, 95.61, 93.35]; ener = [94.57, 95.24, 93.33]
    x = np.arange(len(coh)); w = 0.36
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    ax.bar(x-w/2, hits, w, label="% hits correct", color="#2b6cb0")
    ax.bar(x+w/2, ener, w, label="% energy correct", color="#63b3ed")
    for xi, (h, e) in enumerate(zip(hits, ener)):
        ax.text(xi-w/2, h+0.15, f"{h:.1f}", ha="center", fontsize=9)
        ax.text(xi+w/2, e+0.15, f"{e:.1f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(coh); ax.set_ylim(90, 98)
    ax.set_ylabel("correct t0 (|reco-truth| <= 10 ticks)", fontsize=10.5)
    ax.set_title("v0.3.0 accuracy, 39 events, 100% coverage", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9.5, loc="lower left"); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); fig.savefig(os.path.join(OUT, "fig_results.png"), dpi=200); plt.close(fig)

# ---------------- 4. light + charge panels (from FIXED dumps) ----------------
def load(ftag, ev): return np.load(os.path.join(DUMPS, f"{ftag}_ev{ev}.npz"), allow_pickle=True)

def light_charge(ftag, ev, tpc, tag, title, swap_labels=None):
    d = load(ftag, ev)
    x, y, z, e = [np.asarray(d[k], float) for k in ("x","y","z","e")]
    tp = np.asarray(d["tpc"]); tsf = np.asarray(d["ts_final"], float); tr = np.asarray(d["truth_t0"], float)
    lab = np.asarray(d["labels"]); m = tp == tpc
    act = np.asarray(d["full_light"], float)[tpc].sum(0)
    pred = np.asarray(d["base_image"], float)[tpc].sum(0)
    ok = m & np.isfinite(tr); wrong = ok & np.isfinite(tsf) & (np.abs(tsf-tr) > CUT)
    fig, (axL, axC) = plt.subplots(2, 1, figsize=(7.6, 6.2), gridspec_kw={"height_ratios":[1,1.15]})
    fig.suptitle(title, fontsize=12, fontweight="bold")
    axL.plot(act, color=BLUE, lw=1.3, label="actual light")
    axL.plot(pred, color=ORANGE, lw=1.3, label="predicted light (matcher)")
    if swap_labels:
        for L, col in zip(swap_labels, [RED, "#7a3fbf"]):
            ml = m & (lab == L)
            if ml.any():
                rp = float(np.median(tsf[ml])); tv = tr[ml][np.isfinite(tr[ml])]
                axL.axvline(rp+PEAK, color=col, ls="--", lw=1.4, label=f"L{L} placed t0={rp:.0f}")
                if tv.size: axL.axvline(float(np.median(tv))+PEAK, color=col, ls=":", lw=1.8, label=f"L{L} TRUE t0={np.median(tv):.0f}")
    axL.set_xlim(0, 999); axL.set_xlabel("tick"); axL.set_ylabel("ADC (sum ch)")
    axL.legend(fontsize=7.6, ncol=2, loc="upper right"); axL.set_title("light: actual vs predicted", fontsize=9.5)
    sc = axC.scatter(z[m], y[m], c=x[m], s=6+28*e[m]/max(e[m].max(),1e-9), cmap="viridis", alpha=0.85)
    axC.scatter(z[wrong], y[wrong], s=44, facecolors="none", edgecolors=RED, lw=0.8, label="wrong hits")
    plt.colorbar(sc, ax=axC, label="drift x (cm)")
    axC.set_xlabel("z (cm)"); axC.set_ylabel("y (cm)"); axC.legend(fontsize=8, loc="upper right")
    axC.set_title("charge (this TPC): red-ringed = wrong t0", fontsize=9.5)
    plt.tight_layout(rect=[0,0,1,0.96]); fig.savefig(os.path.join(OUT, f"fig_lc_{tag}.png"), dpi=150); plt.close(fig)

def find_good_tpc():
    """a high-efficiency non-shower TPC where predicted tracks actual well."""
    best=None
    for fp in sorted(glob.glob(os.path.join(DUMPS,"*.npz")))[:6]:
        d=np.load(fp,allow_pickle=True); tp=np.asarray(d["tpc"]); e=np.asarray(d["e"],float)
        tsf=np.asarray(d["ts_final"],float); tr=np.asarray(d["truth_t0"],float)
        for c in np.unique(tp):
            m=tp==c; E=float(e[m].sum())
            if not (300<E<1200): continue
            ok=m&np.isfinite(tr); w=float(e[ok&(np.abs(tsf-tr)>CUT)].sum())
            eff=1-w/max(float(e[ok].sum()),1e-9)
            ft=os.path.basename(fp).split("_ev")[0]; ev=int(str(d["event"]))
            if eff>0.99 and (best is None or E>best[3]): best=(ft,ev,int(c),E,eff)
    return best

if __name__ == "__main__":
    flowchart(); taxonomy(); results()
    light_charge("0000002", 1, 5, "tpc5", "TPC 5: two interactions SWAP flashes (t0 33 <-> 398)", swap_labels=[257,16])
    light_charge("0000001", 12, 38, "tpc38", "TPC 38: SWAP between two nearby flashes (349 vs 386)", swap_labels=[1440,126])
    light_charge("0000002", 12, 37, "tpc37", "TPC 37: OFF-flash near-miss (muon placed 12 ticks late)", swap_labels=[86])
    g = find_good_tpc()
    if g: light_charge(g[0], g[1], g[2], "good", f"Good match: TPC {g[2]} (eff={g[4]:.2f}) - predicted tracks actual")
    print("good tpc:", g)
    print("figures:", sorted(os.listdir(OUT)))
