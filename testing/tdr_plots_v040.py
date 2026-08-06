"""TDR diagrams, old vBeta3 format, regenerated with the newest classical matcher
(v0.4.0, 39 events).  Reproduces:
  hit_t0_residual_energy_hist_allEvents_{notebookStyle,compact}
  vertex_success_hist_all_{min1p0,min5p0}MeV_insideND     (completeness / truth vertex)
  vertex_purity_hist_all_{min1p0,min5p0}MeV_insideND      (purity / reco vertex)
  vertex_tpc_completeness_hist_allTPC_min5MeV
plus a summary json.  Definitions: truth vertex = distinct truth_t0 value per event;
reco vertex = distinct assigned-t0 group; correct hit = |reco-truth| <= 10 ticks;
1 tick = 16 ns.
"""
import glob, json, os, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.abspath(os.path.dirname(__file__))
# usage: tdr_plots_v040.py [dumps_dir] [tag] [out_dir]
DUMPS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "v030v3_out", "dumps")
TAG = sys.argv[2] if len(sys.argv) > 2 else "v0.4.0"
OUT = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "deck_figs", "tdr_v040")
RECO_GAP = float(sys.argv[4]) if len(sys.argv) > 4 else 5.0
TRUTH_GAP = float(sys.argv[5]) if len(sys.argv) > 5 else 5.0
os.makedirs(OUT, exist_ok=True)
CUT, NS = 10, 16.0
if len(sys.argv) > 6:
    CUT = float(sys.argv[6])   # correct-hit cut [ticks], default 10

res_ns, res_w = [], []
vtx = {}          # (file,event,truth_t0_rounded) -> [e_tot, e_ok]
rvx = {}          # (file,event,reco_t0_rounded)  -> [e_tot, e_ok]
vtpc = {}         # (file,event,truth_t0,tpc)     -> [e_tot, e_ok]
rtpc = {}         # (file,event,reco_t0,tpc)      -> [e_tot, e_ok]
N_EVENTS = 0
for fp in sorted(glob.glob(os.path.join(DUMPS, "*.npz"))):
    N_EVENTS += 1
    d = np.load(fp, allow_pickle=True)
    ft = os.path.basename(fp).split("_ev")[0]; ev = int(d["event"])
    e = np.asarray(d["e"], np.float64); tpc = np.asarray(d["tpc"])
    tsf = np.asarray(d["ts_final"], np.float64); tru = np.asarray(d["truth_t0"], np.float64)
    ok = np.isfinite(tru) & np.isfinite(tsf)
    corr = ok & (np.abs(tsf - tru) <= CUT)
    res_ns.append(((tru - tsf)[ok]) * NS); res_w.append(e[ok])

    def time_groups(vals, window=5.0):
        """cluster t0 values: gaps > window start a new vertex; returns group id per entry."""
        o = np.argsort(vals); sv = vals[o]
        newg = np.r_[True, np.diff(sv) > window]
        gid_sorted = np.cumsum(newg) - 1
        gid = np.empty(len(vals), int); gid[o] = gid_sorted
        return gid

    tg = time_groups(tru[ok], window=TRUTH_GAP); rg = time_groups(tsf[ok], window=RECO_GAP)
    for groups, gid, cvals in [(vtx, tg, corr[ok]), (rvx, rg, corr[ok])]:
        et = np.zeros(gid.max() + 1); ec = np.zeros(gid.max() + 1)
        np.add.at(et, gid, e[ok]); np.add.at(ec, gid, np.where(cvals, e[ok], 0.0))
        for i in range(len(et)):
            groups[(ft, ev, i)] = [et[i], ec[i]]
    # vertex-TPC deposits: (truth group, tpc)
    kk = tg.astype(np.int64) * 1000 + tpc[ok].astype(np.int64)
    u2, inv2 = np.unique(kk, return_inverse=True)
    et2 = np.zeros(len(u2)); ec2 = np.zeros(len(u2))
    np.add.at(et2, inv2, e[ok]); np.add.at(ec2, inv2, np.where(corr[ok], e[ok], 0.0))
    for i in range(len(u2)):
        vtpc[(ft, ev, int(u2[i]))] = [et2[i], ec2[i]]
    # reco-TPC groups: (reco group, tpc)
    kk_r = rg.astype(np.int64) * 1000 + tpc[ok].astype(np.int64)
    u3, inv3 = np.unique(kk_r, return_inverse=True)
    et3 = np.zeros(len(u3)); ec3 = np.zeros(len(u3))
    np.add.at(et3, inv3, e[ok]); np.add.at(ec3, inv3, np.where(corr[ok], e[ok], 0.0))
    for i in range(len(u3)):
        rtpc[(ft, ev, int(u3[i]))] = [et3[i], ec3[i]]
    d.close()

res = np.concatenate(res_ns); w = np.concatenate(res_w)
tot_E = float(w.sum())
mean = float(np.average(res, weights=w))
order = np.argsort(res); cw = np.cumsum(w[order]) / tot_E
median = float(res[order][np.searchsorted(cw, 0.5)])
q16 = float(res[order][np.searchsorted(cw, 0.16)]); q84 = float(res[order][np.searchsorted(cw, 0.84)])
std = float(np.sqrt(np.average((res - mean) ** 2, weights=w)))
within400 = float(w[np.abs(res) <= 400].sum() / tot_E)
within_cut = float(w[np.abs(res) <= CUT * 16].sum() / tot_E)   # the standard cut
summary = {"total_energy_MeV": round(tot_E, 1), "weighted_mean_ns": round(mean, 2),
           "weighted_median_ns": round(median, 2), "weighted_std_ns": round(std, 2),
           "weighted_16_84_ns": [round(q16, 2), round(q84, 2)],
           f"energy_within_{int(CUT*16)}ns_{int(CUT)}ticks": round(within_cut, 4),
           "energy_within_400ns": round(within400, 4), "n_events": N_EVENTS, "algorithm": TAG}


def residual_plot(figsize, fname, legend_16_84):
    fig, ax = plt.subplots(figsize=figsize)
    bins = np.arange(-200, 201, 5)
    hw, _ = np.histogram(res, bins=bins, weights=w)      # out-of-range tails not drawn (as in vBeta3)
    ax.bar(bins[:-1] + 2.5, hw / tot_E, width=4.5, color="#4878a8")
    ax.axvline(0, color="black", ls="--", lw=1.5)
    ax.set_xlabel("Truth t0 - Reco t0 [ns]"); ax.set_ylabel("Proportion of total energy")
    ax.set_title("Energy-Weighted Truth t0 - Reco t0", fontsize=14)
    ax.set_xlim(-200, 200); ax.grid(alpha=.25, ls=":")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    lines = [f"Total energy: {tot_E:.1f} MeV", f"Weighted mean: {mean:.2f}",
             f"Weighted median: {median:.2f}", f"Weighted std: {std:.2f}"]
    if legend_16_84:
        lines.append(f"Weighted 16-84%: [{q16:.2f}, {q84:.2f}]")
    lines.append(f"Energy within ±{int(CUT*16)} ns (±{int(CUT)} ticks): {100*within_cut:.1f}%")
    ax.text(0.985, 0.975, "\n".join(lines), transform=ax.transAxes, ha="right", va="top",
            fontsize=10, bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.8"))
    fig.tight_layout(); fig.savefig(os.path.join(OUT, fname), dpi=150); plt.close(fig)
    print("wrote", fname)


residual_plot((11, 6), f"hit_t0_residual_energy_hist_allEvents_notebookStyle_{TAG}.png", True)
residual_plot((7, 7), f"hit_t0_residual_energy_hist_allEvents_compact_{TAG}.png", False)


def frac_hist(groups, min_e, fname, title, xlabel, color):
    fr = np.array([100 * g[1] / g[0] for g in groups.values() if g[0] >= min_e])
    med = float(np.median(fr))
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.hist(fr, bins=np.linspace(0, 100, 22), color=color, edgecolor="white", lw=0.6)
    ax.axvline(med, color="red", ls="--", lw=2, label=f"Median = {med:.2f}%")
    ax.set_xlabel(xlabel); ax.set_ylabel("Count"); ax.set_title(title, fontsize=13)
    ax.set_xlim(0, 100); ax.grid(alpha=.25, ls=":"); ax.legend(loc="upper left", fontsize=10)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, fname), dpi=150); plt.close(fig)
    print(f"wrote {fname}  (n={len(fr)}, median={med:.2f}%)")
    return dict(n=len(fr), median_pct=round(med, 2))


for me, lab in [(1.0, "min1p0MeV"), (5.0, "min5p0MeV")]:
    summary[f"vertex_completeness_{lab}"] = frac_hist(
        vtx, me, f"vertex_success_hist_all_{lab}_insideND_{TAG}.png",
        f"Interaction Completeness (E >= {me:.0f} MeV, inside ND)",
        "Correctly reconstructed energy fraction per truth interaction [%]", "#4878a8")
    summary[f"vertex_purity_{lab}"] = frac_hist(
        rvx, me, f"vertex_purity_hist_all_{lab}_insideND_{TAG}.png",
        f"Interaction Purity (E >= {me:.0f} MeV, inside ND)",
        "Correctly assigned energy fraction per reconstructed interaction [%]", "#5aa14f")
for me, lab in [(1.0, "min1MeV"), (5.0, "min5MeV")]:
    summary[f"vertex_tpc_completeness_{lab}"] = frac_hist(
        vtpc, me, f"vertex_tpc_completeness_hist_allTPC_{lab}_{TAG}.png",
        f"Aggregate Interaction-TPC Completeness (deposited E >= {me:.0f} MeV)",
        "Completeness per interaction-TPC deposit [%]", "#4878a8")
    summary[f"vertex_tpc_purity_{lab}"] = frac_hist(
        rtpc, me, f"vertex_tpc_purity_hist_allTPC_{lab}_{TAG}.png",
        f"Aggregate Interaction-TPC Purity (assigned E >= {me:.0f} MeV)",
        "Correctly assigned energy fraction per reconstructed interaction-TPC group [%]", "#5aa14f")

json.dump(summary, open(os.path.join(OUT, f"{TAG}_final_hist_summary.json"), "w"), indent=2)
print(json.dumps(summary, indent=2))
print("DONE ->", OUT)
