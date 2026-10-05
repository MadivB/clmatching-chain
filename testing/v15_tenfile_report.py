"""v1.5 ten-file performance: per-file assignment accuracy (NS efficiency, all-TPC accuracy, energy-weighted, from the
event rows) and per-hit timing precision from the dumps: residual (t_reco - t_truth) in ns for hits whose final time is
within 5 ticks (and within 2 ticks) of truth, sigma = std and robust (1.4826 MAD), before the precision method
(ts_pre7: the stage-6 time; ts_pre6: the integer-tick assignment) and after (ts_final).  Histograms before/after.
python v15_tenfile_report.py <run_out_dir> <fig_dir>"""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NS = 16.0
out = sys.argv[1]; figdir = sys.argv[2]; os.makedirs(figdir, exist_ok=True)
rows = [json.loads(l) for f in sorted(glob.glob(f"{out}/groups_worker*.jsonl")) for l in open(f)]
# ---- accuracy per file (energy-weighted means over events, as v11_ab does)
byf = {}
for r in rows:
    byf.setdefault(os.path.basename(r["file"]), []).append(r)
print(f"{len(rows)} events in {len(byf)} files")
def wmean(rs, key, sub):
    num = den = 0.0
    for r in rs:
        m = r[sub] if sub else r
        v = m.get(key); w = r.get("total_energy", 1.0)
        if v is None or not np.isfinite(v): continue
        num += v * w; den += w
    return num / max(den, 1e-9)
tot = []
print("file                                NS eff   all-TPC acc   events  t_event median")
for f, rs in sorted(byf.items()):
    ns = wmean(rs, "ns_total_eff", "m_rescued"); acc = wmean(rs, "acc_rescued", None); t = np.median([r["t_event_s"] for r in rs])
    print(f"  {f[-26:]:34s} {100*ns:6.2f}   {100*acc:6.2f}        {len(rs):3d}   {t:6.1f} s"); tot.append((ns, acc))
print(f"  {'ALL (energy-weighted)':34s} {100*wmean(rows, 'ns_total_eff', 'm_rescued'):6.2f}   {100*wmean(rows, 'acc_rescued', None):6.2f}        {len(rows):3d}   {np.median([r['t_event_s'] for r in rows]):6.1f} s (p90 {np.percentile([r['t_event_s'] for r in rows], 90):.0f})")
# ---- precision from dumps
D = {"after": [], "pre7": [], "pre6": []}; E = []; TRU = []; FIN = []
for ff in sorted(glob.glob(f"{out}/full_dumps/*.npz")):
    z = np.load(ff, allow_pickle=True); tru = z["truth_t0"].astype(float); fin = z["ts_final"].astype(float); e = np.clip(z["e"], 0, None)
    m = np.isfinite(tru) & np.isfinite(fin) & (fin >= 0)
    D["after"].append(fin[m]); TRU.append(tru[m]); E.append(e[m])
    D["pre7"].append(z["ts_pre7"].astype(float)[m] if "ts_pre7" in z.files else np.full(m.sum(), np.nan))
    D["pre6"].append(z["ts_pre6"].astype(float)[m] if "ts_pre6" in z.files else np.full(m.sum(), np.nan))
tru = np.concatenate(TRU); e = np.concatenate(E); T = {k: np.concatenate(v) for k, v in D.items()}
res = {k: (T[k] - tru) * NS for k in T}
def stats(x):
    x = x[np.isfinite(x)]; med = np.median(x); mad = 1.4826 * np.median(np.abs(x - med)); return dict(n=int(x.size), mean=float(x.mean()), median=float(med), std=float(x.std()), robust=float(mad), p68=float(np.percentile(np.abs(x - med), 68)))
print("\nper-hit timing residual t_reco - t_truth [ns] (assigned hits with truth), window on the FINAL time:")
for win in (5, 2):
    sel = np.abs(res["after"]) <= win * NS
    print(f"  |dt_final| <= {win} ticks: {sel.sum()} hits ({100*sel.mean():.1f} % of assigned-with-truth), energy {e[sel].sum()/1e3:.1f} GeV")
    for k, lab in (("pre6", "before (integer-tick assignment, pre stage 6)"), ("pre7", "before precision (after subtick, pre stage 7)"), ("after", "after precision (ts_final)")):
        st = stats(res[k][sel]); print(f"     {lab:50s} std {st['std']:6.2f}  robust sigma {st['robust']:6.2f}  |dt| p68 {st['p68']:6.2f}  median {st['median']:6.2f}  mean {st['mean']:6.2f} ns")
# ---- histograms
fig, axs = plt.subplots(1, 2, figsize=(14, 5.2))
for ax, win in zip(axs, (5, 2)):
    sel = np.abs(res["after"]) <= win * NS; bins = np.linspace(-win * NS, win * NS, 161)
    for k, col, lab in (("pre6", "0.55", "before: integer-tick assignment"), ("pre7", "tab:blue", "before precision (after subtick)"), ("after", "tab:red", "after precision (v1.5 final)")):
        x = res[k][sel]; x = x[np.isfinite(x)]; st = stats(x)
        ax.hist(x, bins=bins, histtype="step", lw=1.6, color=col, label=f"{lab}: sigma {st['std']:.2f} ns, robust {st['robust']:.2f} ns")
    ax.set_xlabel("t_reco - t_truth [ns]"); ax.set_ylabel("hits"); ax.set_title(f"hits within {win} ticks of truth ({sel.sum():,} hits, 10 flow files)", fontsize=10); ax.legend(fontsize=8); ax.set_yscale("log")
fn = os.path.join(figdir, "v15_tenfile_precision_hist.png"); fig.tight_layout(); fig.savefig(fn, dpi=130); print("\nfigure", fn)
# linear-scale version
fig, axs = plt.subplots(1, 2, figsize=(14, 5.2))
for ax, win in zip(axs, (5, 2)):
    sel = np.abs(res["after"]) <= win * NS; bins = np.linspace(-win * NS, win * NS, 161)
    for k, col, lab in (("pre7", "tab:blue", "before precision"), ("after", "tab:red", "after precision")):
        x = res[k][sel]; x = x[np.isfinite(x)]; st = stats(x); ax.hist(x, bins=bins, histtype="step", lw=1.6, color=col, label=f"{lab}: sigma {st['std']:.2f} ns, robust {st['robust']:.2f} ns")
    ax.set_xlabel("t_reco - t_truth [ns]"); ax.set_ylabel("hits"); ax.set_title(f"hits within {win} ticks of truth", fontsize=10); ax.legend(fontsize=8)
fn2 = os.path.join(figdir, "v15_tenfile_precision_hist_linear.png"); fig.tight_layout(); fig.savefig(fn2, dpi=130); print("figure", fn2)
