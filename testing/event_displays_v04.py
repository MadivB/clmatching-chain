"""#4 Event displays of events we got wrong, confidence in the title AND filename.
3D charge display: hits colored by whether their t0 is correct (green) / wrong (red)."""
import os, glob, pickle, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.abspath(os.path.dirname(__file__)); OUT = os.path.join(HERE, "deck_figs", "v04_wrong_events")
os.makedirs(OUT, exist_ok=True)
M = pickle.load(open(os.path.join(HERE, "master_v04.pkl"), "rb")); EV = M["EV"]
CUT = 10

# pick the worst events by wrong energy (and record their confidence)
order = np.argsort(EV["wrongE"])[::-1]
picks = [(EV["file"][i], int(EV["event"][i]), float(EV["conf_ewm"][i]), float(EV["conf_max"][i]),
          float(EV["wrongE"][i]), float(EV["acc"][i]) if EV["acc"][i] == EV["acc"][i] else np.nan) for i in order[:6]]
dumps = {os.path.basename(f).split("_ev")[0] + "_ev" + str(int(np.load(f, allow_pickle=True)["event"])): f
         for f in glob.glob(os.path.join(HERE, "v030v3_out", "dumps", "*.npz"))}

for ft, ev, conf, cmax, wE, acc in picks:
    fp = dumps.get(f"{ft}_ev{ev}")
    if fp is None:
        print("no dump for", ft, ev); continue
    d = np.load(fp, allow_pickle=True)
    x = np.asarray(d["x"], float); y = np.asarray(d["y"], float); z = np.asarray(d["z"], float)
    e = np.asarray(d["e"], float); tsf = np.asarray(d["ts_final"], float); tr = np.asarray(d["truth_t0"], float)
    ht = np.isfinite(tr); wrong = ht & (np.abs(tsf - tr) > CUT); right = ht & ~wrong
    fig = plt.figure(figsize=(13, 6))
    for k, (az, el) in enumerate([(-60, 20), (30, 12)]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        # subsample correct hits for speed, show all wrong
        ri = np.where(right)[0]; ri = ri[np.random.default_rng(0).permutation(len(ri))[:20000]] if len(ri) > 20000 else ri
        ax.scatter(z[ri], x[ri], y[ri], s=1, c="#7fbf7f", alpha=.12, label="correct t0")
        wi = np.where(wrong)[0]
        ax.scatter(z[wi], x[wi], y[wi], s=4, c="#d62728", alpha=.6, label="WRONG t0")
        ax.view_init(elev=el, azim=az); ax.set_xlabel("z"); ax.set_ylabel("x"); ax.set_zlabel("y")
        if k == 0: ax.legend(loc="upper left", fontsize=8, markerscale=4)
    frac = 100 * e[wrong].sum() / max(e[ht].sum(), 1e-9)
    fig.suptitle(f"{ft} ev{ev}  |  confidence(ew-mean)={conf:.2f}  max-TPC={cmax:.2f}  |  "
                 f"wrong E={wE:.0f} MeV ({frac:.1f}%)  acc={100*acc:.1f}%", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fn = f"wrongevt_{ft}_ev{ev}_conf{conf:.2f}_maxconf{cmax:.2f}_wrongE{wE:.0f}MeV.png"
    fig.savefig(os.path.join(OUT, fn), dpi=120); plt.close(fig); d.close()
    print("saved", fn)
print("DONE")
