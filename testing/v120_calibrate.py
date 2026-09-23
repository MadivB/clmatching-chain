"""In-situ calibration of the v1.2 precision stage (no truth): per-channel-index
delay k_idx[120] and per-detector-type slew beta_T by alternating medians over
the families of a dump set.  python v120_calibrate.py <dump_dir> <flow_file> <out.json> [max_events] [passes]
Evaluation vs truth (per-hit / per-family sigma before and after) is printed for
information only; nothing in the calibration uses it."""
import glob, json, os, sys
import numpy as np
HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, HERE)
import v120_precision as vp


def evaluate(files, flow, calib_path, tag):
    rE, rN, w = [], [], []
    fE, fN, fw = [], [], []
    tsum = 0.0; nfit = 0
    for ff in files:
        new_t, sig, st, z = vp.run_on_dump(ff, flow, calib_path)
        tsum += st["time_s"]; nfit += st["n_fit"]
        tru = z["truth_t0"].astype(float); ts = z["ts_final"].astype(float); e = np.clip(z["e"], 0, None); tpc = z["tpc"]
        ok = np.isfinite(tru) & (tru >= 0) & np.isfinite(ts) & (ts >= 0) & (np.abs(ts - tru) <= 5) & np.isfinite(new_t)
        changed = ok & (np.abs(new_t - ts) > 1e-9)
        rE.append((ts[changed] - tru[changed]) * vp.NS); rN.append((new_t[changed] - tru[changed]) * vp.NS); w.append(e[changed])
        for c in np.unique(tpc[changed]):
            for tv in np.unique(ts[changed & (tpc == c)]):
                m = changed & (tpc == c) & (ts == tv)
                if e[m].sum() >= 5:
                    fE.append(np.average((ts[m] - tru[m]) * vp.NS, weights=e[m])); fN.append(np.average((new_t[m] - tru[m]) * vp.NS, weights=e[m])); fw.append(e[m].sum())
    rE, rN, w = map(np.concatenate, (rE, rN, w)); fE, fN, fw = map(np.array, (fE, fN, fw))
    def rob(v, ww):
        o = np.argsort(v); cw = np.cumsum(ww[o]); med = v[o][np.searchsorted(cw, 0.5 * cw[-1])]
        d = np.abs(v - med); o2 = np.argsort(d); cw2 = np.cumsum(ww[o2]); return 1.4826 * d[o2][np.searchsorted(cw2, 0.5 * cw2[-1])]
    def wsd(v, ww):
        mu = np.average(v, weights=ww); return np.sqrt(np.average((v - mu) ** 2, weights=ww))
    print(f"[{tag}] {len(files)} ev, {nfit} families, {tsum/len(files):.1f} s/ev | per-hit stage E {wsd(rE,w):.2f}/{rob(rE,w):.2f} -> P {wsd(rN,w):.2f}/{rob(rN,w):.2f} ns | per-family {wsd(fE,fw):.2f}/{rob(fE,fw):.2f} -> {wsd(fN,fw):.2f}/{rob(fN,fw):.2f} ns")


def calibrate(files, flow, out, passes):
    k = np.zeros(120); beta_t = {0: 0.0, 1: 0.0}; type_of = np.zeros(120, int)
    for it in range(passes):
        tmp = out + ".tmp.json"
        json.dump(dict(k_idx=k.tolist(), beta_t={str(a): b for a, b in beta_t.items()}, type_of=type_of.tolist(), beta=0.0), open(tmp, "w"))
        S = []
        for ff in files:
            new_t, sig, st, z = vp.run_on_dump(ff, flow, tmp, collect=True)
            for fam in st["rows"]:
                for (ch, y, A, pile, yc) in fam["chan"]:
                    if pile < 0.1:
                        S.append((yc * vp.NS - (fam["t0"] - fam["t_chain"]) * vp.NS, np.log10(max(A, 1.0)), ch))   # corrected residual
        S = np.array(S)
        upd = np.zeros(120); cnt = np.zeros(120, int)
        for i in range(120):
            m = S[:, 2] == i
            if m.sum() >= 10:
                upd[i] = np.median(S[m, 0]); cnt[i] = m.sum()
        upd -= np.median(upd[cnt > 0])
        k = k + upd
        thr = np.median(k); type_of = (k < thr - 2.0).astype(int)
        for tp in (0, 1):
            m = type_of[S[:, 2].astype(int)] == tp
            if m.sum() > 100:
                c_ = np.polyfit(S[m, 1], S[m, 0] - upd[S[m, 2].astype(int)], 1)
                beta_t[tp] = beta_t[tp] + float(c_[0])
        print(f"pass {it+1}: |upd| median {np.median(np.abs(upd[cnt>0])):.2f} ns, k spread {k.std():.2f} ns, type-1 n={type_of.sum()} mean k {k[type_of==1].mean():+.2f}, type-0 {k[type_of==0].mean():+.2f}; beta_T {beta_t}")
        os.remove(tmp)
    json.dump(dict(k_idx=k.tolist(), beta_t={str(a): b for a, b in beta_t.items()}, type_of=type_of.tolist(), beta=0.0,
                   meta=dict(n_events=len(files), passes=passes, source=os.path.dirname(files[0]))), open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    d, flow, out = sys.argv[1], sys.argv[2], sys.argv[3]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 13
    passes = int(sys.argv[5]) if len(sys.argv) > 5 else 3
    files = sorted(glob.glob(f"{d}/full_dumps/*.npz"))[:n]
    evaluate(files[:2], flow, None, "uncalibrated, 2 ev")
    calibrate(files, flow, out, passes)
    evaluate(files[:2], flow, out, "calibrated, 2 ev")
