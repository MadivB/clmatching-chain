"""Direct Michel-electron metric on ND chain dumps: per true Michel (e+- from mu decay, incl. its brems descendants),
the fraction of its charge energy timed within 10 ticks of truth, for one or two arms (e.g. v1.4 and v1.4 + stage 4.85),
split by merged-with-the-muon vs own cluster, light visible at the Michel time (family within 5 ticks / loose
waveform peak within 5 ticks), delay bins, and whether the Michel is inside the 885-tick window.
python v15_michel_metric.py <arm_dir> [<arm2_dir>] [--events N]"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.signal import find_peaks

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from nd_truth_michel import open_flow, event_trajectories, hit_dominant_traj, ancestry_tag, michel_trajectories  # noqa: E402

CUT = 10.0; PEAK = 105


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("arms", nargs="+"); ap.add_argument("--events", type=int, default=0); ap.add_argument("--out", default=os.path.join(HERE, "michel_nd_out", "michel_metric.json"))
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.arms[0], "full_dumps", "*.npz")))
    if a.events:
        files = files[: a.events]
    rows = []
    for f in files:
        tag = os.path.basename(f)[:-4]
        Z = [np.load(os.path.join(arm, "full_dumps", tag + ".npz"), allow_pickle=True) for arm in a.arms]
        z = Z[0]; h5 = open_flow(str(z["src_file"])); ev_id = int(z["event"])
        ids = np.unique(np.asarray(h5["mc_truth/interactions/data"]["event_id"]))
        mc_event = int(ids[ev_id]) if ev_id < ids.size else None
        if mc_event is None:
            continue
        e = np.clip(z["e"], 0, None).astype(float); tru = z["truth_t0"].astype(float); labs = z["labels"].astype(int); tpc = z["tpc"].astype(int)
        hit_refs = z["hit_refs"].astype(np.int64); X = np.stack([z["x"], z["y"], z["z"]], 1).astype(float)
        traj = event_trajectories(h5, mc_event); dom = hit_dominant_traj(h5, hit_refs, traj)
        mich = michel_trajectories(traj)
        if not mich:
            continue
        tag_m = ancestry_tag(traj, dom, [m for m, _ in mich])
        cnt_e = np.bincount(dom[dom >= 0], weights=e[dom >= 0], minlength=traj["n"])
        TS = [np.asarray(zz["ts_final"], float) for zz in Z]
        full = z["full_light"]; veto = z["veto_mask"]; asg0 = np.isfinite(TS[0]) & (TS[0] >= 0)
        for m_ft, mu_ft in mich:
            hm = np.flatnonzero(tag_m == m_ft); ok = np.isfinite(tru[hm])
            hm = hm[ok]
            if hm.size == 0 or e[hm].sum() < 1.0:
                continue
            # HOST = the charged parent track that stopped here: the muon if it left >= 2 MeV, else its parent (pi+ -> mu+ at rest:
            # the 4 MeV muon is invisible; most ND Michels come from stopped pions), else the muon anyway
            host = mu_ft; host_kind = "mu"
            if cnt_e[mu_ft] < 2.0:
                gp = int(traj["parent_row"][mu_ft])
                if gp >= 0 and abs(int(traj["pdg"][gp])) in (211, 321) and cnt_e[gp] >= 2.0:
                    host = gp; host_kind = "pi" if abs(int(traj["pdg"][gp])) == 211 else "K"
            hmu = np.flatnonzero(dom == host); hmu = hmu[np.isfinite(tru[hmu])]
            E_m = float(e[hm].sum()); t_m = float(np.median(tru[hm])); t_mu = float(np.median(tru[hmu])) if hmu.size else np.nan
            c = int(np.bincount(tpc[hm]).argmax())
            # merged: the cluster holding most Michel energy also holds >= 2 MeV of the muon
            labs_m = labs[hm]; Lm = int(np.bincount(labs_m[labs_m >= 0], weights=e[hm][labs_m >= 0]).argmax()) if (labs_m >= 0).any() else -1
            E_mu_in = float(e[hmu][labs[hmu] == Lm].sum()) if (Lm >= 0 and hmu.size) else 0.0
            merged = Lm >= 0 and E_mu_in >= 2.0
            frac_in_cluster = float(e[hm][labs_m == Lm].sum() / E_m) if Lm >= 0 else 0.0
            # light at the Michel time (arm 0 state): a family within 5 ticks; a loose waveform peak within 5 ticks
            fam = np.unique(np.round(TS[0][(tpc == c) & asg0 & (labs >= 0)])); fam5 = bool(fam.size and np.min(np.abs(fam - t_m)) <= 5)
            ch = np.flatnonzero(~veto[c]); ff5 = False
            if ch.size:
                s = full[c][ch].astype(float).sum(0); sig = max(float(s[:90].std()), 1.0); pk, _ = find_peaks(s, prominence=10 * sig, distance=4)
                tpk = pk - PEAK; ff5 = bool(tpk.size and np.min(np.abs(tpk - t_m)) <= 5)
            corr = [float((e[hm] * (np.abs(ts[hm] - tru[hm]) <= CUT)).sum() / E_m) for ts in TS]
            at_mu = [float((e[hm] * (np.abs(ts[hm] - t_mu) <= CUT)).sum() / E_m) for ts in TS] if np.isfinite(t_mu) else [np.nan] * len(TS)
            j = int(m_ft)
            rows.append(dict(ev=tag, michel=m_ft, muon=mu_ft, host_kind=host_kind, E=E_m, n_hits=int(hm.size), E_mu=float(e[hmu].sum()), t_m=t_m, t_mu=t_mu, delay=t_m - t_mu, in_window=bool(0 <= t_m <= 885),
                             tpc=c, merged=bool(merged), E_mu_in_cluster=E_mu_in, frac_in_cluster=frac_in_cluster, fam5=fam5, ff5=ff5, corr=corr, at_mu=at_mu,
                             KE=float(traj["E_start"][j]) - 0.511, x_end=[float(v) for v in traj["xyz_start"][j]]))
    json.dump(rows, open(a.out, "w"))
    R = rows; n_ev = len(files)
    print(f"{n_ev} events, {len(R)} Michel electrons with >= 1 MeV of charge ({len(R)/n_ev:.1f} per spill); KE p10/50/90 {np.percentile([r['KE'] for r in R],[10,50,90]).round(0)} MeV; charge E p10/50/90 {np.percentile([r['E'] for r in R],[10,50,90]).round(1)} MeV")
    names = [os.path.basename(x.rstrip('/')) for x in a.arms]

    def line(lab, sel):
        if not sel:
            return
        E = np.array([r["E"] for r in sel]); out = f"  {lab:46s} n {len(sel):4d}  E {E.sum()/1e3:5.2f} GeV"
        for k, nm in enumerate(names):
            c = np.array([r["corr"][k] for r in sel]); am = np.array([r["at_mu"][k] for r in sel])
            out += f" | {nm}: E-frac correct {np.average(c, weights=E):.3f}  >=50% {np.mean(c >= 0.5):.2f}  at muon time {np.nanmean(am):.2f}"
        print(out)
    inw = [r for r in R if r["in_window"]]
    line("all Michels", R); line("in window (t <= 885)", inw); line("after the window", [r for r in R if not r["in_window"]])
    line("in window, merged with the muon", [r for r in inw if r["merged"]]); line("in window, own cluster", [r for r in inw if not r["merged"]])
    line("in window, family within 5 ticks (arm 0)", [r for r in inw if r["fam5"]]); line("in window, no family within 5 ticks", [r for r in inw if not r["fam5"]])
    line("in window, loose light peak within 5 ticks", [r for r in inw if r["ff5"]]); line("in window, no light peak", [r for r in inw if not r["ff5"]])
    for lo, hi in ((0, 12), (12, 30), (30, 100), (100, 300), (300, 1e9)):
        line(f"in window, delay {lo:.0f}-{min(hi,999):.0f} ticks", [r for r in inw if lo <= r["delay"] < hi])
    for lo, hi in ((1, 5), (5, 20), (20, 60), (60, 1e9)):
        line(f"in window, Michel charge {lo}-{min(hi,999):.0f} MeV", [r for r in inw if lo <= r["E"] < hi])
    line("in window, merged AND light peak (target)", [r for r in inw if r["merged"] and r["ff5"]])
    for hk in ("mu", "pi", "K"):
        line(f"in window, host = {hk}", [r for r in inw if r["host_kind"] == hk])
    line("in window, host track >= 20 MeV", [r for r in inw if r["E_mu"] >= 20]); line("in window, host track < 5 MeV (invisible)", [r for r in inw if r["E_mu"] < 5])


if __name__ == "__main__":
    main()
