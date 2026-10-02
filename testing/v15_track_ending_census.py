"""Track-ending census (ND, truth-scored) for the ending-classification stage the user described (2026-10-02):
for every finished track cluster, each of its two ends is one of
  EXIT      - leaves the active volume (end within 5 cm of an outer face),
  MICHEL    - a Michel electron starts within 10 cm of the end (truth),
  CONNECT   - the true particle continues beyond the end (its trajectory end point > 10 cm away: track truncated by a
              TPC boundary / dead region / clustering split) or another cluster touches the end (< 3 cm),
  STOP      - the true particle stops within 5 cm of the end with nothing following (proton, pi absorbed, mu captured, ...).
Reco features per end: distance to the nearest outer face, mean dE/dx of the track (MeV/cm, from the skeleton length),
dE/dx of the last 5 cm vs the track mean (Bragg ratio), nearest other cluster (cm), hits within 10 cm of the end belonging
to other clusters.  Truth per end: dominant pdg of the track, nearest Michel start (cm), trajectory end distance,
end_process.  Output: a per-end table (json) and summary tables of ending types vs the reco features, to size the rule
"low dE/dx track stopping away from any face with nothing attached -> expect a Michel or a missing continuation".
python v15_track_ending_census.py <arm_dir> [--events N] [--min-len 15]"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from nd_truth_michel import open_flow, event_trajectories, hit_dominant_traj, michel_trajectories, ancestry_tag  # noqa: E402
from michel_nd import skeleton_vox  # noqa: E402

BOX = (-347.0, 347.0, -215.0, 82.0, 418.0, 913.0)     # ND-LAr active volume (from the event hit boxes)


def face_dist(p):
    return float(min(p[0] - BOX[0], BOX[1] - p[0], p[1] - BOX[2], BOX[3] - p[1], p[2] - BOX[4], BOX[5] - p[2]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("arm"); ap.add_argument("--events", type=int, default=0); ap.add_argument("--min-len", type=float, default=15.0)
    ap.add_argument("--out", default=os.path.join(HERE, "michel_nd_out", "track_ends.json"))
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.arm, "full_dumps", "*.npz")))
    if a.events:
        files = files[: a.events]
    ends = []
    for f in files:
        z = np.load(f, allow_pickle=True); tag = os.path.basename(f)[:-4]
        h5 = open_flow(str(z["src_file"])); ids = np.unique(np.asarray(h5["mc_truth/interactions/data"]["event_id"])); ev_id = int(z["event"])
        if ev_id >= ids.size:
            continue
        traj = event_trajectories(h5, int(ids[ev_id])); hit_refs = z["hit_refs"].astype(np.int64)
        e = np.clip(z["e"], 0, None).astype(float); labs = z["labels"].astype(int); tpc = z["tpc"].astype(int); X = np.stack([z["x"], z["y"], z["z"]], 1).astype(float)
        ts = z["ts_final"].astype(float); tru = z["truth_t0"].astype(float)
        dom = hit_dominant_traj(h5, hit_refs, traj); mich = michel_trajectories(traj)
        mich_start = np.array([traj["xyz_start"][m] for m, _ in mich]) if mich else np.zeros((0, 3)); mich_tree = cKDTree(mich_start) if mich else None
        typ = {r["label"]: r.get("type", "?") for r in json.loads(str(z["lab_rows"]))}
        tree_all = cKDTree(X)
        # per-TPC hit bounds (approximate TPC active bounds): an end within 3 cm of its TPC's x or z bound is at an internal face
        tpc_box = {int(c): (X[tpc == c].min(0), X[tpc == c].max(0)) for c in np.unique(tpc)}
        # Michel host rows: the muon, or its pion/kaon parent when the muon is invisible
        host_of = {}
        cnt_e = np.bincount(dom[dom >= 0], weights=e[dom >= 0], minlength=traj["n"])
        for m_row, mu_row in mich:
            hosts = {mu_row}
            gp = int(traj["parent_row"][mu_row])
            if gp >= 0 and abs(int(traj["pdg"][gp])) in (211, 321):
                hosts.add(gp)
            host_of[m_row] = hosts
        for L in np.unique(labs[labs >= 0]):
            h = np.flatnonzero(labs == L)
            if e[h].sum() < 10 or h.size < 20:
                continue
            tips, D = skeleton_vox(X[h], 3.0, 2.0, 1.0)
            if not tips or D is None:
                continue
            i_max = int(np.argmax(D.max(1))); j_max = int(np.argmax(D[i_max])); extent = float(D[i_max, j_max])
            if extent < a.min_len:
                continue
            a_hit = h[tips[i_max]]; b_hit = h[j_max]; d_from_a = D[i_max]          # geodesic distance of every hit from end a
            d_from_b = np.full(h.size, -1.0)
            # distance from end b: dijkstra row of the tip nearest to b if b is a tip, else approximate by extent - d_from_a along the main path
            tb = [k for k, t in enumerate(tips) if t == j_max]
            d_from_b = D[tb[0]] if tb else np.where(d_from_a >= 0, extent - d_from_a, -1.0)
            E_L = float(e[h].sum()); dEdx_mean = E_L / max(extent, 1.0)
            # dominant particle of the track
            dh = dom[h]; dh = dh[dh >= 0]
            if dh.size == 0:
                continue
            drow = int(np.bincount(dh, weights=e[h][dom[h] >= 0]).argmax()); pdg = int(traj["pdg"][drow])
            for end_name, end_hit, dist in (("a", a_hit, d_from_a), ("b", b_hit, d_from_b)):
                p = X[end_hit]; last = (dist >= 0) & (dist <= 5.0); mid = (dist > 5.0) & (dist <= 30.0)
                dEdx_last = float(e[h][last].sum() / 5.0); dEdx_mid = float(e[h][mid].sum() / max(min(25.0, extent - 5.0), 1.0))
                # other clusters near the end
                nb = tree_all.query_ball_point(p, 10.0); nb = np.array(nb, int); other = nb[(labs[nb] != L)]
                d_other = float(np.min(np.linalg.norm(X[other] - p, axis=1))) if other.size else 99.0
                e_other_10 = float(e[other].sum()); e_other_noise = float(e[other[labs[other] < 0]].sum())
                # truth: trajectory end of the dominant particle, Michel start nearby
                tend = traj["xyz_end"][drow]; d_tend = float(np.linalg.norm(tend - p)); tstart = traj["xyz_start"][drow]; d_tstart = float(np.linalg.norm(tstart - p))
                d_mich, m_idx = (mich_tree.query(p) if mich_tree is not None else (99.0, -1))
                mich_E = 0.0; mich_corr = np.nan; is_host = False
                if m_idx >= 0 and d_mich < 10.0:
                    m_row = mich[m_idx][0]; is_host = drow in host_of.get(m_row, set())
                    tagm = ancestry_tag(traj, dom, [m_row]); hm = np.flatnonzero(tagm == m_row)
                    mich_E = float(e[hm].sum()); okm = np.isfinite(tru[hm]) & np.isfinite(ts[hm])
                    mich_corr = float((e[hm][okm] * (np.abs(ts[hm][okm] - tru[hm][okm]) <= 10)).sum() / max(mich_E, 1e-9)) if hm.size else np.nan
                c_end = int(tpc[end_hit]); lo_b, hi_b = tpc_box[c_end]
                d_tpc = float(min(p[0] - lo_b[0], hi_b[0] - p[0], p[2] - lo_b[2], hi_b[2] - p[2], p[1] - lo_b[1], hi_b[1] - p[1]))
                ends.append(dict(ev=tag, L=int(L), end=end_name, type=typ.get(int(L), "?"), E=E_L, extent=extent, dEdx_mean=dEdx_mean, dEdx_last=dEdx_last, dEdx_mid=dEdx_mid,
                                 face=face_dist(p), d_other=d_other, e_other_10=e_other_10, e_other_noise=e_other_noise, pdg=pdg, d_traj_end=d_tend, d_traj_start=d_tstart,
                                 end_process=int(traj["end_process"][drow]), d_michel=float(d_mich), michel_E=mich_E, michel_corr=mich_corr, michel_host=bool(is_host), d_tpc=d_tpc, t_L=float(np.nanmedian(ts[h]))))
        print(f"{tag}: ends so far {len(ends)}", flush=True)
    json.dump(ends, open(a.out, "w"), default=float)
    # ---- summary
    E = ends; n = len(E)
    def cls(r):
        if r["face"] < 5.0:
            return "exit"
        if r["d_michel"] < 10.0 and r["michel_E"] >= 1.0 and r["michel_host"]:
            return "michel"
        if r["d_tpc"] < 3.0 and min(r["d_traj_end"], r["d_traj_start"]) > 10.0:
            return "tpc-face"
        if min(r["d_traj_end"], r["d_traj_start"]) > 10.0:
            return "truncated"
        if r["d_other"] < 3.0:
            return "connect"
        return "stop"
    for r in E:
        r["cls"] = cls(r)
    print(f"\n{len(files)} events, {n} track ends (tracks >= 10 MeV, extent >= {a.min_len} cm)")
    kinds = ["exit", "tpc-face", "michel", "truncated", "connect", "stop"]
    print("ending classes (truth-scored):", {k: sum(r["cls"] == k for r in E) for k in kinds})
    print("\nby dominant particle (rows) x ending class (cols):")
    for name, pd in (("mu", (13,)), ("pi", (211,)), ("p", (2212,)), ("e/gamma", (11, 22)), ("K", (321,)), ("other", ())):
        sel = [r for r in E if (abs(r["pdg"]) in pd if pd else abs(r["pdg"]) not in (13, 211, 2212, 11, 22, 321))]
        if sel:
            print(f"  {name:8s} n {len(sel):5d}  " + "  ".join(f"{k} {sum(r['cls']==k for r in sel)/len(sel):.2f}" for k in kinds) + f"   dE/dx mean p50 {np.median([r['dEdx_mean'] for r in sel]):.1f} MeV/cm, last-5cm/mid p50 {np.median([r['dEdx_last']/max(r['dEdx_mid'],1e-3) for r in sel]):.2f}")
    print("\nthe rule: not at an outer face (> 5 cm) nor a TPC face (> 3 cm), nothing within 3 cm, low mean dE/dx -> 'weird stop'")
    for dmax in (3.0, 4.0, 5.0, 8.0):
        sel = [r for r in E if r["face"] >= 5.0 and r["d_tpc"] >= 3.0 and r["d_other"] >= 3.0 and r["dEdx_mean"] <= dmax]
        if sel:
            print(f"  dE/dx <= {dmax} MeV/cm: n {len(sel):5d}  " + "  ".join(f"{k} {sum(r['cls']==k for r in sel)/len(sel):.2f}" for k in kinds) + f"  | pdg mu {np.mean([abs(r['pdg'])==13 for r in sel]):.2f} pi {np.mean([abs(r['pdg'])==211 for r in sel]):.2f} p {np.mean([abs(r['pdg'])==2212 for r in sel]):.2f}")
    sel = [r for r in E if r["face"] >= 5.0 and r["d_tpc"] >= 3.0 and r["d_other"] >= 3.0 and r["dEdx_mean"] > 5.0]
    if sel:
        print(f"  dE/dx > 5 MeV/cm (proton-like): n {len(sel):5d}  " + "  ".join(f"{k} {sum(r['cls']==k for r in sel)/len(sel):.2f}" for k in kinds) + f"  | pdg p {np.mean([abs(r['pdg'])==2212 for r in sel]):.2f}")
    mich_ends = [r for r in E if r["cls"] == "michel"]
    if mich_ends:
        print(f"\nMichel ends: n {len(mich_ends)}; Michel energy correctly timed now {np.nansum([r['michel_corr']*r['michel_E'] for r in mich_ends])/sum(r['michel_E'] for r in mich_ends):.2f}; "
              f"host dE/dx mean p10/50/90 {np.percentile([r['dEdx_mean'] for r in mich_ends],[10,50,90]).round(1)}; Bragg ratio last/mid p10/50/90 {np.percentile([r['dEdx_last']/max(r['dEdx_mid'],1e-3) for r in mich_ends],[10,50,90]).round(2)}; "
              f"face dist p10/50/90 {np.percentile([r['face'] for r in mich_ends],[10,50,90]).round(0)}; other cluster within 3 cm {np.mean([r['d_other']<3 for r in mich_ends]):.2f}")


if __name__ == "__main__":
    main()
