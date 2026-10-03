"""Truth photons of the delayed Michels (CPU): produced scintillation photons (segments.n_photons, complete) and detected
photons (mc_truth/light_module tables; only the odd charge TPC of each module carries them in this production), with
dE, for every delayed Michel in the audit list.  Output michel_nd_out/waveforms/michel_photons.json"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from nd_truth_michel import open_flow, event_trajectories, ancestry_tag, michel_trajectories  # noqa: E402

J = json.load(open(sys.argv[1])); AUD = {}
for r in J["audit"]:
    if 0 <= r["t"] <= 885 and r["delay"] > 12 and r["E"] >= 5:
        AUD.setdefault(r["ev"], {})[int(r["michel"])] = r
out = {}
for f in sorted(glob.glob("/pscratch/sd/y/yuxuan/v11arms/V14A/full_dumps/*.npz")):
    tag = os.path.basename(f)[:-4]
    if tag not in AUD:
        continue
    z = np.load(f, allow_pickle=True); h5 = open_flow(str(z["src_file"])); ids = np.unique(np.asarray(h5["mc_truth/interactions/data"]["event_id"])); mc_event = int(ids[int(z["event"])])
    traj = event_trajectories(h5, mc_event); mrows = [m for m, _ in michel_trajectories(traj) if m in AUD[tag]]
    seg = h5["mc_truth/segments/data"]; ev = seg["event_id"][:]; rows = np.flatnonzero(ev == mc_event); blk = seg[rows]
    tr_row = np.array([traj["idx"].get((int(v), int(t)), -1) for v, t in zip(blk["vertex_id"], blk["traj_id"])], np.int64)
    tagseg = ancestry_tag(traj, tr_row, mrows)
    sid = blk["segment_id"].astype(np.int64); nph = blk["n_photons"].astype(float); dE = blk["dE"].astype(float)
    # detected photons per segment (odd TPC only): module table of the Michel's TPC
    for m in mrows:
        r = AUD[tag][m]; c = r["tpc"]; sel = np.flatnonzero(tagseg == m)
        det = 0.0; n_det = 0
        if c % 2 == 1 and sel.size:
            lm = h5[f"mc_truth/light_module{c // 2}/data"]; s_all = lm["segment_id"][:, 0, 0].astype(np.int64); hit = np.flatnonzero(np.isin(s_all, sid[sel]))
            if hit.size:
                det = float(lm["n_photons_det"][hit][:, 0, :].sum()); n_det = int(hit.size)
        out[f"{tag}:{m}"] = dict(ev=tag, michel=int(m), tpc=int(c), E_charge=r["E"], n_seg=int(sel.size), dE=float(dE[sel].sum()), produced=float(nph[sel].sum()), detected=det, n_seg_in_table=n_det, odd_tpc=bool(c % 2 == 1), uncovered=not bool(r.get("piece_rec")))
    print(tag, len(mrows), flush=True)
os.makedirs(os.path.join(HERE, "michel_nd_out", "waveforms"), exist_ok=True)
json.dump(out, open(os.path.join(HERE, "michel_nd_out", "waveforms", "michel_photons.json"), "w"))
R = list(out.values()); P = np.array([r["produced"] for r in R]); E = np.array([r["dE"] for r in R]); Ec = np.array([r["E_charge"] for r in R])
print(f"\n{len(R)} delayed Michels: dE (truth, all segments) p10/50/90 {np.percentile(E,[10,50,90]).round(1)} MeV; charge E p10/50/90 {np.percentile(Ec,[10,50,90]).round(1)}; produced photons p10/50/90 {np.percentile(P,[10,50,90]).round(0)}; photons per MeV median {np.median(P/np.maximum(E,0.1)):.0f}")
odd = [r for r in R if r["odd_tpc"] and r["detected"] > 0]; D = np.array([r["detected"] for r in odd]); Po = np.array([r["produced"] for r in odd])
print(f"odd-TPC Michels with detected photons: {len(odd)}; detected p10/50/90 {np.percentile(D,[10,50,90]).round(0)}; detected / produced p10/50/90 {np.percentile(D/np.maximum(Po,1),[10,50,90]).round(4)}; detected per MeV median {np.median(D/np.maximum([r['dE'] for r in odd],0.1)):.1f}")
for lab, sel in (("uncovered", [r for r in R if r["uncovered"]]), ("covered", [r for r in R if not r["uncovered"]])):
    P = np.array([r["produced"] for r in sel]); od = [r for r in sel if r["odd_tpc"] and r["detected"] > 0]
    print(f"  {lab}: n {len(sel)}, produced photons p10/50/90 {np.percentile(P,[10,50,90]).round(0)}; detected (odd TPCs, n {len(od)}) p10/50/90 {np.percentile([r['detected'] for r in od],[10,50,90]).round(0) if od else None}")
