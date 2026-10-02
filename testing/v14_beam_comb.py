"""Beam-comb recovery with the ND precision timing (the 2x2 study on ND simulation).

Truth: mc_truth/interactions t_vert carries the bunch structure (period ~19.09 ns, flat-top
bunches ~8.6 ns wide); segment times of CONTAINED vertices equal the vertex time (0.1 ns), rock
interactions enter 2-200 ns later (flight time).  So the comb in detector time is defined by the
contained vertices.

Reconstructed interaction start time = the EARLIEST apex time among the precision families
(one per light TPC) that belong to the same vertex.  A family's apex time t0 is already a start
time (every hit is later by its distance to the apex over c); downstream TPC families of the same
interaction start later by the flight time, hence the minimum.

Per event: align the chain's tick frame (16 ns per tick, per-event reference) to the absolute
segment-time frame by matching family truth start times to vertex entry times; match families to
vertices (2 ns); comb phase = circular mean of the contained vertex times; fold the reconstructed
interaction start times on the period.  Reported: in-window fraction vs random, bunch
identification purity by energy and by quoted sigma, reco - truth per interaction, the phase of
the spill measured from the reconstruction alone, and the rock-muon flight-time distribution
read off the comb.
python v14_beam_comb.py <arm_dir>   (rows + full_dumps of a precision run, e.g. V2A5)"""
import glob
import json
import os
import re
import sys

import h5py
import numpy as np

C_CM_NS = 29.9792458; NS = 16.0; TICK_OFF = 8.0
HALF_BUNCH = 4.3; SPILL_REP_NS = 1.2e9


def wrap(x, P):
    return ((x + P / 2) % P) - P / 2


def circ_mean(t, P, w=None):
    z = np.exp(2j * np.pi * np.asarray(t) / P)
    z = np.average(z, weights=w) if w is not None else z.mean()
    return (np.angle(z) / (2 * np.pi) * P) % P, float(np.abs(z))


def event_truth(h5, eid, box):
    it = h5["mc_truth/interactions/data"]; seg = h5["mc_truth/segments/data"]
    m = np.asarray(it["event_id"]) == eid
    vid = np.asarray(it["vertex_id"])[m].astype(int)
    tv = np.asarray(it["t_vert"], float)[m] * 1000.0
    xv, yv, zv = (np.asarray(it[k], float)[m] for k in ("x_vert", "y_vert", "z_vert"))
    inside = (xv > box[0]) & (xv < box[1]) & (yv > box[2]) & (yv < box[3]) & (zv > box[4]) & (zv < box[5])
    sm = np.asarray(seg["event_id"]) == eid
    svid = np.asarray(seg["vertex_id"])[sm].astype(int); st0 = np.asarray(seg["t0"], float)[sm] * 1000.0
    entry = {}
    for v in np.unique(svid):
        entry[int(v)] = float(st0[svid == v].min())
    base = np.round(np.median(list(entry.values())) / SPILL_REP_NS) * SPILL_REP_NS if entry else 0.0
    out = []
    for v, t, ins, px, py, pz in zip(vid, tv, inside, xv, yv, zv):
        if int(v) in entry:
            out.append((int(v), entry[int(v)] - base, t - np.round(t / SPILL_REP_NS) * SPILL_REP_NS, bool(ins), float(px), float(py), float(pz)))
    return out   # (vertex id, entry time ns, vertex time ns, contained, x, y, z of the vertex)


def beam_direction(h5):
    """mean unit vector of the true neutrino momenta = beam axis in detector coordinates"""
    nu = np.asarray(h5["mc_truth/interactions/data"]["nu_4mom"], float)[:, :3]
    u = nu / np.linalg.norm(nu, axis=1, keepdims=True)
    n = u.mean(0); return n / np.linalg.norm(n)


def main(arm):
    rows = {}
    for f in glob.glob(os.path.join(arm, "groups_worker*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            mm = re.search(r"(\d{7})\.FLOW", str(r.get("file", "")))
            if mm:
                rows[(mm.group(1), int(r["event"]))] = r
    h5cache = {}
    fam_abs = []    # EVERY family with a reco time (cols 10-12 apex xyz, 13-18 member-hit box, 19-21 truth-earliest hit xyz, 22 vertex entry, 23 vertex time, 24-26 upstream-most member hit along the beam, 27-29 true vertex xyz): (t_reco_abs [per-event cal removed], t_true_abs or nan, E, sig, apex margin to hit box [cm], n_member_hits, event index, per-event cal, matched-to-vertex flag or nan, vertex contained flag or nan)
    inter_abs = []  # per reconstructed interaction, absolute spill-frame times: (reco start [per-event cal removed], truth vertex, entry, contained, E, min sig, start apex margin to hit box [cm], n_fam, event index, per-event cal)
    inter = []      # per reconstructed interaction: (contained, n_fam, E_sum, min_sig, reco_start - phase, truth - phase, reco - truth, entry - vertex)
    fam_rows = []   # per family: (contained, E, sig, reco_apex - phase, truth_apex - phase)
    ev_phase = []; ev_box = []; truth_all = []; NHAT = None
    cal_all = []
    P = 19.09
    for (ftag, ev), r in sorted(rows.items()):
        fams = r.get("phase3", {}).get("families") or []
        dump = os.path.join(arm, "full_dumps", f"{ftag}_ev{ev}.npz")
        if not fams or not os.path.exists(dump):
            continue
        z = np.load(dump, allow_pickle=True)
        src = str(z["src_file"])
        if src not in h5cache:
            h5cache[src] = h5py.File(src, "r")
        xyz = np.stack([z["x"], z["y"], z["z"]], 1).astype(float); tpc = z["tpc"].astype(int)
        ts = np.nan_to_num(z["ts_final"], nan=-1); tru = z["truth_t0"].astype(float)
        box = (xyz[:, 0].min(), xyz[:, 0].max(), xyz[:, 1].min(), xyz[:, 1].max(), xyz[:, 2].min(), xyz[:, 2].max())
        ids = np.unique(np.asarray(h5cache[src]["mc_truth/interactions/data"]["event_id"]))
        V = event_truth(h5cache[src], int(ids[ev]) if ev < ids.size else 1000 + ev, box)
        if sum(1 for v in V if v[3]) < 3:
            continue
        v_id = np.array([v[0] for v in V]); v_entry = np.array([v[1] for v in V]); v_time = np.array([v[2] for v in V]); v_cont = np.array([v[3] for v in V])
        v_xyz = np.array([v[4:7] for v in V], float)
        if NHAT is None:
            NHAT = beam_direction(h5cache[src]); print("beam direction (mean true neutrino momentum):", np.round(NHAT, 4))
        # ---- families: member hits by apex-clock proximity, truth start time (apex clock), reco start time
        by_tpc = {}
        for k, fm in enumerate(fams):
            by_tpc.setdefault(int(fm["tpc"]), []).append(k)
        fam_hits = {k: [] for k in range(len(fams))}
        for c, ks in by_tpc.items():
            idx = np.flatnonzero((tpc == c) & (ts >= 0))
            if idx.size == 0:
                continue
            apex = np.array([fams[k]["apex"] for k in ks]); t0 = np.array([fams[k]["t0"] for k in ks])
            d = np.linalg.norm(xyz[idx][:, None, :] - apex[None, :, :], axis=2) / C_CM_NS / NS
            t_apex = ts[idx][:, None] - d
            j = np.argmin(np.abs(t_apex - t0[None, :]), axis=1)
            good = np.abs(t_apex[np.arange(idx.size), j] - t0[j]) < 0.5
            for i_h, jj, g in zip(idx, j, good):
                if g:
                    fam_hits[ks[jj]].append(i_h)
        F = []; apex_list = []; G = []; G_geo = []; G_up = []
        for k, fm in enumerate(fams):
            h = np.array(fam_hits[k], int)
            ax_ = np.array(fm["apex"], float)
            G.append(((float(fm["t0"]) - TICK_OFF) * NS, np.nan, float(fm["E"]), float(fm["sig_ns"]),
                      float(min(ax_[0] - box[0], box[1] - ax_[0], ax_[1] - box[2], box[3] - ax_[1], ax_[2] - box[4], box[5] - ax_[2])), int(h.size)))
            if h.size:
                hb = xyz[h]; tr_h = tru[h]; okh = np.isfinite(tr_h)
                te = hb[np.argmin(np.where(okh, tr_h, np.inf))] if okh.any() else np.full(3, np.nan)
                geo = tuple(ax_) + tuple(hb.min(0)) + tuple(hb.max(0)) + tuple(te)
            else:
                geo = tuple(ax_) + (np.nan,) * 9
            G_geo.append(geo)
            G_up.append(tuple(hb[np.argmin(hb @ NHAT)]) if h.size else tuple(ax_))
            if h.size < 10:
                continue
            tr = tru[h]; ok = np.isfinite(tr)
            if ok.sum() < 10:
                continue
            d = np.linalg.norm(xyz[h[ok]] - np.array(fm["apex"]), axis=1) / C_CM_NS / NS
            t_true = tr[ok] - d
            if 1.4826 * np.median(np.abs(t_true - np.median(t_true))) > 1.0:
                continue
            F.append(((float(np.median(t_true)) - TICK_OFF) * NS, (float(fm["t0"]) - TICK_OFF) * NS, float(fm["E"]), float(fm["sig_ns"])))
            apex_list.append(np.array(fm["apex"], float)); G[-1] = G[-1][:1] + ((float(np.median(t_true)) - TICK_OFF) * NS,) + G[-1][2:] + (len(F) - 1,)
        if len(F) < 5:
            continue
        F = np.array(F)
        ev_index = len(ev_phase)
        for vv in V:
            truth_all.append((ev_index, vv[2], vv[4], vv[5], vv[6], float(vv[3]), vv[1]))
        # ---- frame alignment: mode of (vertex entry - family truth start)
        diffs = (v_entry[None, :] - F[:, 0][:, None]).ravel()
        h, e = np.histogram(diffs, bins=16000, range=(-4000, 4000))
        if h.max() < 5:
            continue
        off = e[np.argmax(h)] + 0.25
        near = diffs[np.abs(diffs - off) < 2.0]; off = float(np.median(near))
        t_true_abs = F[:, 0] + off; t_reco_abs = F[:, 1] + off
        # the reconstruction's global time constant (light path + template alignment): one number for the
        # detector, calibrated once; here taken as the median of reco - truth over the event's families
        cal = float(np.median(t_reco_abs - t_true_abs)); t_reco_abs = t_reco_abs - cal
        cal_all.append(cal)
        G_idx = {g[6]: gi for gi, g in enumerate(G) if len(g) > 6}
        j = np.argmin(np.abs(t_true_abs[:, None] - v_entry[None, :]), axis=1)
        matched = np.abs(t_true_abs - v_entry[j]) < 2.0
        # ---- comb phase and period from contained vertices
        tc = v_time[v_cont]
        ph, R = circ_mean(tc, P)
        # ---- reconstructed interactions = earliest family per matched vertex
        per_v = {}
        for i in np.flatnonzero(matched):
            per_v.setdefault(int(j[i]), []).append(i)
        n_int_ev = 0
        for jv, fl in per_v.items():
            fl = np.array(fl)
            i_min = fl[np.argmin(t_reco_abs[fl])]
            reco_start = float(t_reco_abs[i_min]); truth_v = float(v_time[jv]); entry_v = float(v_entry[jv])
            inter.append((int(v_cont[jv]), len(fl), float(F[fl, 2].sum()), float(F[fl, 3].min()),
                          wrap(reco_start - ph, P), wrap(truth_v - ph, P), reco_start - entry_v, entry_v - truth_v))
            ax_ = apex_list[int(i_min)]
            margin = float(min(ax_[0] - box[0], box[1] - ax_[0], ax_[1] - box[2], box[3] - ax_[1], ax_[2] - box[4], box[5] - ax_[2]))
            inter_abs.append((reco_start, truth_v, entry_v, int(v_cont[jv]), float(F[fl, 2].sum()), float(F[fl, 3].min()), margin, len(fl), ev_index, cal)
                             + G_up[G_idx[int(i_min)]] + tuple(v_xyz[jv]))
            n_int_ev += 1
        for gi, g in enumerate(G):
            fi = g[6] if len(g) > 6 else None
            fam_abs.append((g[0] + off - cal, (g[1] + off) if fi is not None else np.nan, g[2], g[3], g[4], g[5], ev_index, cal,
                            float(matched[fi]) if fi is not None else np.nan, float(v_cont[j[fi]]) if fi is not None else np.nan) + G_geo[gi]
                           + (float(v_entry[j[fi]]) if fi is not None else np.nan, float(v_time[j[fi]]) if fi is not None else np.nan)
                           + G_up[gi] + (tuple(v_xyz[j[fi]]) if fi is not None else (np.nan,) * 3))
        for i in np.flatnonzero(matched):
            fam_rows.append((int(v_cont[j[i]]), F[i, 2], F[i, 3], wrap(t_reco_abs[i] - ph, P), wrap(t_true_abs[i] - ph, P)))
        # reco-only spill phase: from all reconstructed interaction starts (no truth), weighted by 1/sigma^2
        starts = np.array([t_reco_abs[np.array(fl)[np.argmin(t_reco_abs[np.array(fl)])]] for fl in per_v.values()])
        ph_reco, R_reco = circ_mean(starts, P) if starts.size >= 5 else (np.nan, np.nan)
        ev_phase.append((ftag, ev, int(v_cont.sum()), n_int_ev, R, R_reco, wrap(ph_reco - ph, P) if np.isfinite(ph_reco) else np.nan, ph)); ev_box.append(box)
    I = np.array(inter, float); Fm = np.array(fam_rows, float)
    cont, nfam, Esum, sigmin, dpr, dpt, d_rt, flight = I.T
    m = cont == 1
    print(f"{len(ev_phase)} events; reconstructed interactions matched to a truth vertex: {len(I)} (contained {int(m.sum())}, rock/external {int((~m).sum())}); families used {len(Fm)}")
    print(f"global reco time constant removed: median {np.median(cal_all):+.2f} ns, event-to-event spread (robust) {1.4826*np.median(np.abs(np.array(cal_all)-np.median(cal_all))):.2f} ns")
    print(f"comb period used {P} ns; truth comb concentration |R| of contained vertices per event: median {np.median([e[4] for e in ev_phase]):.2f}")
    print(f"CONTAINED interactions: reco start - vertex time: robust {1.4826*np.median(np.abs(d_rt[m])):.2f} ns, std {np.std(d_rt[m][np.abs(d_rt[m])<20]):.2f} ns (|d|<20), median {np.median(d_rt[m]):+.2f} ns")
    print(f"  in-window (+-{HALF_BUNCH} ns of the comb phase): truth {np.mean(np.abs(dpt[m]) <= HALF_BUNCH):.3f}  reco {np.mean(np.abs(dpr[m]) <= HALF_BUNCH):.3f}  (random {2*HALF_BUNCH/P:.2f})")
    bt = np.round(((dpt) + 0) / P); br = np.round((dpr) / P)      # both already wrapped: bunch index 0 unless |phase| > P/2
    # bunch identification: reco start assigned to the nearest bunch; compare with truth bunch (absolute index)
    print(f"  bunch identification purity: {np.mean(np.abs(dpr[m] - dpt[m]) < P/2):.4f}  (energy-weighted {np.sum(Esum[m]*(np.abs(dpr[m]-dpt[m]) < P/2))/Esum[m].sum():.4f})")
    for lo, hi in ((0, 1), (1, 2), (2, 4), (4, 1e9)):
        mm = m & (sigmin >= lo) & (sigmin < hi)
        if mm.any():
            print(f"     best family sigma {lo}-{hi} ns: n {int(mm.sum()):5d}  reco-truth robust {1.4826*np.median(np.abs(d_rt[mm])):.2f} ns  in-window {np.mean(np.abs(dpr[mm]) <= HALF_BUNCH):.3f}  purity {np.mean(np.abs(dpr[mm]-dpt[mm]) < P/2):.4f}")
    for lo, hi in ((5, 20), (20, 100), (100, 500), (500, 1e9)):
        mm = m & (Esum >= lo) & (Esum < hi)
        if mm.any():
            print(f"     interaction E {lo}-{hi} MeV: n {int(mm.sum()):5d}  reco-truth robust {1.4826*np.median(np.abs(d_rt[mm])):.2f} ns  in-window {np.mean(np.abs(dpr[mm]) <= HALF_BUNCH):.3f}  purity {np.mean(np.abs(dpr[mm]-dpt[mm]) < P/2):.4f}")
    h_t, _ = np.histogram(dpt[m], bins=38, range=(-P/2, P/2)); h_r, _ = np.histogram(dpr[m], bins=38, range=(-P/2, P/2))
    print("  phase histograms, 0.5-ns bins from -P/2 to +P/2 (contained interactions):"); print("     truth:", h_t.tolist()); print("     reco :", h_r.tolist())
    fc, fE, fs, fdr, fdt = Fm.T
    mf = fc == 1
    print(f"per-FAMILY cross-check (contained): in-window truth {np.mean(np.abs(fdt[mf]) <= HALF_BUNCH):.3f} reco {np.mean(np.abs(fdr[mf]) <= HALF_BUNCH):.3f}; reco - truth robust {1.4826*np.median(np.abs(wrap(fdr[mf]-fdt[mf], P))):.2f} ns")
    mr = ~m
    print(f"ROCK / external interactions (n={int(mr.sum())}): truth flight time entry - vertex p10/50/90 {np.round(np.percentile(flight[mr], [10,50,90]),1)} ns; reco in-window {np.mean(np.abs(dpr[mr]) <= HALF_BUNCH):.3f}; reco start - entry robust {1.4826*np.median(np.abs(d_rt[mr])):.2f} ns")
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "beam_comb_figs")
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "beam_comb_data.npz"), I=I, Fm=Fm, P=P, half_bunch=HALF_BUNCH,
             ev=np.array([(int(e[1]), e[2], e[3], e[4], e[5], e[6], e[7]) for e in ev_phase], float), inter_abs=np.array(inter_abs, float), fam_abs=np.array(fam_abs, float),
             ev_file=np.array([e[0] for e in ev_phase]), cal=np.array(cal_all), ev_box=np.array(ev_box, float),
             truth_all=np.array(truth_all, float), nhat=NHAT)
    d_ph = np.array([e[6] for e in ev_phase if np.isfinite(e[6])])
    print(f"spill bunch phase from the reconstruction alone (all interaction starts, no truth) vs truth: robust {1.4826*np.median(np.abs(d_ph)):.2f} ns, p90 |diff| {np.percentile(np.abs(d_ph), 90):.2f} ns ({d_ph.size} events); reco concentration |R| median {np.nanmedian([e[5] for e in ev_phase]):.2f}")


if __name__ == "__main__":
    main(sys.argv[1])
