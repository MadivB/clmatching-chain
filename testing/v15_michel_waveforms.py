"""How faint are the delayed Michels the ND stage never sees?  For every delayed Michel (> 12 ticks after its host,
>= 5 MeV) in the v1.4 dumps: truth photons detected (mc_truth/light_module<module> rows of the Michel's own segments,
one copy of the four-fold replicated (4, 64) block), the Michel's own light predicted by the perceiver from its hits
(amplitude per channel, template-peak units = ADC), the pile-up it sits in (actual and chain-model amplitude at the
Michel time on its brightest channels), and example figures: charge view, full-TPC waveform with host / Michel times,
and zooms on the channels where the Michel's light should be largest.  GPU.
python v15_michel_waveforms.py <dump_dir> --audit michel_nd_out/michel_audit5_63.json [--files ...] [--examples 8]"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, "..", "CLMatching_AlphaRelease"))
for p in (BASE, os.path.join(BASE, "M5p1"), os.path.join(BASE, "NewMLSection"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from michel_nd import render, PEAK, CUT  # noqa: E402
from nd_truth_michel import open_flow, event_trajectories, hit_dominant_traj, ancestry_tag, michel_trajectories  # noqa: E402

PULSE = os.path.join(BASE, "assets", "avg_pulse.npy")
SMALL = "/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/NewMLSection/runs/ndfull_small4x/best_model_arch.pt"
TMP0 = 1_000_000


def michel_segments(h5, traj, mc_event, m_rows):
    """segment ids of this event whose trajectory has one of m_rows in its ancestry -> {m_row: set(segment ids)}"""
    seg = h5["mc_truth/segments/data"]; ev = seg["event_id"][:]; rows = np.flatnonzero(ev == int(mc_event))
    vx = seg["vertex_id"][rows].astype(np.int64); tid = seg["traj_id"][rows].astype(np.int64); sid = seg["segment_id"][rows].astype(np.int64)
    tr_row = np.array([traj["idx"].get((int(v), int(t)), -1) for v, t in zip(vx, tid)], np.int64)
    tag = ancestry_tag(traj, tr_row, m_rows)
    out = {int(m): set() for m in m_rows}
    for s_, t_ in zip(sid, tag):
        if t_ >= 0:
            out[int(t_)].add(int(s_))
    return out


def truth_photons(h5, module, seg_ids):
    """detected photons (one copy of the replicated block) and photon-weighted t0 (ticks) for these segments in this module"""
    lm = h5[f"mc_truth/light_module{int(module)}/data"]; s_all = lm["segment_id"][:, 0, 0].astype(np.int64)
    sel = np.flatnonzero(np.isin(s_all, np.fromiter(seg_ids, np.int64))) if seg_ids else np.zeros(0, int)
    if sel.size == 0:
        return 0.0, np.nan, 0
    blk = lm[sel]; ph = blk["n_photons_det"][:, 0, :].astype(float); t0 = blk["t0_det"][:, 0, :].astype(float)
    tot = float(ph.sum()); tt = float((ph * t0).sum() / tot * 62.5) if tot > 0 else np.nan
    return tot, tt, int(sel.size)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dump_dir"); ap.add_argument("--audit", required=True); ap.add_argument("--files", default=None); ap.add_argument("--events", type=int, default=0)
    ap.add_argument("--examples", type=int, default=8); ap.add_argument("--out", default=os.path.join(HERE, "michel_nd_out", "waveforms")); ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    J = json.load(open(a.audit)); AUD = {}
    for r in J["audit"]:
        if 0 <= r["t"] <= 885 and r["delay"] > 12 and r["E"] >= 5:
            AUD[(r["ev"], int(r["michel"]))] = r
    files = sorted(glob.glob(os.path.join(a.dump_dir, "full_dumps", "*.npz")))
    if a.files:
        tags = a.files.split(","); files = [f for f in files if any(os.path.basename(f).startswith(t) for t in tags)]
    if a.events:
        files = files[: a.events]
    from M5p1.first_stage_matching import load_first_stage_models
    from M5p1.first_stage_matching.config import FirstStageConfig
    from M5p1.first_stage_matching.streaming_prediction import process_clusters_to_imageMaps_streaming
    fcfg = FirstStageConfig(); fcfg.model.light_checkpoint = SMALL
    models = load_first_stage_models(fcfg); tmpl = np.asarray(models.waveform_template, np.float64)
    T = np.load(PULSE).astype(np.float64).ravel(); T = T / T.max()
    print("models on", models.device, flush=True)
    rows = []; t_begin = time.time()
    # example selection: uncovered (no probe piece), spread over reasons, E >= 10, delay 25-400
    want_ex = {}
    for (ev, m), r in AUD.items():
        if r.get("piece_rec") or r["E"] < 10 or not (25 <= r["delay"] <= 400):
            continue
        key = tuple(sorted(r.get("why_uncovered") or ["none"]))[:1]
        want_ex.setdefault(key, []).append((r["E"], ev, m))
    chosen = set(); n_per = max(1, a.examples // max(len(want_ex), 1) + 1)
    for key, lst in sorted(want_ex.items()):
        lst.sort(reverse=True)
        for E, ev, m in lst[:n_per]:
            chosen.add((ev, m))
    for f in files:
        z = np.load(f, allow_pickle=True); tag = os.path.basename(f)[:-4]
        sel_m = [m for (ev, m) in AUD if ev == tag]
        if not sel_m:
            continue
        h5 = open_flow(str(z["src_file"])); ids = np.unique(np.asarray(h5["mc_truth/interactions/data"]["event_id"])); ev_id = int(z["event"]); mc_event = int(ids[ev_id])
        e = np.clip(z["e"], 0, None).astype(np.float64); ts = z["ts_final"].astype(np.float64); tru = z["truth_t0"].astype(np.float64)
        labs = z["labels"].astype(int); tpc = z["tpc"].astype(int); x, y, zz = (z[k].astype(np.float32) for k in ("x", "y", "z"))
        A = z["full_light"].astype(np.float64); B = z["base_image"].astype(np.float64); veto = z["veto_mask"]; ceil = z["sat_ceiling"].astype(np.float64)
        traj = event_trajectories(h5, mc_event); dom = hit_dominant_traj(h5, z["hit_refs"].astype(np.int64), traj); mich = michel_trajectories(traj)
        mrows = [m for m, _ in mich if m in sel_m]
        if not mrows:
            continue
        tagm = ancestry_tag(traj, dom, mrows); segs = michel_segments(h5, traj, mc_event, mrows)
        # one batched perceiver call: each Michel's hits in its TPC as a temporary cluster
        lab2 = np.full(labs.size, -1, np.int64); want = set(); keys = {}
        for j, m in enumerate(mrows):
            r = AUD[(tag, m)]; c = r["tpc"]; hm = np.flatnonzero((tagm == m) & (tpc == c))
            if hm.size < 2:
                continue
            lab2[hm] = TMP0 + j; want.add(TMP0 + j); keys[m] = (TMP0 + j, c, hm)
        if not want:
            continue
        _, meta = process_clusters_to_imageMaps_streaming(x, y, zz, e.astype(np.float32), tpc, lab2, model=models.light_model, template=tmpl, batch_size=a.batch,
                                                          restrict_clusters=want, device_policy="auto", amplitudes_only=True, min_prediction_threshold=0.0)
        amps = {(int(c_), int(t_)): meta["amplitudes"][i] for i, (c_, t_) in enumerate(zip(meta["group_cls"], meta["group_tpcs"]))}; n_ch = A.shape[1]
        for m, (L2, c, hm) in keys.items():
            r = AUD[(tag, m)]; ap_ = amps.get((L2, c)); ap_ = np.zeros(n_ch) if ap_ is None else np.asarray(ap_, np.float64).copy(); ap_[veto[c]] = 0.0
            t_m = r["t"]; k_m = int(round(t_m)) + PEAK; k_h = int(round(r["t_host"])) + PEAK if np.isfinite(r["t_host"]) else -1
            ph, t_ph, nseg = truth_photons(h5, c // 2, segs.get(m, set()))
            top = np.argsort(-ap_)[:6]; sl = slice(max(0, k_m - 3), min(A.shape[2], k_m + 4))
            act_top = A[c][top][:, sl].max(1); mod_top = B[c][top][:, sl].max(1)
            # isolated-flash check: no assigned family within 15 ticks of the Michel time (other than the Michel's own cluster time if it is right)
            fam = np.unique(np.round(ts[(tpc == c) & (labs >= 0) & np.isfinite(ts)])); near = fam[np.abs(fam - t_m) <= 15]
            resid_top = (A[c][top][:, sl] - B[c][top][:, sl]).max(1)
            rec = dict(ev=tag, michel=int(m), E=r["E"], E_tpc=float(e[hm].sum()), n_hits=int(hm.size), delay=r["delay"], host=r["host_kind"], tpc=c, merged=r["merged"], reasons=r.get("why_uncovered"),
                       uncovered=not bool(r.get("piece_rec")), covered_frac=r.get("covered_frac"), photons=ph, t_photons=t_ph, n_seg=nseg,
                       pred_max=float(ap_.max()), pred_sum=float(ap_.sum()), pred_n_sup90=int((np.cumsum(np.sort(ap_)[::-1]) <= 0.9 * ap_.sum()).sum() + 1) if ap_.sum() > 0 else 0,
                       top_ch=[int(k) for k in top], actual_top=[float(v) for v in act_top], model_top=[float(v) for v in mod_top], resid_top=[float(v) for v in resid_top],
                       ceil_top=[float(ceil[c, k]) for k in top], n_family_15=int(near.size), corr_before=r["corr_before"], corr_after=r["corr_after"])
            rows.append(rec)
            if (tag, m) in chosen:
                draw(a.out, rec, z, c, hm, labs, tpc, e, x, y, zz, A[c], B[c], veto[c], ap_, T, t_m, r["t_host"], r["cluster"])
        print(f"{tag}: Michels {len(keys)}  ({time.time() - t_begin:.0f}s)", flush=True)
    json.dump(rows, open(os.path.join(a.out, "michel_light.json"), "w"), default=float)
    summarize(rows)


def draw(outdir, rec, z, c, hm, labs, tpc, e, x, y, zz, A, B, veto, ap_, T, t_m, t_h, Lm):
    n_s = A.shape[1]; k_m = int(round(t_m)) + PEAK
    fig = plt.figure(figsize=(16, 12)); gs = fig.add_gridspec(3, 6, height_ratios=[1.1, 1.0, 1.0], hspace=0.45, wspace=0.35)
    # charge view
    inT = np.flatnonzero(tpc == c); inC = inT[labs[inT] == Lm] if Lm >= 0 else np.zeros(0, int)
    for col, (u, v, lu, lv) in enumerate(((zz, y, "z [cm]", "y [cm]"), (x, y, "x [cm]", "y [cm]"), (zz, x, "z [cm]", "x [cm]"))):
        ax = fig.add_subplot(gs[0, 2 * col: 2 * col + 2])
        ax.scatter(u[inT], v[inT], s=2, c="0.85", label="other hits in TPC")
        if inC.size:
            ax.scatter(u[inC], v[inC], s=4, c="0.4", label=f"host cluster {Lm} ({e[inC].sum():.0f} MeV)")
        ax.scatter(u[hm], v[hm], s=10, c="red", label=f"Michel hits ({e[hm].sum():.1f} MeV)")
        ax.set_xlabel(lu); ax.set_ylabel(lv)
        if inC.size:
            pad = 12; ax.set_xlim(min(u[inC].min(), u[hm].min()) - pad, max(u[inC].max(), u[hm].max()) + pad); ax.set_ylim(min(v[inC].min(), v[hm].min()) - pad, max(v[inC].max(), v[hm].max()) + pad)
        if col == 0:
            ax.legend(fontsize=8, loc="best")
    # full-window summed waveform
    ax = fig.add_subplot(gs[1, :]); good = ~veto; tt = np.arange(n_s) - PEAK
    sa = A[good].sum(0); sb = B[good].sum(0); sm = render(ap_, t_m, n_s, T)[good].sum(0)
    ax.plot(tt, np.clip(sa, 1, None), "k-", lw=0.8, label="actual, summed over channels")
    ax.plot(tt, np.clip(sb, 1, None), "b--", lw=0.8, label="chain model (all placed families)")
    ax.plot(tt, np.clip(sm, 1, None), "r-", lw=1.2, label=f"Michel's own light (perceiver from its hits): sum {ap_.sum():.0f}, max channel {ap_.max():.0f} ADC")
    if np.isfinite(t_h):
        ax.axvline(t_h, color="green", ls=":", lw=1.5, label=f"host time {t_h:.0f}")
    ax.axvline(t_m, color="red", ls=":", lw=1.5, label=f"Michel time {t_m:.0f} (delay {rec['delay']:.0f} ticks)")
    ax.set_yscale("log"); ax.set_xlim(-20, 900); ax.set_ylim(1, max(sa.max(), 10) * 2); ax.set_xlabel("tick (16 ns)"); ax.set_ylabel("summed amplitude [ADC]"); ax.legend(fontsize=8, loc="upper right", ncol=2)
    # zooms on the Michel's brightest channels
    top = rec["top_ch"]
    for i, k in enumerate(top):
        ax = fig.add_subplot(gs[2, i]); lo, hi = max(0, k_m - 45), min(n_s, k_m + 90); t_ = np.arange(lo, hi) - PEAK
        ax.plot(t_, A[k, lo:hi], "k-", lw=0.9, label="actual"); ax.plot(t_, B[k, lo:hi], "b--", lw=0.8, label="chain model")
        ax.plot(t_, (B[k] + render(ap_, t_m, n_s, T)[k])[lo:hi], "r-", lw=1.0, label="model + Michel")
        ax.axvline(t_m, color="red", ls=":", lw=1); ax.set_title(f"ch {k}: Michel {ap_[k]:.0f} ADC; actual {rec['actual_top'][i]:.0f}, model {rec['model_top'][i]:.0f}", fontsize=8)
        ax.set_xlabel("tick"); ax.tick_params(labelsize=7)
        if i == 0:
            ax.set_ylabel("ADC"); ax.legend(fontsize=7)
    why = ", ".join(rec["reasons"]) if rec["reasons"] else "none of the recorded reasons"
    fig.suptitle(f"{rec['ev']}  TPC {c}  Michel {rec['michel']} ({rec['host']}-hosted, {rec['E']:.1f} MeV, {rec['n_hits']} hits, delay {rec['delay']:.0f} ticks, {'merged' if rec['merged'] else 'own cluster'})\n"
                 f"truth detected photons {rec['photons']:.0f} ({rec['n_seg']} segments); perceiver: max channel {rec['pred_max']:.0f} ADC, sum {rec['pred_sum']:.0f} ADC over {rec['pred_n_sup90']} support channels; "
                 f"families within 15 ticks: {rec['n_family_15']}; not a candidate because: {why}", fontsize=10)
    fn = os.path.join(outdir, f"michel_{rec['ev']}_m{rec['michel']}.png"); fig.savefig(fn, dpi=110); plt.close(fig); print("  figure", fn, flush=True)


def summarize(rows):
    def q(v, ps=(10, 25, 50, 75, 90)):
        v = np.asarray(v, float); v = v[np.isfinite(v)]; return np.percentile(v, ps).round(1) if v.size else None
    for lab, sel in (("ALL delayed Michels >= 5 MeV", rows), ("UNCOVERED (no probe piece)", [r for r in rows if r["uncovered"]]), ("covered by a probe piece", [r for r in rows if not r["uncovered"]])):
        if not sel:
            continue
        E = np.array([r["E"] for r in sel]); ph = np.array([r["photons"] for r in sel]); pm = np.array([r["pred_max"] for r in sel]); ps = np.array([r["pred_sum"] for r in sel])
        act = np.array([r["actual_top"][0] for r in sel]); mod = np.array([r["model_top"][0] for r in sel]); ce = np.array([r["ceil_top"][0] for r in sel])
        print(f"\n{lab}: n {len(sel)}, E {E.sum()/1e3:.2f} GeV, Michel E p10/25/50/75/90 {q(E)} MeV")
        print(f"  truth detected photons (one copy, Michel segments, own module) p10/25/50/75/90: {q(ph)}; photons per MeV median {np.median(ph / np.maximum(E, 1e-9)):.0f}")
        print(f"  perceiver: max-channel amplitude {q(pm)} ADC; summed amplitude {q(ps)} ADC; support channels (90 %) median {np.median([r['pred_n_sup90'] for r in sel]):.0f}")
        print(f"  on the Michel's brightest channel at the Michel time: actual {q(act)} ADC, chain model {q(mod)} ADC, ceiling {q(ce)}; Michel / actual ratio p25/50/75 {np.percentile(pm / np.maximum(act, 1), [25, 50, 75]).round(2)}")
        iso = [r for r in sel if r["n_family_15"] == 0]; print(f"  Michels with no placed family within 15 ticks: {len(iso)} ({sum(r['E'] for r in iso)/1e3:.2f} GeV)")
        if len(iso) >= 5:
            rr = np.array([r["resid_top"][0] for r in iso]); pmi = np.array([r["pred_max"] for r in iso]); phi = np.array([r["photons"] for r in iso])
            ok = (phi > 0) & (rr > 0); print(f"     their actual-minus-model residual on the brightest channel {q(rr)} ADC vs perceiver {q(pmi)} ADC; residual-sum per truth photon (calibration): median {np.median(np.array([sum(r['resid_top']) for r in iso])[ok] / phi[ok]):.2f} ADC per photon over the top-6 channels")
    # by reason
    unc = [r for r in rows if r["uncovered"]]; reasons = {}
    for r in unc:
        for w in (r["reasons"] or ["(none recorded)"]):
            reasons.setdefault(w, []).append(r)
    print("\nuncovered, by reason: n, E GeV, photons median, perceiver max ADC median, actual-at-Michel-time median, Michel/actual median")
    for w, sel in sorted(reasons.items(), key=lambda kv: -sum(r["E"] for r in kv[1])):
        pm = np.array([r["pred_max"] for r in sel]); act = np.array([r["actual_top"][0] for r in sel])
        print(f"  {w:48s} n {len(sel):3d}  E {sum(r['E'] for r in sel)/1e3:.2f}  photons {np.median([r['photons'] for r in sel]):7.0f}  pred max {np.median(pm):6.0f}  actual {np.median(act):7.0f}  ratio {np.median(pm/np.maximum(act,1)):.2f}")


if __name__ == "__main__":
    main()
