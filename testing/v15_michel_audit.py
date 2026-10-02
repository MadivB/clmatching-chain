"""Per-Michel recall audit of the ND Michel stage (GPU): for every true Michel electron (>= 1 MeV, in window, delayed by
> 12 ticks from its host) trace it through the stage: covered by a candidate piece / whole cluster? did the scan
produce a hypothesis at its true time (within 5 ticks)? what were mf / alpha / apc / cov / unexplained there and which
gates failed? was it moved, and correctly?  Also the host geometry (extent, dE/dx, Bragg stop) so the geometric gates
can be audited.   python v15_michel_audit.py <dump_dir> [--events N] [--files ...] [--set k=v ...] [--out JSON]"""
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
from michel_nd import run_michel_nd, MichelNDConfig, CUT  # noqa: E402
from nd_truth_michel import open_flow, event_trajectories, hit_dominant_traj, ancestry_tag, michel_trajectories  # noqa: E402
from michel_nd import skeleton_vox, dense_points  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

PULSE = os.path.join(BASE, "assets", "avg_pulse.npy")
SMALL = "/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/NewMLSection/runs/ndfull_small4x/best_model_arch.pt"
TMP0 = 1_000_000


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dump_dir"); ap.add_argument("--events", type=int, default=0); ap.add_argument("--files", default=None)
    ap.add_argument("--out", default=os.path.join(HERE, "michel_nd_out", "michel_audit.json")); ap.add_argument("--set", nargs="*", default=[]); ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    ALL_HYPS = []; cfg = MichelNDConfig()
    for kv in a.set:
        k, v = kv.split("="); cur = getattr(cfg, k)
        setattr(cfg, k, type(cur)(float(v)) if isinstance(cur, (int, float)) and not isinstance(cur, bool) else (v.lower() in ("1", "true") if isinstance(cur, bool) else v))
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
    audit = []; t_begin = time.time()
    for f in files:
        z = np.load(f, allow_pickle=True); tag = os.path.basename(f)[:-4]
        h5 = open_flow(str(z["src_file"])); ids = np.unique(np.asarray(h5["mc_truth/interactions/data"]["event_id"])); ev_id = int(z["event"])
        if ev_id >= ids.size:
            continue
        e = np.clip(z["e"], 0, None).astype(np.float64); ts = z["ts_final"].astype(np.float64); tru = z["truth_t0"].astype(np.float64)
        labs = z["labels"].astype(int); tpc = z["tpc"].astype(int); x, y, zz = (z[k].astype(np.float32) for k in ("x", "y", "z"))
        A = z["full_light"].astype(np.float32); B = z["base_image"].astype(np.float32).copy(); veto = z["veto_mask"]; ceil = z["sat_ceiling"].astype(np.float32)
        al, at, ar = z["amp_lab"].astype(int), z["amp_tpc"].astype(int), z["amp_row"].astype(np.float64)
        amp = {(int(L), int(c)): ar[i] for i, (L, c) in enumerate(zip(al, at))}
        traj = event_trajectories(h5, int(ids[ev_id])); dom = hit_dominant_traj(h5, z["hit_refs"].astype(np.int64), traj); mich = michel_trajectories(traj)
        if not mich:
            continue
        tagm = ancestry_tag(traj, dom, [m for m, _ in mich])          # per hit: Michel row or -1
        cnt_e = np.bincount(dom[dom >= 0], weights=e[dom >= 0], minlength=traj["n"])
        # per Michel: truth properties
        info = {}
        for m_row, mu_row in mich:
            hm = np.flatnonzero(tagm == m_row); hm = hm[np.isfinite(tru[hm])]
            if hm.size == 0 or e[hm].sum() < 1.0:
                continue
            host = mu_row; hk = "mu"
            if cnt_e[mu_row] < 2.0:
                gp = int(traj["parent_row"][mu_row])
                if gp >= 0 and abs(int(traj["pdg"][gp])) in (211, 321) and cnt_e[gp] >= 2.0:
                    host = gp; hk = "pi"
            hh = np.flatnonzero(dom == host); hh = hh[np.isfinite(tru[hh])]
            t_m = float(np.median(tru[hm])); t_h = float(np.median(tru[hh])) if hh.size else np.nan
            lm = labs[hm]; Lm = int(np.bincount(lm[lm >= 0], weights=e[hm][lm >= 0]).argmax()) if (lm >= 0).any() else -1
            E_host_in = float(e[hh][labs[hh] == Lm].sum()) if (Lm >= 0 and hh.size) else 0.0
            c = int(np.bincount(tpc[hm]).argmax())
            info[m_row] = dict(ev=tag, michel=int(m_row), E=float(e[hm].sum()), n=int(hm.size), t=t_m, t_host=t_h, delay=t_m - t_h, host_kind=hk, E_host=float(e[hh].sum()),
                               cluster=Lm, merged=bool(Lm >= 0 and E_host_in >= 2.0), E_cluster=float(e[labs == Lm].sum()) if Lm >= 0 else 0.0, tpc=c,
                               corr_before=float((e[hm] * (np.abs(ts[hm] - tru[hm]) <= CUT)).sum() / e[hm].sum()))
        if not info:
            continue
        dbg = []

        def predict_fn(items):
            out = {}; passes = []
            for j, (k, h) in enumerate(items):
                for pm in passes:
                    if not pm["used"][h].any():
                        pm["lab"][h] = TMP0 + j; pm["used"][h] = True; pm["want"].add(TMP0 + j); pm["keys"].append((j, k)); break
                else:
                    pm = {"lab": np.full(labs.size, -1, np.int64), "used": np.zeros(labs.size, bool), "want": set(), "keys": []}
                    pm["lab"][h] = TMP0 + j; pm["used"][h] = True; pm["want"].add(TMP0 + j); pm["keys"].append((j, k)); passes.append(pm)
            for pm in passes:
                _, meta = process_clusters_to_imageMaps_streaming(x, y, zz, e.astype(np.float32), tpc, pm["lab"], model=models.light_model, template=tmpl,
                                                                  batch_size=a.batch, restrict_clusters=pm["want"], device_policy="auto", amplitudes_only=True, min_prediction_threshold=0.0)
                amps = {(int(c_), int(t_)): meta["amplitudes"][i] for i, (c_, t_) in enumerate(zip(meta["group_cls"], meta["group_tpcs"]))}
                for j, k in pm["keys"]:
                    v = amps.get((TMP0 + j, k[1])); out[k] = np.asarray(v, np.float64) if v is not None else np.zeros(A.shape[1])
            return out
        st = run_michel_nd(xyz=np.stack([x, y, zz], 1).astype(np.float64), e=e, tpc=tpc, labels=labs, ts=ts, amp=amp, actual=A, base=B, veto=veto, ceil=ceil,
                           template=T, predict_fn=predict_fn, cfg=cfg, truth_t0=tru, apply=True, debug_tag=tagm, debug_out=dbg)
        geo = next((d for d in dbg if d["kind"] == "geometry"), {}); covered = geo.get("covered", {}); covered_w = geo.get("covered_whole", {})
        hyps = [d for d in dbg if d["kind"] not in ("geometry", "cand", "candw")]; cand_recs = [d for d in dbg if d["kind"] in ("cand", "candw")]
        for d in hyps:
            ALL_HYPS.append(dict(ev=tag, **{k: v for k, v in d.items() if k != "comp"}, michel_E=float(sum(d["comp"].values()))))
        X = np.stack([x, y, zz], 1).astype(np.float64)
        for m_row, rec in info.items():
            Em = rec["E"]; rec["covered_frac"] = float(max(covered.get(m_row, 0.0), covered_w.get(m_row, 0.0)) / Em); rec["covered_piece"] = float(covered.get(m_row, 0.0) / Em); rec["covered_whole"] = float(covered_w.get(m_row, 0.0) / Em)
            # geometric reasons: the Michel's cluster in its TPC
            Lm, c = rec["cluster"], rec["tpc"]; why = []
            if Lm < 0:
                why.append("michel hits unclustered")
            else:
                h = np.flatnonzero((labs == Lm) & (tpc == c)); E_L = float(e[labs == Lm].sum()); rec["E_L"] = E_L; rec["templated"] = bool((Lm, c) in amp)
                if not rec["templated"]: why.append("no template in TPC")
                if E_L < cfg.host_min_mev: why.append("cluster < host_min")
                hm = np.flatnonzero((tagm == m_row) & (labs == Lm) & (tpc == c))
                if h.size >= 6 and hm.size:
                    tips, D = skeleton_vox(X[h], cfg.link_cm, cfg.tip_min_cm, cfg.vox_cm)
                    if tips and D is not None:
                        i_max = int(np.argmax(D.max(1))); j_max = int(np.argmax(D[i_max])); ext = float(D[i_max, j_max]); rec["host_extent"] = ext; rec["host_dedx"] = E_L / max(ext, 1)
                        if ext < cfg.host_extent_min_cm: why.append("extent < 20")
                        if E_L / max(ext, 1) > cfg.host_dedx_max: why.append("dE/dx > max")
                        # is there a tip inside the Michel hits?  dense points of this cluster; free tips; growth stopped by a dense ball?
                        loc = {int(hh): i for i, hh in enumerate(h)}; m_loc = np.array([loc[int(hh)] for hh in hm])
                        tip_in = [ti for ti, t in enumerate(tips) if t in set(m_loc.tolist())]
                        rec["tip_in_michel"] = len(tip_in) > 0
                        dp = dense_points(X[h], e[h], cfg.dense_r, cfg.dense_sep, cfg.dense_min_mev, cfg.dense_rel); dxyz = X[h][dp] if dp else np.zeros((0, 3))
                        if not tip_in: why.append("no tip in michel")
                        else:
                            ti = tip_in[0]; t = tips[ti]
                            pt = X[h][t]; bx = cfg.box; rec["tip_face_cm"] = float(min(pt[0] - bx[0], bx[1] - pt[0], pt[1] - bx[2], bx[3] - pt[1], pt[2] - bx[4], bx[5] - pt[2]))
                            if cfg.tip_face_min_cm > 0 and rec["tip_face_cm"] < cfg.tip_face_min_cm: why.append("tip within the outer-face margin")
                            free = bool(np.min(np.linalg.norm(dxyz - X[h][t], axis=1)) > cfg.free_cm) if dxyz.shape[0] else True
                            if not free: why.append("tip not free (dense point within 3 cm)")
                            excl = np.zeros(h.size, bool)
                            if dxyz.shape[0]:
                                tr = cKDTree(X[h])
                                for k_ in dp: excl[tr.query_ball_point(X[h][k_], cfg.dense_r)] = True
                            bragg = False; grew = 0.0
                            for Lc in cfg.piece_cm:
                                mk = (D[ti] >= 0) & (D[ti] <= Lc)
                                if e[h][mk & excl].sum() > cfg.dense_leak_mev: bragg = True; break
                                if mk.sum() > h.size - cfg.piece_min_hits or e[h][mk].sum() > cfg.max_frac * E_L or e[h][mk].sum() > cfg.piece_max_mev: break
                                grew = Lc
                            rec["bragg_stop"] = bragg; rec["grew_to"] = grew
                            if bragg:
                                band = (D[ti] > float(Lc)) & (D[ti] <= float(Lc) + cfg.band_cm); rec["band_dedx"] = float(e[h][band].sum()) / cfg.band_cm
                                if cfg.band_dedx_max < 1e8 and rec["band_dedx"] > cfg.band_dedx_max: why.append("band dE/dx > max")
                            if not bragg: why.append("no Bragg ball within 16 cm")
                            if bragg and cfg.bragg_end_cm < 1e8:
                                mk = (D[ti] >= 0) & (D[ti] <= 16.0); stop_xyz = X[h][mk & excl].mean(0) if (mk & excl).any() else X[h][t]
                                ends_xyz = X[h][[tips[i_max], j_max]]
                                if np.min(np.linalg.norm(ends_xyz - stop_xyz, axis=1)) > cfg.bragg_end_cm: why.append("Bragg not at a host end")
                    else:
                        why.append("no skeleton tips")
                else:
                    why.append("michel/cluster too small in TPC")
            rec["why_uncovered"] = why
            # hypotheses holding >= 50 % of this Michel's energy
            H = [d for d in hyps if d["comp"].get(m_row, 0.0) >= 0.5 * Em]
            rec["n_hyp"] = len(H)
            near = [d for d in H if abs(d["tf"] - rec["t"]) <= 5.0]
            rec["n_hyp_true_time"] = len(near)
            # candidate records holding >= 50 % of this Michel: status of the best one (scored > no_minima > zero_pred > no_pred)
            C = [d for d in cand_recs if d["comp"].get(m_row, 0.0) >= 0.5 * Em]
            order = {"scored": 3, "no_minima": 2, "zero_pred": 1, "no_pred": 0}
            if C:
                bc = max(C, key=lambda d: order[d["status"]]); rec["cand_status"] = bc["status"]; rec["cand_n"] = len(C)
                rec["cand_curve"] = {k: bc.get(k) for k in ("tt", "in_range", "curve_tt", "curve_min", "neg_frac", "n_min", "tfs", "E_p", "t_c")}
            else:
                rec["cand_status"] = "none"
            # piece (probe) records and whole-cluster records separately; also the piece record holding the MOST Michel energy
            # (even < 50 %: the probe is <= 5 cm and the Michel may be longer)
            for kind, key in (("cand", "piece_rec"), ("candw", "whole_rec")):
                K = [d for d in cand_recs if d["kind"] == kind and d["comp"].get(m_row, 0.0) > 0]
                if K:
                    bk = max(K, key=lambda d: (d["comp"].get(m_row, 0.0) >= 0.5 * Em, order[d["status"]], d["comp"].get(m_row, 0.0)))
                    rec[key] = dict(status=bk["status"], Lc=bk["Lc"], E_p=bk["E_p"], t_c=bk["t_c"], michel_frac=bk["comp"].get(m_row, 0.0) / Em, michel_purity=bk["comp"].get(m_row, 0.0) / max(bk["E_p"], 1e-9),
                                    tt=bk.get("tt"), in_range=bk.get("in_range"), curve_tt=bk.get("curve_tt"), n_min=bk["n_min"], tfs=bk["tfs"], n_recs=len(K))
            if near:
                b = max(near, key=lambda d: d["mf"]); rec["best"] = {k: b[k] for k in ("kind", "E_p", "mf", "alpha", "apc", "sat", "cov", "unexpl", "ok", "tf")}
                purity = b["comp"].get(m_row, 0.0) / max(b["E_p"], 1e-9); rec["best"]["michel_purity"] = float(purity)
                fails = []
                if b["mf"] < cfg.mf_min: fails.append("mf")
                if not (cfg.alpha_lo <= b["alpha"] <= (cfg.whole_alpha_hi if b["kind"] == "whole" else cfg.alpha_hi)): fails.append("alpha")
                if not (np.isfinite(b["apc"]) and b["apc"] <= cfg.apc_max): fails.append("apc")
                if b["sat"] > cfg.sat_max: fails.append("sat")
                if b["cov"] < cfg.cov_min: fails.append("cov")
                if b["unexpl"] < cfg.unexpl_min_frac: fails.append("unexpl")
                if b["E_p"] < cfg.piece_min_mev: fails.append("E_p")
                rec["fails"] = fails
            elif H:
                b = max(H, key=lambda d: d["mf"]); rec["best_other_time"] = {k: b[k] for k in ("kind", "E_p", "mf", "tf", "ok")}
            rec["corr_after"] = float((e[tagm == m_row] * (np.abs(ts[tagm == m_row] - tru[tagm == m_row]) <= CUT)).sum() / Em)
            audit.append(rec)
        print(f"{tag}: Michels {len(info)}, hypotheses recorded {len(hyps)}, stage {st['time_s']:.1f}s  (total {time.time()-t_begin:.0f}s)", flush=True)
    json.dump(dict(cfg={k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("_")}, audit=audit, hyps=ALL_HYPS), open(a.out, "w"), default=float)
    # ---- summary
    R = [r for r in audit if 0 <= r["t"] <= 885 and r["delay"] > 12 and r["E"] >= 5]
    E = np.array([r["E"] for r in R]); print(f"\n{len(files)} events; delayed Michels (in window, delay > 12 ticks, >= 5 MeV): {len(R)}, {E.sum()/1e3:.2f} GeV; correct before {np.average([r['corr_before'] for r in R], weights=E):.3f} after {np.average([r['corr_after'] for r in R], weights=E):.3f}")
    def frac(sel, lab):
        if not sel: return
        Es = np.array([r["E"] for r in sel]); print(f"  {lab:62s} n {len(sel):4d}  E {Es.sum()/1e3:5.2f} GeV  ({Es.sum()/E.sum():.2f} of delayed Michel E)  correct after {np.average([r['corr_after'] for r in sel], weights=Es):.2f}")
    unc = [r for r in R if r["covered_frac"] < 0.5]; frac(unc, "NOT covered by any candidate piece / whole cluster (geometry)")
    reasons = {}
    for r in unc:
        for w in (r.get("why_uncovered") or ["(none recorded)"]):
            reasons.setdefault(w, []).append(r)
    for w, sel in sorted(reasons.items(), key=lambda kv: -sum(r["E"] for r in kv[1])):
        frac(sel, f"   reason: {w}")
    frac([r for r in unc if r["merged"]], "   merged & uncovered"); frac([r for r in unc if not r["merged"]], "   own cluster & uncovered")
    # piece-level view: the probe piece (what is actually scanned) vs the whole cluster
    print("  --- piece-level view (probe pieces are what the scan uses; a merged cluster <= 80 MeV also enters as a whole, which is host-dominated) ---")
    pr = [r for r in R if r.get("piece_rec")]; frac(pr, "a probe piece holds some of the Michel")
    frac([r for r in pr if r["piece_rec"]["michel_frac"] >= 0.5], "   probe holds >= 50 % of the Michel")
    frac([r for r in pr if r["piece_rec"]["michel_frac"] < 0.5], "   probe holds < 50 % of the Michel (Michel longer than the probe / tip not at the Michel end)")
    frac([r for r in pr if r["piece_rec"]["michel_purity"] >= 0.8], "   probe >= 80 % Michel-pure")
    frac([r for r in pr if r["piece_rec"]["michel_purity"] < 0.8], "   probe < 80 % Michel-pure (host hits inside the probe)")
    for stt in ("no_pred", "zero_pred", "no_minima", "scored"):
        sel = [r for r in pr if r["piece_rec"]["status"] == stt]
        if sel:
            frac(sel, f"   probe status {stt}")
            cc = [r["piece_rec"] for r in sel if r["piece_rec"].get("curve_tt") is not None]
            if cc:
                print(f"         true time inside the scan range {np.mean([c['in_range'] for c in cc]):.2f}; curve at the true time < 0 for {np.mean([c['curve_tt'] < 0 for c in cc]):.2f}; probe E med {np.median([c['E_p'] for c in cc]):.1f} MeV, Lc med {np.median([c['Lc'] for c in cc]):.0f} cm, Michel purity med {np.median([c['michel_purity'] for c in cc]):.2f}, Michel frac med {np.median([c['michel_frac'] for c in cc]):.2f}")
    frac([r for r in R if not r.get("piece_rec") and r.get("whole_rec")], "no probe piece; only the whole cluster holds the Michel")
    frac([r for r in R if not r.get("piece_rec") and not r.get("whole_rec")], "neither a probe piece nor a whole cluster holds the Michel")
    nop = [r for r in R if not r.get("piece_rec") and r["merged"]]
    reasons = {}
    for r in nop:
        for w in (r.get("why_uncovered") or ["(none recorded)"]):
            reasons.setdefault(w, []).append(r)
    for w, sel in sorted(reasons.items(), key=lambda kv: -sum(r["E"] for r in kv[1])):
        frac(sel, f"   merged, no probe piece: reason {w}")
    cov = [r for r in R if r["covered_frac"] >= 0.5]; frac(cov, "covered by a candidate (>= 50 % of the Michel energy)")
    nh = [r for r in cov if r["n_hyp"] == 0]; frac(nh, "   covered but no hypothesis scored")
    for stt in ("none", "no_pred", "zero_pred", "no_minima", "scored"):
        sel = [r for r in nh if r.get("cand_status") == stt]
        if sel:
            frac(sel, f"      candidate status {stt}")
            if stt in ("no_minima", "scored"):
                cc = [r["cand_curve"] for r in sel if r.get("cand_curve") and r["cand_curve"].get("curve_tt") is not None]
                if cc:
                    ir = np.mean([c["in_range"] for c in cc]); ct = np.array([c["curve_tt"] for c in cc]); nf = np.array([c["neg_frac"] for c in cc if c["neg_frac"] is not None])
                    print(f"         true time inside the scan range {ir:.2f}; curve at the true time < 0 for {np.mean(ct < 0):.2f}; median negative fraction of the curve {np.median(nf) if nf.size else float('nan'):.2f}; piece E med {np.median([c['E_p'] for c in cc]):.1f} MeV")
    frac([r for r in cov if r["n_hyp"] > 0 and r["n_hyp_true_time"] == 0], "   hypotheses only at other times (scan missed the true time)")
    tt = [r for r in cov if r["n_hyp_true_time"] > 0]; frac(tt, "   hypothesis at the true time")
    frac([r for r in tt if r["best"]["ok"]], "      passing all gates")
    frac([r for r in tt if not r["best"]["ok"]], "      failing gates")
    for g in ("mf", "alpha", "apc", "sat", "cov", "unexpl", "E_p"):
        frac([r for r in tt if not r["best"]["ok"] and g in r["fails"]], f"         fails {g}")
    frac([r for r in tt if not r["best"]["ok"] and len(r["fails"]) == 1], "      failing exactly one gate")
    ok1 = [r for r in tt if r["best"]["ok"]]
    if ok1:
        print(f"  gate values at the true time (passing / failing): mf {np.median([r['best']['mf'] for r in ok1]):.2f} / {np.median([r['best']['mf'] for r in tt if not r['best']['ok']]):.2f}, "
              f"apc {np.median([r['best']['apc'] for r in ok1]):.2f} / {np.median([r['best']['apc'] for r in tt if not r['best']['ok']]):.2f}, unexpl {np.median([r['best']['unexpl'] for r in ok1]):.2f} / {np.median([r['best']['unexpl'] for r in tt if not r['best']['ok']]):.2f}, "
              f"michel purity of the piece {np.median([r['best']['michel_purity'] for r in tt]):.2f}")
    for hk in ("mu", "pi"):
        frac([r for r in R if r["host_kind"] == hk], f"host {hk}"); frac([r for r in R if r["host_kind"] == hk and r["covered_frac"] >= 0.5], f"   host {hk}: covered")
    frac([r for r in R if r["merged"]], "merged with the host"); frac([r for r in R if not r["merged"]], "own cluster")
    frac([r for r in R if not r["merged"] and r["covered_frac"] < 0.5], "   own cluster, not covered (whole-cluster window 5-80 MeV / templated?)")


if __name__ == "__main__":
    main()
