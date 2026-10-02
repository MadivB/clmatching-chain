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

PULSE = os.path.join(BASE, "assets", "avg_pulse.npy")
SMALL = "/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/NewMLSection/runs/ndfull_small4x/best_model_arch.pt"
TMP0 = 1_000_000


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dump_dir"); ap.add_argument("--events", type=int, default=0); ap.add_argument("--files", default=None)
    ap.add_argument("--out", default=os.path.join(HERE, "michel_nd_out", "michel_audit.json")); ap.add_argument("--set", nargs="*", default=[]); ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args(); cfg = MichelNDConfig()
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
                                                                  batch_size=a.batch, restrict_clusters=pm["want"], device_policy="auto", amplitudes_only=True)
                amps = {(int(c_), int(t_)): meta["amplitudes"][i] for i, (c_, t_) in enumerate(zip(meta["group_cls"], meta["group_tpcs"]))}
                for j, k in pm["keys"]:
                    v = amps.get((TMP0 + j, k[1])); out[k] = np.asarray(v, np.float64) if v is not None else np.zeros(A.shape[1])
            return out
        st = run_michel_nd(xyz=np.stack([x, y, zz], 1).astype(np.float64), e=e, tpc=tpc, labels=labs, ts=ts, amp=amp, actual=A, base=B, veto=veto, ceil=ceil,
                           template=T, predict_fn=predict_fn, cfg=cfg, truth_t0=tru, apply=True, debug_tag=tagm, debug_out=dbg)
        covered = next((d["covered"] for d in dbg if d["kind"] == "geometry"), {})
        hyps = [d for d in dbg if d["kind"] != "geometry"]
        for m_row, rec in info.items():
            Em = rec["E"]; rec["covered_frac"] = float(covered.get(m_row, 0.0) / Em)
            # hypotheses holding >= 50 % of this Michel's energy
            H = [d for d in hyps if d["comp"].get(m_row, 0.0) >= 0.5 * Em]
            rec["n_hyp"] = len(H)
            near = [d for d in H if abs(d["tf"] - rec["t"]) <= 5.0]
            rec["n_hyp_true_time"] = len(near)
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
    json.dump(dict(cfg={k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("_")}, audit=audit), open(a.out, "w"), default=float)
    # ---- summary
    R = [r for r in audit if 0 <= r["t"] <= 885 and r["delay"] > 12 and r["E"] >= 5]
    E = np.array([r["E"] for r in R]); print(f"\n{len(files)} events; delayed Michels (in window, delay > 12 ticks, >= 5 MeV): {len(R)}, {E.sum()/1e3:.2f} GeV; correct before {np.average([r['corr_before'] for r in R], weights=E):.3f} after {np.average([r['corr_after'] for r in R], weights=E):.3f}")
    def frac(sel, lab):
        if not sel: return
        Es = np.array([r["E"] for r in sel]); print(f"  {lab:62s} n {len(sel):4d}  E {Es.sum()/1e3:5.2f} GeV  ({Es.sum()/E.sum():.2f} of delayed Michel E)  correct after {np.average([r['corr_after'] for r in sel], weights=Es):.2f}")
    frac([r for r in R if r["covered_frac"] < 0.5], "NOT covered by any candidate piece / whole cluster (geometry)")
    cov = [r for r in R if r["covered_frac"] >= 0.5]; frac(cov, "covered by a candidate (>= 50 % of the Michel energy)")
    frac([r for r in cov if r["n_hyp"] == 0], "   covered but no hypothesis scored (probe mf < 0.5 / no scan minimum)")
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
