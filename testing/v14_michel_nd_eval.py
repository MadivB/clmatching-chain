"""Offline evaluation of the shared ND Michel kernel (testing/michel_nd.py) on chain dumps with truth labels (GPU).
Same physics as v14_michel_nd_fast.py but runs the production kernel, so what is measured is what the harness runs.
python v14_michel_nd_eval.py <dump_dir> [--events N] [--files ...] [--out JSON] [--set key=value ...] [--batch 64]"""
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

PULSE = os.path.join(BASE, "assets", "avg_pulse.npy")
SMALL = "/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/NewMLSection/runs/ndfull_small4x/best_model_arch.pt"
TMP0 = 1_000_000


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dump_dir"); ap.add_argument("--events", type=int, default=0); ap.add_argument("--files", default=None)
    ap.add_argument("--out", default=os.path.join(HERE, "michel_nd_out", "eval.json")); ap.add_argument("--set", nargs="*", default=[]); ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args(); cfg = MichelNDConfig()
    for kv in a.set:
        k, v = kv.split("="); cur = getattr(cfg, k)
        setattr(cfg, k, float(v) if cur is None else (type(cur)(float(v)) if isinstance(cur, (int, float)) and not isinstance(cur, bool) else (v.lower() in ("1", "true") if isinstance(cur, bool) else v)))
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
    print("models on", models.device, "cfg", {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("_")}, flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True); all_moves = []; ev_stats = []; t_begin = time.time()
    for f in files:
        z = np.load(f, allow_pickle=True); tag = os.path.basename(f)[:-4]
        e = np.clip(z["e"], 0, None).astype(np.float64); ts = z["ts_final"].astype(np.float64); tru = z["truth_t0"].astype(np.float64)
        labs = z["labels"].astype(int); tpc = z["tpc"].astype(int); x, y, zz = (z[k].astype(np.float32) for k in ("x", "y", "z"))
        A = z["full_light"].astype(np.float32); B = z["base_image"].astype(np.float32).copy(); veto = z["veto_mask"]; ceil = z["sat_ceiling"].astype(np.float32)
        al, at, ar = z["amp_lab"].astype(int), z["amp_tpc"].astype(int), z["amp_row"].astype(np.float64)
        amp = {(int(L), int(c)): ar[i] for i, (L, c) in enumerate(zip(al, at))}
        ok = np.isfinite(tru)
        # delayed class
        dmask = np.zeros(labs.size, bool)
        for L in np.unique(labs[labs >= 0]):
            h = np.flatnonzero(labs == L); m = h[ok[h] & (tru[h] >= 0) & (tru[h] <= 885)]
            if e[h].sum() < 10 or m.size < 5:
                continue
            tmaj = np.median(tru[m]); dmask[m[tru[m] - tmaj > 12]] = True
        E_del = float(e[dmask].sum()); Eb = float((e[dmask] * (np.abs(ts[dmask] - tru[dmask]) <= CUT)).sum())

        pred_t = []

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
                pred_t.append(meta["timings"])
                for j, k in pm["keys"]:
                    v = amps.get((TMP0 + j, k[1])); out[k] = np.asarray(v, np.float64) if v is not None else np.zeros(A.shape[1])
            return out
        st = run_michel_nd(xyz=np.stack([x, y, zz], 1).astype(np.float64), e=e, tpc=tpc, labels=labs, ts=ts, amp=amp, actual=A, base=B, veto=veto, ceil=ceil,
                           template=T, predict_fn=predict_fn, cfg=cfg, truth_t0=tru, apply=True)
        Ea = float((e[dmask] * (np.abs(ts[dmask] - tru[dmask]) <= CUT)).sum())
        mv = st.pop("moves")
        for m in mv:
            m.pop("hits", None); m["ev"] = tag
        all_moves += mv
        pt = {k: round(float(sum(t_[k] for t_ in pred_t)), 2) for k in ("grouping_s", "voxelize_s", "model_s", "materialize_s")} if pred_t else {}
        ev_stats.append(dict(ev=tag, **{k: v for k, v in st.items()}, E_del=E_del, E_del_before=Eb, E_del_after=Ea, pred_t=pt))
        print(f"{tag}: pieces {st['n_cand']} wholes {st['n_whole']} pred {st['n_pred']} moves {st['n_moves']} fixed {st['fixed_e']:.0f} broke {st['broke_e']:.0f} MeV {st['time_s']:.1f}s {st['tim']}  (total {time.time()-t_begin:.0f}s)", flush=True)
    json.dump(dict(cfg={k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("_")}, moves=all_moves, events=ev_stats), open(a.out, "w"), default=float)
    fx = sum(s_["fixed_e"] for s_ in ev_stats); br = sum(s_["broke_e"] for s_ in ev_stats); Ed = sum(s_["E_del"] for s_ in ev_stats); Eb = sum(s_["E_del_before"] for s_ in ev_stats); Ea = sum(s_["E_del_after"] for s_ in ev_stats)
    print(f"\n{len(files)} events: moves {len(all_moves)} (whole {sum(m['whole'] for m in all_moves)}); fixed {fx/1e3:.2f} broke {br/1e3:.2f} GeV net {(fx-br)/1e3:+.2f} purity {fx/max(fx+br,1e-9):.2f}; "
          f"events better/worse {sum(s_['fixed_e']>s_['broke_e']+0.5 for s_ in ev_stats)}/{sum(s_['broke_e']>s_['fixed_e']+0.5 for s_ in ev_stats)}; delayed class {Ed/1e3:.2f} GeV {Eb/max(Ed,1e-9):.3f} -> {Ea/max(Ed,1e-9):.3f}; time/ev {np.mean([s_['time_s'] for s_ in ev_stats]):.1f} s")
    for lab, sel in (("pieces", [m for m in all_moves if not m["whole"]]), ("whole", [m for m in all_moves if m["whole"]])):
        if sel:
            fx = sum(max(m["now"] - m["was"], 0) for m in sel); br = sum(max(m["was"] - m["now"], 0) for m in sel); print(f"   {lab:7s} n {len(sel):3d} fixed {fx:6.0f} broke {br:5.0f} purity {fx/max(fx+br,1e-9):.2f}")
    tt = {k: round(float(np.mean([s_["tim"][k] for s_ in ev_stats])), 2) for k in ev_stats[0]["tim"]} if ev_stats else {}
    print("mean time per event by part:", tt)


if __name__ == "__main__":
    main()
