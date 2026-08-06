"""Assemble the v0.4 master dataset for all deliverables:
 - per-TPC: energy, truth wrong-fraction/energy, confidence v2 (+components)
 - per-cluster (from lab_rows): energy, correct(|reco-truth|<=10), type
 - per-hit: energy, correct, backbone(ts_p1 finite), has_truth
 - per-event: stage timings, total/truth energy, accuracy, aggregate confidence
"""
import glob, os, re, json, pickle, numpy as np
from confidence_v04 import Calibration

HERE = os.path.abspath(os.path.dirname(__file__))
CUT = 10
tag = lambda s: (re.search(r"(\d{7})", str(s)) or ["", "0000000"])[0] if re.search(r"(\d{7})", str(s)) else "0000000"

# ---------- confidence v2 from validated components (cache order) ----------
cache = pickle.load(open(os.path.join(HERE, "lightcache.pkl"), "rb"))
V1 = json.load(open(os.path.join(HERE, "eval_final_out.json")))
VF = json.load(open(os.path.join(HERE, "confidence_vfable_out.json")))
FR = json.load(open(os.path.join(HERE, "eval_fullres_out.json")))
chi2 = np.array(V1["chi2"]); disp = np.array(V1["disp"]); maxk = np.array(VF["maxk"])
NF = np.array(FR["nflash"], float); NL = np.array(FR["nlit"], float)
E_tpc = np.array([r["E"] for r in cache]); wf = np.array([r["wf"] for r in cache])
conf = Calibration().fit_score({"chi2_flash": chi2, "displaced": disp, "maxk": maxk}, NF, E_tpc, NL)
key_tpc = [(r["file"], int(r["event"]), int(r["tpc"])) for r in cache]
conf_by = {k: float(c) for k, c in zip(key_tpc, conf)}

# sanity: reproduce AUC / rhoFrac
bad = wf > 0.20; good = wf < 0.05; sel = bad | good; y = bad[sel].astype(int)
def auc(s, yy):
    p = s[yy == 1]; n = s[yy == 0]; o = np.argsort(np.concatenate([p, n])); rk = np.empty_like(o, float); rk[o] = np.arange(1, len(o) + 1)
    return (rk[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n))
rho = np.corrcoef(np.argsort(np.argsort(conf)), np.argsort(np.argsort(wf)))[0, 1]
print(f"confidence v2 check: AUC={auc(conf[sel], y):.3f} rhoFrac={rho:+.3f}  (expect 0.935/0.654)")

TPC = dict(file=[], event=[], tpc=[], E=[], wf=[], wrongE=[], conf=[], chi2=[], disp=[], maxk=[], nflash=[])
for k, r in zip(key_tpc, cache):
    TPC["file"].append(k[0]); TPC["event"].append(k[1]); TPC["tpc"].append(k[2])
    TPC["E"].append(r["E"]); TPC["wf"].append(r["wf"]); TPC["wrongE"].append(r["wf"] * r["E"])
    TPC["conf"].append(conf_by[k])
for a, arr in [("chi2", chi2), ("disp", disp), ("maxk", maxk), ("nflash", NF)]:
    TPC[a] = list(map(float, arr))
TPC = {k: np.array(v) for k, v in TPC.items()}

# ---------- timing per event (from jsonl) ----------
timing = {}
for fp in sorted(glob.glob(os.path.join(HERE, "v030v3_out", "groups_worker*.jsonl"))):
    for l in open(fp):
        if not l.strip(): continue
        r = json.loads(l); k = (tag(r["file"]), int(r["event"]))
        timing[k] = dict(t_event=r.get("t_event_s"), tim=r.get("timing", {}),
                         total_E=r.get("total_energy"), truth_E=r.get("truth_energy"),
                         acc=r.get("acc_rescued"))

# ---------- per-cluster + per-hit from dumps ----------
CL = dict(file=[], event=[], energy=[], e_wrong=[], reco=[], truth=[], correct=[], ctype=[])
HIT = dict(file=[], event=[], tpc=[], e=[], correct=[], backbone=[], has_truth=[])
EV = dict(file=[], event=[], t_event=[], first=[], backbone=[], rescue=[], staged=[], rest=[],
          total_E=[], truth_E=[], acc=[], conf_ewm=[], conf_max=[], wrongE=[])
for fp in sorted(glob.glob(os.path.join(HERE, "v030v3_out", "dumps", "*.npz"))):
    d = np.load(fp, allow_pickle=True); ft = os.path.basename(fp).split("_ev")[0]; ev = int(d["event"])
    e = np.asarray(d["e"], float); tp = np.asarray(d["tpc"]); tsf = np.asarray(d["ts_final"], float)
    tsp1 = np.asarray(d["ts_p1"], float); tr = np.asarray(d["truth_t0"], float)
    ht = np.isfinite(tr); corr = ht & (np.abs(tsf - tr) <= CUT)
    HIT["file"] += [ft] * len(e); HIT["event"] += [ev] * len(e)
    HIT["tpc"] += list(tp); HIT["e"] += list(e); HIT["correct"] += list(corr)
    HIT["backbone"] += list(np.isfinite(tsp1)); HIT["has_truth"] += list(ht)
    for c in json.loads(str(d["lab_rows"])):
        t = c.get("truth"); en = c.get("e_ok", 0) + c.get("e_wrong", 0)
        CL["file"].append(ft); CL["event"].append(ev); CL["energy"].append(en)
        CL["e_wrong"].append(c.get("e_wrong", 0)); CL["reco"].append(c.get("reco_t0"))
        CL["truth"].append(t); CL["ctype"].append(c.get("type", "?"))
        CL["correct"].append(bool(t is not None and np.isfinite(t) and abs(c.get("reco_t0", 1e9) - t) <= CUT))
    # per-event aggregate confidence over its TPCs
    m = (TPC["file"] == ft) & (TPC["event"] == ev)
    cE = TPC["E"][m]; cC = TPC["conf"][m]
    ewm = float((cC * cE).sum() / cE.sum()) if cE.sum() > 0 else 0.0
    tk = timing.get((ft, ev), {}); ti = tk.get("tim", {}) or {}
    fs = ti.get("first_stage", np.nan); ab = ti.get("after_backbone", np.nan)
    ar = ti.get("after_rescue_adopt", np.nan); ast = ti.get("after_staged", np.nan); te = tk.get("t_event", np.nan)
    EV["file"].append(ft); EV["event"].append(ev); EV["t_event"].append(te)
    EV["first"].append(fs); EV["backbone"].append(ab - fs if ab == ab else np.nan)
    EV["rescue"].append(ar - ab if ar == ar else np.nan); EV["staged"].append(ast - ar if ast == ast else np.nan)
    EV["rest"].append(te - ast if (te == te and ast == ast) else np.nan)
    EV["total_E"].append(tk.get("total_E")); EV["truth_E"].append(tk.get("truth_E")); EV["acc"].append(tk.get("acc"))
    EV["conf_ewm"].append(ewm); EV["conf_max"].append(float(cC.max()) if len(cC) else 0.0)
    EV["wrongE"].append(float(TPC["wrongE"][m].sum()))
    d.close()

CL = {k: np.array(v, dtype=object if k in ("ctype",) else float) if k not in ("file",) else np.array(v) for k, v in CL.items()}
HIT = {k: np.array(v) for k, v in HIT.items()}
EV = {k: np.array(v, dtype=object) if k == "file" else np.array(v, float) for k, v in EV.items()}
pickle.dump(dict(TPC=TPC, CL=CL, HIT=HIT, EV=EV), open(os.path.join(HERE, "master_v04.pkl"), "wb"))

# quick summary
ht = HIT["has_truth"].astype(bool); bb = HIT["backbone"].astype(bool); cor = HIT["correct"].astype(bool)
def eff(mask):
    m = mask & ht; ew = HIT["e"][m]; return 100 * ew[cor[m]].sum() / ew.sum(), 100 * cor[m].mean(), int(m.sum())
print(f"HITS: n={len(HIT['e'])}, with truth={int(ht.sum())}")
for nm, mk in [("all", np.ones(len(ht), bool)), ("backbone", bb), ("non-backbone", ~bb)]:
    e_eff, h_eff, n = eff(mk); print(f"  {nm:>13}: energy-eff={e_eff:.2f}%  hit-eff={h_eff:.2f}%  N={n}")
print(f"CLUSTERS: {len(CL['energy'])}  with truth={int(np.isfinite(CL['truth'].astype(float)).sum())}")
print(f"EVENTS: {len(EV['event'])}  mean t_event={np.nanmean(EV['t_event']):.1f}s")
print("saved master_v04.pkl"); print("DONE")
