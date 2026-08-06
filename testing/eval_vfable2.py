"""Synthesize v1 + vFable: keep each family's strongest, complexity-normalize, combine,
event-split validate.  Strengths: v1 displaced (amount -> energy), vFable maxk (spatial
coherence) + Anscombe chi2 (low-PE), vFable z_zmis_y (localized displaced detector).
"""
import os, json, numpy as np

HERE = os.path.abspath(os.path.dirname(__file__))
V1 = json.load(open(os.path.join(HERE, "eval_final_out.json")))       # chi2(flash), disp
VF = json.load(open(os.path.join(HERE, "confidence_vfable_out.json")))  # chi2(ansc), maxk, z_zmis_y
FR = json.load(open(os.path.join(HERE, "eval_fullres_out.json")))      # nflash, nlit, E, wf

n = len(V1["wf"]); assert len(VF["wf"]) == n == len(FR["wf"])
WF = np.array(V1["wf"]); EE = np.array(V1["E"])
NF = np.array(FR["nflash"], float); NL = np.array(FR["nlit"], float); logE = np.log(np.maximum(EE, 1))
feat = dict(v1_chi2=np.array(V1["chi2"]), v1_disp=np.array(V1["disp"]),
            an_chi2=np.array(VF["chi2"]), maxk=np.array(VF["maxk"]), zmis_y=np.array(VF["z_zmis_y"]))
bad = WF > 0.20; good = WF < 0.05; sel = bad | good; y = bad[sel].astype(int)
keys_ev = None  # event split below uses FR order; need event ids -> load from cache
import pickle
cache = pickle.load(open(os.path.join(HERE, "lightcache.pkl"), "rb"))
EVID = [(r["file"], r["event"]) for r in cache]


def rank(x): return np.argsort(np.argsort(np.asarray(x, float))).astype(float) / len(x)
def auc(sc, yy):
    sc = np.asarray(sc); pos = sc[yy == 1]; neg = sc[yy == 0]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    o = np.argsort(np.concatenate([pos, neg])); rk = np.empty_like(o, float); rk[o] = np.arange(1, len(o) + 1)
    return (rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
def spr(a, b): return float(np.corrcoef(rank(a), rank(b))[0, 1])
def presid(metric, mask=None):
    m = metric if mask is None else metric[mask]
    ctr = [NF, logE, NL] if mask is None else [NF[mask], logE[mask], NL[mask]]
    rm = rank(m); X = np.c_[[np.ones_like(rm)] + [rank(c) for c in ctr]].T
    return rm - X @ np.linalg.lstsq(X, rm, rcond=None)[0]


print(f"{n} TPCs. singles (complexity-normalized):")
print(f"{'feature':>10} {'AUCraw':>7} {'AUCcx':>7} {'rhoCx':>8}")
res = {}
for k, v in feat.items():
    cx = presid(v); res[k] = cx
    print(f"{k:>10} {auc(v[sel], y):>7.3f} {auc(cx[sel], y):>7.3f} {spr(cx, WF):>+8.3f}")

print("\ncombinations (complexity-normalized, mean-rank):")
combos = {
    "v1(chi2+disp)": ["v1_chi2", "v1_disp"],
    "v1disp+maxk": ["v1_disp", "maxk"],
    "v1disp+maxk+anchi2": ["v1_disp", "maxk", "an_chi2"],
    "v1(2)+maxk": ["v1_chi2", "v1_disp", "maxk"],
    "v1(2)+maxk+zmisY": ["v1_chi2", "v1_disp", "maxk", "zmis_y"],
    "all5": ["v1_chi2", "v1_disp", "an_chi2", "maxk", "zmis_y"],
}
best = None
for name, ks in combos.items():
    c = np.mean([rank(res[k]) for k in ks], axis=0)
    a = auc(c[sel], y); rf = spr(c, WF)
    print(f"   {name:>22} AUC={a:.3f} rhoFrac={rf:+.3f}")
    if best is None or a > best[0]: best = (a, rf, name, ks)
print(f"\nBEST: {best[2]} AUC={best[0]:.3f} rhoFrac={best[1]:.3f}")

# event-split validation of the best (complexity-normalization recomputed within split)
uk = sorted(set(EVID)); half = set(uk[::2]); inA = np.array([e in half for e in EVID])
print("\nEVENT-SPLIT (complexity-norm within split):")
for grp, mask in [("A", inA), ("B", ~inA)]:
    parts = []
    for k in best[3]:
        parts.append(rank(presid(feat[k], mask)))
    c = np.mean(parts, axis=0)
    s = sel[mask]; yy = bad[mask][s].astype(int)
    print(f"  split{grp} (BAD={int(yy.sum())} GOOD={int((yy==0).sum())}): AUC={auc(c[s], yy):.3f} rhoFrac={spr(c, WF[mask]):+.3f}")

# save winning metric
cbest = np.mean([rank(res[k]) for k in best[3]], axis=0)
json.dump({"score": list(map(float, cbest)), "wf": list(map(float, WF)), "E": list(map(float, EE)),
           "components": best[3]}, open(os.path.join(HERE, "confidence_v2_out.json"), "w"))
print("\nsaved confidence_v2_out.json"); print("DONE")
