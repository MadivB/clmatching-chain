"""Offline gate scan over the hypotheses saved by v15_michel_audit.py (--out JSON, key 'hyps'): emulate the kernel's selection
(pass gates -> one hypothesis per cluster, best mf) for many gate sets and report moved energy right / wrong (truth), net,
purity and the delayed-Michel energy recovered.  Probe pieces stand in for the refined move (the refine step extends the
piece along the host; its correctness follows the probe's).
python v15_michel_gatescan.py michel_nd_out/michel_audit5_63.json"""
import itertools
import json
import sys

import numpy as np

J = json.load(open(sys.argv[1])); H = [h for h in J["hyps"] if h.get("right_now") is not None]
P = [h for h in H if h["kind"] == "probe"]; W = [h for h in H if h["kind"] == "whole"]
n_ev = len(set(h["ev"] for h in H))
print(f"{n_ev} spills; hypotheses probe {len(P)} whole {len(W)}; delayed-Michel energy inside probes (michel_E, >=50 % pure) "
      f"{sum(h['michel_E'] for h in P if h['michel_E'] >= 0.5 * h['E_p'])/1e3:.2f} GeV")


def select(rows, mf_min=0.8, apc_max=-0.5, apc_pfrac=None, alpha=(0.8, 6.0), walpha=2.0, unexpl_min=0.1, cov_min=0.2, sat_max=0.5, ext_min=0.0, band_max=1e9, dedx_max=1e9, face_min=0.0, n_sup_min=0):
    best = {}
    for h in rows:
        whole = h["kind"] == "whole"
        if h["mf"] < mf_min or h["alpha"] < alpha[0] or h["alpha"] > (walpha if whole else alpha[1]):
            continue
        apc_ok = (h["apc"] is not None and h["apc"] <= apc_max)
        if apc_pfrac is not None and not apc_ok:
            apc_ok = (h.get("pfrac") is not None and h["pfrac"] < apc_pfrac)      # apc has no power when the piece is faint under its host
        if not apc_ok or h["unexpl"] < unexpl_min or h["cov"] < cov_min or h["sat"] > sat_max or h["E_p"] < 5:
            continue
        if not whole:
            if (h.get("ext") or 0) < ext_min or (h.get("band") if h.get("band") is not None else 0) > band_max or (h.get("dedx") or 0) > dedx_max or (h.get("face") if h.get("face") is not None else 1e9) < face_min:
                continue
        if (h.get("n_sup") or 0) < n_sup_min:
            continue
        k = (h["ev"], h["L"])
        if k not in best or h["mf"] > best[k]["mf"]:
            best[k] = h
    return list(best.values())


def report(sel, lab):
    E = np.array([h["E_p"] for h in sel]); rn = np.array([h["right_now"] for h in sel]); rb = np.array([h["right_before"] for h in sel])
    good = float((E * rn).sum()); bad = float((E * (1 - rn)).sum()); fixed = float((E * np.clip(rn - rb, 0, None)).sum()); broke = float((E * np.clip(rb - rn, 0, None)).sum())
    mich = float(sum(h["michel_E"] * h["right_now"] for h in sel))
    print(f"  {lab:64s} moves {len(sel):4d}  moved E {E.sum()/1e3:5.2f} GeV  right {good/max(good+bad,1e-9):.2f}  fixed {fixed/1e3:5.2f} broke {broke/1e3:5.2f} net {(fixed-broke)/1e3:+5.2f} GeV  Michel E recovered {mich/1e3:.2f} GeV")
    return fixed - broke


print("\n=== reference (kernel gates: mf 0.8, apc -0.5, alpha 0.8-6 / whole 2, unexpl 0.1, cov 0.2, sat 0.5), geometry as run")
report(select(P + W), "certified light gates")
report(select(P), "   pieces only"); report(select(W), "   wholes only")
print("\n=== geometry (pieces only, certified light gates)")
for ext in (10, 20):
    for band in (1e9, 5, 4):
        for face in (0, 5):
            report(select(P, ext_min=ext, band_max=band, face_min=face), f"extent >= {ext}, band dE/dx <= {band if band < 1e8 else 'inf'}, face >= {face}")
print("\n=== light gates (pieces, extent >= 20, face >= 5)")
for mf in (0.75, 0.8, 0.85):
    for apc in (-0.5, -0.25, 0.0, 1e9):
        report(select(P, mf_min=mf, apc_max=apc, ext_min=20, face_min=5), f"mf >= {mf}, apc <= {apc if apc < 1e8 else 'none'}")
print("\n=== conditional apc: apc <= -0.5 OR piece light < pfrac x host light on its support (pieces, extent >= 20, face >= 5)")
for mf in (0.8, 0.85, 0.9):
    for pf in (0.1, 0.2, 0.3):
        for u in (0.1, 0.2, 0.3):
            report(select(P, mf_min=mf, apc_pfrac=pf, unexpl_min=u, ext_min=20, face_min=5), f"mf >= {mf}, apc or pfrac < {pf}, unexpl >= {u}")
print("\n=== unexpl / alpha (pieces, extent >= 20, face >= 5, mf 0.8, apc -0.5)")
for u in (0.1, 0.2, 0.3, 0.4):
    for al in ((0.8, 6.0), (0.5, 6.0), (0.8, 10.0)):
        report(select(P, unexpl_min=u, alpha=al, ext_min=20, face_min=5), f"unexpl >= {u}, alpha {al}")
