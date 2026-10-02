"""Gate scan on the labelled ND stage-M hypothesis sample (michel_nd_out/label_*.json): simulate the gates and the
per-flash selection (fullest passing piece per (cluster, tip), best mf wins, a cluster moves once, flashes in the
recorded order = brightest first) and score good = energy moved to the right time, bad = energy that was right and is
now wrong.   python v14_michel_nd_gatescan.py michel_nd_out/label_13.json"""
import itertools
import json
import sys

import numpy as np

CUT = 10.0


def simulate(H, g):
    good = bad = 0.0; n_mv = n_bad = 0
    by_flash = {}
    for h in H:
        by_flash.setdefault((h["ev"], h["tpc"], round(h["tf"], 1)), []).append(h)
    moved = set()
    for key in by_flash:              # dict preserves insertion order = the recorded (brightest-first) order per event
        hs = by_flash[key]; ev = key[0]
        passing = []
        for h in hs:
            if (ev, h["L"]) in moved:
                continue
            whole = h["tip"] < 0
            ok = (h["mf"] >= (g["mf_whole"] if whole else g["mf"]) and g["alpha_lo"] <= h["alpha"] <= g["alpha_hi"] and np.isfinite(h["apc"]) and h["apc"] <= g["apc"]
                  and h["sat"] <= 0.5 and h["cov"] >= g["cov"] and (not whole or h["E_p"] >= g["whole_min"]) and (whole or h["Lc"] <= g["Lmax"]))
            if ok:
                passing.append(h)
        if not passing:
            continue
        reps = {}
        for h in passing:
            k = (h["L"], h["tip"])
            if k not in reps or h["E_p"] > reps[k]["E_p"]:
                reps[k] = h
        top = max(reps.values(), key=lambda h: h["mf"]); best = top
        for h in reps.values():
            if h is not top and h["L"] == top["L"] and h["mf"] >= top["mf"] - 0.02 and h["E_p"] > best["E_p"] and h["tip"] >= 0 and top["tip"] >= 0:
                best = h
        d = best["E_p"] * (best["true_frac"] - best["cur_frac"])
        good += max(d, 0.0); bad += max(-d, 0.0); n_mv += 1; n_bad += int(d < -0.5)
        moved.add((ev, best["L"]))
    return good, bad, n_mv, n_bad


def main(path):
    J = json.load(open(path)); H = J["hyps"]
    base = dict(mf=0.8, mf_whole=0.8, alpha_lo=0.8, alpha_hi=3.0, apc=-0.5, cov=0.1, whole_min=5.0, Lmax=16.0)
    print(f"{len(H)} hypotheses, {len(J['events'])} events; true energy available {sum(h['E_p'] for h in H if h['true_frac'] >= 0.5)/1e3:.2f} GeV (counting every hypothesis, nested pieces overlap)")
    rows = []
    for mf, ah, apc, cov, lo in itertools.product((0.7, 0.8, 0.9), (3.0, 6.0, 10.0), (-0.3, -0.5, -1.0, 0.0), (0.05, 0.1, 0.2), (0.6, 0.8)):
        g = dict(base, mf=mf, mf_whole=mf, alpha_hi=ah, apc=apc, cov=cov, alpha_lo=lo)
        good, bad, n, nb = simulate(H, g); rows.append((good - bad, good, bad, n, nb, g))
    rows.sort(key=lambda r: -r[0])
    print("\nbest 12 gate sets by net energy (good - bad) [MeV]:")
    for net, good, bad, n, nb, g in rows[:12]:
        print(f"  net {net:7.0f}  good {good:6.0f}  bad {bad:5.0f}  moves {n:3d} (bad {nb:2d})  mf>={g['mf']} alpha[{g['alpha_lo']},{g['alpha_hi']}] apc<={g['apc']} cov>={g['cov']}")
    print("\n2x2 v2 gates:", end=" ")
    good, bad, n, nb = simulate(H, base); print(f"net {good-bad:.0f} good {good:.0f} bad {bad:.0f} moves {n} (bad {nb})")
    for name, g in (("alpha_hi 6", dict(base, alpha_hi=6.0)), ("alpha_hi 10", dict(base, alpha_hi=10.0)), ("alpha_hi 6, apc -1", dict(base, alpha_hi=6.0, apc=-1.0)),
                    ("alpha_hi 6, mf 0.7", dict(base, alpha_hi=6.0, mf=0.7, mf_whole=0.7)), ("alpha_hi 6, cov 0.05", dict(base, alpha_hi=6.0, cov=0.05)), ("no apc gate, alpha_hi 6", dict(base, alpha_hi=6.0, apc=99.0))):
        good, bad, n, nb = simulate(H, g); print(f"{name:26s} net {good-bad:.0f} good {good:.0f} bad {bad:.0f} moves {n} (bad {nb})")


if __name__ == "__main__":
    main(sys.argv[1])
