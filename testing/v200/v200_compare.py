"""Side-by-side class table for several v2.0 result files (same events) plus the chain reference.
python v200_compare.py A.json B.json ...  (labels = file stems)"""
import json
import os
import sys

CLASSES = ["noise/blind", "<1.5", "1.5-5", "5-20", "20-100", ">=100"]


def main(paths):
    arms = {os.path.basename(p)[:-5]: json.load(open(p)) for p in paths}
    ref = next(iter(arms.values()))
    tags = sorted(set.intersection(*[set(r["tag"] for r in a) for a in arms.values()]))
    print(f"{len(tags)} common events")
    cols = ["chain"] + list(arms)
    def rows_of(name):
        if name == "chain":
            return {r["tag"]: r["chain"] for r in ref}
        return {r["tag"]: r["v200"] for r in arms[name]}
    R = {c: rows_of(c) for c in cols}
    print(f"{'':>12} " + " ".join(f"{c:>12}" for c in cols))
    line = []
    for c in cols:
        Ew = sum(R[c][t]["E_wrong"] for t in tags); Eo = sum(R[c][t]["E_ok"] for t in tags)
        line.append(f"{100*(1-Ew/Eo):12.3f}")
    print(f"{'eff %':>12} " + " ".join(line))
    line = [f"{sum(R[c][t]['E_light'] for t in tags)/1e6:12.3f}" for c in cols]
    print(f"{'E_light/1e6':>12} " + " ".join(line))
    print("wrong energy fraction per class (%):")
    for k in CLASSES:
        line = []
        for c in cols:
            wv = sum(R[c][t][k][0] for t in tags); Ek = sum(R[c][t][k][1] for t in tags)
            line.append(f"{100*wv/max(Ek,1e-9):12.2f}")
        print(f"{k:>12} " + " ".join(line))
    print("wrong energy per class (MeV):")
    for k in CLASSES:
        line = [f"{sum(R[c][t][k][0] for t in tags):12.0f}" for c in cols]
        print(f"{k:>12} " + " ".join(line))
    # paired per-event better/worse vs chain
    for c in cols[1:]:
        b = sum(1 for t in tags if R[c][t]["E_wrong"] < R["chain"][t]["E_wrong"] - 1e-6)
        w = sum(1 for t in tags if R[c][t]["E_wrong"] > R["chain"][t]["E_wrong"] + 1e-6)
        print(f"{c}: events better/worse than chain {b}/{w}; median time {sorted(R[c][t].get('time_s', 0) for t in tags)[len(tags)//2]:.0f} s")


if __name__ == "__main__":
    main(sys.argv[1:])
