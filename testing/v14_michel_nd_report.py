"""Summarise ND stage-M apply outputs (michel_nd_out/*.json): per file fixed / broken energy, moves, delayed-class
recovery, move outcomes by piece energy and by delay.   python v14_michel_nd_report.py dev2_A.json [hold2_A.json ...]"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); O = os.path.join(HERE, "michel_nd_out")


def main(paths):
    for p in paths:
        J = json.load(open(os.path.join(O, p) if not os.path.isabs(p) else p))
        ev = J["events"]; mv = J["moves"]
        print(f"\n=== {p}: {len(ev)} events, apply={J['apply']}, gates mf>={J['cfg']['mf_min']} alpha[{J['cfg']['alpha_sel_lo']},{J['cfg']['alpha_sel_hi']}] apc<={J['cfg']['apc_max']} cov>={J['cfg']['cov_min']} unexpl>={J['cfg']['unexpl_frac']}")
        by = {}
        for e_ in ev:
            by.setdefault(e_["ev"][:7], []).append(e_)
        for f, es in sorted(by.items()):
            fx = sum(x["fixed"] for x in es); br = sum(x["broke"] for x in es); n = sum(x["n_moves"] for x in es)
            Ed = sum(x.get("E_del", 0) for x in es); Eb = sum(x.get("E_del_before", 0) for x in es); Ea = sum(x.get("E_del_after", 0) for x in es)
            print(f"  {f}: {len(es)} ev, moves {n}, fixed {fx:.0f} broke {br:.0f} MeV, net {fx-br:+.0f}; delayed class {Ed/1e3:.2f} GeV correct {Eb/max(Ed,1e-9):.3f} -> {Ea/max(Ed,1e-9):.3f}; time/ev {np.mean([x['t'] for x in es]):.0f} s")
        fx = sum(x["fixed"] for x in ev); br = sum(x["broke"] for x in ev); Ed = sum(x.get("E_del", 0) for x in ev); Eb = sum(x.get("E_del_before", 0) for x in ev); Ea = sum(x.get("E_del_after", 0) for x in ev)
        print(f"  TOTAL: moves {len(mv)}, fixed {fx/1e3:.2f} broke {br/1e3:.2f} GeV (net {(fx-br)/1e3:+.2f}); delayed class {Ed/1e3:.2f} GeV correct {Eb/max(Ed,1e-9):.3f} -> {Ea/max(Ed,1e-9):.3f}; events better/worse {sum(x['fixed']>x['broke']+0.5 for x in ev)}/{sum(x['broke']>x['fixed']+0.5 for x in ev)}")
        if mv:
            EL = {(h["ev"], h["L"]): h["E_L"] for h in J["hyps"]}
            print("  post-filters chosen on the dev files (E_p >= 5 MeV, delay <= 300 ticks, host cap on end-piece moves):")
            for emin, dmax, cap in ((0, 1e9, None), (5, 300, None), (5, 300, 200.0), (5, 300, 150.0)):
                keep = [m for m in mv if m["E_p"] >= emin and (m["tf"] - m["t_c"]) <= dmax and (cap is None or m["tip"] < 0 or EL.get((m["ev"], m["L"]), 0) <= cap)]
                fx = sum(max(m["now"] - m["was"], 0) for m in keep); br = sum(max(m["was"] - m["now"], 0) for m in keep)
                nb = sum(1 for m in keep if m["now"] < m["was"] - 0.5)
                print(f"    E_p >= {emin}  delay <= {dmax:5.0f}  host cap {str(cap):5s}: moves {len(keep):3d} (bad {nb:3d})  fixed {fx:6.0f}  broke {br:5.0f}  net {fx-br:+6.0f} MeV  purity {fx/max(fx+br,1e-9):.2f}")
            M = mv; Ep = np.array([m["E_p"] for m in M]); d = np.array([m["now"] - m["was"] for m in M]); whole = np.array([m["tip"] < 0 for m in M])
            print(f"  moves: pieces {int((~whole).sum())} whole {int(whole.sum())}; outcome right {np.mean(d > 0.5):.2f} neutral {np.mean(np.abs(d) <= 0.5):.2f} wrong {np.mean(d < -0.5):.2f}; E_p p10/50/90 {np.percentile(Ep,[10,50,90]).round(1)} MeV")
            for lo, hi in ((0, 5), (5, 20), (20, 100), (100, 1e9)):
                m = (Ep >= lo) & (Ep < hi)
                if m.any():
                    print(f"    piece E {lo:3.0f}-{min(hi,999):3.0f} MeV: n {m.sum():3d}  right {np.mean(d[m] > 0.5):.2f}  wrong {np.mean(d[m] < -0.5):.2f}  net {d[m].sum():+.0f} MeV")
            dl = np.array([m["tf"] - m["t_c"] for m in M]) if "t_c" in M[0] else None
            if dl is not None:
                for lo, hi in ((12, 30), (30, 100), (100, 300), (300, 1e9)):
                    m = (dl >= lo) & (dl < hi)
                    if m.any():
                        print(f"    delay {lo:3.0f}-{min(hi,999):3.0f} ticks: n {m.sum():3d}  right {np.mean(d[m] > 0.5):.2f}  wrong {np.mean(d[m] < -0.5):.2f}  net {d[m].sum():+.0f} MeV")


if __name__ == "__main__":
    main(sys.argv[1:])
