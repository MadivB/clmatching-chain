"""Run the v2.0_test solver on chain dumps and compare with the chain's assignment.

python v200_run.py <dump_dir> [--events tag,tag,...] [--beta beta.json] [--sweeps 3]
                   [--workers 8] [--out results.json] [--verbose]
Prints per-event: efficiency (all light clusters + nearest-family coverage) for v2.0 and for
the chain's assignment re-rendered in the same loss, the light energies of both, and the
wrong-energy split by cluster class; then totals.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import v200_core as core  # noqa: E402

CLASSES = ["noise/blind", "<1.5", "1.5-5", "5-20", "20-100", ">=100"]


def load_beta(ev: core.Event, table: dict | None):
    """beta per contact pair from the calibrated table (see v200_calib.py)."""
    ev.beta = {}; ev.dtable = None; ev.pnew = None
    if not table:
        return
    if "beta" in table and isinstance(table["beta"], dict):      # new format with spatial prior
        ev.dtable = table.get("dprior") or None
        ev.pnew = table.get("pnew") or None
        table = table["beta"]
    out = {}
    for (L1, L2), d in ev.contacts.items():
        if L1 not in ev.lab_tpcs or L2 not in ev.lab_tpcs:
            continue
        key = feature_key(ev.lab_E[L1], ev.lab_E[L2], d)
        b = table.get(key)
        if b is not None and b != 0.0:
            out[(L1, L2)] = float(b)
    ev.beta = out


def ebucket(E: float) -> str:
    return "s" if E < 5 else ("m" if E < 30 else "b")


def feature_key(E1: float, E2: float, d: float) -> str:
    lo, hi = sorted([E1, E2])
    return f"{ebucket(lo)}{ebucket(hi)}_{'d1' if d <= 1.0 else 'd3'}"


def run_one(args):
    path, beta_table, sweeps, verbose = args
    ev = core.Event(path)
    load_beta(ev, beta_table)
    ref = core.chain_reference(ev)
    res = core.solve_event(ev, n_sweeps=sweeps, verbose=verbose)
    # contact-pair observations for the in-situ prior calibration (decidable pairs only)
    obs = []
    dec = set()
    slots = {}
    clusters = []
    for L in ev.light_labels:
        own, decid, margin, _, n_dom = ev.candidates(L, slots)
        if decid:
            dec.add(L)
        idx = ev.lab_hits[L]
        tr = ev.truth[idx]; tr = tr[np.isfinite(tr)]
        t_true = float(np.median(tr)) if tr.size else float("nan")
        tv = res["assign"].get(L); tc = ref["assign"].get(L)
        clusters.append([int(L), float(ev.lab_E[L]), float(ev.sig.get(L, 0.0)), float(margin if np.isfinite(margin) else 1e9),
                         int(decid), int(n_dom), -1 if tv is None else int(tv), -1 if tc is None else int(tc), t_true,
                         len(own), int(any(L in k for k in ev.contacts))])
    for (L1, L2), d in ev.contacts.items():
        if L1 in dec and L2 in dec and L1 in res["assign"] and L2 in res["assign"]:
            same = abs(res["assign"][L1] - res["assign"][L2]) <= core.SAME_TICKS
            obs.append((feature_key(ev.lab_E[L1], ev.lab_E[L2], d), bool(same),
                        float(ev.lab_E[L1]), float(ev.lab_E[L2]), float(d)))
    dobs, nobs = core.prior_observations(ev, set(res.get("decidable", [])))
    out = {"tag": ev.tag, "v200": {k: v for k, v in res.items() if k not in ("assign", "decidable")},
           "dprior_obs": dobs, "new_obs": nobs,
           "chain": {k: v for k, v in ref.items() if k != "assign"}, "pair_obs": obs,
           "assign_v200": {str(k): int(v) for k, v in res["assign"].items()},
           "clusters": clusters}
    line = (f"{ev.tag:>14}  eff v200 {res['eff']:.4f}  chain {ref['eff']:.4f} (chain all-hits {ref['eff_all_hits_chain']:.4f})"
            f"  E_light v200 {res['E_light']:.0f} chain {ref['E_light']:.0f}  decid {res['n_decidable']}  {res['time_s']:.0f}s")
    print(line, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir")
    ap.add_argument("--events", default="")
    ap.add_argument("--beta", default="")
    ap.add_argument("--sweeps", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.dump_dir, "full_dumps", "*.npz")))
    if a.events:
        want = set(a.events.split(","))
        files = [f for f in files if os.path.basename(f)[:-4] in want]
    table = json.load(open(a.beta)) if a.beta else None
    jobs = [(f, table, a.sweeps, a.verbose) for f in files]
    t0 = time.time()
    if a.workers > 1 and len(jobs) > 1:
        with Pool(a.workers) as pool:
            results = pool.map(run_one, jobs, chunksize=1)
    else:
        results = [run_one(j) for j in jobs]
    # totals
    def tot(key):
        Ew = sum(r[key]["E_wrong"] for r in results); Eo = sum(r[key]["E_ok"] for r in results)
        return 1.0 - Ew / max(Eo, 1e-9), Ew, Eo
    e_v, Ew_v, Eo = tot("v200"); e_c, Ew_c, _ = tot("chain")
    print(f"\n=== {len(results)} events: eff v200 {e_v:.4f}  chain {e_c:.4f}  (delta {100*(e_v-e_c):+.2f} pp; wrong E {Ew_v:.0f} vs {Ew_c:.0f} of {Eo:.0f} MeV)")
    print(f"    light energy sum: v200 {sum(r['v200']['E_light'] for r in results):.0f}  chain {sum(r['chain']['E_light'] for r in results):.0f}"
          f"   time/ev {np.median([r['v200']['time_s'] for r in results]):.0f} s  wall {time.time()-t0:.0f} s")
    print(f"    {'class':>12} {'wrong v200':>11} {'wrong chain':>11} {'class E':>9} {'v200 %':>7} {'chain %':>8}")
    for k in CLASSES:
        wv = sum(r["v200"][k][0] for r in results); wc = sum(r["chain"][k][0] for r in results); Ek = sum(r["v200"][k][1] for r in results)
        print(f"    {k:>12} {wv:11.0f} {wc:11.0f} {Ek:9.0f} {100*wv/max(Ek,1e-9):7.2f} {100*wc/max(Ek,1e-9):8.2f}")
    if a.out:
        json.dump(results, open(a.out, "w"))
        print("wrote", a.out)


if __name__ == "__main__":
    main()
