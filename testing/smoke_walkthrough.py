"""Headless smoke test of the v0.4.1 walkthrough engine (notebook path).

Mirrors every W.* call of v040_step_by_step.ipynb with a print-only callback;
passes iff the whole chain runs and prints SMOKE_OK with the TPC summary.
"""
import sys

TESTING = "/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing"
sys.path.insert(0, TESTING)
import v040_walkthrough as W  # noqa: E402

FTAG, EV, TPC = "0000003", 6, 12
DATA = ("/global/cfs/cdirs/dunepro/people/abooth/nd-production/output/MiniProdN5/"
        "run-ndlar-flow/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift/FLOW/0000000")
FP = f"{DATA}/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.{FTAG}.FLOW.hdf5"

import os as _os
if _os.environ.get("WT_FRONTEND_REPO"):
    W.FRONTEND_REPO = _os.environ["WT_FRONTEND_REPO"]
_sub = _os.environ.get("WT_SUBJECT", "").strip()
if _sub:
    _parts = _sub.split(",")
    if len(_parts) != 3:
        raise SystemExit(f"WT_SUBJECT must be 'tag,event,tpc', got {_sub!r}")
    FTAG, EV, TPC = _parts[0], int(_parts[1]), int(_parts[2])
    FP = f"{DATA}/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.{FTAG}.FLOW.hdf5"
ctx = W.load_chain()
S = W.first_stage(ctx, FP, EV, TPC)
rows, n_noise = W.cluster_table(S)
print(f"cluster table: {len(rows)} rows, {n_noise} noise hits")

frames = {"n": 0}


def cb(stage, note, base_view=None, force=False):
    frames["n"] += 1
    print(f"  frame {frames['n']:>3} [{stage}] {note[:120]}")


W.stack2_backbone(S, cb)
assert S.get("merged_placed") is not None
W.rescue_stage(S, cb)
W.staged_stage(S, cb)
W.smallcluster_stage(S, cb)

print("\ntimings:")
for name, sec in S["timings"]:
    print(f"  {sec:7.1f}s  {name}")
summ = W.tpc_summary(S)
print(f"\nwatched TPC {TPC}: E {summ['e_truth']:.0f} MeV | coverage "
      f"{summ['coverage']*100:.1f}% | eff {summ['eff']*100:.2f}%")
print(f"merged pool: {len(S['merged_set'])} P2-class labels, "
      f"{len(S['merged_placed'])} placed")
print("SMOKE_OK")
