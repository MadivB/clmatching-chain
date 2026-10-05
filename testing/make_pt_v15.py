"""Assemble v1.5 .pt output files (one per flow file) from the chain's full_dumps - no re-run needed.
Schema = the v1.0 prompt-hit schema (make_pt_v1.py) + v1.2 precision timing:
  calib_hit_t0_reco        float32 (n_calib_prompt_hits,)  reconstructed t0 in matching ticks (16 ns), sub-tick (stage 7 precision), -1 = unassigned
  prompt_hit_t_cluster_id  int32                             chain cluster id (-1 = noise / unclustered)
  calib_hit_t0_conf        float32                           per-hit confidence 0-1 (pcu_vote10 / conf93), NaN = no estimate
  calib_hit_t0_sigma       float32                           per-hit timing sigma in TICKS (precision stage; v1.0-compatible), NaN = none
  calib_hit_t0_sigma_ns    float32                           the same in ns
  calib_hit_t0_reco_ns     float32                           t0 in ns (ticks x 16), -1 = unassigned
Usage: python make_pt_v15.py <run_out_dir> <pt_out_dir>"""
import glob
import os
import sys

import h5py
import numpy as np
import torch

TICK_NS = 16.0
ALGO = ("CLMatching v1.5 (tag CLMatching_v1.5, 2026-10-05): external frontend clustering + stack2 backbone + light-coupled "
        "permutation pool + joint mean-field + matrix + spatial trading + exact-block re-solve (v200) + swap-fix + residual-flash "
        "filling + Michel stages 4.85/4.86 + light-vote force coverage + subtick + per-hit precision timing (3 rounds, pieces); "
        "hit_t0_conf = pcu_vote10; prompt-hit arrays only")


def main(outdir, ptdir):
    os.makedirs(ptdir, exist_ok=True)
    by_file = {}
    for ff in sorted(glob.glob(f"{outdir}/full_dumps/*.npz")):
        fd = np.load(ff, allow_pickle=True)
        by_file.setdefault(str(fd["src_file"]), []).append((ff, int(fd["event"])))
    for src, evs in sorted(by_file.items()):
        with h5py.File(src, "r") as h5:
            n = len(h5["charge/calib_prompt_hits/data"])
        t0 = np.full(n, -1.0, np.float32); cid = np.full(n, -1, np.int32); conf = np.full(n, np.nan, np.float32); sig = np.full(n, np.nan, np.float32)
        ev_ids = []
        for ff, ev_id in sorted(evs, key=lambda t: t[1]):
            fd = np.load(ff, allow_pickle=True); refs = fd["hit_refs"].astype(np.int64); ok = (refs >= 0) & (refs < n)
            ts = fd["ts_final"].astype(np.float32); ts_w = np.where(np.isfinite(ts) & (ts >= 0), ts, -1.0)
            t0[refs[ok]] = ts_w[ok]; cid[refs[ok]] = fd["labels"].astype(np.int32)[ok]
            if "hit_conf93" in fd.files:
                conf[refs[ok]] = fd["hit_conf93"].astype(np.float32)[ok]
            if "hit_sigma_p" in fd.files and fd["hit_sigma_p"].size == refs.size:
                sp = fd["hit_sigma_p"].astype(np.float32)
                sig[refs[ok]] = sp[ok]                                                         # ticks (v1.0-compatible); ns copy below
            ev_ids.append(ev_id)
        out = {"version": "v1.5", "algorithm": ALGO, "input_file": src,
               "calib_hit_t0_reco": torch.from_numpy(t0), "calib_hit_t0_reco_ns": torch.from_numpy(np.where(t0 >= 0, t0 * TICK_NS, -1.0).astype(np.float32)),
               "prompt_hit_t_cluster_id": torch.from_numpy(cid), "calib_hit_t0_conf": torch.from_numpy(conf), "calib_hit_t0_sigma": torch.from_numpy(sig),
               "calib_hit_t0_sigma_ns": torch.from_numpy((sig * TICK_NS).astype(np.float32)),
               "tick_ns": TICK_NS, "n_calib_hits": int(n), "n_assigned": int((t0 >= 0).sum()), "n_unassigned": int((t0 < 0).sum()),
               "calib_hit_t0_unassigned_value": -1.0, "processed_event_ids": sorted(ev_ids)}
        tag = os.path.basename(src).replace(".hdf5", ""); path = os.path.join(ptdir, f"{tag}.v1_5.pt"); torch.save(out, path)
        print(f"{path}: {n} hits, {out['n_assigned']} assigned ({100*out['n_assigned']/n:.1f}%), {len(ev_ids)} events, conf finite {np.isfinite(conf).mean()*100:.0f}%, sigma finite {np.isfinite(sig).mean()*100:.0f}%")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
