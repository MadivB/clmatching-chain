"""Assemble v1.0 .pt output files (vAlpha prompt-hit schema + v1.0 extras)
from the chain's full_dumps + subtick dumps - no re-run needed.

Adds over vAlpha: calib_hit_t0_conf (pcu_vote10 confidence, NaN = no
estimate) and calib_hit_t0_sigma (per-group timing sigma, ticks).
calib_final_* tables are NOT included (prompt-hit arrays only).

Usage: python make_pt_v1.py <dump_out_dir> <pt_out_dir>
"""
import glob
import os
import sys

import h5py
import numpy as np
import torch

ALGO = ("CLMatching v1.0 (tag CLMatching_v1.0): external frontend clustering "
        "+ stack2 backbone + merged permutation pool + joint mean-field + "
        "spatial trading + swap-fix + KD force coverage (25cm isolation veto) "
        "+ subtick continuous t0s; hit_t0_conf = pcu_vote10 (zero-parameter, "
        "data-only); prompt-hit arrays only")


def main(outdir, ptdir):
    os.makedirs(ptdir, exist_ok=True)
    by_file = {}
    for ff in sorted(glob.glob(f"{outdir}/full_dumps/*.npz")):
        fd = np.load(ff, allow_pickle=True)
        src = str(fd["src_file"])
        by_file.setdefault(src, []).append((ff, int(fd["event"])))
    for src, evs in sorted(by_file.items()):
        with h5py.File(src, "r") as h5:
            n = len(h5["charge/calib_prompt_hits/data"])
        t0 = np.full(n, -1.0, np.float32)
        cid = np.full(n, -1, np.int32)
        conf = np.full(n, np.nan, np.float32)
        sig = np.full(n, np.nan, np.float32)
        ev_ids = []
        for ff, ev_id in sorted(evs, key=lambda t: t[1]):
            fd = np.load(ff, allow_pickle=True)
            sd_path = os.path.join(outdir, "dumps", os.path.basename(ff))
            sd = np.load(sd_path) if os.path.exists(sd_path) else None
            refs = fd["hit_refs"].astype(np.int64)
            ok = (refs >= 0) & (refs < n)
            ts = fd["ts_final"].astype(np.float32)
            ts_w = np.where(np.isfinite(ts) & (ts >= 0), ts, -1.0)
            t0[refs[ok]] = ts_w[ok]
            cid[refs[ok]] = fd["labels"].astype(np.int32)[ok]
            if "hit_conf93" in fd.files:
                conf[refs[ok]] = fd["hit_conf93"].astype(np.float32)[ok]
            if sd is not None and "hit_sigt" in sd.files and sd["hit_sigt"].size == refs.size:
                sig[refs[ok]] = sd["hit_sigt"].astype(np.float32)[ok]
            ev_ids.append(ev_id)
        out = {
            "version": "v1.0",
            "algorithm": ALGO,
            "input_file": src,
            "calib_hit_t0_reco": torch.from_numpy(t0),
            "prompt_hit_t_cluster_id": torch.from_numpy(cid),
            "calib_hit_t0_conf": torch.from_numpy(conf),
            "calib_hit_t0_sigma": torch.from_numpy(sig),
            "n_calib_hits": int(n),
            "n_assigned": int((t0 >= 0).sum()),
            "n_unassigned": int((t0 < 0).sum()),
            "calib_hit_t0_unassigned_value": -1.0,
            "processed_event_ids": sorted(ev_ids),
        }
        tag = os.path.basename(src).replace(".hdf5", "")
        path = os.path.join(ptdir, f"{tag}.v1_0.pt")
        torch.save(out, path)
        print(f"{path}: {n} hits, {out['n_assigned']} assigned "
              f"({100*out['n_assigned']/n:.1f}%), {len(ev_ids)} events, "
              f"conf finite {np.isfinite(conf).mean()*100:.0f}%")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
