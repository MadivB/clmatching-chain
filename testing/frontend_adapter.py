"""External front-end clustering adapter (clmatching-frontend repo).

Keeps the front end a SEPARATE git repo (developed/pulled independently)
while plugging it into the chain as a drop-in replacement for
first_stage_matching.clustering.run_global_track_clustering:

    from frontend_adapter import make_external_clustering
    import M5p1.first_stage_matching.pipeline as fsp
    fsp.run_global_track_clustering = make_external_clustering(repo, params_json)

The repo's own recluster.run_clustering(ev, params) is the single entry
point — whatever chain the repo evolves to (stages, params) is picked up
without touching this adapter, as long as that signature holds.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np


def _file_load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def make_external_clustering(repo_dir: str, params_path: str | None = None,
                             shower_min_tracks: int = 0):
    import torch

    from M5p1.first_stage_matching.clustering import ClusteringResult

    repo_dir = os.path.abspath(repo_dir)
    params_path = params_path or os.path.join(repo_dir, "params_full.json")
    params = json.load(open(params_path))
    rec = _file_load("_ext_frontend_recluster", os.path.join(repo_dir, "recluster.py"))
    tag = f"external:{os.path.basename(repo_dir)}:{os.path.basename(params_path)}"

    def run(*, x, y, z, io_group, config=None, energy=None):
        if energy is None:
            energy = np.zeros(np.asarray(x).shape[0], np.float32)
        ev = {
            "x": torch.from_numpy(np.ascontiguousarray(x, np.float32)),
            "y": torch.from_numpy(np.ascontiguousarray(y, np.float32)),
            "z": torch.from_numpy(np.ascontiguousarray(z, np.float32)),
            "io_group": torch.from_numpy(np.ascontiguousarray(io_group, np.int64)),
            "energy": torch.from_numpy(np.ascontiguousarray(energy, np.float32)),
        }
        labels, split_index, label_info, debug = rec.run_clustering(ev, params)
        if shower_min_tracks > 0 and label_info:
            # the external chain types every backbone object "track"; restore the
            # builtin convention: a vertex group of >= N merged tracks is a shower
            # (placed first in Phase 1, full-channel support downstream)
            n_re = 0
            for lab, li in label_info.items():
                # count TRACK members only: merged_from may contain leftover-DBSCAN
                # blob labels (>= split_index) when vp_include_clusters is on
                n_trk = 1 + sum(1 for m in li.get("merged_from", [])
                                if int(m) < int(split_index))
                if (int(lab) < int(split_index) and li.get("vertex_merged")
                        and li.get("n_hits", 1) > 0
                        and n_trk >= shower_min_tracks):
                    li["type"] = "shower"
                    n_re += 1
            debug = dict(debug or {})
            debug["shower_retype"] = n_re
        labels_global = np.asarray(labels, dtype=np.int32)
        n_noise = int(np.count_nonzero(labels_global < 0))
        n_labeled = int(np.count_nonzero(labels_global >= 0))
        n_labels = int(labels_global[labels_global >= 0].max() + 1) if n_labeled else 0
        track_shower_labels = list(range(int(split_index)))
        cluster_labels = list(range(int(split_index), int(n_labels)))
        backbone_type_counts: dict[str, int] = {}
        for label in track_shower_labels:
            t = str(label_info.get(int(label), {}).get("type", "track"))
            backbone_type_counts[t] = backbone_type_counts.get(t, 0) + 1
        debug = dict(debug or {})
        debug["frontend"] = tag
        return ClusteringResult(
            labels_global=labels_global,
            split_index=int(split_index),
            label_info={int(k): dict(v) for k, v in label_info.items()},
            debug=debug,
            track_shower_labels=track_shower_labels,
            cluster_labels=cluster_labels,
            n_noise=n_noise,
            n_labeled=n_labeled,
            n_labels=n_labels,
            backbone_type_counts=backbone_type_counts,
        )

    run.frontend_tag = tag
    return run
