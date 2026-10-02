"""ND truth helpers for Michel-electron and track-ending studies (MC only; never used in the chain).

Per charge event: the dominant trajectory of every calibrated prompt hit (backtrack fraction_traj argmax -> file_traj_id),
the trajectory table of that event (pdg, parent, start process, start / end points and times, KE), and the Michel
electrons: e+- trajectories with start_process 6 (decay) whose parent is a mu+-.  A hit is "Michel energy" when the
Michel trajectory is in its dominant trajectory's ancestry (bremsstrahlung photons and their electrons included).
"""
from __future__ import annotations

import os

import h5py
import numpy as np

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
_CACHE: dict = {}


def open_flow(path):
    if path not in _CACHE:
        _CACHE[path] = h5py.File(path, "r")
    return _CACHE[path]


def event_trajectories(h5, event_id):
    """trajectory rows of one charge event as a dict of arrays; keys are (vertex_id, traj_id) -> row ('idx')."""
    key = ("traj", id(h5), int(event_id))
    if key in _CACHE:
        return _CACHE[key]
    tr = h5["mc_truth/trajectories/data"]
    ev = tr["event_id"][:]
    rows = np.flatnonzero(ev == int(event_id))
    sub = tr[rows]
    vx = sub["vertex_id"].astype(np.int64); tid = sub["traj_id"].astype(np.int64)
    idx = {(int(v), int(t)): i for i, (v, t) in enumerate(zip(vx, tid))}
    parent = np.array([idx.get((int(v), int(p)), -1) for v, p in zip(vx, sub["parent_id"])], np.int64)   # row of the parent (-1 = none)
    out = dict(idx=idx, vertex_id=vx, traj_id=tid, pdg=sub["pdg_id"].astype(np.int64), parent_row=parent, start_process=sub["start_process"].astype(np.int64),
               xyz_start=np.asarray(sub["xyz_start"], float), xyz_end=np.asarray(sub["xyz_end"], float), t_start=np.asarray(sub["t_start"], float),
               t_end=np.asarray(sub["t_end"], float), E_start=np.asarray(sub["E_start"], float), E_end=np.asarray(sub["E_end"], float),
               end_process=sub["end_process"].astype(np.int64), dist_travel=np.asarray(sub["dist_travel"], float), n=rows.size)
    _CACHE[key] = out
    return out


def _segment_lookup(h5):
    key = ("seg", id(h5))
    if key not in _CACHE:
        sg = h5["mc_truth/segments/data"]
        sid = sg["segment_id"][:].astype(np.int64); o = np.argsort(sid)
        _CACHE[key] = dict(sid=sid[o], ev=sg["event_id"][:].astype(np.int64)[o], vx=sg["vertex_id"][:].astype(np.int64)[o],
                           tid=sg["traj_id"][:].astype(np.int64)[o], pdg=sg["pdg_id"][:].astype(np.int64)[o])
    return _CACHE[key]


def hit_dominant_traj(h5, hit_refs, traj):
    """row (in traj) of the dominant trajectory of each hit via its dominant backtrack SEGMENT (-1 without truth)"""
    bt = h5["mc_truth/calib_prompt_hit_backtrack/data"]
    hr = np.asarray(hit_refs, np.int64); order = np.argsort(hr); rows = bt[hr[order]]
    sid = np.asarray(rows["segment_ids"], np.int64); fr = np.asarray(rows["fraction"], float)
    valid = sid >= 0; fr = np.where(valid, fr, -np.inf); best = np.argmax(fr, axis=1)
    dom_sid = np.where(valid.any(1), sid[np.arange(sid.shape[0]), best], -1)
    S = _segment_lookup(h5); pos = np.searchsorted(S["sid"], dom_sid); pos = np.clip(pos, 0, S["sid"].size - 1)
    ok = (dom_sid >= 0) & (S["sid"][pos] == dom_sid)
    out = np.full(hr.size, -1, np.int64)
    for i in np.flatnonzero(ok):
        out[i] = traj["idx"].get((int(S["vx"][pos[i]]), int(S["tid"][pos[i]])), -1)
    res = np.empty_like(out); res[order] = out
    return res


def ancestry_tag(traj, dom_row, targets):
    """for each hit: the first target trajectory ROW found walking up the ancestry of its dominant trajectory (-1 if none)"""
    targets = set(int(t) for t in targets); cache = {}
    out = np.full(dom_row.size, -1, np.int64)
    for i, r in enumerate(dom_row):
        r = int(r)
        if r < 0:
            continue
        if r in cache:
            out[i] = cache[r]; continue
        cur = r; hit = -1; steps = 0; path = []
        while cur >= 0 and steps < 60:
            if cur in targets:
                hit = cur; break
            path.append(cur); cur = int(traj["parent_row"][cur]); steps += 1
        for p_ in path:
            cache[p_] = hit
        cache[r] = hit; out[i] = hit
    return out


def michel_trajectories(traj):
    """(michel row, muon row) pairs: e+- with start_process 6 (decay) whose parent is a mu+-"""
    pdg = traj["pdg"]; par = traj["parent_row"]; sp = traj["start_process"]
    out = []
    for i in np.flatnonzero((np.abs(pdg) == 11) & (sp == 6)):
        j = int(par[i])
        if j >= 0 and abs(int(pdg[j])) == 13:
            out.append((int(i), j))
    return out
