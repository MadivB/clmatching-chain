"""v0.4.5 — STAGE-2 joint family expansion (geodesic Voronoi).

Motivation (user, 2026-07-10): the v0.3 family expansion runs in the phase-3
slot, long after every backbone/P2/V2 placement is frozen.  By then a small
cluster's light has already been "filled in" by whatever else was scanned onto
it — a later cluster can be mis-placed precisely because the light belonging to
an unabsorbed neighbour looked unexplained.  Absorbing the neighbours EARLY, as
soon as their parent's t0 is decided in the stage-2 permutation placement, makes
the predicted waveform look full at the moment the remaining backbone clusters
are scanned.

What this module does differently from v030_expansion.run_family_expansion:

  * JOINT, not per-family.  One multi-source Dijkstra over the 3 cm hit graph,
    sourced from EVERY decided family's hits at once.  scipy's `min_only=True`
    mode returns, per hit, the source node that reached it first, so each hit is
    owned by its geodesically nearest family.  Expansion fronts therefore MEET
    AND STOP at the geodesic midpoint — no family's boundary passes through
    another's (the user's constraint) — and it costs ONE Dijkstra for all
    families instead of one per family.

  * NO global rescan loop.  v030 re-scores every remaining candidate against
    every family after each single commit (O(assignments x candidates x
    families)); here each candidate is offered to exactly ONE family (its
    Voronoi owner), so a pass is O(candidates).

  * Never absorbs a candidate above LARGE_MEV (30 MeV) — those are Phase 2's
    to place.  Only already-decided (large) clusters pick up smaller ones.

The pass is idempotent-safe: absorbed labels are removed from the candidate
pool, and `absorbed` is returned so the driver can purge them from the Phase-2 /
Phase-3 worklists (otherwise the error matrix re-places them => double light,
the v0.3 integration bug).
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

GAIN = 200.0
CLIP = 60780.0

# ---- tunables ----
LARGE_MEV = 30.0        # never absorb a cluster this big (Phase 2 owns it)
R_LINK = 3.0            # hit-graph edge radius [cm]
D_GEO_MAX = 20.0        # expansion reach through the cloud [cm]
BETA_GEO = 0.05         # score penalty per cm of geodesic distance
TAU_ACCEPT = 0.0        # absorb only if score < TAU
FAM_DT = 2.0            # family time window [ticks]
EVERY_N_COMMITS = 4     # run a joint pass after this many accepted commits


def build_hit_graph(xyz: np.ndarray, r_link: float = R_LINK):
    """Sparse symmetric radius graph over hit positions (edge weight = cm)."""
    tree = cKDTree(xyz)
    pairs = tree.query_pairs(r_link, output_type="ndarray")
    n = xyz.shape[0]
    if pairs.size == 0:
        return coo_matrix((n, n)).tocsr()
    d = np.linalg.norm(xyz[pairs[:, 0]] - xyz[pairs[:, 1]], axis=1)
    g = coo_matrix((np.concatenate([d, d]),
                    (np.concatenate([pairs[:, 0], pairs[:, 1]]),
                     np.concatenate([pairs[:, 1], pairs[:, 0]]))), shape=(n, n))
    return g.tocsr()


class Stage2Expander:
    """Joint geodesic-Voronoi expansion of the decided t0s during stage 2."""

    def __init__(self, *, xyz, e_pos, labels, image_maps, cluster_to_tpcs,
                 actual, base, ti_fn, backbone_labels, n_light, n_ch, n_ticks,
                 params: dict[str, float] | None = None, verbose: bool = False):
        p = {k: v for k, v in globals().items() if k.isupper()}
        if params:
            p.update({k: v for k, v in params.items() if v is not None})
        self.p = p
        self.xyz = np.asarray(xyz, np.float64)
        self.e_pos = np.asarray(e_pos, np.float64)
        self.labels = np.asarray(labels)
        self.image_maps = image_maps
        self.actual = np.asarray(actual, np.float64)
        self.base = base                     # mutated in place (the stage-2 baseA)
        self.ti_fn = ti_fn
        self.n_light, self.n_ch, self.n_ticks = n_light, n_ch, n_ticks
        self.w_all = 1.0 / (GAIN * np.maximum(self.actual, 100.0))
        self.verbose = verbose
        self.stats = {"passes": 0, "absorbed": 0, "e_absorbed": 0.0,
                      "skipped_large": 0, "n_cands0": 0, "dijkstra_calls": 0}

        # --- candidate table: unassigned, imaged, NOT backbone, < LARGE_MEV ---
        backbone = set(int(l) for l in backbone_labels)
        self.cands: dict[int, dict[str, Any]] = {}
        for lab in np.unique(self.labels[self.labels >= 0]):
            lab = int(lab)
            if lab in backbone:
                continue
            tpcs = sorted(int(t) for t in cluster_to_tpcs.get(lab, [])
                          if int(t) < n_light and (lab, int(t)) in image_maps)
            if not tpcs:
                continue
            m = self.labels == lab
            e_c = float(self.e_pos[m].sum())
            if e_c <= 0:
                continue
            if e_c >= p["LARGE_MEV"]:
                self.stats["skipped_large"] += 1
                continue
            sup = {}
            for c in tpcs:
                img = np.asarray(image_maps[(lab, c)])
                ch = np.flatnonzero(img.max(axis=1) > 1.0)
                if ch.size:
                    sup[c] = ch
            if not sup:
                continue
            self.cands[lab] = {"hits": np.flatnonzero(m), "e": e_c,
                               "tpcs": tpcs, "sup": sup}
        self.stats["n_cands0"] = len(self.cands)
        self.graph = build_hit_graph(self.xyz, p["R_LINK"]) if self.cands else None
        self.absorbed: dict[int, float] = {}      # label -> t0

    # ---------------- internals ----------------
    def _families(self, decided: dict[int, float], hit_tpc: np.ndarray):
        """Union-find decided labels over (|dt0|<=FAM_DT and shared TPC)."""
        labs = sorted(decided)
        if not labs:
            return []
        tpcs_of, hits_of = {}, {}
        for l in labs:
            m = self.labels == l
            hits_of[l] = np.flatnonzero(m)
            tpcs_of[l] = set(int(t) for t in np.unique(hit_tpc[m]) if t < self.n_light)
        parent = {l: l for l in labs}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for i, a in enumerate(labs):
            for b in labs[i + 1:]:
                if abs(decided[a] - decided[b]) <= self.p["FAM_DT"] and tpcs_of[a] & tpcs_of[b]:
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[rb] = ra
        groups: dict[int, list[int]] = {}
        for l in labs:
            groups.setdefault(find(l), []).append(l)
        fams = []
        for mem in groups.values():
            hits = np.concatenate([hits_of[l] for l in mem]) if mem else np.array([], int)
            w = np.array([max(self.e_pos[self.labels == l].sum(), 1e-9) for l in mem])
            t0 = float(np.average([decided[l] for l in mem], weights=w))
            fams.append({"id": len(fams), "t0": t0, "hits": hits})
        return fams

    def _dlight(self, lab: int, t0: float) -> float:
        """Poisson-clipped delta-chi2 per MeV of adding lab's image at t0."""
        c = self.cands[lab]
        tot = 0.0
        for tc, ch in c["sup"].items():
            img = np.asarray(self.image_maps[(int(lab), int(tc))], np.float64)[ch]
            sh = self.ti_fn(img, t0)
            B = np.asarray(self.base[tc], np.float64)[ch]
            A = self.actual[tc][ch]
            W = self.w_all[tc][ch]
            tot += float(np.sum(W * ((np.minimum(B + sh, CLIP) - A) ** 2
                                     - (np.minimum(B, CLIP) - A) ** 2)))
        return tot / max(c["e"], 1e-9)

    def _add_light(self, lab: int, t0: float) -> None:
        for tc in self.cands[lab]["tpcs"]:
            img = np.asarray(self.image_maps[(int(lab), int(tc))], np.float64)
            self.base[tc] += self.ti_fn(img, t0).astype(self.base.dtype)

    # ---------------- public ----------------
    def expand(self, decided: dict[int, float], hit_tpc: np.ndarray) -> dict[int, float]:
        """One joint pass. Returns {label: t0} newly absorbed (also mutates base)."""
        if not self.cands or self.graph is None or not decided:
            return {}
        fams = self._families(decided, hit_tpc)
        if not fams:
            return {}

        # --- ONE multi-source Dijkstra: every family's hits are sources ---
        src_hits, src_fam = [], []
        for f in fams:
            if f["hits"].size:
                src_hits.append(f["hits"])
                src_fam.append(np.full(f["hits"].size, f["id"], np.int32))
        if not src_hits:
            return {}
        src_hits = np.concatenate(src_hits)
        src_fam = np.concatenate(src_fam)
        fam_of_hit = np.full(self.xyz.shape[0], -1, np.int32)
        fam_of_hit[src_hits] = src_fam        # source hit -> its family

        dist, _pred, sources = dijkstra(
            self.graph, directed=False, indices=np.unique(src_hits),
            min_only=True, return_predecessors=True, limit=self.p["D_GEO_MAX"])
        self.stats["dijkstra_calls"] += 1
        # `sources[h]` = the SOURCE HIT that claimed hit h  ->  its family = owner.
        # Fronts meet at the geodesic midpoint => boundaries never cross.
        owner = np.where(sources >= 0, fam_of_hit[np.clip(sources, 0, None)], -1)

        newly: dict[int, float] = {}
        for lab in list(self.cands):
            h = self.cands[lab]["hits"]
            d = dist[h]
            if not np.isfinite(d).any():
                continue
            j = int(np.argmin(d))
            dg = float(d[j])
            if not np.isfinite(dg) or dg > self.p["D_GEO_MAX"]:
                continue
            fid = int(owner[h[j]])            # Voronoi owner = the ONLY candidate family
            if fid < 0:
                continue
            t0 = fams[fid]["t0"]
            score = self._dlight(lab, t0) + self.p["BETA_GEO"] * dg
            if score >= self.p["TAU_ACCEPT"]:
                continue
            self._add_light(lab, t0)
            newly[lab] = t0
            self.absorbed[lab] = t0
            self.stats["absorbed"] += 1
            self.stats["e_absorbed"] += self.cands[lab]["e"]
            del self.cands[lab]
        self.stats["passes"] += 1
        if self.verbose and newly:
            print(f"[v045 stage2] pass {self.stats['passes']}: absorbed {len(newly)} "
                  f"({sum(self.e_pos[self.labels == l].sum() for l in newly):.0f} MeV) "
                  f"into {len(fams)} families", flush=True)
        return newly


def make_expansion_hook(inner_hook: Callable | None, expander: Stage2Expander,
                        decided: dict[int, float], hit_tpc: np.ndarray,
                        on_absorb: Callable[[dict], None] | None = None,
                        every: int = EVERY_N_COMMITS):
    """Compose a stage-2 expansion onto place_pool_v02's existing commit_hook.

    place_pool_v02.try_commit adds the label's light, then calls commit_hook;
    a False return REJECTS (light removed, label back to pool).  We therefore
    only expand after the inner hook has ACCEPTED, so a rejected placement never
    leaves absorbed light behind.
    """
    state = {"since": 0}

    def hook(lab, t0, baseA):
        ok = True if inner_hook is None else inner_hook(lab, t0, baseA)
        if ok is False:
            return False
        decided[int(lab)] = float(t0)
        state["since"] += 1
        if state["since"] >= int(every):
            state["since"] = 0
            got = expander.expand(decided, hit_tpc)
            if got and on_absorb is not None:
                on_absorb(got)
        return True

    return hook
