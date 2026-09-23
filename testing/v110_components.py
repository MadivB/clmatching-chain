"""v1.1 pool: light-coupled COMPONENT batching + exact COMBINATION search,
with optional per-cluster amplitude profiling at commit.

Replaces the v0.2-v1.0 energy-banded ordering search (place_pool_v02) when
the harness runs with --pool-mode light.  Same contract: mutates baseA,
returns (tmap, stats); honours `pick` (cosine-softened argmin) for the
single-cluster case and `commit_hook` (defer gate) for every commit.

Physics of the design
---------------------
* The Poisson chi2 is additive over channels, so two placed clusters interact
  through the loss ONLY where their predicted light overlaps on the same
  channels.  Clusters with dissimilar channel patterns (tracks far apart in y,
  different modules) are independent placements; clusters with similar
  patterns are mutually confusable and must be decided TOGETHER.  Batches are
  therefore connected components of the light-coupling graph - an edge when
  the cosine of the two amplitude patterns on a shared light-TPC is >= COS_MIN
  - instead of energy bands.
* Within a component the joint loss over placement COMBINATIONS is exactly
  pairwise-quadratic (v0.7): L = const + sum_i a_i(t_i) + sum_{i<j}
  C_ij(t_i, t_j), so ALL k^n combinations of the members' k candidate times
  are scored by table sums, and the best few are re-scored with the exact
  clipped union loss before the winner is committed.  The energy-order greedy
  path (what v1.0 does first) is always one of the candidates, so the joint
  minimum is never worse than v1.0's single path.  Candidate times are the
  local minima of each member's own isolated scan curve.
* Amplitude profiling (GAIN band b): the light model's per-cluster amplitude
  is uncertain at the 10-20 % level (v1.0 dumps: correctly placed families are
  under-predicted by 2-10 % median, p84 ~1.2).  A committed cluster's
  amplitude is profiled at its chosen time within [1-b, 1+b] (closed form, on
  its own support channels, against the residual without itself) so that its
  own template error does not remain in the residual to attract look-alike
  small clusters - the measured piggyback mechanism behind ~1/3 of the wrong
  energy.  The template is rescaled IN PLACE in image_maps so every later
  stage (rescue, pass2, subtick sum-mode, confidence) stays consistent.
"""
from __future__ import annotations

import itertools
import time

import numpy as np

import v02_ordering_pipeline as _vop
from v02_ordering_pipeline import (CLIP, SEARCH, add_label, scan_label,
                                   weighted_scan, _edge_penalty_vec)

# ---- tunables (few, physical) ----
COS_MIN = 0.80      # light-coupling edge: cosine of amplitude patterns on a shared TPC
K_CAND = 6          # candidate times per member (isolated-curve local minima)
N_MAX = 8           # exact enumeration / chunk size (k^n <= 1.7M at k=6)
SEP = 10            # ticks separating candidate minima (= the success window)
N_EXACT = 8         # top table combinations re-scored with the exact clipped loss
GAIN_BAND = 0.0     # 0 = off; else profile each committed amplitude in [1-b, 1+b]
SPATIAL_FRAC = 0.0  # 0 = off; else charge-contact prior: connected members get a
                    # same-time bonus of SPATIAL_FRAC x (median unary spread of the
                    # chunk) - the v0.7 energy-unit convention (3-cm contact ~ same
                    # interaction with P~0.99, measured 2026-08-06)
R_CONTACT = 3.0     # cm: hit-cloud contact distance (the chain's hit-graph link radius)
SAME_T = 2.0        # ticks: "same time" for the contact bonus


class PlacedContacts:
    """Contact of a cluster's hit cloud with ALREADY-PLACED clusters: returns
    the set of their times.  One KD-tree over all event hits; a member's
    (subsampled) hits are queried within r cm, neighbours are mapped to their
    label and to that label's current time (placed_t dict, maintained by the
    caller).  This is the v0.7 'connectivity to placed mass' certificate."""

    def __init__(self, xyz, labels, r=R_CONTACT, max_pts=3000):
        from scipy.spatial import cKDTree
        self.xyz = np.asarray(xyz, np.float64)
        self.labels = np.asarray(labels)
        self.tree = cKDTree(self.xyz)
        self.r = float(r)
        self.max_pts = int(max_pts)
        self._nbr_labels = {}

    def neighbour_labels(self, lab, hits):
        lab = int(lab)
        if lab not in self._nbr_labels:
            h = np.asarray(hits, dtype=int)
            if h.size > self.max_pts:
                h = h[np.linspace(0, h.size - 1, self.max_pts).astype(int)]
            nb = self.tree.query_ball_point(self.xyz[h], self.r)
            idx = np.unique(np.concatenate([np.asarray(v, int) for v in nb])) if len(nb) else np.array([], int)
            ls = np.unique(self.labels[idx])
            self._nbr_labels[lab] = set(int(v) for v in ls if v >= 0 and int(v) != lab)
        return self._nbr_labels[lab]

    def contact_times(self, lab, hits, placed_t, exclude=()):
        out = set()
        for l in self.neighbour_labels(lab, hits):
            if l in exclude:
                continue
            t = placed_t.get(l)
            if t is not None and np.isfinite(t) and t >= 0:
                out.add(float(t))
        return out


def contact_pairs(B, hits_of, xyz, r=R_CONTACT, max_pts=3000):
    """Set of (i, j) index pairs of B whose hit clouds come within r cm."""
    from scipy.spatial import cKDTree
    pts, trees = [], []
    for l in B:
        h = np.asarray(hits_of(l), dtype=int)
        if h.size > max_pts:
            h = h[np.linspace(0, h.size - 1, max_pts).astype(int)]
        P = xyz[h]
        pts.append(P)
        trees.append(cKDTree(P) if P.shape[0] else None)
    out = set()
    for i in range(len(B)):
        for j in range(i + 1, len(B)):
            if trees[i] is None or trees[j] is None:
                continue
            # cheap bounding-box reject first
            lo = np.maximum(pts[i].min(0), pts[j].min(0)) - r
            hi = np.minimum(pts[i].max(0), pts[j].max(0)) + r
            if np.any(lo > hi):
                continue
            d, _ = trees[i].query(pts[j], k=1, distance_upper_bound=r)
            if np.isfinite(d).any():
                out.add((i, j))
    return out


# --------------------------------------------------------------------------
# light-coupling graph
# --------------------------------------------------------------------------
def amp_pattern(lab, c, image_maps, veto_mask):
    a = np.asarray(image_maps[(int(lab), int(c))]).max(axis=1).astype(np.float64)
    a[np.asarray(veto_mask[int(c)], bool)] = 0.0
    return a


def build_components(pool, info, image_maps, veto_mask, cos_min=COS_MIN):
    """Connected components of the light-coupling graph over `pool`.

    Returns (components sorted by their most energetic member, descending;
    each component energy-descending), number of edges."""
    parent = {int(l): int(l) for l in pool}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    by_tpc = {}
    for l in pool:
        for c in info[l]["tpcs"]:
            by_tpc.setdefault(int(c), []).append(int(l))
    n_edges = 0
    for c, ll in by_tpc.items():
        if len(ll) < 2:
            continue
        P = np.stack([amp_pattern(l, c, image_maps, veto_mask) for l in ll])
        nrm = np.linalg.norm(P, axis=1)
        ok = nrm > 0
        Pn = np.where(ok[:, None], P / np.maximum(nrm[:, None], 1e-12), 0.0)
        C = Pn @ Pn.T
        iu, ju = np.triu_indices(len(ll), 1)
        for i, j in zip(iu, ju):
            if C[i, j] >= cos_min:
                union(ll[i], ll[j])
                n_edges += 1
    comps = {}
    for l in pool:
        comps.setdefault(find(int(l)), []).append(int(l))
    out = [sorted(v, key=lambda l: -info[l]["energy"]) for v in comps.values()]
    out.sort(key=lambda v: -info[v[0]]["energy"])
    return out, n_edges


# --------------------------------------------------------------------------
# per-member isolated curve + candidate times
# --------------------------------------------------------------------------
def member_arrays(lab, baseA, info, image_maps, actual, w_full):
    prs, bss, acs, wrs = [], [], [], []
    for c, ch in info[lab]["sup"].items():
        prs.append(np.asarray(image_maps[(int(lab), int(c))], np.float64)[ch])
        bss.append(baseA[c][ch])
        acs.append(actual[c][ch])
        wrs.append(w_full[c][ch])
    return (np.concatenate(prs), np.concatenate(bss), np.concatenate(acs),
            np.concatenate(wrs))


def _local_minima(curve):
    n = curve.size
    idx = np.flatnonzero((curve[1:-1] <= curve[:-2]) & (curve[1:-1] <= curve[2:])) + 1
    ends = []
    if n > 1 and curve[0] <= curve[1]:
        ends.append(0)
    if n > 1 and curve[-1] <= curve[-2]:
        ends.append(n - 1)
    return np.concatenate([idx, np.asarray(ends, int)]) if ends else idx


def candidate_times(curve, P, B, A, W, pick, k=K_CAND, sep=SEP):
    """Soft-pick argmin (v1.0 single-cluster behaviour) + the best-(k-1)
    separated local minima of the plain curve."""
    t0 = float(np.argmin(curve)) if pick is None else float(pick(curve, P, B, A, W))
    cands = [int(round(t0))]
    lm = _local_minima(curve)
    for t in lm[np.argsort(curve[lm])]:
        t = int(t)
        if all(abs(t - c) > sep for c in cands):
            cands.append(t)
            if len(cands) >= k:
                break
    return cands


# --------------------------------------------------------------------------
# amplitude profiling
# --------------------------------------------------------------------------
def profile_gain(lab, t0, baseA, info, image_maps, actual, w_full, ti, band):
    """Fit g in [1-band, 1+band] per (cluster, TPC) on the cluster's own support
    channels against the residual WITHOUT itself; apply (g-1)*template to the
    base and rescale the template in place.  Returns {tpc: g} (for undo)."""
    gains = {}
    if band <= 0:
        return gains
    for c in info[lab]["tpcs"]:
        ch = info[lab]["sup"].get(int(c))
        if ch is None or len(ch) == 0:
            continue
        arr = image_maps[(int(lab), int(c))]
        img = np.asarray(arr, np.float64)
        sh = ti(img, t0)
        s = sh[ch]
        R = actual[c][ch] - (baseA[c][ch] - s)
        w = w_full[c][ch]
        den = float(np.sum(w * s * s))
        if den <= 0.0:
            continue
        g = float(np.clip(float(np.sum(w * s * R)) / den, 1.0 - band, 1.0 + band))
        if abs(g - 1.0) < 1e-6:
            continue
        baseA[c] += (g - 1.0) * sh
        if isinstance(arr, np.ndarray) and arr.flags.writeable:
            arr *= g
        else:
            image_maps[(int(lab), int(c))] = (np.asarray(arr) * g).astype(np.float32)
        gains[int(c)] = g
    return gains


def undo_gain(lab, t0, baseA, image_maps, ti, gains):
    for c, g in gains.items():
        arr = image_maps[(int(lab), int(c))]
        img = np.asarray(arr, np.float64)          # currently g * original
        baseA[c] -= (1.0 - 1.0 / g) * ti(img, t0)  # remove (g-1)*original
        if isinstance(arr, np.ndarray) and arr.flags.writeable:
            arr /= g
        else:
            image_maps[(int(lab), int(c))] = (np.asarray(arr) / g).astype(np.float32)


# --------------------------------------------------------------------------
# combination search over one chunk
# --------------------------------------------------------------------------
def _enumerate(a_list, C_pairs, const, use_torch=False):
    """Total table over all combinations. a_list[i]: (k_i,), C_pairs[(i,j)]:
    (k_i, k_j).  Returns flat float64 array in C order over (k_0..k_{n-1})."""
    n = len(a_list)
    shape = tuple(int(a.size) for a in a_list)
    if use_torch:
        import torch
        dev = "cuda"
        tot = torch.full(shape, float(const), dtype=torch.float64, device=dev)
        for i, a in enumerate(a_list):
            sh = [1] * n
            sh[i] = a.size
            tot += torch.from_numpy(a).to(dev).reshape(sh)
        for (i, j), C in C_pairs.items():
            sh = [1] * n
            sh[i], sh[j] = C.shape
            tot += torch.from_numpy(C).to(dev).reshape(sh)
        return tot.reshape(-1).cpu().numpy()
    tot = np.full(shape, float(const), dtype=np.float64)
    for i, a in enumerate(a_list):
        sh = [1] * n
        sh[i] = a.size
        tot += a.reshape(sh)
    for (i, j), C in C_pairs.items():
        sh = [1] * n
        sh[i], sh[j] = C.shape
        tot += C.reshape(sh)
    return tot.reshape(-1)


def search_chunk(B, baseA, info, image_maps, actual, w_full, ti, pick,
                 k=K_CAND, n_exact=N_EXACT, stats=None, extra_cands=None,
                 incumbent=None, spatial_frac=0.0, hits_of=None, xyz=None,
                 placed_contacts=None, placed_t=None):
    """Joint placement of the members of B (energy-descending) on baseA.

    extra_cands: {lab: [t, ...]} forced into the candidate sets (e.g. the
    current placement); incumbent: {lab: [t]} -> its exact loss is reported
    as diag["loss_incumbent"] (re-decision mode).
    Returns (ts: {lab: t}, diag dict).  Does NOT mutate baseA."""
    t_start = time.time()
    n = len(B)
    # union support channels per TPC
    uch = {}
    for l in B:
        for c, ch in info[l]["sup"].items():
            uch[c] = np.union1d(uch.get(c, np.array([], dtype=int)), ch)
    act_u = {c: actual[c][ch] for c, ch in uch.items()}
    w_u = {c: w_full[c][ch] for c, ch in uch.items()}
    sb0 = {c: baseA[c][ch].copy() for c, ch in uch.items()}
    R_u = {c: act_u[c] - sb0[c] for c in uch}
    P_u = {}
    for l in B:
        for c in info[l]["tpcs"]:
            if c in uch:
                P_u[(l, c)] = np.asarray(image_maps[(int(l), int(c))], np.float64)[uch[c]]

    # 1) isolated curves -> candidates (+ edge penalty per candidate)
    cands, pen_at, n_scans = [], [], 0
    for l in B:
        P, Bm, A, W = member_arrays(l, baseA, info, image_maps, actual, w_full)
        curve = weighted_scan(P, Bm - A, W)
        n_scans += 1
        ct = candidate_times(curve, P, Bm, A, W, pick, k=k)
        for t in (extra_cands or {}).get(l, []):
            t = int(round(t))
            if 0 <= t <= SEARCH and t not in ct:
                ct.append(t)
        cands.append(ct)
        if _vop.EDGE_PENALTY:
            pen = _edge_penalty_vec(P, W, SEARCH)
            pen_at.append({t: float(pen[t]) for t in ct})
        else:
            pen_at.append({t: 0.0 for t in ct})

    # 2) energy-order greedy path (v1.0's single path) -> guaranteed candidate
    sb = {c: sb0[c].copy() for c in uch}
    greedy = {}
    for i, l in enumerate(B):
        if i == 0:
            t = cands[0][0]
        else:
            prs, bss, acs, wrs = [], [], [], []
            for c, ch in info[l]["sup"].items():
                ii = np.searchsorted(uch[c], ch)
                prs.append(P_u[(l, c)][ii])
                bss.append(sb[c][ii])
                acs.append(act_u[c][ii])
                wrs.append(w_u[c][ii])
            Pn, Bn, An, Wn = map(np.concatenate, (prs, bss, acs, wrs))
            curve = weighted_scan(Pn, Bn - An, Wn)
            n_scans += 1
            t = int(round(float(np.argmin(curve)) if pick is None
                          else float(pick(curve, Pn, Bn, An, Wn))))
            if t not in cands[i]:
                cands[i].append(t)
                if _vop.EDGE_PENALTY:
                    pen_at[i][t] = float(_edge_penalty_vec(Pn, Wn, SEARCH)[t])
                else:
                    pen_at[i][t] = 0.0
        greedy[l] = t
        for c in info[l]["tpcs"]:
            if (l, c) in P_u:
                sb[c] += ti(P_u[(l, c)], t)

    # 3) shifted templates at candidates (on U), unary + pairwise tables
    S = {}          # (i, ci) -> {c: shifted rows}
    for i, l in enumerate(B):
        for ci, t in enumerate(cands[i]):
            S[(i, ci)] = {c: ti(P_u[(l, c)], t) for c in info[l]["tpcs"] if (l, c) in P_u}
    a_list = []
    for i, l in enumerate(B):
        a = np.zeros(len(cands[i]))
        for ci, t in enumerate(cands[i]):
            v = pen_at[i][t]
            for c, s in S[(i, ci)].items():
                v += float(np.sum(w_u[c] * (s * s - 2.0 * s * R_u[c])))
            a[ci] = v
        a_list.append(a)
    C_pairs = {}
    for i in range(n):
        for j in range(i + 1, n):
            shared = [c for c in info[B[i]]["tpcs"] if c in uch and c in info[B[j]]["tpcs"]
                      and (B[i], c) in P_u and (B[j], c) in P_u]
            if not shared:
                continue
            C = np.zeros((len(cands[i]), len(cands[j])))
            for c in shared:
                Si = np.stack([S[(i, ci)][c].ravel() for ci in range(len(cands[i]))])
                Sj = np.stack([(w_u[c] * S[(j, cj)][c]).ravel() for cj in range(len(cands[j]))])
                C += 2.0 * (Si @ Sj.T)
            C_pairs[(i, j)] = C
    const = float(sum(np.sum(w_u[c] * R_u[c] * R_u[c]) for c in uch))
    # optional charge-contact prior (energy units): connected members at the
    # same time get -prior_e; enters the tables AND the exact re-score
    contacts, prior_e, n_unary_contact = set(), 0.0, 0
    if spatial_frac > 0 and hits_of is not None and xyz is not None:
        contacts = contact_pairs(B, hits_of, xyz)
        ctimes = []
        if placed_contacts is not None and placed_t is not None:
            excl = set(int(l) for l in B)
            ctimes = [placed_contacts.contact_times(l, hits_of(l), placed_t, exclude=excl) for l in B]
        if contacts or any(ctimes):
            prior_e = float(spatial_frac) * float(np.median([max(float(np.ptp(a)), 1.0) for a in a_list]))
            # unary: member at a time where a PLACED contact cluster sits
            for i, l in enumerate(B):
                if not ctimes or not ctimes[i]:
                    continue
                ct = np.asarray(sorted(ctimes[i]), float)
                for ci, t in enumerate(cands[i]):
                    if np.min(np.abs(ct - t)) <= SAME_T:
                        a_list[i][ci] -= prior_e
                        pen_at[i][t] -= prior_e
                        n_unary_contact += 1
            for (i, j) in contacts:
                ti_ = np.asarray(cands[i], float)[:, None]
                tj_ = np.asarray(cands[j], float)[None, :]
                bonus = -prior_e * (np.abs(ti_ - tj_) <= SAME_T)
                C_pairs[(i, j)] = C_pairs.get((i, j), np.zeros((len(cands[i]), len(cands[j])))) + bonus
    n_comb = int(np.prod([len(cc) for cc in cands]))
    use_torch = False
    if n_comb > 400000:
        try:
            import torch
            use_torch = torch.cuda.is_available()
        except Exception:
            use_torch = False
    tot = _enumerate(a_list, C_pairs, const, use_torch=use_torch)

    # 4) exact clipped re-score of the top table combinations (+ greedy)
    shape = tuple(len(cc) for cc in cands)
    order = np.argsort(tot)[: min(n_exact, tot.size)]
    combos = [np.unravel_index(int(f), shape) for f in order]
    g_idx = tuple(cands[i].index(greedy[B[i]]) for i in range(n))
    if g_idx not in combos:
        combos.append(g_idx)
    inc_idx = None
    if incumbent is not None:
        inc_idx = tuple(cands[i].index(int(round(incumbent[B[i]][0]))) for i in range(n))
        if inc_idx not in combos:
            combos.append(inc_idx)

    def exact_loss(idx):
        sbx = {c: sb0[c].copy() for c in uch}
        pen = 0.0
        for i, ci in enumerate(idx):
            for c, s in S[(i, ci)].items():
                sbx[c] += s
            pen += pen_at[i][cands[i][ci]]
        for (i, j) in contacts:
            if abs(cands[i][idx[i]] - cands[j][idx[j]]) <= SAME_T:
                pen -= prior_e
        return pen + float(sum(np.sum(w_u[c] * (np.minimum(sbx[c], CLIP) - act_u[c]) ** 2)
                               for c in uch))

    exact = [exact_loss(idx) for idx in combos]
    best = int(np.argmin(exact))
    win = combos[best]
    ts = {B[i]: float(cands[i][win[i]]) for i in range(n)}
    g_loss = exact[combos.index(g_idx)]
    diag = {"n_scans": n_scans, "n_comb": n_comb, "n_exact": len(combos),
            "torch": use_torch, "loss_best": float(exact[best]), "loss_greedy": float(g_loss),
            "loss_incumbent": (float(exact[combos.index(inc_idx)]) if inc_idx is not None else None),
            "differs": any(abs(ts[l] - greedy[l]) > 0.5 for l in B),
            "table_rank_of_winner": int(best), "time_s": time.time() - t_start,
            "n_contacts": len(contacts), "n_unary_contact": n_unary_contact, "prior_e": prior_e,
            "greedy": {l: float(greedy[l]) for l in B}, "n_cand": [len(cc) for cc in cands]}
    if stats is not None:
        stats["n_scans"] += n_scans
        stats["n_leaves"] += n_comb
        stats["n_exact_checks"] += len(combos)
        stats["perm_total"] += 1
        if diag["differs"] and exact[best] < g_loss - 1e-9:
            stats["perm_better"] += 1
            stats["perm_rel_gain_sum"] += (g_loss - exact[best]) / max(abs(g_loss), 1e-12)
        stats["t_search"] += diag["time_s"]
    return ts, diag


# --------------------------------------------------------------------------
# ordering search over a chunk (v1.0's decision rule on a light component)
# --------------------------------------------------------------------------
def search_orderings(B, baseA, info, image_maps, actual, w_full, ti, pick, stats=None):
    """All orderings of B; each ordering places greedily at the (soft-picked)
    scan optimum given base-so-far; the ordering with the smallest clipped
    union-support loss wins.  Identical decision rule to place_pool_v02's
    batches, applied to a light-coupled component.  Returns (ts, diag)."""
    t_start = time.time()
    uch = {}
    for l in B:
        for c, ch in info[l]["sup"].items():
            uch[c] = np.union1d(uch.get(c, np.array([], dtype=int)), ch)
    act_u = {c: actual[c][ch] for c, ch in uch.items()}
    w_u = {c: w_full[c][ch] for c, ch in uch.items()}
    sb0 = {c: baseA[c][ch].copy() for c, ch in uch.items()}
    P_u, idx = {}, {}
    for l in B:
        for c in info[l]["tpcs"]:
            if c in uch:
                P_u[(l, c)] = np.asarray(image_maps[(int(l), int(c))], np.float64)[uch[c]]
        for c, ch in info[l]["sup"].items():
            idx[(l, c)] = np.searchsorted(uch[c], ch)
    energy_order = tuple(B)
    best, eloss, counters = [None], [None], [0, 0]

    def uloss(sb):
        return float(sum(np.sum(w_u[c] * (np.minimum(sb[c], CLIP) - act_u[c]) ** 2) for c in uch))

    def dfs(remaining, sb, order, ts):
        if not remaining:
            lv = uloss(sb)
            counters[0] += 1
            if best[0] is None or lv < best[0][0]:
                best[0] = (lv, list(order), dict(ts))
            if tuple(order) == energy_order:
                eloss[0] = lv
            return
        for nxt in remaining:
            prs, bss, acs, wrs = [], [], [], []
            for c in info[nxt]["sup"]:
                ii = idx[(nxt, c)]
                prs.append(P_u[(nxt, c)][ii]); bss.append(sb[c][ii])
                acs.append(act_u[c][ii]); wrs.append(w_u[c][ii])
            Pn, Bn, An, Wn = map(np.concatenate, (prs, bss, acs, wrs))
            curve = weighted_scan(Pn, Bn - An, Wn)
            t0n = float(np.argmin(curve)) if pick is None else pick(curve, Pn, Bn, An, Wn)
            counters[1] += 1
            sb2 = {c: a.copy() for c, a in sb.items()}
            for c in info[nxt]["tpcs"]:
                if (nxt, c) in P_u:
                    sb2[c] += ti(P_u[(nxt, c)], t0n)
            order.append(nxt); ts[nxt] = t0n
            dfs([r for r in remaining if r != nxt], sb2, order, ts)
            order.pop(); del ts[nxt]

    dfs(list(B), sb0, [], {})
    loss_b, order_b, ts_b = best[0]
    diag = {"n_scans": counters[1], "n_leaves": counters[0], "loss_best": float(loss_b),
            "loss_greedy": float(eloss[0]) if eloss[0] is not None else None,
            "order": [int(l) for l in order_b],
            "differs": tuple(order_b) != energy_order, "time_s": time.time() - t_start}
    if stats is not None:
        stats["n_scans"] += counters[1]
        stats["n_leaves"] += counters[0]
        stats["perm_total"] += 1
        if eloss[0] is not None and loss_b < eloss[0] - 1e-9:
            stats["perm_better"] += 1
            stats["perm_rel_gain_sum"] += (eloss[0] - loss_b) / max(eloss[0], 1e-12)
        stats["t_search"] += diag["time_s"]
    return {l: float(ts_b[l]) for l in B}, diag


# --------------------------------------------------------------------------
# the pool
# --------------------------------------------------------------------------
def place_pool_v110(pool, baseA, info, image_maps, actual, w_full, ti, veto_mask,
                    pick=None, commit_hook=None, cos_min=COS_MIN, k=K_CAND,
                    n_max=N_MAX, gain_band=GAIN_BAND, n_exact=N_EXACT,
                    gain_log=None, spatial_frac=SPATIAL_FRAC, hits_of=None, xyz=None,
                    labels=None, preplaced=None, truth_fn=None, audit_cap=400,
                    search="combination", joint_min=1, commit_min=0.0):
    """Light-coupled component placement.  Mutates baseA (and, with
    gain_band > 0, rescales committed templates in image_maps in place).
    Returns (tmap, stats) with the place_pool_v02 stats keys plus v1.1 ones."""
    assert _vop.LOSS_MODE == "chi2", "v1.1 pool is chi2-only"
    t_start = time.time()
    pool = [int(l) for l in pool]
    stats = {"n_batches": 0, "n_multi_batches": 0, "n_leaves": 0, "n_scans": 0,
             "batch_sizes": [], "perm_total": 0, "perm_better": 0,
             "perm_rel_gain_sum": 0.0, "n_exact_checks": 0, "t_search": 0.0,
             "n_components": 0, "n_edges": 0, "comp_size_hist": {},
             "n_chunked": 0, "n_rejected": 0, "n_gained": 0, "gains": [],
             "chunk_audit": [], "n_uncommitted": 0, "e_uncommitted": 0.0}
    committed, deferred_q = {}, []
    gains_of = {}
    placed_t = dict(preplaced or {})          # label -> time, grows with commits
    pc = None
    if spatial_frac > 0 and xyz is not None and labels is not None and hits_of is not None:
        pc = PlacedContacts(xyz, labels)

    def try_commit(lab, t0):
        add_label(baseA, lab, t0, info, image_maps, ti)
        g = profile_gain(lab, t0, baseA, info, image_maps, actual, w_full, ti, gain_band)
        if commit_hook is None or commit_hook(lab, t0, baseA):
            committed[lab] = t0
            placed_t[int(lab)] = float(t0)
            if g:
                gains_of[lab] = g
                stats["n_gained"] += 1
                stats["gains"].extend(float(v) for v in g.values())
                if gain_log is not None:
                    gain_log.append({"label": int(lab), "t0": float(t0),
                                     "e": float(info[lab]["energy"]),
                                     "gains": {int(c): float(v) for c, v in g.items()}})
            return True
        undo_gain(lab, t0, baseA, image_maps, ti, g)
        add_label(baseA, lab, t0, info, image_maps, ti, sign=-1.0)
        stats["n_rejected"] += 1
        return False

    comps, n_edges = build_components(pool, info, image_maps, veto_mask, cos_min)
    stats["n_components"] = len(comps)
    stats["n_edges"] = int(n_edges)
    for comp in comps:
        key = str(min(len(comp), 99))
        stats["comp_size_hist"][key] = stats["comp_size_hist"].get(key, 0) + 1

    for comp in comps:
        remaining = list(comp)          # energy-descending
        if len(comp) > n_max:
            stats["n_chunked"] += 1
        while remaining:
            B = remaining[:n_max]
            remaining = remaining[n_max:]
            stats["n_batches"] += 1
            stats["batch_sizes"].append(len(B))
            if len(B) == 1:
                lab = B[0]
                if info[lab]["energy"] < commit_min:
                    stats["n_uncommitted"] += 1
                    stats["e_uncommitted"] += float(info[lab]["energy"])
                    continue
                if pc is not None and pc.contact_times(lab, hits_of(lab), placed_t, exclude={int(lab)}):
                    ts1, _d1 = search_chunk([lab], baseA, info, image_maps, actual, w_full, ti,
                                            pick, k=k, n_exact=n_exact, stats=stats,
                                            spatial_frac=spatial_frac, hits_of=hits_of, xyz=xyz,
                                            placed_contacts=pc, placed_t=placed_t)
                    t = ts1[lab]
                    stats["n_single_contact"] = stats.get("n_single_contact", 0) + 1
                else:
                    t = scan_label(lab, baseA, info, image_maps, actual, w_full, pick=pick)
                    stats["n_scans"] += 1
                if not try_commit(lab, t):
                    deferred_q.append(lab)
                continue
            stats["n_multi_batches"] += 1
            todo = list(B)
            while todo:
                if len(todo) == 1:
                    lab = todo[0]
                    t = scan_label(lab, baseA, info, image_maps, actual, w_full, pick=pick)
                    stats["n_scans"] += 1
                    if not try_commit(lab, t):
                        deferred_q.append(lab)
                    todo = []
                    break
                if search == "ordering" or len(todo) < int(joint_min):
                    # v1.0 decision rule (all orderings, soft-pick greedy scans) for
                    # components too small to carry an exclusion structure
                    ts, _diag = search_orderings(todo[:5], baseA, info, image_maps, actual,
                                                 w_full, ti, pick, stats=stats)
                    _diag["greedy"] = {}
                    if len(todo) > 5:
                        todo_rest = todo[5:]
                        todo = todo[:5]
                    else:
                        todo_rest = []
                else:
                    ts, _diag = search_chunk(todo, baseA, info, image_maps, actual, w_full, ti,
                                             pick, k=k, n_exact=n_exact, stats=stats,
                                             spatial_frac=spatial_frac, hits_of=hits_of, xyz=xyz,
                                             placed_contacts=pc, placed_t=placed_t)
                stats["n_contacts"] = stats.get("n_contacts", 0) + int(_diag.get("n_contacts", 0))
                stats["n_unary_contact"] = stats.get("n_unary_contact", 0) + int(_diag.get("n_unary_contact", 0))
                if _diag.get("differs") and _diag.get("greedy") and len(stats["chunk_audit"]) < audit_cap:
                    _g = _diag["greedy"]
                    rec = {"labels": [int(l) for l in todo],
                           "e": [float(info[l]["energy"]) for l in todo],
                           "greedy_t": [float(_g[l]) for l in todo],
                           "chosen_t": [float(ts[l]) for l in todo],
                           "loss_greedy": float(_diag["loss_greedy"]),
                           "loss_best": float(_diag["loss_best"]),
                           "rel_gain": float((_diag["loss_greedy"] - _diag["loss_best"])
                                             / max(abs(_diag["loss_greedy"]), 1e-12)),
                           "n_contacts": int(_diag.get("n_contacts", 0)),
                           "n_unary_contact": int(_diag.get("n_unary_contact", 0))}
                    if truth_fn is not None:
                        tr = [truth_fn(int(l)) for l in todo]
                        rec["truth"] = [None if t is None or not np.isfinite(t) else float(t) for t in tr]
                        rec["greedy_ok"] = [None if t is None or not np.isfinite(t) else bool(abs(_g[l] - t) <= 10.0)
                                            for l, t in zip(todo, tr)]
                        rec["chosen_ok"] = [None if t is None or not np.isfinite(t) else bool(abs(ts[l] - t) <= 10.0)
                                            for l, t in zip(todo, tr)]
                    stats["chunk_audit"].append(rec)
                rejected = None
                for lab in list(todo):
                    if info[lab]["energy"] < commit_min:
                        # light-weak member: its light hypothesis shaped the joint
                        # decision, but its own placement is left to the tail solver
                        todo.remove(lab)
                        stats["n_uncommitted"] += 1
                        stats["e_uncommitted"] += float(info[lab]["energy"])
                        continue
                    if try_commit(lab, ts[lab]):
                        todo.remove(lab)
                    else:
                        rejected = lab
                        break
                if rejected is not None:
                    deferred_q.append(rejected)
                    todo.remove(rejected)
                if "todo_rest" in locals() and todo_rest:
                    todo = todo + todo_rest
                    todo_rest = []
                    # the others are re-decided on the updated base (their
                    # chosen times assumed the rejected light was present)

    for lab in deferred_q:
        t = scan_label(lab, baseA, info, image_maps, actual, w_full, pick=pick)
        stats["n_scans"] += 1
        try_commit(lab, t)

    g = np.asarray(stats.pop("gains"), float)
    if g.size:
        stats["gain_med"] = float(np.median(g))
        stats["gain_p16"] = float(np.percentile(g, 16))
        stats["gain_p84"] = float(np.percentile(g, 84))
        stats["gain_frac_at_edge"] = float(np.mean(np.abs(np.abs(g - 1.0) - gain_band) < 1e-6))
    stats["time_s"] = time.time() - t_start
    stats["gains_of"] = {int(l): {int(c): float(v) for c, v in d.items()} for l, d in gains_of.items()}
    return committed, stats
