"""v2.0_test: charge-light assignment from scratch as an energy minimisation over clusters.

Offline on chain dumps (first-stage products only: clusters, per-cluster predicted light
peaks a_c[ch], actual light).  See V200_DESIGN.md.

    E(t) = sum_TPC sum_{ch in U} sum_tau w (A - min(sum_c a_c P(tau - t_c), rail))^2
         + sum_{contact pairs} beta [ |t_c - t_c'| > SAME_TICKS ]

The light term is pairwise-quadratic in the assignment when the clip is ignored, so any block
of clusters is solved exactly by enumeration from k x k tables (see block_tables); the clip is
honoured when a candidate configuration is accepted (exact delta on the rendered model).
"""
from __future__ import annotations

import os
import time
import itertools
import numpy as np
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree

HERE = os.path.abspath(os.path.dirname(__file__))
BASE = os.path.abspath(os.path.join(HERE, "..", "..", "CLMatching_AlphaRelease"))
TMPL = np.load(os.path.join(BASE, "assets", "avg_pulse.npy")).astype(np.float64)
TMPL = TMPL / TMPL.max()
NT = TMPL.size                    # 1000 samples
GAIN = float(os.environ.get("V200_GAINADC", "200"))   # ADC^2 per ADC of Poisson variance (single-PE gain scale)
PEAK, SEARCH, CUT = 105, 895, 10.0
VAR_FLOOR = float(os.environ.get("V200_VARFLOOR", "100"))   # A floor [ADC] in the Poisson variance
VAR_DIR = os.environ.get("V200_VARDIR", "")                 # dir with <tag>.npy per-sample variance maps from the trained predictor
SAME_TICKS = 3                    # |dt| <= SAME_TICKS counts as "same time" for the prior
DEC_MARGIN = 25.0                 # second-best self-scan minimum worse by >= this: decidable
DEC_SIG = 25.0                    # own light significance sum w a^2 P^2 >= this: has light
MIN_SEP = 10                      # ticks between distinct minima
CONTACT_CM = 3.0
K_CAND = 6                        # own minima kept per cluster
K_MAX = 10                        # candidate cap per cluster in a block
BLOCK_MAX = 8                     # exact block size
ENUM_MAX = 2_000_000              # product of candidate counts in one enumeration
SUPPORT_FRAC = 0.90
LEGACY_RAIL = 60780.0
DOM_MIN = int(os.environ.get("V200_DOM", "0"))   # >0: decidable also needs this many own-dominated channels
MIN_A = 300.0                                  # own predicted peak [ADC] for a channel to count as dominated
USE_SUP = os.environ.get("V200_SUP", "0") not in ("", "0")   # 1: a cluster's own terms only on its support channels (chain-like)
BLIND_F = float(os.environ.get("V200_BLIND", "0"))   # >0: a cluster whose own light is below this fraction of the
                                                      # others' light on every candidate is light-blind: light terms off, prior/proximity decides
BLIND_NCH = 3
COS_MIN = float(os.environ.get("V200_COS", "0"))     # >0: a candidate counts as light-supported only if the residual
                                                      # pattern on the support channels matches the cluster's own pattern (cosine)
WARM_COLD = os.environ.get("V200_COLD", "0") not in ("", "0")   # with WARM: re-initialise the re-solved clusters by greedy self-placement
FREE_MIN_E = float(os.environ.get("V200_FREE", "0"))   # >0: clusters below this energy are never light-decidable and only choose among existing slots
WARM_DEC_E = float(os.environ.get("V200_WARMDEC", "0"))   # >0 with WARM: clusters in [WARMDEC, WARM) also re-solved, but only if light-decidable
WARM_MIN_E = float(os.environ.get("V200_WARM", "0"))   # >0: start from the chain's assignment and re-solve only clusters >= this energy [MeV]
GAIN_SIG = float(os.environ.get("V200_GAIN", "0"))   # >0: per-family light gains with prior sigma (rel.); 0 = fixed gains
GAIN_LO, GAIN_HI = 0.5, 2.0
FSYS = float(os.environ.get("V200_FSYS", "0"))     # prediction-systematics term: sigma^2 = GAIN max(A,100) + (FSYS A)^2


# ----------------------------------------------------------------------------- template
_SH: dict[int, np.ndarray] = {}
_SH2: dict[int, np.ndarray] = {}


def P_shift(t: int) -> np.ndarray:
    t = int(t)
    if t not in _SH:
        out = np.zeros(NT)
        if 0 <= t < NT:
            out[t:] = TMPL[: NT - t]
        _SH[t] = out
        _SH2[t] = out * out
    return _SH[t]


def P_shift2(t: int) -> np.ndarray:
    P_shift(t)
    return _SH2[int(t)]


def _corr_full(q: np.ndarray, kern: np.ndarray) -> np.ndarray:
    """c(t) = sum_tau q(tau) kern(tau - t) for t = 0..SEARCH (kern = P or P^2 at shift 0)."""
    return fftconvolve(q, kern[::-1], mode="full")[NT - 1: NT - 1 + SEARCH + 1]


# ----------------------------------------------------------------------------- event
class Event:
    def __init__(self, npz_path: str | None = None, max_hits_per_label_tpc: int = 3000, seed: int = 0, **arrays):
        if npz_path is not None:
            z = np.load(npz_path, allow_pickle=True)
            self.path = npz_path
            self.tag = os.path.basename(npz_path)[:-4]
            rail = z["sat_ceiling"].astype(np.float64) if "sat_ceiling" in z.files else None
            amp_rows = (z["amp_lab"].astype(int), z["amp_tpc"].astype(int), z["amp_row"].astype(np.float64))
            self._setup(e=z["e"], xyz=np.stack([z["x"], z["y"], z["z"]], 1), tpc=z["tpc"], labels=z["labels"],
                        truth=z["truth_t0"], ts_chain=z["ts_final"], A=z["full_light"], veto=z["veto_mask"],
                        rail=rail, amp_rows=amp_rows, cap=max_hits_per_label_tpc, seed=seed)
        else:
            self.path = None
            self.tag = str(arrays.pop("tag", "event"))
            self._setup(cap=max_hits_per_label_tpc, seed=seed, **arrays)

    @classmethod
    def from_arrays(cls, *, e, xyz, tpc, labels, ts_chain, A, veto, amp, rail=None, truth=None, tag="event",
                    cap: int = 3000, seed: int = 0):
        """In-chain constructor: `amp` is {(label, tpc): (n_ch,) predicted peak amplitude}."""
        ev = cls.__new__(cls)
        ev.path = None; ev.tag = str(tag)
        al = np.array([k[0] for k in amp], int); at = np.array([k[1] for k in amp], int)
        ar = np.stack([np.asarray(v, np.float64) for v in amp.values()]) if amp else np.zeros((0, A.shape[1]))
        ev._setup(e=e, xyz=xyz, tpc=tpc, labels=labels, truth=truth, ts_chain=ts_chain, A=A, veto=veto,
                  rail=rail, amp_rows=(al, at, ar), cap=cap, seed=seed)
        return ev

    def _setup(self, *, e, xyz, tpc, labels, truth, ts_chain, A, veto, rail, amp_rows, cap, seed):
        self.e = np.clip(np.asarray(e), 0, None).astype(np.float64)
        self.xyz = np.asarray(xyz, np.float64)
        self.tpc = np.asarray(tpc).astype(int)
        self.labels = np.asarray(labels).astype(int)
        self.truth = (np.full(self.e.size, np.nan) if truth is None else np.asarray(truth, np.float64))
        tsc = np.asarray(ts_chain, np.float64)
        self.ts_chain = np.where(np.isfinite(tsc), tsc, -1.0)
        self.A = np.asarray(A, np.float64)
        self.n_light, self.n_ch, _ = self.A.shape
        self.veto = np.asarray(veto).astype(bool)
        if rail is not None:
            self.rail = np.asarray(rail, np.float64)
        else:
            mx = self.A.max(-1)
            self.rail = np.where(mx >= LEGACY_RAIL - 80.0, mx, LEGACY_RAIL)
        # Poisson noise plus a relative prediction-systematics term (v2.0 test 3): where a lot of
        # light is present, whoever made it, a small cluster cannot be timed against it.
        var = GAIN * np.maximum(self.A, VAR_FLOOR) + (FSYS * np.maximum(self.A, 0.0)) ** 2
        if VAR_DIR:
            vpath = os.path.join(VAR_DIR, f"{self.tag}.npy")
            if os.path.exists(vpath):
                # trained variance predictor (waveform noise) + optional relative model-error term
                var = np.maximum(np.load(vpath).astype(np.float64), 1.0) + (FSYS * np.maximum(self.A, 0.0)) ** 2
            else:
                print(f"[v200] no variance map for {self.tag} in {VAR_DIR}; Poisson used", flush=True)
        self.w = 1.0 / var
        # per-cluster predicted peak amplitudes
        al, at, ar = amp_rows
        self.amp: dict[tuple[int, int], np.ndarray] = {}
        for i in range(al.size):
            if 0 <= at[i] < self.n_light:
                a = ar[i].copy()
                a[self.veto[at[i]]] = 0.0          # vetoed channels never enter the loss
                if a.max() > 0:
                    self.amp[(int(al[i]), int(at[i]))] = a
        self.lab_tpcs: dict[int, list[int]] = {}
        for (L, c) in self.amp:
            self.lab_tpcs.setdefault(L, []).append(c)
        labs_all = np.unique(self.labels[self.labels >= 0])
        self.lab_hits = {int(L): np.flatnonzero(self.labels == L) for L in labs_all}
        self.lab_E = {L: float(self.e[idx].sum()) for L, idx in self.lab_hits.items()}
        self.light_labels = sorted(L for L in self.lab_tpcs if L in self.lab_hits)
        # support channels (90 % of predicted light) and the TPC channel unions
        self.support: dict[tuple[int, int], np.ndarray] = {}
        U = [set() for _ in range(self.n_light)]
        for (L, c), a in self.amp.items():
            order = np.argsort(a)[::-1]
            cs = np.cumsum(a[order])
            n = int(np.searchsorted(cs, SUPPORT_FRAC * cs[-1])) + 1
            ch = np.sort(order[:n])
            ch = ch[a[ch] > 0]
            if ch.size == 0:
                continue
            self.support[(L, c)] = ch
            U[c].update(int(v) for v in ch)
        self.U = [np.array(sorted(s), dtype=int) for s in U]
        # cluster significance per TPC: sum_ch a^2 sum_tau w P^2 at a typical placement (use mean w)
        self.sig: dict[int, float] = {}
        for L in self.light_labels:
            s = 0.0
            for c in self.lab_tpcs[L]:
                a = self.amp[(L, c)]; ch = self.U[c]
                s += float(np.sum(a[ch] ** 2 * self.w[c][ch].mean(axis=1)) * np.sum(TMPL ** 2))
            self.sig[L] = s
        self._rng = np.random.default_rng(seed)
        self.contacts = self._contact_graph(cap)
        # runtime state
        self.assign: dict[int, int] = {}
        self.model = np.zeros_like(self.A)
        self.beta: dict[tuple[int, int], float] = {}
        # spatial prior for undecidable clusters (v2.0 test 2): per-cluster unary penalties by
        # candidate time, built each sweep from the in-situ tables dtable / pnew (None = off)
        self.uprior: dict[int, dict[int, float]] = {}
        self.g: dict[tuple[int, int], float] = {}     # per (cluster, TPC) light gain (family-tied)
        self.blind: set[int] = set()
        self.supp: dict[int, set[int]] = {}             # cluster -> candidate times supported by the light pattern
        self.dtable: dict[str, float] | None = None
        self.pnew: dict[str, float] | None = None

    # ---- geometry -------------------------------------------------------------------
    def _contact_graph(self, cap: int) -> dict[tuple[int, int], float]:
        """label pairs with minimum hit distance <= CONTACT_CM -> that distance."""
        out: dict[tuple[int, int], float] = {}
        for c in np.unique(self.tpc):
            m_tpc = self.tpc == c
            labs = np.unique(self.labels[m_tpc & (self.labels >= 0)])
            if labs.size < 2:
                continue
            pts, box = {}, {}
            for L in labs:
                idx = np.flatnonzero(m_tpc & (self.labels == L))
                if idx.size > cap:
                    idx = self._rng.choice(idx, cap, replace=False)
                p = self.xyz[idx]
                pts[int(L)] = p
                box[int(L)] = (p.min(0), p.max(0))
            labs = [int(v) for v in labs]
            trees = {}
            for i, L1 in enumerate(labs):
                lo1, hi1 = box[L1]
                for L2 in labs[i + 1:]:
                    lo2, hi2 = box[L2]
                    if np.any(lo1 > hi2 + CONTACT_CM) or np.any(lo2 > hi1 + CONTACT_CM):
                        continue
                    a, b = (L1, L2) if pts[L1].shape[0] <= pts[L2].shape[0] else (L2, L1)
                    if b not in trees:
                        trees[b] = cKDTree(pts[b])
                    d, _ = trees[b].query(pts[a], k=1, distance_upper_bound=CONTACT_CM)
                    dm = float(d.min()) if np.isfinite(d).any() else np.inf
                    if dm <= CONTACT_CM:
                        key = (min(L1, L2), max(L1, L2))
                        out[key] = min(out.get(key, np.inf), dm)
        return out

    # ---- light model ----------------------------------------------------------------
    def reset(self):
        self.assign = {}
        self.g = {}
        self.model[:] = 0.0

    def set_time(self, L: int, t: int | None):
        told = self.assign.get(L)
        if told == t:
            return
        for c in self.lab_tpcs[L]:
            a = self.amp[(L, c)]
            if told is not None:
                self.model[c] -= self.g.get((L, c), 1.0) * a[:, None] * P_shift(told)[None, :]
            self.g.pop((L, c), None)               # a moved cluster joins a new family: gain reset
            if t is not None:
                self.model[c] += a[:, None] * P_shift(t)[None, :]
        if t is None:
            self.assign.pop(L, None)
        else:
            self.assign[L] = int(t)

    # ---- per-family light gains (v2.0 test 4) ------------------------------------------
    def families_tpc(self, c: int) -> list[list[int]]:
        """clusters placed in TPC c grouped by time (within SAME_TICKS)."""
        items = sorted(((self.assign[L], L) for L in self.assign if (L, c) in self.amp))
        fams, cur, t_ref = [], [], None
        for t, L in items:
            if cur and t - t_ref > SAME_TICKS:
                fams.append(cur); cur = []
            if not cur:
                t_ref = t
            cur.append(L)
        if cur:
            fams.append(cur)
        return fams

    def profile_tpc(self, c: int) -> None:
        """Least squares for one gain per family in TPC c with prior N(1, GAIN_SIG); updates
        self.g and the model.  Families are the placed clusters grouped by time."""
        if GAIN_SIG <= 0:
            return
        ch = self.U[c]
        if ch.size == 0:
            return
        fams = self.families_tpc(c)
        if not fams:
            return
        w = self.w[c][ch]; A = self.A[c][ch]
        X = []
        for f in fams:
            x = np.zeros((ch.size, NT))
            for L in f:
                x += self.amp[(L, c)][ch][:, None] * P_shift(self.assign[L])[None, :]
            X.append(x)
        n = len(fams)
        G = np.empty((n, n)); b = np.empty(n)
        wX = [w * x for x in X]
        for i in range(n):
            b[i] = float(np.sum(wX[i] * A))
            for j in range(i, n):
                G[i, j] = G[j, i] = float(np.sum(wX[i] * X[j]))
        lam = 1.0 / GAIN_SIG ** 2
        g = np.linalg.solve(G + lam * np.eye(n), b + lam)
        g = np.clip(g, GAIN_LO, GAIN_HI)
        for f, gf in zip(fams, g):
            for L in f:
                g_old = self.g.get((L, c), 1.0)
                if abs(gf - g_old) > 1e-12:
                    self.model[c] += (gf - g_old) * self.amp[(L, c)][:, None] * P_shift(self.assign[L])[None, :]
                self.g[(L, c)] = float(gf)

    def gain_energy(self, tpcs=None) -> float:
        if GAIN_SIG <= 0:
            return 0.0
        tot = 0.0
        rng_c = range(self.n_light) if tpcs is None else tpcs
        for c in rng_c:
            for f in self.families_tpc(c):
                gf = self.g.get((f[0], c), 1.0)
                tot += (gf - 1.0) ** 2 / GAIN_SIG ** 2
        return tot

    def chi2_tpc(self, c: int, model_c: np.ndarray | None = None) -> float:
        ch = self.U[c]
        if ch.size == 0:
            return 0.0
        M = self.model[c] if model_c is None else model_c
        m = np.minimum(M[ch], self.rail[c][ch, None])
        return float(np.sum(self.w[c][ch] * (self.A[c][ch] - m) ** 2))

    def light_energy(self) -> float:
        return float(sum(self.chi2_tpc(c) for c in range(self.n_light) if self.U[c].size))

    def prior_energy(self, assign: dict[int, int] | None = None) -> float:
        a = self.assign if assign is None else assign
        tot = 0.0
        for (L1, L2), b in self.beta.items():
            t1, t2 = a.get(L1), a.get(L2)
            if t1 is None or t2 is None:
                continue
            if abs(t1 - t2) > SAME_TICKS:
                tot += b
        for L, pen in self.uprior.items():
            t = a.get(L)
            if t is not None:
                tot += pen.get(int(t), 0.0)
        return tot

    def energy(self) -> float:
        return self.light_energy() + self.prior_energy() + self.gain_energy()

    def delta_exact(self, moves: dict[int, int]) -> float:
        """Exact (clipped) change of the light energy if the clusters in `moves` took the new times."""
        tpcs = set()
        for L in moves:
            tpcs.update(self.lab_tpcs[L])
        d = 0.0
        for c in tpcs:
            ch = self.U[c]
            if ch.size == 0:
                continue
            M = self.model[c][ch].copy()
            for L, t in moves.items():
                if (L, c) not in self.amp:
                    continue
                a = self.amp[(L, c)][ch]
                told = self.assign.get(L)
                if told is not None:
                    M -= a[:, None] * P_shift(told)[None, :]
                if t is not None:
                    M += a[:, None] * P_shift(t)[None, :]
            rail = self.rail[c][ch, None]
            m_new = np.minimum(M, rail)
            m_old = np.minimum(self.model[c][ch], rail)
            d += float(np.sum(self.w[c][ch] * ((self.A[c][ch] - m_new) ** 2 - (self.A[c][ch] - m_old) ** 2)))
        return d

    def prior_delta(self, moves: dict[int, int]) -> float:
        new = dict(self.assign); new.update(moves)
        return self.prior_energy(new) - self.prior_energy()

    # ---- self scan on the residual (energy-consistent: TPC channel unions) -------------
    def qr_vectors(self, L: int, exclude_self: bool = True):
        """q_LL(tau) = sum_ch a^2 w ; r_L(tau) = sum_ch a w R_ext, summed over the cluster's TPCs."""
        q = np.zeros(NT); r = np.zeros(NT)
        for c in self.lab_tpcs[L]:
            ch = self.support[(L, c)] if USE_SUP else self.U[c]
            a = self.amp[(L, c)][ch]
            w = self.w[c][ch]
            R = self.A[c][ch] - self.model[c][ch]
            if exclude_self and L in self.assign:
                R = R + a[:, None] * P_shift(self.assign[L])[None, :]
            q += (a ** 2)[:, None].T @ w if False else np.einsum("i,ij->j", a ** 2, w)
            r += np.einsum("i,ij->j", a, w * R)
        return q, r

    def scan(self, L: int, exclude_self: bool = True) -> np.ndarray:
        """chi2 change (up to a constant) of placing L at t, t = 0..SEARCH, on the current residual."""
        q, r = self.qr_vectors(L, exclude_self)
        return _corr_full(q, TMPL * TMPL) - 2.0 * _corr_full(r, TMPL)

    @staticmethod
    def minima(curve: np.ndarray, k: int = K_CAND, sep: int = MIN_SEP):
        n = curve.size
        loc = np.flatnonzero((curve[1:-1] < curve[:-2]) & (curve[1:-1] <= curve[2:])) + 1
        if curve[0] < curve[1]:
            loc = np.append(loc, 0)
        if curve[-1] < curve[-2]:
            loc = np.append(loc, n - 1)
        loc = loc[np.argsort(curve[loc])]
        out = []
        for t in loc:
            if all(abs(int(t) - u) > sep for u, _ in out):
                out.append((int(t), float(curve[t])))
            if len(out) >= k:
                break
        return out

    def candidates(self, L: int, slots: dict[int, list[int]]):
        """(own minima, decidable flag, margin) on the current residual excluding L itself."""
        curve = self.scan(L, exclude_self=True)
        mins = self.minima(curve)
        if not mins:
            return [], False, 0.0, curve, 0
        margin = (mins[1][1] - mins[0][1]) if len(mins) > 1 else np.inf
        n_dom = self.n_dominated(L, mins[0][0])
        decid = bool(margin >= DEC_MARGIN and self.sig.get(L, 0.0) >= DEC_SIG and (DOM_MIN <= 0 or n_dom >= DOM_MIN)
                     and (FREE_MIN_E <= 0 or self.lab_E.get(L, 0.0) >= FREE_MIN_E))
        return [t for t, _ in mins], decid, float(margin), curve, n_dom

    def n_dominated(self, L: int, t: int) -> int:
        """support channels where the cluster's own predicted peak at t is at least the light of
        everything else there and above MIN_A: where light can time THIS cluster."""
        k = int(t) + PEAK
        if not (0 <= k < NT):
            return 0
        n = 0
        for c in self.lab_tpcs[L]:
            ch = self.support.get((L, c))
            if ch is None:
                continue
            own = self.amp[(L, c)][ch]
            others = self.model[c][ch, k]
            if L in self.assign:
                others = others - own * P_shift(self.assign[L])[k]
            n += int(np.sum((own >= MIN_A) & (own >= others)))
        return n

    def n_visible(self, L: int, t: int, frac: float) -> int:
        """support channels where the cluster's own peak at t is at least `frac` of the light of
        everything else there (and above MIN_A): where its light is above the systematics floor."""
        k = int(t) + PEAK
        if not (0 <= k < NT):
            return 0
        n = 0
        for c in self.lab_tpcs[L]:
            ch = self.support.get((L, c))
            if ch is None:
                continue
            own = self.amp[(L, c)][ch]
            others = self.model[c][ch, k]
            if L in self.assign:
                others = others - self.g.get((L, c), 1.0) * own * P_shift(self.assign[L])[k]
            n += int(np.sum((own >= MIN_A) & (own >= frac * np.maximum(others, 0.0))))
        return n

    def pattern_cos(self, L: int, t: int) -> float:
        """cosine between the cluster's predicted channel pattern and the per-channel template
        amplitude fitted on the residual (own light removed) at time t, over its support channels."""
        a_all, r_all = [], []
        P = P_shift(t); P2 = P * P
        for c in self.lab_tpcs[L]:
            ch = self.support.get((L, c))
            if ch is None:
                continue
            a = self.amp[(L, c)][ch]
            w = self.w[c][ch]
            R = self.A[c][ch] - self.model[c][ch]
            if L in self.assign:
                R = R + self.g.get((L, c), 1.0) * a[:, None] * P_shift(self.assign[L])[None, :]
            num = (w * R) @ P
            den = w @ P2
            r = np.where(den > 0, num / np.maximum(den, 1e-30), 0.0)
            a_all.append(a); r_all.append(r)
        if not a_all:
            return 0.0
        a = np.concatenate(a_all); r = np.concatenate(r_all)
        na, nr = np.linalg.norm(a), np.linalg.norm(r)
        if na == 0 or nr == 0:
            return 0.0
        return float(a @ r / (na * nr))

    # ---- block tables and exact solve ---------------------------------------------------
    def block_tables(self, S: list[int], K: dict[int, list[int]]):
        """unary u[L] (k,), pairwise p[(L,L')] (k,k) for the block S with everything else fixed."""
        # residual of everything outside S, per TPC (unclipped model)
        tpcs = sorted({c for L in S for c in self.lab_tpcs[L]})
        Rext = {}
        for c in tpcs:
            ch = self.U[c]
            M = self.model[c][ch].copy()
            for L in S:
                if (L, c) in self.amp and L in self.assign:
                    M -= self.g.get((L, c), 1.0) * self.amp[(L, c)][ch][:, None] * P_shift(self.assign[L])[None, :]
            Rext[c] = self.A[c][ch] - M
        Pm = {L: np.stack([P_shift(t) for t in K[L]]) for L in S}
        Pm2 = {L: Pm[L] ** 2 for L in S}
        u = {L: np.zeros(len(K[L])) for L in S}
        p: dict[tuple[int, int], np.ndarray] = {}
        for c in tpcs:
            ch = self.U[c]; w = self.w[c][ch]
            here = [L for L in S if (L, c) in self.amp]
            avec = {L: self.amp[(L, c)][ch] for L in here}
            if USE_SUP:
                # chain-like: a cluster only sees its own support channels
                pos = {int(v): i for i, v in enumerate(ch)}
                msk = {}
                for L in here:
                    m = np.zeros(ch.size, bool)
                    m[[pos[int(v)] for v in self.support[(L, c)] if int(v) in pos]] = True
                    msk[L] = m
                avec_own = {L: np.where(msk[L], avec[L], 0.0) for L in here}
            else:
                avec_own = avec
            for L in here:
                if L in self.blind:
                    continue
                q = np.einsum("i,ij->j", avec_own[L] ** 2, w)
                r = np.einsum("i,ij->j", avec_own[L], w * Rext[c])
                u[L] += Pm2[L] @ q - 2.0 * (Pm[L] @ r)
            for i, L1 in enumerate(here):
                for L2 in here[i + 1:]:
                    if L1 in self.blind or L2 in self.blind:
                        continue
                    q12 = np.einsum("i,ij->j", avec_own[L1] * avec_own[L2], w)
                    tab = 2.0 * ((Pm[L1] * q12[None, :]) @ Pm[L2].T)
                    key = (L1, L2) if L1 < L2 else (L2, L1)
                    p[key] = p.get(key, 0.0) + (tab if L1 < L2 else tab.T)
        if self.supp:
            # light evidence only counts where the residual pattern matches the cluster's own pattern
            mask = {L: np.array([t in self.supp.get(L, ()) for t in K[L]]) for L in S}
            for L in S:
                if L in self.supp:
                    u[L] = np.where(mask[L], u[L], 0.0)
            for (L1, L2), tab in p.items():
                if L1 in self.supp:
                    tab = tab * mask[L1][:, None]
                if L2 in self.supp:
                    tab = tab * mask[L2][None, :]
                p[(L1, L2)] = tab
        return u, p

    def solve_block(self, S: list[int], K: dict[int, list[int]], n_top: int = 3):
        """Exact enumeration of the block energy (quadratic light + prior); returns the best
        `n_top` configurations (list of dict L->t) by table energy."""
        S = list(S)
        # cap the enumeration size: drop the least useful candidates of the least significant clusters
        while np.prod([len(K[L]) for L in S]) > ENUM_MAX:
            L = min(S, key=lambda x: (len(K[x]) <= 2, self.sig.get(x, 0.0)))
            if len(K[L]) <= 2:
                break
            K[L] = K[L][:-1]
        u, p = self.block_tables(S, K)
        shape = tuple(len(K[L]) for L in S)
        E = np.zeros(shape)
        for i, L in enumerate(S):
            sh = [1] * len(S); sh[i] = shape[i]
            E += u[L].reshape(sh)
        idx = {L: i for i, L in enumerate(S)}
        for (L1, L2), tab in p.items():
            i, j = idx[L1], idx[L2]
            sh = [1] * len(S); sh[i] = shape[i]; sh[j] = shape[j]
            E += tab.reshape(sh)
        for (L1, L2), b in self.beta.items():
            if L1 in idx and L2 in idx:
                i, j = idx[L1], idx[L2]
                t1 = np.asarray(K[L1])[:, None]; t2 = np.asarray(K[L2])[None, :]
                pen = b * (np.abs(t1 - t2) > SAME_TICKS)
                sh = [1] * len(S); sh[i] = shape[i]; sh[j] = shape[j]
                E += pen.reshape(sh)
            elif L1 in idx and L2 in self.assign:
                i = idx[L1]; t2 = self.assign[L2]
                pen = b * (np.abs(np.asarray(K[L1]) - t2) > SAME_TICKS)
                sh = [1] * len(S); sh[i] = shape[i]
                E += pen.reshape(sh)
            elif L2 in idx and L1 in self.assign:
                j = idx[L2]; t1 = self.assign[L1]
                pen = b * (np.abs(np.asarray(K[L2]) - t1) > SAME_TICKS)
                sh = [1] * len(S); sh[j] = shape[j]
                E += pen.reshape(sh)
        for L, pen in self.uprior.items():
            if L in idx:
                i = idx[L]
                sh = [1] * len(S); sh[i] = shape[i]
                E += np.asarray([pen.get(int(t), 0.0) for t in K[L]]).reshape(sh)
        flat = E.ravel()
        n_top = min(n_top, flat.size)
        best = np.argpartition(flat, n_top - 1)[:n_top]
        best = best[np.argsort(flat[best])]
        out = []
        for b in best:
            combo = np.unravel_index(int(b), shape)
            out.append(({L: int(K[L][combo[i]]) for i, L in enumerate(S)}, float(flat[b])))
        return out

    def try_block(self, S: list[int], K: dict[int, list[int]]) -> float:
        """Solve the block exactly on the tables, then accept the best configuration whose
        EXACT (clipped, gain-profiled) energy change is negative.  Returns the delta (<= 0)."""
        cands = self.solve_block(S, K)
        if GAIN_SIG <= 0:
            best_d, best_cfg = 0.0, None
            for cfg, _ in cands:
                moves = {L: t for L, t in cfg.items() if self.assign.get(L) != t}
                if not moves:
                    continue
                d = self.delta_exact(moves) + self.prior_delta(moves)
                if d < best_d - 1e-9:
                    best_d, best_cfg = d, moves
            if best_cfg:
                for L, t in best_cfg.items():
                    self.set_time(L, t)
            return best_d
        # gains on: apply tentatively, re-profile the affected TPCs, compare exact energies, revert if worse
        tpcs = sorted({c for L in S for c in self.lab_tpcs[L]})
        def local_E():
            return sum(self.chi2_tpc(c) for c in tpcs) + self.gain_energy(tpcs) + self.prior_energy()
        def snapshot():
            return ({L: self.assign.get(L) for L in S},
                    {(L, c): self.g.get((L, c), 1.0) for c in tpcs for L in self.assign if (L, c) in self.amp},
                    {c: self.model[c].copy() for c in tpcs})
        def restore(snap):
            asg, g, mdl = snap
            for L, t in asg.items():
                if t is None:
                    self.assign.pop(L, None)
                else:
                    self.assign[L] = t
            self.g = {k: v for k, v in self.g.items() if k[1] not in tpcs}
            self.g.update(g)
            for c in tpcs:
                self.model[c] = mdl[c]
        E0 = local_E()
        snap0 = snapshot()
        best_d, best_snap = 0.0, None
        for cfg, _ in cands:
            moves = {L: t for L, t in cfg.items() if self.assign.get(L) != t}
            if not moves:
                continue
            for L, t in moves.items():
                self.set_time(L, t)
            for c in tpcs:
                self.profile_tpc(c)
            d = local_E() - E0
            if d < best_d - 1e-9:
                best_d, best_snap = d, snapshot()
            restore(snap0)
        if best_snap is not None:
            restore(best_snap)
        return best_d

    # ---- coverage for light-blind clusters and noise ----------------------------------
    def hit_times(self) -> np.ndarray:
        ts = np.full(self.e.size, -1.0)
        for L, t in self.assign.items():
            ts[self.lab_hits[L]] = float(t)
        # nearest placed hit in the same TPC decides the rest
        for c in np.unique(self.tpc):
            m = self.tpc == c
            have = m & (ts >= 0)
            need = m & (ts < 0)
            if not need.any():
                continue
            if not have.any():
                continue
            tree = cKDTree(self.xyz[have])
            _, j = tree.query(self.xyz[need], k=1)
            ts[np.flatnonzero(need)] = ts[np.flatnonzero(have)][j]
        return ts

    def chain_assignment(self) -> dict[int, int]:
        out = {}
        for L in self.light_labels:
            v = self.ts_chain[self.lab_hits[L]]
            v = v[v >= 0]
            if v.size:
                out[L] = int(round(float(np.median(v))))
        return out

    # ---- scoring ------------------------------------------------------------------------
    def score(self, ts: np.ndarray):
        ok = np.isfinite(self.truth) & (ts >= 0)
        good = ok & (np.abs(ts - self.truth) <= CUT)
        e = self.e
        res = {"eff": float(e[good].sum() / max(e[ok].sum(), 1e-9)), "E_ok": float(e[ok].sum()),
               "E_wrong": float(e[ok & ~good].sum()), "cover": float(e[ts >= 0].sum() / max(e.sum(), 1e-9))}
        labE = np.array([self.lab_E.get(int(L), 0.0) if L >= 0 else 0.0 for L in self.labels])
        has = np.array([int(L) in self.lab_tpcs for L in self.labels])
        cls = {"noise/blind": (~has), "<1.5": has & (labE < 1.5), "1.5-5": has & (labE >= 1.5) & (labE < 5),
               "5-20": has & (labE >= 5) & (labE < 20), "20-100": has & (labE >= 20) & (labE < 100), ">=100": has & (labE >= 100)}
        for k, m in cls.items():
            mm = ok & m
            res[k] = (float(e[mm & ~good].sum()), float(e[mm].sum()))
        return res


# ----------------------------------------------------------------------------- solver
def light_edges(ev: Event, min_cos: float = 0.5):
    """cluster pairs sharing a TPC whose predicted patterns overlap (cosine on U >= min_cos)."""
    edges = {}
    for c in range(ev.n_light):
        here = [L for L in ev.light_labels if (L, c) in ev.amp]
        if len(here) < 2:
            continue
        ch = ev.U[c]
        M = np.stack([ev.amp[(L, c)][ch] for L in here])
        n = np.linalg.norm(M, axis=1); n[n == 0] = 1.0
        cosm = (M / n[:, None]) @ (M / n[:, None]).T
        ii, jj = np.nonzero(np.triu(cosm >= min_cos, 1))
        for i, j in zip(ii, jj):
            key = (min(here[i], here[j]), max(here[i], here[j]))
            edges[key] = max(edges.get(key, 0.0), float(cosm[i, j]))
    return edges


def components(nodes, edges):
    adj = {n: set() for n in nodes}
    for (a, b) in edges:
        if a in adj and b in adj:
            adj[a].add(b); adj[b].add(a)
    seen, comps = set(), []
    for n in nodes:
        if n in seen:
            continue
        stack, comp = [n], []
        seen.add(n)
        while stack:
            x = stack.pop(); comp.append(x)
            for y in adj[x]:
                if y not in seen:
                    seen.add(y); stack.append(y)
        comps.append(comp)
    return comps, adj


def slots_of(ev: Event, decidable: set[int]) -> dict[int, list[int]]:
    """per TPC: clustered times of the decidable clusters placed there."""
    out: dict[int, list[int]] = {}
    for L in decidable:
        if L not in ev.assign:
            continue
        for c in ev.lab_tpcs[L]:
            out.setdefault(c, []).append(ev.assign[L])
    for c, ts in out.items():
        ts = sorted(ts); merged = []
        for t in ts:
            if not merged or t - merged[-1] > SAME_TICKS:
                merged.append(t)
        out[c] = merged
    return out


def candidate_set(ev: Event, L: int, own: list[int], decid: bool, slots: dict[int, list[int]], K_max: int = K_MAX):
    if FREE_MIN_E > 0 and ev.lab_E.get(L, 0.0) < FREE_MIN_E:
        cand = []                                   # slot-only: no own minima, no new slots
    else:
        cand = list(own[:K_CAND]) if decid else list(own[:3])
    if L in ev.assign and ev.assign[L] not in cand:
        cand.append(ev.assign[L])
    pool = []
    for c in ev.lab_tpcs[L]:
        pool.extend(slots.get(c, []))
    # nearest slots first (by their unary value on the current scan is costly; use distance to own best)
    ref = own[0] if own else (ev.assign.get(L, 0))
    pool = sorted(set(pool), key=lambda t: abs(t - ref))
    for t in pool:
        if all(abs(t - u) > SAME_TICKS for u in cand):
            cand.append(t)
        if len(cand) >= K_max:
            break
    if not cand:                                    # slot-only cluster with no slot in its TPCs: keep / own best
        cand = [ev.assign[L]] if L in ev.assign else list(own[:1])
    return cand[:K_max]


def solve_event(ev: Event, n_sweeps: int = 3, seed: int = 0, verbose: bool = False, log=None,
                warm_min_e: float | None = None, cold: bool | None = None) -> dict:
    WARM = WARM_MIN_E if warm_min_e is None else float(warm_min_e)
    COLD = WARM_COLD if cold is None else bool(cold)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    ev.reset()
    labs = sorted(ev.light_labels, key=lambda L: -ev.lab_E[L])
    decidable: set[int] = set()
    own_min: dict[int, list[int]] = {}
    slots: dict[int, list[int]] = {}
    if WARM > 0:
        # warm start: the chain's assignment for everything; only clusters >= WARM are re-solved,
        # the rest stay frozen at the chain's times (they still shape the residual)
        for L, t in ev.chain_assignment().items():
            ev.set_time(L, t)
        if WARM_DEC_E > 0:
            keep = []
            for L in labs:
                if ev.lab_E[L] >= WARM:
                    keep.append(L)
                elif ev.lab_E[L] >= WARM_DEC_E:
                    _own, _decid, _m, _c, _nd = ev.candidates(L, slots)
                    if _decid:
                        keep.append(L)
            labs = keep
        else:
            labs = [L for L in labs if ev.lab_E[L] >= WARM]
        if COLD:
            for L in labs:
                ev.set_time(L, None)
            for L in labs:                      # greedy re-initialisation on the residual (small clusters frozen)
                own, decid, margin, curve, _nd = ev.candidates(L, slots)
                own_min[L] = own
                if decid:
                    decidable.add(L)
                    ev.set_time(L, own[0])
                else:
                    cand = candidate_set(ev, L, own, False, slots)
                    if cand:
                        ev.set_time(L, int(cand[int(np.argmin([curve[t] for t in cand]))]))
                slots = slots_of(ev, decidable)
        else:
            for L in labs:
                own, decid, margin, curve, _nd = ev.candidates(L, slots)
                own_min[L] = own
                if decid:
                    decidable.add(L)
            slots = slots_of(ev, decidable)
    else:
      # ---- initial state: energy-ordered greedy on the residual (decidable at own best, others at slots)
      for L in labs:
        own, decid, margin, curve, _nd = ev.candidates(L, slots)
        own_min[L] = own
        if decid:
            decidable.add(L)
            ev.set_time(L, own[0])
        else:
            cand = candidate_set(ev, L, own, False, slots)
            if cand:
                ev.set_time(L, int(cand[int(np.argmin([curve[t] for t in cand]))]))
        slots = slots_of(ev, decidable)
    if GAIN_SIG > 0:
        for c in range(ev.n_light):
            ev.profile_tpc(c)
    E0 = ev.energy()
    hist = [E0]
    # ---- graph
    ledges = light_edges(ev)
    edges = set(ledges) | set(ev.beta)
    comps, adj = components(labs, edges)
    if verbose:
        print(f"[{ev.tag}] init E={E0:.0f} clusters {len(labs)} decidable {len(decidable)} comps {len(comps)} "
              f"(max {max(len(c) for c in comps)}) contacts {len(ev.contacts)} {time.time()-t0:.1f}s", flush=True)
    # ---- sweeps
    for sweep in range(n_sweeps):
        # refresh candidates on the current residual
        K: dict[int, list[int]] = {}
        for L in labs:
            own, decid, margin, _, _nd = ev.candidates(L, slots)
            own_min[L] = own
            if decid:
                decidable.add(L)
            elif L in decidable and margin < DEC_MARGIN * 0.5:
                decidable.discard(L)
        slots = slots_of(ev, decidable)
        for L in labs:
            K[L] = candidate_set(ev, L, own_min[L], L in decidable, slots)
        ev.supp = {}
        if COS_MIN > 0:
            for L in labs:
                ok = {t for t in K[L] if ev.pattern_cos(L, t) >= COS_MIN}
                ev.supp[L] = ok
                if L in decidable and own_min[L] and own_min[L][0] not in ok:
                    decidable.discard(L)
            slots = slots_of(ev, decidable)
        ev.blind = set()
        if COS_MIN > 0:
            ev.blind = {L for L in labs if L not in decidable and not ev.supp.get(L)}
        if BLIND_F > 0:
            for L in labs:
                if L in decidable:
                    continue
                if max((ev.n_visible(L, t, BLIND_F) for t in K[L]), default=0) < BLIND_NCH:
                    ev.blind.add(L)
        build_uprior(ev, labs, decidable, K, build_families(ev, decidable))
        gain = 0.0; n_blocks = 0
        order = list(range(len(comps)))
        rng.shuffle(order)
        for ci in order:
            comp = comps[ci]
            if len(comp) <= BLOCK_MAX:
                gain += ev.try_block(comp, {L: list(K[L]) for L in comp}); n_blocks += 1
            else:
                # random connected sub-blocks covering the component ~2x
                n_sub = max(2, 2 * len(comp) // 6)
                for _ in range(n_sub):
                    start = comp[int(rng.integers(len(comp)))]
                    blk, frontier = [start], [start]
                    seen = {start}
                    while frontier and len(blk) < 6:
                        x = frontier.pop(0)
                        nb = [y for y in adj[x] if y not in seen]
                        rng.shuffle(nb)
                        for y in nb:
                            if len(blk) >= 6:
                                break
                            seen.add(y); blk.append(y); frontier.append(y)
                    gain += ev.try_block(blk, {L: list(K[L]) for L in blk}); n_blocks += 1
        # twin-swap kicks on light edges
        n_swap = 0
        lab_set = set(labs)
        for (L1, L2), cosv in ledges.items():
            if L1 not in lab_set or L2 not in lab_set:
                continue
            if L1 not in ev.assign or L2 not in ev.assign:
                continue
            t1, t2 = ev.assign[L1], ev.assign[L2]
            if abs(t1 - t2) <= SAME_TICKS:
                continue
            moves = {L1: t2, L2: t1}
            d = ev.delta_exact(moves) + ev.prior_delta(moves)
            if d < -1e-9:
                ev.set_time(L1, t2); ev.set_time(L2, t1); gain += d; n_swap += 1
        if GAIN_SIG > 0:
            for c in range(ev.n_light):
                ev.profile_tpc(c)
        E = ev.energy(); hist.append(E)
        if verbose:
            print(f"[{ev.tag}] sweep {sweep}: E={E:.0f} (block gain {gain:.0f}, {n_blocks} blocks, {n_swap} swaps, "
                  f"decidable {len(decidable)}) {time.time()-t0:.1f}s", flush=True)
        if gain > -1.0:
            break
    ts = ev.hit_times()
    res = ev.score(ts)
    res.update({"E_light": ev.light_energy(), "E_prior": ev.prior_energy(), "E_hist": hist,
                "n_decidable": len(decidable), "time_s": time.time() - t0, "assign": dict(ev.assign),
                "decidable": sorted(int(v) for v in decidable), "n_blind": len(ev.blind),
                "gain_stats": [float(v) for v in np.percentile(list(ev.g.values()), [5, 25, 50, 75, 95])] if ev.g else []})
    return res


def chain_reference(ev: Event) -> dict:
    """Score the chain's own assignment and its energy under the same loss."""
    ev.reset()
    for L, t in ev.chain_assignment().items():
        ev.set_time(L, t)
    if GAIN_SIG > 0:
        for c in range(ev.n_light):
            ev.profile_tpc(c)
    res = ev.score(ev.hit_times())
    res_all = ev.score(ev.ts_chain)
    res.update({"E_light": ev.light_energy(), "E_prior": ev.prior_energy(), "eff_all_hits_chain": res_all["eff"],
                "assign": dict(ev.assign)})
    return res


# ----------------------------------------------------------------------------- spatial prior
D_BUCKETS = [(0.0, 3.0, "d3"), (3.0, 10.0, "d10"), (10.0, 30.0, "d30"), (30.0, 100.0, "d100"), (100.0, 1e9, "dinf")]


def e_bucket(E: float) -> str:
    return "s" if E < 5 else ("m" if E < 30 else "b")


def d_bucket(d: float) -> str:
    for lo, hi, name in D_BUCKETS:
        if lo <= d < hi:
            return name
    return "dinf"


def build_families(ev: Event, decidable: set[int], cap: int = 500, seed: int = 0):
    """per TPC: slot time -> (hit index array of the decidable clusters at that time, member labels)."""
    rng = np.random.default_rng(seed)
    fam: dict[int, dict[int, tuple[np.ndarray, list[int]]]] = {}
    for L in decidable:
        t = ev.assign.get(L)
        if t is None:
            continue
        for c in ev.lab_tpcs[L]:
            idx = ev.lab_hits[L]
            idx = idx[ev.tpc[idx] == c]
            if idx.size == 0:
                continue
            if idx.size > cap:
                idx = rng.choice(idx, cap, replace=False)
            d = fam.setdefault(c, {})
            key = None
            for ts in d:
                if abs(ts - t) <= SAME_TICKS:
                    key = ts; break
            if key is None:
                d[t] = (idx, [L])
            else:
                d[key] = (np.concatenate([d[key][0], idx]), d[key][1] + [L])
    return fam


def family_distances(ev: Event, L: int, fam, cap: int = 200, seed: int = 0):
    """{slot time: min distance [cm] from L's hits to that family's hits}, over L's TPCs; the
    cluster's own hits never count (families made of L alone are skipped)."""
    rng = np.random.default_rng(seed)
    out: dict[int, float] = {}
    for c in ev.lab_tpcs[L]:
        if c not in fam:
            continue
        idx = ev.lab_hits[L]
        idx = idx[ev.tpc[idx] == c]
        if idx.size == 0:
            continue
        if idx.size > cap:
            idx = rng.choice(idx, cap, replace=False)
        P = ev.xyz[idx]
        for ts, (fidx, members) in fam[c].items():
            if members == [L]:
                continue
            fidx2 = fidx if L not in members else fidx[ev.labels[fidx] != L]
            if fidx2.size == 0:
                continue
            d, _ = cKDTree(ev.xyz[fidx2]).query(P, k=1)
            dm = float(d.min())
            out[ts] = min(out.get(ts, np.inf), dm)
    return out


def build_uprior(ev: Event, labs, decidable: set[int], K: dict[int, list[int]], fam) -> None:
    """unary spatial prior for undecidable clusters from the in-situ tables (chi2 units)."""
    ev.uprior = {}
    for L in labs:
        if L in decidable or L not in K:
            continue
        if ev.dtable is None and L not in ev.blind:
            continue
        eb = e_bucket(ev.lab_E[L])
        dist = family_distances(ev, L, fam)
        pen = {}
        for t in K[L]:
            best = None
            for ts, d in dist.items():
                if abs(ts - t) <= SAME_TICKS:
                    if ev.dtable is not None:
                        v = ev.dtable.get(f"{eb}_{d_bucket(d)}")
                    else:
                        v = d                      # no table: nearest family wins (blind clusters only)
                    if v is not None and (best is None or v < best):
                        best = v
            if best is None:
                best = (ev.pnew or {}).get(eb, 0.0) if ev.dtable is not None else 1e6   # own time forbidden when blind & no table
            pen[int(t)] = float(best)
        ev.uprior[L] = pen


def prior_observations(ev: Event, decidable: set[int]):
    """in-situ statistics for the tables: for every decidable cluster and every OTHER family in
    its TPCs -> (E bucket, distance bucket, same time?); and (E bucket, has no family at its
    time?) for the new-slot rate."""
    fam = build_families(ev, decidable)
    dobs, nobs = [], []
    for L in decidable:
        t = ev.assign.get(L)
        if t is None:
            continue
        eb = e_bucket(ev.lab_E[L])
        dist = family_distances(ev, L, fam)
        if not dist:
            continue
        is_new = True
        for ts, d in dist.items():
            same = abs(ts - t) <= SAME_TICKS
            if same:
                is_new = False
            dobs.append((eb, d_bucket(d), bool(same)))
        nobs.append((eb, bool(is_new)))
    return dobs, nobs
