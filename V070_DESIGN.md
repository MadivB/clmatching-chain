# v0.7 — joint probabilistic small-cluster assignment  **[ADOPTED 2026-08-06]**

## ADOPTION (126 events / 10 files, jobs 56415736 + 56416591)

| | v0.6 | **v0.7 hybrid** |
|---|---|---|
| NS total-eff | 96.05% | **96.19%** (+0.14pp paired, 96/30; ALL 10 files better) |
| all-TPC eff | 95.06% | 95.07% (no regression) |
| time/event | 268 s | **233 s** (-13%) |

Canonical v0.7 flags = v0.6 flags with `--phase3 joint` (implies the hybrid:
the mean-field solver owns clusters in TPCs <= `--joint-tpc-emax` 1400 MeV,
the v11 matrix keeps shower TPCs; expansion retired from the canonical chain).
Implementation: `testing/v070_joint.py`. Development history: iter-1
(prior annealed away -> matrix-like), iter-2 (energy-units prior, +NS but
shower-TPC regression), iter-3 (hybrid split, both metrics clear).

# v0.7 design — joint probabilistic small-cluster assignment
### (formalizing the "[t, p(t)|S, p(t)|L] iterative array" idea, 2026-08-05)

## 1. The user's idea, formalized

Every cluster i carries a distribution over candidate placement times instead
of a single greedy choice; light and space each contribute a likelihood; an
iterative step reshapes all distributions together until a collective loss is
minimized, so swaps emerge dynamically instead of being enumerated.

Written as a posterior over the joint assignment T = (t_1..t_N):

    P(T | A, X)  ∝  P(A | T) · P(T | X)

- **Light likelihood** P(A|T): the waveform model A ≈ Σ_i s_i(t_i) + noise
  with s_i(t) = cluster i's template shifted to t. With our fixed Poisson
  weights w = 1/(200·max(A,100)) this is exp(−χ²/2) with the chain's own loss.
- **Spatial prior** P(T|X): geodesically connected clusters prefer a common
  time — a pairwise (Potts-like) coupling
  J_ij = λ·exp(−d_geo(i,j)/d0) rewarding t_i = t_j, plus per-cluster priors
  anchoring backbone placements.

The user's per-cluster array [t, p(t)|S, p(t)|L] is exactly the **mean-field
marginal** q_i(t) with its two log-likelihood components; the "iterative step"
is the mean-field / annealed update:

    q_i(t) ∝ exp(−β·ΔL_i(t | q_−i))          · exp(+β·Σ_j J_ij·q_j(t))     · exp(−β·κ_i·[t ≠ t_i^prior])
             └── light: p(t)|L given the      └── space: p(t)|S given the   └── backbone prior, weakened by
                 EXPECTED base image of           soft "family" F(t) =          LOW per-cluster confidence
                 everyone else's mass             neighbors' mass at t          (enables backbone correction)

## 2. The key structural fact: the light loss is exactly pairwise

Because the Poisson weights depend only on the measured waveform (not on the
assignment), expanding the χ² gives — exactly, up to the saturation clip —

    χ²(T) = const + Σ_i a_i(t_i) + Σ_{i<j} C_ij(t_i, t_j)

    a_i(t)      = Σ w·s_i(t)·(s_i(t) − 2·R_fixed)      (R_fixed = A − B_fixed)
    C_ij(t,t')  = 2·Σ w·s_i(t)·s_j(t')                 (template overlap)

with B_fixed = the frozen high-confidence backbone light. So the collective
problem is a **quadratic assignment / sparse pairwise MRF**, not a black box:
C_ij is nonzero only for same-TPC pairs whose supports overlap and slot pairs
within a template length. Mean-field, belief propagation, and annealing all
apply *rigorously*; no re-rendering of waveforms inside the loop — a_i and
C_ij are precomputed once per event.

This also explains, in one line, everything the chain already learned:
- pairwise **swaps** are the C_ij cross terms the greedy chain cannot see;
- **expansion** = spatial term with hard light deltas; **matrix** = a_i only;
- **swap-fix / ejection chains** = hand-coded local moves on this energy;
- v0.5.0's failure = collapsing this structured energy into one signed scalar.

## 3. Algorithm sketch (v0.7-b prototype)

1. **Slot set** per cluster: registered t0Candidates of its TPC + its own
   top-k scan minima (k≈3) + slots of geodesic neighbors → K ≈ 10–30 slots.
2. **Precompute** per event: a_i(τ) for every (cluster, slot); sparse
   C_ij(τ,τ') for coupled pairs; J_ij from the expansion hit-graph Dijkstra;
   priors κ_i from per-cluster confidence (confidence_cluster components) —
   backbone strong, P2-class medium, tail zero.
3. **Initialize** q from the current chain output (softened deltas) — warm
   start keeps the solver in the right basin and makes it a *refiner* first.
4. **Anneal**: damped mean-field sweeps at decreasing temperature; the
   collective loss E(q) = Σ q·a + ΣΣ q·C·q − spatial − T·H(q) is monotone
   under damping; stop at ΔE < tol per sweep.
5. **Harden**: argmax per cluster → exact clipped-χ² greedy polish of the
   winners (full-range rescan, clip restored) → police probe → the residual
   no-template clusters keep the current force path (they have no light term;
   their spatial term alone already reproduces geodesic attach).
6. **Audit** exactly like every stage: cohort accuracy vs truth, per-swap
   fixed/broke bookkeeping, energy budget.

## 4. What must be validated (the honest risk list)

- **Local minima**: annealing schedule + warm start; add 2–3 perturbation
  restarts on the worst-loss TPCs only.
- **Clip nonlinearity**: ignored in the soft phase (few % of channels),
  exact in the polish; saturation-vetoed channels already excluded.
- **Slot discretization**: 1-tick precision recovered by the final rescan.
- **Spatial calibration**: λ, d0 fitted from truth on the 126-event set —
  P(same interaction | d_geo) is directly measurable from the dumps.
- **Runtime**: sweeps are sparse mat-vec ops; target ≤ the ~45 s the
  expansion+matrix pair costs today; torch-GPU if numpy falls short.
- **Backbone opening** (v0.7-c) is the riskiest step: keep κ high until the
  small-cluster solver is validated, then lower κ for the LOW-confidence
  backbone tail only (confidence AUC 0.69–0.94 says which).

## 5. Staged plan

| step | content | gate |
|---|---|---|
| v0.7-a | second matrix pass, relaxed worsen tolerance (running) | NS on file 0000001 vs 96.02 |
| v0.7-b | joint solver replacing expansion+matrix+force for the imaged tail; backbone frozen (`--phase3 joint`) | tail cohort acc vs 70.4% / force 54.6%; NS ≥ 96.0 |
| v0.7-c | confidence-weighted backbone priors (correction enabled) | net backbone fixed-vs-broke > 0 |
| v0.7-d | vectorization/GPU + 10-file statistics | ≤ v0.6 runtime; NS > 96.05 |

Fallback at every step: frozen stable release at
/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.6 (read-only,
includes the frontend snapshot + commit); the working tree keeps every v0.7
mechanism flag-gated with v0.6 defaults.
