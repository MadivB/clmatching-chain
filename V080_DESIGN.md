# v0.8 design — iterative light–spatial refinement with a structural spatial score
### (user scheme, 2026-08-06)

## The scheme

Coordinate descent between two views of the same assignment, at two
granularities:

```
PASS 1  LIGHT  (cluster level)   : v0.6/v0.7 placement, RECORDING each
                                   cluster's loss-delta curve  a_c(t)
                                   ("delta_t at every candidate t")
PASS 2  SPATIAL (family level)   : score every cluster against every nearby
                                   family with a STRUCTURAL spatial metric
                                   (contact, one-piece vs crossing,
                                   directionality, shower cone, multi-TPC);
                                   TRADE clusters between families (not 1:1),
                                   light curves a_c(t) as prior/veto
PASS 3  LIGHT  (family level)    : families as rigid super-clusters — move
                                   whole families, family<->family swaps,
                                   +-few-tick tuning  (attacks the measured
                                   #1 failure mode: family-level t0 swaps)
PASS 4  SPATIAL again            : re-trade on the improved geometry
...iterate until the event "regains a feasible shape"
```

Backbone correction happens implicitly: a family move in PASS 3 carries its
backbone members; a PASS 2 trade can pull a mis-clustered piece out of a
backbone family.

## The spatial score S(c, F) — requirements (user)

1. **Multi-TPC with directionality**: understands what a shower IS (vertex,
   axis, cone, propagation direction) across TPC boundaries in global (x,y,z).
2. **Contact ≠ one piece**: two crossing tracks touch but are two pieces; a
   track through a shower likewise. PCA-based tests for whether structures
   "emerge from the same space" (common vertex) vs merely intersect.

## Candidate features (Task 1 — to be validated on truth BEFORE any physics)

Per cluster: energy-weighted PCA (centroid, axes, eigenvalues, linearity,
length, endpoints). Per family: member PCAs + family axis + a vertex estimate
(both axis ends tried; directionality from longitudinal energy profile).

| feature | captures |
|---|---|
| d_eu, d_geo (3-cm linking) | contact / connectivity (P≈0.99 certificate when linked) |
| continuation: endpoint gap + collinearity + lateral offset vs nearest member | broken track = one piece |
| crossing flag: contact + large axis angle + member continues on BOTH sides | in contact but TWO pieces |
| vertex pointing: DCA of c's backward axis to family vertex | "emerges from the same space" |
| cone: angle of (μ_c − vertex) vs family axis; longitudinal position | shower membership + propagation |
| light prior: a_c(t_F) − min_t a_c(t) | the recorded delta_t curve (used in-chain, not in the truth study) |

Combined score: logistic on the validated features (interpretable, monotone).

## Development order (user-prescribed)

1. **Task 1**: implement features (`testing/spatial_score.py`); build a
   truth-labeled (cluster, family) dataset from the v0.6/v0.7 dumps; report
   per-feature and combined separation (AUC), overall and per energy bin;
   plus the operational test — detecting actually-misplaced clusters in the
   v0.7 output. GATE: combined AUC and misplacement-detection precision
   must clearly beat the geodesic-only certificate before anything moves.
2. **Task 2**: PASS 2 trading engine (score + light-prior veto, ordered by
   margin, not 1:1), truth-audited fixed/broke ledger.
3. **Task 3**: PASS 3 family-level light (rigid-family FFT scans, family
   swaps, tick tuning), truth-audited.
4. **Task 4**: iteration loop + convergence criterion; full-chain A/B
   (both metrics: NS AND all-TPC) on file 0000001, then 10-file gate.

Safety: v0.6 frozen (CLMatching_v0.6) and v0.7 tagged on GitHub; every v0.8
mechanism flag-gated off by default.
