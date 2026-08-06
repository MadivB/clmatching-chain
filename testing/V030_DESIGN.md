# v0.3.0 — Family expansion for small clusters (design)

Research-backed design (deep-research 2026-07-07: 25/25 claims verified 3-0;
primary framework = GAP/ALNS move-based local search; alpha-expansion+label-costs
noted as a future global polish). Target: the 11.6% of event energy sitting in
~1300 unassigned <50 MeV clusters/event + 1.3% noise hits.

## Objects

**Family** = union-find of PLACED clusters over (|Δt0| ≤ 2 ticks AND shared
TPC). Carries: t0_f (energy-weighted), member hit cloud, TPC set. This is the
"gradually reconstructed interaction".

**Candidate** = unassigned cluster (has a perceiver image; production
prediction threshold restored so small clusters are predicted). Noise hits
(label −1) are attached at the very end by pure geometry (no image).

## Cost of putting candidate c into family f

    score(c,f) = Δχ²_pois(c's image added at t0_f | current base) / E_c
               + β · d_geo(c,f)                       [β ≈ 0.05 / cm]

- Δχ² on c's support channels, prediction clipped at saturation — negative
  when c's light explains residual light at t0_f (the physics driver).
- d_geo = **geodesic distance through the hit cloud** from f's hits to c's
  hits (radius graph r=3 cm over all hits, multi-source Dijkstra). This is the
  verified cheap proxy for "transverse distance to the nearest track": it
  follows multi-prong skeletons where PCA/ellipsoids fail. Gate: d_geo ≤ 20 cm.
- Assign only if score < 0 (leaving unassigned costs 0 — conservative).

## Order of assignment: regret-k insertion (verified OR result)

At each step assign the candidate with the LARGEST regret = score(2nd-best
family) − score(best family): items with the most to lose commit first; truly
ambiguous items (regret ≈ 0) wait, and if still ambiguous at the end they are
left for the error-matrix fallback. Processed in two energy bands (≥10 MeV
first) so bright items shape the families early. Base image and family hit
clouds update after every acceptance (geodesics only shrink; refreshed lazily
per family after 20% growth).

## Q1 — stopping rule: 2-dissent hysteresis (user spec)

A frontier candidate whose LIGHT term clearly prefers a different family
(margin δ) is a "dissent": it is skipped (left for its preferred family), the
family's dissent counter increments. At 2 consecutive dissents the family's
expansion FREEZES (substitution still allowed). Acceptance resets the counter.
We never stop on "light looks full" (ML amplitude systematics, later backbones
may still arrive — user constraint).

## Q2a — substitution (ejection chains, bounded)

If adding c to f fires the overflow gate (locked trigger: ≥3 ch excess ≥500,
z_pois > 2 at t0_f), try evicting the most RELUCTANT expansion members of f
(largest own Δχ² at f, best alternative elsewhere; never evict ≥50 MeV
backbone seeds): chains of depth ≤ 2 (evict m → insert c → re-place m at its
2nd family or unassign). Accept iff the true summed loss improves. Oscillation
guard: O(1) timestamp tabu — an evicted (label, family) pair may not return
for 5 iterations (verified GAP tabu practice).

## Q2b — missed flashes: new-t0 opening (label cost)

A candidate with E ≥ 15 MeV whose OWN full-range FFT scan finds a minimum
that beats its best family score by a label cost κ AND sits >10 ticks from
every family t0 opens a NEW family at that t0 (registered as a t0 candidate).
Conservative: κ prices the "activate a new flash" hypothesis (verified
label-costs idea); later candidates may join the new family.

## <50 MeV tail strategy (user spec)

After expansion, everything still unassigned goes to the EXISTING error-matrix
Phase 3 (`run_small_cluster_matrix_phase_v11`) — no permutation search for the
small tail. Measured variants: (A) matrix only [old way], (B) expansion only,
(C) expansion + matrix tail. All after the full v0.2.5 chain (stack2 + rescue
+ P2/V2 + police), metric = energy-weighted accuracy AND assigned fraction
(the point of v0.3.0 is coverage at good purity).

## First-run compute posture

Slow-but-correct: full radius-graph Dijkstra per family refresh, direct Δχ²
per (candidate, family) pair at one shift (no FFT except new-t0 scans),
production-level prediction of all small clusters. Caching: per-candidate
support rows; per-family loss deltas recomputed only against the mutated base.
