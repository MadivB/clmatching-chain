# CLMatching v0.4.1 — full algorithm description (2026-07-15; v0.4.0 defined 2026-07-10)

Canonical flags (testing harness):
`--backbone stack2 --merge-p2 --sigma-mode poisson --max-clip-ticks 4
--police locked --merge-vertex --phase3 both --force-coverage --swap-fix
--swap-ratio-max 0.85 --p2-threshold 30 --predict-min-energy 1.5`

v0.4.1 changes vs v0.4.0 (each A/B-validated on the 39-event set):
**stages 3.1+3.3 MERGED** (`--merge-p2`): every P2-class cluster is placed by
the permutation pool, and the 3.2 corrections run after ALL big placements
(NS 95.78 vs 95.73, 22/17 paired, −12 s/event); **plain expansion score**
(raw clipped-Poisson Δχ², tie vs the per-MeV+geodesic score); defer-gate
fixes in the pool (stale-t0 commit chain, true singleton deferral); full-model
trigger in 3.2; merged placements registered as assigned flash slots.

Benchmarks (39 MiniProdN5p1 events, energy-weighted, |reco−truth| ≤ 10 ticks):
non-shower TPCs (≤ 1.4 GeV, 67% of energy) **95.8% total efficiency at 100%
coverage**; all-TPC final ~95.0%; shower TPCs (> 1.5 GeV) 93.3% (dedicated
path shelved 2026-07-10 after redo/refine attempts; standard chain handles
them). Compute: **133 s/event/GPU** (2 workers/GPU) — inside the 150 s budget.

Core loss everywhere: Poisson-weighted chi², σ²_ADC = 200·max(actual,100)
(g = 200 ADC/photon), prediction clipped at the 60780 ADC saturation ceiling,
evaluated on support channels; full-range shifts scanned at once by FFT.

---

## STAGE 1 — Clustering & ML prediction

1.0 Inputs: calibrated prompt hits (x, y, z, E, io_group → TPC) and the
    120-channel × 1000-tick light waveform per light-TPC.
1.1 **Global track clustering** (RANSAC toolbox): labels + types
    (track / shower / cluster), track-vs-backbone split; includes the
    length-gate fix and track noise absorption (transverse radius, scale 1.5).
1.2 **Vertex-track merge** (v0.2.5): track clusters whose ENDS converge on a
    common origin are merged into one multi-prong object. Five
    fragmentation-biased gates: end-to-end ≤ 4 cm; both end-lines pass within
    2.5 cm of the vertex; opening angle 15–165°; crossing guard (endpoint vs
    other track's core ≥ 3 cm); local end linearity ≥ 0.75. Union-find
    groups; > 6 prongs dropped whole. (97% merge precision.)
1.3 **Perceiver light prediction** (GPU) for every cluster ≥ 1.5 MeV → per
    (cluster, TPC) light-image template.
1.4 **Support channels**: per cluster-TPC, the smallest channel set holding
    90% of its predicted light (gap bridging ≤ 2), minus the saturation veto
    (channels clipped > 4 ticks are excluded from every scan and probe).

## STAGE 2 — Phase 1 assignment + STAGE 3.1 (merged permutation pool)

2.1 Showers placed first (standard machinery).
2.2 Tracks crossing ≥ 3 TPCs placed by the standard scan — the most
    constrained objects commit early.  (End of Phase 1 proper.)
3.1 The MERGED pool (`--merge-p2`, since v0.4.1): the remaining 1–2-TPC
    backbone tracks AND every P2-class cluster — multi-TPC or > 30 MeV with a
    template (single-TPC ≤ 30 MeV backbone tracks ride along — known
    impurity): **energy-banded permutation batches**:
    seed = most energetic unplaced; batch = pool mates sharing a TPC with
    E > 0.4·E_seed (s = 0.6), cap 5; try ALL orderings, each placing greedily
    at the scan optimum given base-so-far; scan score is cosine-softened:
    loss × (1 + (1 − cos(own peak pattern, unexplained light))) — penalise,
    never forbid. Best ordering by union-support loss commits its first N−1
    placements; the LAST-placed returns to the pool for the next band.
    **Defer gate**: a commit that fires the overflow trigger is rejected once.
    A rejected mid-batch member invalidates the rest of its batch (they are
    re-batched, not committed at stale t0s); a rejected mate-less seed is set
    aside and re-placed after the pool drains (true deferral).
    Clean-up passes: **swap** (shared-TPC pairs exchange t0s when the pair's
    union loss improves) and **trigger rescue** (flagged placements rescanned
    against the finished backbone).

## STAGE 3.2 — Overflow trigger + saving algorithm  [correction]

Runs after ALL big placements (the old "move 3.2 behind 3.3" — automatic with
the merge). Trigger (locked, at probe tick t0+105, per TPC, evaluated against
the FULL stacked model): ≥ 3 channels with excess (pred − actual) ≥ 500 ADC
and z_pois > 2, OR cos(own, residual) < 0.75. Merged P2-class placements are
probed, flagged, swapped and rescued exactly like backbone tracks.
For each flagged placement, by slot population (|Δt0| ≤ 5, shared TPC, ≤ 3):
- **Stage A** (N ≥ 2): anchored permutations — one member keeps the slot, all
  orders tried, others full-range rescanned; adopt the best arrangement only
  if it beats everyone-stays.
- **Stage B** (all-stay): distrust the cross-TPC stitch — cut the firing
  TPC's part off the merged track under a NEW cluster id; it re-associates
  downstream at its own energy scale.
- **N = 1** (lone): the claimed flash may not exist — unanchored full-range
  rescan (move whole) or part-detach + remainder rescan; fair union-loss
  comparison with every piece placed.

## STAGE 3.3 — Phase 2 large-cluster scan + police

With the merge, this scan only places what 3.2 detached (Stage B / N=1 parts
get NEW cluster ids) plus any imaged stragglers; the merged pool owns the
P2-class population. Boundary 30 MeV (was 50): any multi-TPC cluster, or
single-TPC cluster > 30 MeV, is Phase-2-class. **Placement police**: every
new placement is probed with the trigger the moment it lands; on alarm it is
resolved immediately — keep / rewind / unanchored rescan, whichever minimises
the clipped-Poisson union loss (in practice the rescan always beats rewind).

## STAGE 3.5 — V2 spatial + light rescue + police  [correction]

Default V2 configuration (its knobs are saturated); every V2 move is policed
identically to Phase 2 placements.

## STAGE 3.9 — Spatial family expansion (v0.3.0; part of Phase 2)

Families = union-find of ALL placed clusters over (|Δt0| ≤ 2 ticks AND shared
TPC) — the reconstructed interactions. Candidates = still-unassigned clusters
with templates (≥ 1.5 MeV).
- score(c, f) = Δχ²_pois(add c's light at t0_f | current model) — the regular
  pipeline loss (plain, since 2026-07-15; the original per-MeV + 0.05/cm
  geodesic-distance score is kept as `--expand-score special`; A/B on 39
  events was a statistical tie, NS 95.73 both). Candidate–family
  **eligibility** still requires geodesic reach ≤ 20 cm through the
  3-cm-radius hit graph (multi-source Dijkstra) — the
  transverse-distance-to-track proxy that handles multi-prong shapes.
- **Regret-k insertion**: assign next the candidate with the largest
  best-vs-second-best gap; two energy bands (≥ 10 MeV first); accept only if
  score < 0 (leaving unassigned costs nothing).
- **Dissent rule**: a candidate whose light clearly prefers a different
  reachable family is banned from this one (stays available elsewhere); two
  consecutive dissents freeze the family's growth.
- **Ejection chains** (depth ≤ 2, timestamp tabu): an overflow-blocked insert
  may evict the most reluctant expansion member (never backbone seeds) if the
  total score improves.
- **New-t0 opening**: a ≥ 15 MeV candidate whose own scan beats every family
  by the label cost, > 10 ticks from all family t0s, opens a new family
  (covers flashes missed by flash finding).
Assigned candidates are purged from the matrix worklists.

## STAGE 4.1 — Phase 3: error-matrix association (the tail)

The v11 small-cluster matrix association sweeps everything expansion left
(single-TPC ≤ 30 MeV clusters ≥ 0.5 MeV) — the "old way", still the best
tool for the ambiguous tail. No cosine anywhere in the small-cluster path.

## STAGE 4.5 — Per-TPC swap fix  [correction]

Co-located cluster-part pairs (per charge-TPC) whose JOINT t0 exchange lowers
the support-masked per-TPC light chi² to < 0.85 of the current value AND
whose parts both spatially prefer the swap are exchanged (≤ 3 passes,
min part energy 25 MeV). Fixes interactions that traded flashes wholesale.

## STAGE 4.9 — Force coverage (coverage mandate: 100%)

Everything still unassigned is placed:
- clusters WITH templates: at the better (absolute loss) of the nearest
  families' t0 vs their OWN full-range scan minimum — a loss increase is
  accepted (correct placements can raise chi² under ~20–30% ML amplitude
  systematics);
- clusters WITHOUT templates (< 1.5 MeV): geodesically nearest family's t0
  (≤ 30 cm), Euclidean-nearest assigned hit as fallback;
- noise hits (label −1): nearest family through the cloud (≤ 15 cm), else
  nearest assigned hit's t0.

---

## Provenance of each design decision

| stage | validated by |
|---|---|
| vertex merge | 97% precision both samples; +0.05pp held-out |
| stack2 ordering | 95.1 → 96.4 phase-1 (s-sweep, permutation audit) |
| trigger + A/B/N=1 | 64%-fix Stage A; 37/37 N=1 moves; +1.3pp backbone |
| police | +0.24pp, P2 cohort 92.5 → 94.0; rewind never beat rescan |
| boundary 30 / predict 1.5 | monotone boundary sweep; budget scan 144 s/GPU |
| expansion + matrix + force | 93.2 → 95.6 NS total-eff at 100% coverage |
| swap 0.85 | 21 good / 6 bad (purer than 0.9 at equal efficiency) |
| merged 3.1+3.3 pool | A/B 39 ev: NS 95.78 vs 95.73 (22/17 paired), −12 s/event |
| plain expansion score | A/B 39 ev: dead tie (95.73/95.73) — simpler wins |
| shelved: shower-TPC path, shower absorb, support < 0.90, relax pass | measured null or negative |
