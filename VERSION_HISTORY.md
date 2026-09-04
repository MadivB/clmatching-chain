# Charge-Light Matching — Version History

*The evolution of the pre-reconstruction charge-light matching chain.
Success = |reco − truth| ≤ 10 ticks (±160 ns), energy-weighted.
Headline metrics: **NS** = non-shower TPCs (≤1.4 GeV truth) at forced 100 %
coverage; **all-TPC** = every TPC. 126-event reference sample = files
0000001–0000010. Adopted versions are tagged on `MadivB/clmatching-chain`.*

## The ledger (adopted versions, 126 events unless noted)

| version | date | NS | all-TPC | one-line summary |
|---|---|---|---|---|
| v0.2 | 2026-07 | 97.25* | — | permutation placement + police (39-ev, pre-fix numbers) |
| v0.4.0 | 07-10 | 95.7 | — | 4-stage architecture defined, per-stage instrumentation |
| v0.4.1 | 07-15 | 95.78 | — | merged permutation pool (3.1+3.3) |
| v0.6 | 07-2x | 96.05 | 95.06 | external frontend clustering (96 % target met) |
| v0.7 | 08-0x | 96.19 | 95.07 | joint mean-field simultaneous assignment |
| v0.8 | 08-06 | 96.49 | 95.49 | PASS 2 spatial trading |
| v0.8.1 | 08-07 | 96.53 | 95.55 | light-nominated trade candidates |
| v0.8.3 | 08-11 | (output mode) | | confidence hit-pruning, neutron-safe |
| v0.8.5 | 08-13 | 96.46† | 95.65† | 4×-smaller amplitude perceiver (−33 % stage 1) |
| v0.8.6 | 08-14 | 96.47 | 95.60 | + skip V2 → **115 s/ev single-node latency** |
| v0.8.7 | 08-14 | 96.46 | 95.61 | support-restricted overflow gate (semantics fix) |
| v0.8.8 | 08-14 | 96.49 | 95.64 | stage 5: LE blip re-timing (LE class 58.7→60.9) |
| v0.8.9 | 08-23 | 96.50 | 95.65 | 895 t0 window + edge penalty (2x2-ported boundary-hiding guard) |
| v0.9 | 09-02 | 96.50 | 95.65 | continuous t0s (--subtick): sum-mode stage E, sigma 9.2->8.6 ns, sigma_t confidence output; 61.8 s/ev/GPU |
| v0.9.2 | 09-02 | 96.51* | 95.71* | calibrated 0-1 t0 confidence (hit_t0_conf): quadratic-fit + waveform evidence, LOO-AUC 0.881, reliability 22->99.8% |
| v0.10.0 | 09-02 | 96.50 | 95.65 | cluster-level 3-family confidence (--conf-cluster): spatial 40f + group 10f + scan-landscape 8f, HGB+isotonic, LOO-AUC 0.850 (0.970 E-wt); +4.2 s/ev median |
| v0.10.1 | 09-03 | 96.50 | 95.65 | confidence goes RULE-BASED (4 JSON thresholds on scan-landscape: rel_margin/pcu_vote10/pcu_sig_med/c_cosres, 5-level purity lookup): E-wt AUC 0.961, 0.78 s/ev, no ML in data path |
| v0.10.2 | 09-04 | 96.50 | 95.65 | lean product confidence (user design): pcu_vote10 x 2 side-help penalties + 8-knot purity calibration; correct median 0.88 (none <0.3), E-wt AUC 0.933, 0.71 s/ev |
| v0.10.3 | 09-04 | 96.50 | 95.65 | ZERO-MC confidence: conf = pcu_vote10 x rel_margin/(1+rel_margin) (witness agreement x rival-defeat posterior, no tuned numbers); E-wt AUC 0.955, 0.74 s/ev |
| **v1.0** | 09-04 | 96.50 | 95.65 | **FIRST RELEASE**: conf = pcu_vote10 directly (zero parameters, data-only); 3-file confirm: out-of-sample AUC 0.777/0.936 E-wt, reading key 33->99% purity, 0.65 s/ev; run_v1.sh |

\* 39-event sample, pre length-gate fix — not comparable to later numbers.
† file-0000001 same-node pair; 126-ev v0.8.6 row is the certified one.

**2026-08-15 chain-wide corrections (no version bump):** (1) t0 search
window 700/800 → **895** everywhere (user correction; upper edge = 1000
ticks − PEAK 105), applied to all clones incl. frozen v0.2/v0.4.1/v0.6,
AlphaRelease core, and the frontend; file-1 verification: metrics
unchanged within noise (NS 96.49/95.70, LE 58.25), +2.4 % latency (the
12 % longer scans), zero assignments >800 on this file. Required an
off-by-one fix in the cos matched-filter family (peak past waveform end
→ cos 0). (2) NERSC refresh killed the harness python (torch gone,
user-site wiped) → all launch scripts repointed to the pytorch/2.8.0
module python; the resulting 5× slowdown was an uncached failed
`import cupy` per call in `pulse_shapes._get_xp` (~115k stat/ev) —
probe now cached, 141 s/ev restored. (3) Eager shower absorption
(`--shower-eat`, unversioned candidate) measured NULL — showers are 86 %
light-fulfilled at recognition; see SHOWER_EAT_DESIGN.md.

---

## v0.1 – v0.3: foundations

- **v0.1 backbone-only** — fresh-clone baseline; stack2 backbone, quick-test
  mode. The working tree all later versions grew inside.
- **v0.2** (frozen `CLMatching_v0.2`) — energy-banded permutation placement
  (s=0.60), the overflow **defer gate** ("locked" trigger: cos<0.75 OR ≥3
  channels excess≥500 & z>2), placement police, swap + rescue passes.
  Post-V2 97.25 on the old 39-event sample.
- **v0.3.0** — bad-TPC event-display study: **91 % of residual wrong energy
  was pairwise family t0 SWAPS** (this diagnosis drove v0.7/v0.8 design);
  family expansion for the <50 MeV tail; error-matrix tail stage.

## v0.4: architecture

- **v0.4.0** (07-10) — the 4-stage numbering the chain still uses:
  1 clustering+ML → 2 Phase-1 (showers, ≥3-TPC) → 3.x Phase-2 (3.1 backbone
  pool, 3.2 trigger+rescue, 3.3 P2 scan, 3.5 V2, 3.9 expansion) → 4.x
  Phase-3 (4.1 matrix, 4.5 swap-fix, 4.9 force). Per-stage timings in
  `row["stage_t"]`. Shower-TPC dedicated path tested and SHELVED.
- **v0.4.1** (07-15, frozen) — **merged pool** (`--merge-p2`): every P2-class
  cluster (>30 MeV or multi-TPC) goes through the same all-orderings
  permutation pool as the backbone; corrections after all placements.
  NS 95.78.
- **Cheating study** (07-15) — backbone is SOLVED (perfect type-1 worth only
  +0.5pp); the potential lives in type-2 merged clusters (~1.7pp) and
  type-3 small clusters (~2.6pp). Guided everything after.

## v0.5: the loss experiment and the release

- **v0.5.0** (07-16) — user's signed/cubic loss idea. TESTED AND REJECTED:
  both arms lose ~1.5pp (|ΣL| balance is degenerate to cross-channel
  cancellation). Lesson: never collapse the per-channel comparison.
- **v0.5.6** (travel release) — 10 flow-file `.pt` outputs + per-cluster
  confidence + offline clustering-tuning package (`CLMatching_v0.5.6_release`).

## v0.6: the external frontend  *(96 % target met)*

- User's `clmatching-frontend` repo wired in via `frontend_adapter.py`
  (`--frontend-repo`). Full stats: **NS 96.05 / all-TPC 95.06** — +0.44pp
  over builtin, every file better. Frozen (`CLMatching_v0.6`), pushed to
  GitHub with the chain (`clmatching-chain` + submodule). TDR plot set
  regenerated ("Interaction" naming, ±10-tick standard).
- **v0.6.2** (08-13) — v0.6 flags + the 4×-smaller perceiver: 95.97/95.17
  (noise-level vs big), stage 1 −40 %.

## v0.7: joint assignment  *(user's [t, p|S, p|L] idea)*

- The light χ² is EXACTLY pairwise-quadratic in assignment one-hots →
  sparse MRF/QUBO, annealed mean-field solver (`v070_joint.py`). Unaries
  from one FFT scan per cluster; geodesic-connectivity prior **in energy
  units** (temperature-free priors get annealed away — measured lesson);
  hybrid split: solver owns TPCs ≤1.4 GeV. **NS 96.19 / 95.07.**
  (all-TPC guard added after an iter-2 regression caught ONLY by the
  user's insistence on reporting both metrics.)

## v0.8 family: the light↔spatial loop  *(user's Q1/Q2 philosophy)*

- **v0.8** (08-06) — **PASS 2 spatial trading** (`--pass2-trade`): truth-
  validated spatial score v4 (typed member PCA, vertex, cone, best-member
  pooling; HGB), KD prefilter, light-curve veto, margin-ordered non-1:1
  trades. **96.49 / 95.49** (125/1 events). Q1-SWAP built and measured
  DEAD: 0 true swap pairs in 126 events — the v0.3 swap failure mode was
  eaten by merged pool + joint solver.
- **v0.8.1** (08-07) — spatial-score campaign (user's SPATIAL_SCORE.md
  reimplemented + extended; truth AUC 0.930 but operational transfer
  fails — all scorers ~0.72, v4 stays). Census: candidate generation was
  the binding constraint (57 % reachable / 92 % exists / 8 % no-slot) →
  **light-nominated candidates**: families at the cluster's own curve
  minima join the candidate set. **96.53 / 95.55** (114/10 events;
  via-light trades FIXED 740 / BROKE 164). `tpc_adjacency.json` built
  (7×5 module grid ×2 drift; gaps 5–6.5 cm).
- **v0.8.2** (08-07) — loop-deepening: iteration ×3, full Q1↔Q2
  alternation, symmetric light-proposal. ALL NULL — the loop is at a
  fixed point after one pass. Headroom is outside the loop.
- **v0.8.3** (08-11/12) — **confidence hit-pruning output mode**
  (`--prune-frac`). Charge+light badness; user requirement simplified it
  to 4-5 thresholds; user physics constraint (neutron deposits 0.5–6 MeV
  must stay assigned) set the discard gate: final rule **E<2 MeV AND all
  of [d_eu>20 cm, dz>0.25, z_own>−2]** — 0.34 % of hits, assigned-acc
  +0.30, purity 50 %, nothing ≥2 MeV ever unassigned. "Tiny" defined by
  measurement: light-blind <3 MeV, weak 3–20, trustworthy ≥20 MeV
  (matches k*(E)=0 at 20.2 MeV).
- **v0.8.4** (08-12) — decline-or-repair side path. NULL with a lesson:
  "fits nowhere" selects broken TEMPLATES, not misplacements (85–93 % of
  declined clusters were correctly placed) — misplaced clusters by
  definition fit well somewhere (that low-r signal IS PASS-2 nomination).
- **v0.8.9** (08-23, tag) — **895 t0 window + edge penalty**. The window
  correction (user; upper edge = 1000 − PEAK) naively opens boundary
  hiding: predicted light shifted past the window end leaves the χ²
  (FFT zero-padding), so no-match clusters sink at t0=895 (synthetic:
  −53k below interior). Guard ported from the user's 2x2 development
  (HANDOFF_ND_threshold_association.md): hidden template columns charged
  against the channel quiet-tail weight, suffix-summed, in every
  weighted_scan. File 1: 12/13 events up, 0 down. 126-ev certification:
  **96.50 / 95.65**, LE 59.22 (all ≥ v0.8.8), +2 % latency (longer
  scans). Canonical flags += `--edge-penalty`.
- **Variance weighting** (08-13) — restored the never-loading variance
  perceiver, tested quadrature weighting: tie in-band, degrades above;
  pulls show Poisson-only already over-covers (0.76 vs 1.0) — any additive
  variance worsens coverage. Reverted; model dropped for speed. A useful σ
  model must REPLACE the error model targeting pull≈1 (retrain item).
- **v0.8.5** (08-13) — **4×-smaller amplitude perceiver** (HybridPerceiver3D
  10.24M, b32/e256; Val R² 0.9850 vs prod 0.9854). Arch-aware loader,
  `--light-ckpt`. Accuracy noise-level; stage 1 −33 %.
- **v0.8.6** (08-14) — **the speed release**: + `--skip-v2` (V2 rescue fully
  superseded by joint+PASS 2 — 126-ev NS delta −0.06 at a PERFECT 63/63
  split). **147 s/ev wall, 18.4 s/ev node-throughput, 115 s/ev
  single-worker latency — the ≤120 s/event single-node requirement MET.**
  Rejected on the way: `--fast-track-stage` (5 s for −0.32pp — track-stage
  refinements feed t0/flash quality downstream).
- **v0.8.7 (candidate, 08-14)** — **support-restricted overflow gate**
  (`--gate-support`). Superposition test on the family that deferred
  cluster 446: sum-of-separate renders +4.3 % total and **+92/+115 % on
  non-support channels** (pure summation floor; collective render cures
  them), and 2 of the 3 gate-tripping channels were exactly that garbage.
  Fix: both trigger branches restricted to the subject's support channels.
  126-ev certification: NS 96.46 / all-TPC 95.61 (deltas −0.004/+0.003,
  splits 43/61 and 52/61 — noise), deferrals 5778→3353 (−42 %), rescue
  flags −40 %. Threshold scan (excess 300–1000, z 1.5–3, nch 2–4, cos
  0.6–0.85): a PLATEAU — ±0.03pp across the whole grid despite 6× defer
  variation; historical defaults kept. ADOPTED, tagged v0.8.7. The gate is
  not a fragile knob; its thresholds need no future retuning campaigns.

## v0.8.8 (08-14): stage 5 — LE blip re-timing  *(the neutron-ToF campaign)*

User thesis: the LE territory (noise hits + sub-1.5 MeV clusters, ~2 % of
truth energy; SPINE-unreachable) is uncontested CLM value; the borrowed
class sat at 56 %.  Anatomy census: correct family EXISTS for 93 % of wrong
blips but is the nearest one only 2 % (displaced deposits — the neutron
signature); rank-3 contains truth 57 %; ambiguity census: 96.4 % of LE
energy has time-DISAGREEING candidates (geometry near its information
ceiling).  Three ranker iterations failed instructively (candidate sets
missing the incumbent; forced no-incumbent changes; linear model flat);
HGB on feature interactions extracted the available signal.  Stage 5
(`--le-retime`): 3-cm LE blobs, k=8 member-distance candidates + incumbent,
batched HGB, incumbent-margin gating, 4.8 s/ev, on the placed backbone
only.  **126-ev certification: LE 58.7→60.9 (+2.2), NS 96.46→96.49
(123/3), all-TPC 95.61→95.64 (124/2).**  Lesson (user): spatial isolation
is the PHYSICS signature of displaced deposits, not wrongness — retired as
a give-up signal in the LE band; the per-blip association margin is the
confidence product for neutron ToF selection.

## Standing constraints and lessons (do not relearn)

1. Both metrics (NS AND all-TPC) headline every result table.
2. Neutron deposits ≥2 MeV are never unassigned (physics requirement).
3. Light-driven decisions respect the information boundaries: blind <3 MeV,
   weak prior 3–20 MeV, trustworthy ≥20 MeV.
4. Per-channel light comparison, never a collapsed scalar (v0.5.0).
5. Priors in energy units, annealing-consistent (v0.7).
6. Truth-side score gains don't transfer operationally — validate on reco
   populations (v0.8.1).
7. Data-facing algorithms: few thresholds, 1D-checkable (v0.8.3).
8. Old stages don't announce obsolescence — measure supersession
   (production placement, V2, swap machinery).
9. The per-cluster-render SUM is not additive: +3 % median / ×1.65 p90 on
   bright channels, ×2 on far dim channels. Gates must read support
   channels; superposition-consistency is a Charge2Light training item.
