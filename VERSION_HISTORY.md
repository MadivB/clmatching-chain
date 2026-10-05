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
| v0.10.5 | 09-04 | 96.50 | 95.65 | conf = pcu_vote10 directly (zero parameters, data-only); 3-file confirm: out-of-sample AUC 0.777/0.936 E-wt, reading key 33->99% purity, 0.65 s/ev; run_v0105.sh. v1.0 deferred: speed campaign first |
| v0.10.6 | 09-04 | 96.50 | 95.65 | low-E simplification: KD-tree force (11.8->0.3 s) + 25cm isolation veto (refuses 0.04% hits) + LE-ranker stage DELETED (5.0->0 s); eff -0.006 pp; ~50 s/ev/GPU; run_v0106.sh |
| v0.10.7 | 09-04 | 96.50 | 95.65 | fast-scan ADOPTED (rank-1 row-split collapse, 6834 calls/ev at 100% fast, bit-identical; backbone 19.7->16.3); TF32 null; 55.0 s/ev/GPU broad accounting (was 64.8); run_v0107.sh |
| **v1.0** | 09-04 | 96.50 | 95.65 | **FIRST RELEASE** = v0.10.7 chain + vote10 confidence: continuous t0s + sigma_t + zero-parameter data-only hit_t0_conf (out-of-sample AUC 0.777/0.936 E-wt), 25cm isolation veto, 55.0 s/ev/GPU; run_v1.sh |
| **v1.1** | 09-22 | +0.26‡ | +0.17‡ | light-coupled component pool (exact combination search, 20 MeV floor) + stage 3.98 component re-decision (>= 20 MeV) + 0.5 MeV prediction floor in non-shower TPCs; 5-file cert 63 ev: NS 96.44->96.70 (52/11), all-TPC 95.37->95.54 (46/17), LE 57.9->61.0; +5 % time; run_v11.sh |
| **v1.2** | 09-23 | = v1.1 | = v1.1 | + stage P per-hit PRECISION TIMING (2x2 v2 line, ND-adapted): per-hit sigma 6.00 -> **2.85 ns robust** (9.13 -> 6.43 std), per-family 5.87 -> 2.56 ns robust, hit_t0_sigma_p delivered (98 % of hits, median 0.44 ns); assignment identical; +9.4 s/ev; run_v12.sh |

\* 39-event sample, pre length-gate fix — not comparable to later numbers.
† file-0000001 same-node pair; 126-ev v0.8.6 row is the certified one.
‡ v1.1 certified on files 0000001-0000005 (63 events, paired vs the v1.0 outputs of the same events; user: 5 files suffice for ND validation); deltas are vs v1.0 on those events.

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


## v1.1 (2026-09-22): light-coupled component placement, re-decision, 0.5 MeV templates  *(user's "more permutations" direction)*

User brief: better assignment under a constrained budget, data-driven (few
thresholds, no MC reverse-engineering), physically sound; core idea = more
placement combinations against the light loss.  Full record: V110_DESIGN.md.

Census first (v1.0 dumps, 39 ev): no flash finder exists (slots come only from
cluster self-placement); the undiscovered-flash pot is 13.7 % of wrong energy
but light-visible only to 0.1 pp; 52 % of wrong energy sits under residual
deficits at EXISTING families (a member placed elsewhere); wrong clusters have
a different-interaction light twin (cos > 0.8) 75-93 % of the time while the
energy-banded pool co-decides the pair 0-38 %; correct families are under-
predicted 2-10 % and wrong clusters at a bigger correct cluster's time fill
exactly that deficit (piggyback, 32 % of wrong E).

Adopted (tag CLMatching_v1.1, launcher run_v11.sh = v1.0 flags +
`--pool-mode light --p2-threshold 20 --redecide --redecide-emin 20
--predict-min-energy 0.5 --predict-min-energy-shower 1.5`):
1. **Stage 3.1 light-coupled components** (testing/v110_components.py):
   batches = connected components of the pattern-cosine graph (>= 0.8 on a
   shared TPC) instead of energy bands; each chunk (<= 8) decided by an EXACT
   combination search on the pairwise-quadratic loss (k = 6 candidate minima
   per member + the greedy path; top-8 re-scored with the clipped loss).
   Accuracy-neutral alone (twin swaps are degenerate at the template-
   systematic level) but 42 % less pool time (scans 4867 -> 319/ev).
2. **Pool floor 20 MeV** = the light-information boundary (k* crossing):
   20-30 MeV single-TPC clusters join the component search.  (10 MeV gives
   more all-TPC but loses 0.7 GeV in the light-weak 5-20 MeV band.)
3. **Stage 3.98 component re-decision** (testing/v110_redecide.py): after
   pass2, components of PLACED clusters >= 20 MeV in non-shower TPCs are
   re-decided jointly on the leave-component-out base; the winner must beat
   the incumbent's exact loss; singletons untouched.  1.3 s/ev, ~12 moves/ev,
   truth-scored +2 GeV/39 ev (58 fixes / 32 breaks).  At >= 5 MeV the same
   stage LOSES 3 GeV in 5-20 MeV: the light-weak band must not be re-decided
   by light (standing lesson, now re-measured).
4. **0.5 MeV prediction floor in non-shower TPCs** (M5p1 PredictionConfig.
   min_cluster_energy_shower_tpc_mev = 1.5): 0.5-1.5 MeV clusters get light
   templates and go to the tail solver instead of nearest-hit force placement
   - the "small peaks" population.  Alone: NS +0.15 (36/3), LE 58 -> 61;
   shower TPCs gain nothing from it, so they keep 1.5 MeV (matrix stage back
   to 16 s).

Measured NULL / NEGATIVE (all flag-gated off, do not re-test):
- amplitude profiling at commit (`--pool-gain`): gains pile at the +25 % edge
  (steals unplaced co-timed light), NS -0.11..-0.17, three arms;
- charge-contact prior for big clusters (`--pool-spatial`): 99 % of correct
  clusters' disagreeing 3-cm contacts are genuine cross-interaction contacts
  (47 GeV at risk vs 3.9 GeV fixable) - the v0.7 certificate holds for the
  disconnected tail only; not run in-chain;
- ordering search on component batches (batching rule alone): wash;
- participate-but-do-not-commit hybrid (`--pool-commit-min`): negative and slow;
- 5 / 10 MeV pool floors: more all-TPC, but the light-weak band pays;
- the in-component greedy path is a strawman (joint beats it by +8 GeV per
  33 ev yet end-to-end is flat): only end-to-end ledgers count.

Ledger (paired vs v1.0 on the same events; chain deterministic - a v1.0 rerun
is bit-identical):

| arm (39 ev, files 1-3) | NS | all-TPC | events NS b/w |
|---|---|---|---|
| B light components | +0.03 | -0.03 | 23/16 |
| D2 + floor 10 | +0.07 | +0.08 | 22/17 |
| F20 + redecide 20 | +0.14 | +0.05 | 26/13 |
| K0 light + predict 0.5/1.5sh | +0.15 | +0.05 | 36/3 |
| **K1 = v1.1** | **+0.29** | **+0.15** | **33/6** |
| K2 (floor 10) | +0.24 | +0.19 | 33/6 |
| **v1.1 CERT files 1-5, 63 ev** | **+0.26 (96.44->96.70)** | **+0.17 (95.37->95.54)** | **52/11 (all-TPC 46/17)** |

Time: 110 -> 116 s/ev/worker (+5 %; stage 1 +6 s for the extra templates,
pool -5 s, re-decision +1.3 s).  Confidence output unchanged (pcu_vote10).
Tooling: testing/v11_ab.py (paired ledger, multi-dir reference), v11_hitflips.py,
v11_flips.py, v11_chunk_audit.py, v11_census.py, v11_contact_potential.py,
run_v11arm.sh / run_v11cert.sh (files 1-3 / 1-5 runners).
Next (user): v1.2 = v1.1 + the 2x2 per-hit precision-timing module
(/pscratch/sd/y/yuxuan/2x2QLMatching/V2Design/ND_HANDOFF.md).


## v1.2 (2026-09-23): per-hit precision timing  *(the 2x2 v2 line, ND-adapted)*

v1.2 = v1.1 + stage P (`--precision --precision-calib testing/v120_calib_lowi.json
--precision-rounds 3`, launcher run_v12.sh).  Membership is untouched; every
hit gets a deposition time and a per-family uncertainty (`hit_t0_sigma_p`).
Design and all measurements: V120_DESIGN.md; handoff source
/pscratch/sd/y/yuxuan/2x2QLMatching/V2Design/ND_HANDOFF.md.

What the census said (v1.1 outputs, truth used to measure): the 8.6 ns per-hit
residual is a per-FAMILY time error (family offsets 9.4 ns, within-family
1.4 ns); track propagation and light time of flight are ~1 ns effects at ND
and the apex clock t_dep = t_apex + d/c is exact in truth; per-channel leading
edges already reach 4 ns per channel (2.9 ns per family) but half of all
family-channel pairs carry >= 10 % pile-up light; a +5.8 ns/decade amplitude
slew and a period-12 channel pattern (a ~10 ns offset between the two light
detector types hidden in the single average template) sit on top.

Stage P (testing/v120_precision.py), per family = hits sharing one final t0,
per light TPC: support channels = the channels holding 90 % of the family's
own predicted light; per-channel delay-aware model = the family's light-
arrival density from its hits (apex clock + n d/c, c_LAr = 13.4 cm/ns fixed,
1/d^2 visibility proxy) convolved with the single template; fit on the
residual after all other families, 0.25-tick scan +-3 ticks with the
amplitude profiled and a 9-point parabola; per-channel-index delay and per-
type slew removed; family apex time = weighted median over channels
(sqrt(A) x (1 - pile)); sigma_t = 1.4826 MAD / sqrt(n); apex = best of the
two principal-axis ends and the centroid by total loss; 3 pile-up iterations
with the other families re-rendered at their refined arrival-equivalent
times; per-hit time = t0 + |r - apex| / c.  Calibration
(testing/v120_calibrate.py) = alternating medians on the low-intensity
sample (60 events of MiniProdN5p1 ... lowintensity, no truth): k_idx[120] and
beta_T per type; it agrees with a file-1 calibration to 0.67 ns rms.

Certified (files 0000001-0000005, 63 events, paired vs the v1.1 outputs of
the same events):

| metric (core hits, E-weighted) | v1.1 stage E | v1.2 stage P |
|---|---|---|
| per-hit std / robust | 9.13 / 6.00 ns | 6.43 / **2.85 ns** |
| per-family std / robust | 8.75 / 5.87 ns | 5.74 / **2.56 ns** |
| per-file robust | 5.85-6.13 | 2.79-2.92 ns |
| per-hit std with a 20 ns clip | | 3.4-3.5 ns (1.1-1.6 % of energy beyond 20 ns) |
| delivered sigma_p | | 98.3 % of hits, median 0.44 ns |
| assignment (NS / all-TPC) | 96.70 / 95.54 | identical within 0.01 pp |
| stage cost | | 9.4 s/ev median on the GPU (p90 16.7); total 116 -> 128 s/ev/worker |

sigma_p is an honest uncertainty: per-hit std 2.4 / 3.1 / 4.2 / 6.7 / 12 /
23 ns in the sigma_p bins < 0.3 / 0.3-0.6 / 0.6-1 / 1-2 / 2-4 / > 4 ns (24 /
50 / 17 / 7 / 1.8 / 0.6 % of the energy), and stage P beats stage E in every
bin, so no fallback is applied.  Pile-up-free floor (held-out low-intensity
events): 2.62 ns robust, 3.0 ns clipped std - the same as on the standard
sample, so the floor is the per-channel timing with one average template and a
per-index delay, not pile-up.  Next levers (v1.2.x, in the 2x2 order):
per-CHANNEL k_j from the 767-file low-intensity sample, per-type templates,
sub-sample pulse-shape fit.

Measured and not adopted on the way: the 2x2 topology corrector (home rules:
endpoint-at-shower 96 % pure, embedded fragment 93 %, isolated fragment 35 % in
ND; v1.1 already agrees with the pure ones; the crossing-track failure mode has
0 cases in 39 ND events - testing/v12_home_rules.py); pile-up iterations with
the other families at their apex time (convention mismatch, diverges); the
cumulative calibration update (must use corrected residuals).
Tooling: v12_decompose.py, v12_channel_edges.py, v12_proto_fast.py (offline
prototype with all knobs), v12_eval.py, v12_calib_ab.py, sbatch_v12_P1.sh.
Gotchas: h5py cannot open the dunepro /pscratch flow files from a login node
(HDF5_USE_FILE_LOCKING=FALSE, or the file-1 flow for geometry); the stored
template carries ~100 pre-trigger samples before its rise.

### v1.2.1 (2026-09-23): multiple starting times per family (pieces)  *(user: time-of-flight needs per-fragment t0)*

`--precision-pieces` (now in run_v12.sh): each family is split into 3-cm
contact-chained pieces; the family (core) fit is the certified v1.2 fit; every
displaced piece >= 5 MeV gets a FREE time from its own support channels on the
residual after everything else (scan +-10 ticks), but only on channels where
its own predicted light is at least comparable to everything else there — with
fewer than three such channels it keeps the family's apex-clock time (sigma
NaN).  Delivered per piece: delay to the core apex time, distance, implied
beta, sigma (`row["phase3"]["pieces"]`); per family: apex time, sigma, apex
position, energy (`row["phase3"]["families"]`) for time-of-flight pairing.
Measured (39 events, offline == in-chain): bulk per-hit resolution unchanged
(6.43 / 2.95 -> 6.41 / 2.96 ns), 22.7 light-resolvable pieces per event of
~1000, delay resolution 4.5 ns robust, +3.5 s/ev.  Physics found on the way:
(1) a truth apex time taken from the earliest 2 % of core hits is pulled early
by pathological hits in big families and fakes 20-100 ns "delays" — use the
median of (truth - d/c) over the core; (2) pieces that then still looked
delayed were impure clusters (a minority of hits with late truth) whose light
sits at the chain tick — the fits are right; within the +-10-tick assignment
window this sample has no genuinely delayed light-resolvable fragment
population; (3) the genuinely late population (clusters >= 5 MeV with truth
> 10 ticks after the nearest bigger neighbour within 1 m: 207 GeV / 39 ev,
median delay 2.95 us) is timed by the chain at its own flash for 95 % of the
energy, i.e. as separate families with their own v1.2 apex time — a
time-of-flight delay is the difference of two family times (2.6 ns robust
each); < 5 MeV deposits are light-blind and carry only their association time.
Tools: v121_fragments.py, v121_piece_test.py, v121_piece_debug.py,
v121_piece_eval.py, v121_slow_census.py, v121_bulk_check.py.

### v1.2.2 (2026-09-23): light saturation ceiling = rail minus dynamic baseline  *(user: predicted pulses allowed above the saturated actual)*

The light digitiser stores int16 samples: pedestal about -28000, rail 2**15-1 = 32767
(the FLOW `clipped` flag is exactly "touches 32767").  The chain works on
baseline-subtracted waveforms (raw minus the mean of the first 75 samples), so the rail
in those units is `32767 - baseline`, per channel and per event (60765-60770 in this MC).
Every stage hard-coded 60780 (about 100 sites: `adc_clip=60780`, `CLIP`, `ADC_CLIP`,
`raw_clip=(0, 60780)`, and the veto `actual > 60700 for > N ticks`), and the displays drew
the unclipped prediction, which is what showed up as orange above the flat blue top.
Now one definition, `M5p1/light_saturation.py`: the first stage computes
`sat_ceiling (n_tpc, 120) = 32767 - baseline` next to the formatted waveform
(`EventData.sat_ceiling`, FLOW `clipped` kept as `EventData.light_clipped`), the
saturation veto counts samples within 80 counts of that rail (same margin as before) and
publishes the array as `saturated_channel_cache["ceiling"]`; every clip in the submodule
takes `adc_clip` as scalar-or-array through `clip_hi / clip_hi_stack / clip_hi_full /
stacked_row_ceiling`, the harness registers the array once per event in
`testing/sat_clip.py` and all testing-side stages clip through `clip_rows / clip_col /
ceiling`; the release entry (QLmatchingND -> v11_plotting_purpose_eval) computes the same
ceiling per event; dumps carry `sat_ceiling`; demo_tpc_displays.py draws the prediction
clipped at it (older dumps: rail channels reveal it as their maximum).  The constant
survives only as `LEGACY_CLIP`, the fallback when no raw samples exist.  Old tags are not
rewritten; run_v1.sh / run_v11.sh / run_v12.sh all pick the fix up through the shared code.
Measured (file 0000001, 13 events, dumps in /pscratch/sd/y/yuxuan/v11arms/SAT1 vs the same
code with CL_SAT_LEGACY=1 in SAT1_LEG): on every channel that touches the rail the actual
maximum equals the new ceiling to 0.00 counts and the clipped prediction never exceeds the
plateau; phase-1 times, veto masks and 12 of 13 events' final times are bit-identical
(2.24 M hits within 0.5 tick), the stored base image differs by the 16-18 count clip level;
in event 9 one light-degenerate twin pair in a saturated TPC (119 and 26 MeV clusters at
249 / 288 ticks, 4 rail-touching channels still in the fit) swapped slots, -0.013 pp all-TPC
over the file, NS +0.000.  The MC pedestal is flat, so the physics content of the change is
for data (pedestal drifts, pre-trigger tails); against the v1.2.1 arm P2 the assignment is
also identical to 0.01 pp (P2 predates the final v1.2.1 precision commit by 34 min, hence
its sub-tick times differ).  Tools: testing/v122_sat_check.py, v122_ab_breakdown.py,
sbatch_v122sat.sh (CL_SAT_LEGACY=1 for the A/B), demo_tpc_displays.py re-render in
testing/v12_displays_v122/.

### v2.0_test (2026-09-24): assignment as one energy, exact block search  *(user: light chi2 + beta neighbour prior, try combinations, break clusters, data-driven)*

`testing/v200/` + V200_DESIGN.md.  One time per cluster; E = TPC-level Poisson chi2 of the
whole light model (rail-clipped, v1.2.2 ceiling) + a contact Potts term with beta measured in
situ; the light term is pairwise-quadratic, so blocks of <= 8 clusters are solved EXACTLY from
k x k tables (q/r vectors, no GPU) inside large-neighbourhood sweeps with twin-swap kicks;
candidate times = own residual-scan minima (new slots allowed for light-decidable clusters)
plus the TPC slots.  Offline on the 39 dev dumps the solver goes far deeper in the light
energy than the chain (33.7M vs 45.7M) but lands at 95.53 % vs 96.03 %: better above 20 MeV,
worse below.  Measured cause: 17 % per-channel light-model systematics (in situ), small
clusters' light 5-15 % of the ambient light, and the chi2 rewarding any small cluster that
fills a big family's deficit; six regularisations (dominance, support channels, systematics
weights, per-family gains with a 17 % prior, blind-to-nearest-family, residual-pattern cosine)
do not repair it, and the in-situ beta table is selection-biased for small-big contacts
(P(same) = 0.008 among decidable pairs) so it cannot be used.  What transfers: re-solve only
clusters >= 20 MeV (cold re-initialised) with everything below frozen: offline +0.19 pp
(27/11 events); floor scan W10 -0.03 / W50 +0.07 confirms 20 MeV.
In-chain as stage 3.99 (`--v200-min-e 20`, after the v1.1 re-decision, +5 s/ev): files 1-3
vs v1.2.1 arm P2: all-TPC 96.05 -> 96.24 (+0.19 pp, 27/11 events), NS +0.03, all three files
up; >= 100 MeV wrong 1.46 -> 1.29 %, 20-100 MeV 8.16 -> 7.65 %, < 20 MeV unchanged.
5-file certification (arm V2A5 vs CERT_P, 63 paired events): all-TPC 95.54 -> 95.76 (+0.22 pp, 42/20
events better/worse), NS 96.70 -> 96.77 (+0.08, 35/25), all five files up (all-TPC +0.07/+0.29/+0.23/
+0.38/+0.14); >= 100 MeV wrong 1.91 -> 1.68 %, 20-100 MeV 8.62 -> 8.28 %, < 20 MeV and LE unchanged
(60.95 -> 60.89); +5-12 s/ev for the stage.  Passes the 5-file standard; adoption = user's call
(flag `--v200-min-e 20` in the launcher).  Outputs /pscratch/sd/y/yuxuan/v11arms/V2A (files 1-3), V2A5 (1-5).
Floor / variance scan (offline, 39 ev): 15 MeV 96.15, 10 MeV 96.09, 10 MeV slot-only 95.80, 20 MeV +
decidable 10-20 96.17 (all below the 20 MeV floor's 96.21); Poisson 300 A identical to 200 A; Poisson +
(0.17 A)^2 95.68; trained variance predictor 94.12 (predictor + model error 94.67): the predictor describes
waveform noise, not the 17 % light-model error, so it cannot be the placement weight (V200_DESIGN.md s.7).
Open for v2.0 proper: an unbiased contact prior for the < 20 MeV population (physics rule or
truth-validated purity), and cluster splitting with per-piece light from the perceiver.

### v1.3 (2026-09-24) ADOPTED: v1.2.1 + stage 3.99 (exact-block re-solve >= 20 MeV)

`run_v13.sh` = run_v12.sh + `--v200-min-e 20` (V200_DOM=3).  Certified on files 1-5 as above (arm V2A5);
numbers for the record: all-TPC 95.54 -> 95.76, NS 96.70 -> 96.77, LE 60.95 -> 60.89, +5-12 s/ev.  The v1.2.2
saturation ceiling is part of the same commit; run_v12.sh reproduces v1.2.2 behaviour (no stage 3.99).
Tagged CLMatching_v1.2.2 and CLMatching_v1.3 on the same commit (repo + submodule); not pushed.

### Study 2026-09-24: beam-comb recovery with the v1.3 precision timing (testing/v14_beam_comb.py)

ND simulation truth carries the bunch structure: period 19.09 ns, flat-top bunches ~8.6 ns wide,
batch dips every ~1.6 us; contained vertices (25 of ~150 interactions per spill in ND-LAr) have
segment times equal to the vertex time, rock interactions enter 2-88 ns (p10-90) later.
Reconstructed interaction start = earliest precision-family apex time per vertex (one family per
light TPC).  63 events (V2A5), 1471 contained + 3978 external interactions:
  * global reco time constant -5.59 ns (light path + template), stable to 0.25 ns event to event
  * contained: reco start - vertex time 3.3 ns robust (core std 3.1 ns); 20-100 MeV 2.8 ns,
    100-500 MeV 2.9 ns, > 500 MeV 4.0 ns
  * comb: truth 99.0 % inside the bunch window, reco 71.8 % (random 45 %); bunch identification
    (nearest bunch) 97.1 % correct (96.9 % energy-weighted), 96-98 % in every energy class
  * spill bunch phase from the reconstruction alone (no truth): 1.8 ns robust per spill
  * external interactions sit off the comb by their flight time; reco start - entry 3.3 ns, so
    the comb turns the detector into a time-of-flight measurement for rock muons (mod 19 ns).

Figure in the 2x2 style (testing/v14_beam_comb_nd_figure.py -> testing/beam_comb_figs/beam_comb_nd_v13.png,
support figures fig1-fig6 from testing/v14_beam_comb_plots.py, arrays beam_comb_figs/beam_comb_data.npz).
The reconstruction side is truth-free: families are linked into interactions by reconstructed time alone
(gap 10 ns), the start is the earliest family timed to sigma < 1 ns, "not entering" = no family has hits
within 10 cm of the upstream, x-side or top faces of the event hit box, plus E > 100 MeV and (elite) start
sigma < 0.5 ns.  Truth supplies only the per-spill frame (the RWM analogue: the bunch phase is locked to the
spill start, spread 0.22 ns) and one global constant.  Rayleigh power at 19.09 ns over 63 spills:
  * all 6108 reconstructed interactions: P = 155, |R| = 0.16 -- 73 % of them are rock/external by truth,
    which is the physics limit of an unselected ND-LAr comb (rock arrivals are spread by flight time)
  * not entering & E > 100 MeV (717): P = 140, |R| = 0.44, truth-contained purity 0.74, in-window 0.73
  * elite (382): P = 105, |R| = 0.53, purity 0.78, in-window 0.80, residual to the true vertex 2.07 ns robust
  * truth contained vertices (1471): P = 885, |R| = 0.78 (99 % in-window)
  * the apex margin to the hit box does NOT separate contained from external interactions (rock neutrons
    and photons make deep, low-energy blips); face-touching plus energy does.
Finding about the precision stage: the apex choice (two principal-axis ends + centroid, by fit loss) is at
chance -- wins 32 / 31 / 38 %, apex within 20 cm of the true start point in only 27 % of > 60 cm families
(median 41 cm away) -- while the true start sits at the LOW-z end 72-77 % of the time (median 6 cm).  The
family time is insensitive (shift <= L/4c) but per-hit apex-clock times carry up to L/2c of this; a
beam-direction or vertex-anchored apex prior is the next per-hit precision lever (not implemented).

Projection to a common beam-time reference (user's proposal 2026-09-24: do not cut entering interactions,
assume c and project every start point to the beam origin; testing/v14_beam_comb_projected.py ->
testing/beam_comb_figs/beam_comb_nd_v13_projected.png).  t_beam = t_start - (r_start . n)/c with n = mean
true neutrino direction (-0.0001, -0.102, 0.995) and r_start = the upstream-most hit of the start family
along n (beam-direction prior; the fitted apex is unreliable).  Two measured facts first:
  * the ND MC assigns the BUNCH TIME AT THE VERTEX for every interaction, rock vertices (z -20 m .. +6 m)
    included: all 8272 true vertices have |R| = 0.77 unprojected, and subtracting any fraction of the
    target flight (r.n)/c lowers it (contained 0.78 -> 0.14 at the full 1/c).  Only the products' flight
    from the vertex is simulated (rock entry delay = distance/c to 0.1 %, vertex-to-entry line at
    cos 0.98 to the beam).  So in the MC as delivered the projection SMEARS the comb (reco |R| 0.16 ->
    0.03) and cannot repair rock muons, whose delay is set by an unknown rock-vertex position.
  * emulating the physical convention (each true interaction shifted by + (r_v . n)/c, the flight the MC
    left out, applied identically to its reconstructed time; 136 of 6108 reco interactions without a
    truth vertex left out), the truth-free projection works with NO containment cut: all 5972 reconstructed
    interactions P = 139 -> 644 (|R| 0.15 -> 0.33), in-window 0.65 (random 0.45); rock interactions go
    from a +27.5 ns flight-delay smear (robust 28.7 ns) to 4.7 ns robust, 60 % in-bunch; contained 4.1 ns
    robust (3.7 ns before projection: the start-point term adds ~1 ns).  Truth bunch times of all 8272
    interactions: P = 4923, |R| = 0.77.  => for DATA the projection is the right operation; for this MC any
    comb study must either stay unprojected on contained interactions or apply the emulated shift.

### ND Michel / delayed-daughter stage: port of the 2x2 stage M (started 2026-10-01, in progress)
*(user: Michel electrons must be characterised for the stopped-kaon study; see the 2x2 handoff
/pscratch/sd/y/yuxuan/2x2QLMatching/spine_kaon/centirun7/kaon_light/HANDOFF_CLM_stopped_kaons.md)*

Status of that handoff on the 2x2 side (done by the 2x2-project session, v2.0_alpha_12/DESIGN_v2.0_alpha_12.md):
Task A = stage M (QLMatching2x2/assoc_michel_2x2.py): unexplained late waveform-finder flashes -> end pieces of
clusters from free tips bounded by dense points (Bragg peaks / vertices) -> perceiver light of the piece -> gates
mf (pattern at the flash) >= 0.8, alpha 0.8-3, apc (PROMPT DEFICIT = projection of the piece's prompt pattern on the
current residual; ~ -1 absent, ~ 0 present) <= -0.5, sat <= 0.5, cov >= 0.1, L <= 16 cm.  Full K -> mu nu list (1,264
events): Michel within 5 ticks 16.9 -> 35.7 % (merged 6.6 -> 30.3 %), K+ unchanged, mu+ -0.5 (muon-end false
pieces); 10-file production gate 98.483 -> 98.524 %, arm alpha13.  Task B: truth-piece K/mu timing resolution 11.8 ns
= the lifetime, unbiased, statistical recovery only; reco pieces at the stopping point not attempted.  Not yet in the
release repo; no K/mu precision display yet.

ND port (`testing/v14_michel_nd.py`, GPU, offline on the v1.4 cert dumps; `run_michel_nd.sh`, `run_michel_nd_devhold.sh`,
gate scan `v14_michel_nd_gatescan.py`, outputs `testing/michel_nd_out/`): late flashes = residual flashes (stage-4.8
finder, tail 0.5) with unexplained fraction >= 0.2 and >= 2000 ADC; candidates = clusters > 12 ticks earlier with
whole-pattern cosine >= 0.25 at the flash (top 3); hypotheses = whole cluster (5-80 MeV) or end pieces (free tips,
dense points, L 2-16 cm, <= 50 % of the cluster); piece light from the small4x perceiver (amp pattern x template);
scores mf / alpha / apc / sat / cov as in 2x2 with Poisson weights and the per-channel saturation ceiling.
Labelled sample, 13 spills (file 1): 7,298 hypotheses, 104 true (>= 50 % of the piece energy at the flash time,
1.01 GeV) vs 6,903 currently-right pieces (83 GeV).  The discriminants transfer: mf true median 0.91 vs 0.59; alpha
2.33 vs 0.11; apc -1.88 vs +0.22 (2x2: -1.85 vs +1.94); cov 0.35 vs 0.88.  With the 2x2 v2 gates 32.7 % of the true
hypotheses pass (0.51 GeV) and 0.3 % of the currently-right ones (0.22 GeV).  Simulated selection (gate scan):
2x2 gates 184 good / 49 bad MeV in 13 moves; mf >= 0.9 and apc <= -0.3: 168 / 0 in 7 moves; no apc gate: 233 / 157.
Cost offline ~20 s per spill (300-500 perceiver piece predictions).  Applied on the same 13 spills with the 2x2 gates:
13 moves, correctly timed energy 49 -> 184 MeV (fixed 0.18, broken 0.05 GeV) = the simulation exactly.
BUT the stage-4.8 residual-flash finder reaches only 9 % of the ND delayed-daughter energy (4.04 GeV in these 13
spills, 5 % correct): 65 % of the delayed pieces' true flashes lie within 5 ticks of an EXISTING family time (part of
the Michel, or another deposit, is already placed there) and most sit on their host's tail (residual / model median
0.21), both of which that finder rejects by design.  Round 2 therefore takes the candidate late times from the TPC's
family times + 10-sigma residual peaks, kept when the light there is unexplained after allowing the placed families a
30 % under-prediction (gain-corrected residual >= 10 % of the actual and >= 5000 ADC; a plain 20 % test passes nearly
every family time at ND's systematics), capped at 8 per TPC: 64 % of the delayed energy passes this test at its true
time; ~100 s per spill offline (1,500-2,200 piece predictions).  Labelled sample, 13 spills: 47,120 hypotheses, 579
true (5.5 GeV, nested) vs 45,242 currently right (554 GeV); mf true 0.81 vs 0.62, apc -1.04 vs +0.16, alpha 1.49 vs
0.30.  Simulated selection, 2x2 gates: 52 moves, 497 good / 239 bad MeV; no apc gate: 617 / 1,075 (apc is again the
decisive gate).  The bad moves are END PIECES OF BIG CLUSTERS (host median 384 MeV vs 70 for good moves, piece
fraction 2 % vs 31 %, mf 0.85 vs 0.93) matched to another family's unexplained residual - the ND analogue of the 2x2
muon-end false pieces; a host-energy cap on end-piece moves removes them: host <= 150 MeV 381 / 77 MeV in 28 moves,
host <= 100 MeV 301 / 8 in 21 moves.
DEV RESULT (files 1-3, 39 spills, gates mf >= 0.8, alpha 0.8-6, apc <= -0.5, cov >= 0.2, unexplained >= 10 %; `michel_nd_out/dev2_A.json`):
144 moves (130 end pieces, 14 whole clusters), fixed 1.12 GeV, broken 0.52 GeV, net +0.60 GeV, spills better / worse
27 / 10, every file positive (+222 / +176 / +204 MeV); the delayed-daughter class (11.7 GeV in these spills) goes from
5.1 % to 11.1 % correctly timed; moved energy 0.70 GeV inside the class, 1.87 GeV outside it.  By piece energy:
< 5 MeV pieces are 77 % wrong (net -67 MeV), 5-20 MeV 71 % right (+593), 20-100 MeV 56 % right (+75); by delay:
12-30 ticks 75 % right, 30-100 65 %, 100-300 60 %, > 300 ticks 32 % (net -40; other interactions' families).  The
host-energy cap seen on file 1 does not pay on files 2-3 (it removes good moves too).  Post-filters chosen on dev:
piece >= 5 MeV and delay <= 300 ticks -> 96 moves, 1,015 / 323 MeV (net +692, purity 0.76); with a 200 MeV host cap
on end pieces 70 moves, 821 / 205 (purity 0.80).  Cost 100 s per spill offline.
HOLDOUT (files 4-5, 24 spills, same gates, `michel_nd_out/hold2_A.json`): 97 moves, fixed 0.53 / broken 0.30 GeV
(net +0.23), spills 17 / 7, both files positive (+171 / +63 MeV); delayed class 12.9 GeV 7.7 -> 10.4 % correct.  With
the dev-chosen post-filter (piece >= 5 MeV, delay <= 300 ticks): 48 moves, 423 / 164 MeV, purity 0.72 (dev 0.76); the
host cap does not transfer (holdout 247 / 100).  OVER THE 63 SPILLS with the post-filter: 144 moves, fixed 1.44 GeV,
broken 0.49 GeV, net +0.95 GeV (~ +0.025 pp all-TPC), purity 0.75; the delayed-daughter class (24.6 GeV = 0.64 % of the
energy) goes from 6.4 % to 10.7 % correctly timed - a quarter of what the light reaches, against 36 % of the Michel
energy for the 2x2 stage (16.9 -> 35.7 %): ND pile-up puts another family within 5 ticks of most Michel flashes and the
17 % model systematics blunt the prompt-deficit test (apc true -1.0 vs right +0.2, against -1.9 vs +1.9 in 2x2).
NOT adopted: 100 s per spill offline (1,500-2,200 perceiver piece predictions) against the 3 s budget; an in-chain
version needs the probe-only pass, fewer candidate flashes and a cheaper piece light (amp-native prediction) - a
10-20 s per spill target, i.e. a budget decision.  Scripts: testing/v14_michel_nd.py (stage + labelling),
v14_michel_nd_gatescan.py, v14_michel_nd_report.py, michel_nd_round2b.sh; outputs testing/michel_nd_out/.

**Piece-first rewrite (2026-10-02, `testing/v14_michel_nd_fast.py`)** - user: 100 s is far too much, inject physics,
Michels are critical.  Physics injected: a Michel is a short branch (5-55 MeV, <= 16 cm) that LEAVES the Bragg peak at
the end of a stopping muon and has its own flash 12-900 ticks later.  Order inverted: (1) dense points; (2) candidate
pieces = free tips of clusters >= 10 MeV whose geodesic growth is STOPPED by a dense ball within 16 cm ("bragg"), plus
whole templated clusters 5-80 MeV; (3) ONE batched perceiver call for the probe pieces (L <= 5 cm); (4) per piece a
matched-filter scan (FFT) of its pattern over all later times in its TPC on the residual with its prompt light removed,
best 2 minima = its candidate flashes; mf / alpha / apc / cov / sat at those; gates mf >= 0.8, alpha 0.8-6, apc <= -0.5,
sat <= 0.5, cov >= 0.2, piece >= 5 MeV, delay <= 300 ticks; (5) passing probes refined over all lengths (second small
batch), fullest within 0.02 of the best mf; one move per cluster.  Same 13 spills (file 1), apply mode:

| variant | moves | fixed / broken [GeV] | net | purity | spills +/- | s per spill (dense / geometry / perceiver / scan) |
|---|---|---|---|---|---|---|
| flash-first round 2 (above) | 50 | 0.47 / 0.25 | +0.22 | 0.65 | | 100 |
| piece-first, Bragg required | 37 (30 pieces + 7 whole) | 0.44 / 0.10 | +0.34 | 0.81 | 9 / 3 | 13.5 (1.2 / 2.1 / 7.8+0.4 / 1.5) |
| piece-first, any free tip | 50 | 0.47 / 0.30 | +0.17 | 0.61 | 8 / 5 | 22.9 |

Pieces leaving a Bragg ball: 304 / 98 MeV (purity 0.76); pieces from other free ends: 42 / 209 MeV (0.17) - the Bragg
attachment IS the Michel discriminant that the light cannot provide; whole-cluster moves ~136 / 2 MeV.  Delayed class on
these spills 5.0 -> 9.7 % (same as flash-first) at 8x less cost and twice the purity.  Remaining cost is the perceiver on
~600 probe pieces per spill (7.8 s); physics gates on the host (geodesic extent >= 20 cm, Bragg ball within 20 cm of a
host end, piece <= 60 MeV) measured on the same 13 spills: 28 moves (20 pieces + 8 whole), fixed 0.38 / broken 0.08 GeV,
net +0.30, purity 0.83 (pieces 0.76, whole clusters 156 / 5 MeV = 0.97), 10.0 s per spill (perceiver 4.8 s on ~400
probe pieces, geometry 2.2, dense points 1.2, scans 1.1).
Piece-first (Bragg required, ungated) on dev / holdout: files 1-3 (39 spills) 109 moves (89 pieces + 20 whole), fixed
1.04 / broken 0.44 GeV, net +0.59, purity 0.70, spills 29 / 9, delayed class 5.1 -> 10.7 %; files 4-5 (24 spills) 77
moves (51 + 26), 0.47 / 0.33, net +0.14, purity 0.59, 15 / 8, class 7.7 -> 9.9 % (file 5 net -30 MeV).  Same physics as
flash-first on the same spills (1.12 / 0.52 and 0.53 / 0.30) at 13 s instead of 100 s per spill.  Host properties do not
separate good from bad pieces (in-TPC energy, extent, linear density, type: purity stays 0.63-0.69 under any cut); the
gates mf / apc / alpha do not either inside the passing set.  What does: (a) the GAIN-CORRECTED UNEXPLAINED LIGHT at the
destination (placed families allowed a 30 % under-prediction; what remains >= 10 % of the actual): pieces 0.63 -> 0.72
purity, net +461 -> +616 MeV on the 140 dev+holdout pieces - bad moves land on other families' systematics deficits;
(b) whole-cluster moves with alpha <= 2: purity 0.76 -> 0.85.  Both folded into the production kernel
`testing/michel_nd.py` (`run_michel_nd`, `MichelNDConfig`; harness flag `--michel-nd`, stage 4.85 after 4.8 / before 4.9),
evaluated by `v14_michel_nd_eval.py`.  Speed: dense points vectorised (1.9 -> 1.4 s), skeleton on 1-cm voxel
representatives (2.9 -> 1.1 s, same tips); the perceiver batch size does not matter (per-cluster overhead), so the
remaining lever is the probe-piece count; physics gates on the host (extent >= 20 cm, Bragg within 20 cm of a host end,
piece <= 60 MeV) cut the pieces ~2x at a 10-20 % loss of fixes.
Host gates on dev (39 spills): 80 moves, 0.81 / 0.20 GeV, net +0.60 (= ungated), purity 0.70 -> 0.80, 25 / 6, 8.1 s per spill;
holdout 66 moves, 0.43 / 0.27, net +0.16, purity 0.62, 16 / 6.
PRODUCTION KERNEL (`testing/michel_nd.py`, all gates: host extent >= 20 cm, Bragg ball within 20 cm of a host end, piece
5-60 MeV, delay 12-300 ticks, mf >= 0.8, alpha 0.8-6 (whole clusters <= 2), apc <= -0.5, sat <= 0.5, cov >= 0.2, gain-corrected
unexplained light at the destination >= 10 % of the actual) on dev: 64 moves (51 pieces + 13 whole), fixed 0.74 / broken
0.11 GeV, net +0.63, PURITY 0.87 (pieces 0.83, whole clusters 1.00), spills 27 / 4, delayed class 5.1 -> 9.4 %, 8.3 s per
spill (perceiver 4.2 s on ~400 probe pieces at ~11 ms each - voxelisation / per-cluster overhead, independent of batch
size and of materialising the image maps, which the new `amplitudes_only` option of
`process_clusters_to_imageMaps_streaming` skips).  In-chain smoke (harness flag `--michel-nd`, stage 4.85, file 1
events 0 and 8): 7 moves, 6 right, all-TPC +0.12 pp on both spills, 8-15 s.  Holdout of the kernel and the 5-file
certification V14M = v1.4 + `--michel-nd --mnd-host-extent-cm 20 --mnd-bragg-end-cm 20 --mnd-piece-max-mev 60`
(`testing/sbatch_v14m_V14M.sh`, reference V14A) running.
Kernel holdout (files 4-5, 24 spills): 44 moves (33 pieces + 11 whole), fixed 0.35 / broken 0.14 GeV, net +0.21, purity
0.71 (pieces 0.70, whole 0.75), spills 13 / 5, delayed class 7.7 -> 9.5 %, 8.7 s per spill.  Dev + holdout together:
108 moves, 1.09 / 0.25 GeV, purity 0.81, 40 / 9 spills.  First certification attempt crashed 3 of 8 workers with CUDA
out-of-memory (perceiver batch 64 = 11 GiB of dense voxel grids with two workers per GPU); relaunched with batch 16.

**CERTIFIED (arm V14M vs V14A, 63 spills, in-chain, 2026-10-02; `testing/v14m_cert_V14M.log`):**

| | v1.4 | v1.4 + stage 4.85 | delta | spills better / worse |
|---|---|---|---|---|
| NS efficiency | 96.89 | 96.92 | +0.028 pp | 38 / 8 |
| all-TPC accuracy | 95.86 | 95.88 | +0.021 pp | 41 / 11 |
| LE class | 65.55 | 65.57 | | |

Every file up on both metrics (NS +0.01 to +0.07).  Wrong energy: 5-20 MeV 20.59 -> 20.45 GeV, 20-100 MeV 32.64 ->
32.35, >= 100 MeV 52.75 -> 52.39, total 160.63 -> 159.83.  Stage 4.85 in-chain: 388 probe pieces + 390 whole candidates
per spill, 394 perceiver predictions, 1.94 moves per spill (122 in all, 34 whole clusters), 68 % right / 27 % wrong,
fixed 1.14 / broken 0.34 GeV (pieces 791 / 245, whole 345 / 92; purity 0.77), 10.0 s per spill median, p90 14.5
(dense points 1.0, geometry 1.3, perceiver 5.6 + 0.4, scans 1.6).  Chain total 148 s per spill per worker on this node
(v1.4 136).  The ledger gain is small by construction (the delayed-daughter class is 0.64 % of the energy); the product
is the Michel tag: the class goes from ~6 % to ~10 % correctly timed with 77 % purity on the moves.
Launcher `run_v15.sh` (v1.5 CANDIDATE = v1.4 + `--michel-nd` with the gates above); adoption as the default is the
user's call on the time budget (10 s per spill against the 3 s asked for the earlier late stages).

**Track-ending census (2026-10-02, `testing/v15_track_ending_census.py`, truth helper `nd_truth_michel.py`; 20 spills,
7,566 ends of track clusters >= 10 MeV and >= 15 cm; user's proposal: classify every finished track end as exit /
Michel / continuation into another TPC / genuine stop, with dE/dx as the proton handle, right after track merging):**
truth-scored classes: exit (outer face) 18 %, TPC face (continuation into the next module: per-TPC clustering truncates
every crossing track) 36 %, Michel 0.9 % (68 ends), truncated elsewhere 8 %, touching another cluster 7 %, genuine stop 29 %.
By particle: muon ends 31 % exit / 61 % TPC face / 0.0 % Michel (mu- are captured on argon; through-going rock muons
dominate); pion ends 5 % Michel / 19 % stop (absorption); proton ends 41 % stop; e/gamma 61 % stop.  THE MICHEL HOST IN
THIS MC IS A STOPPED PI+ (16 of the 19 isolated Michel-ended tracks; pi+ -> mu+ at rest -> e+, the 4 MeV muon is
invisible), not a muon track.  The proposed rule "low dE/dx track, away from any face, nothing attached" selects
300-1,000 ends per 20 spills of which 2-4 % are Michel ends, 12-19 % truncated, 80-85 % genuine stops (electron
'tracks', absorbed pions, protons) - as a Michel tag by geometry alone it is 4 % pure; Michel hosts read dE/dx 1.9-2.8
MeV/cm (MIP) vs 1.9-5.3 for the other stops, Bragg ratio last-5-cm / mid 0.83 (diluted by the Michel's own hits when
merged) vs 1.08, and only 19 of the 68 Michel ends are isolated (26 within 3 cm of a TPC face, 9 touching another
cluster).  Michel energy at those ends is 29 % correctly timed with stage 4.85.  So the ending classifier is useful for
(a) continuation / truncation bookkeeping (TPC faces are 36 % of all ends) and (b) restricting WHERE the Michel stage
looks (host dE/dx <= 5 MeV/cm, tip >= 5 cm from an outer face: kernel options `host_dedx_max`, `tip_face_min_cm`, being
measured), but the Michel decision itself needs the delayed light, which is what stage 4.85 tests.

**Direct Michel metric (2026-10-02, `testing/v15_michel_metric.py`; truth = e+- from mu decay incl. brems descendants,
host = the stopped muon or, when the muon is invisible, its parent pion; 63 spills):** 916 Michel electrons with >= 1 MeV
of charge (14.5 per spill, 19.7 GeV; KE 23-50 MeV, charge 5-29 MeV); 879 inside the window; 587 (14.8 GeV) merged with
the host at the front end, 292 own clusters; host = pion 561 (10.4 GeV), muon 317 (8.7 GeV); a family within 5 ticks of
the Michel time exists for 533 (13.8 GeV), none for 346 (5.3 GeV); a loose waveform peak for 770 (17.5 GeV).

| Michel energy correctly timed (within 10 ticks) | v1.4 | v1.4 + stage 4.85 |
|---|---|---|
| all (19.7 GeV) | 33.3 % | 36.5 % |
| delay 0-12 ticks (4.5 GeV: shares the host's window) | 94.7 % | 95.0 % |
| delay 12-30 / 30-100 / 100-300 / > 300 ticks | 13 / 13 / 11 / 12 % | 15 / 16 / 19 / 12 % |
| merged with the host (14.8 GeV) | 33.0 % | 36.9 % |
| own cluster (4.3 GeV) | 39.8 % | 41.1 % |
| family within 5 ticks of the Michel time | 46.4 % | 49.3 % |
| no family within 5 ticks | 3.7 % | 8.0 % |
| host = muon / host = pion | 54.7 / 17.6 % | 58.4 / 20.6 % |
| Michels >= 50 % right | 19 % | 23 % |

80 % of the Michel energy sits at the host's time.  The stage's gain on the delayed Michels (> 12 ticks, 13.4 GeV) is
13 -> 17 %, against 17 -> 36 % for the 2x2 stage M on its Michels: ND pile-up (a family within 5 ticks of most Michel
flashes), the 17 % model systematics that forced the purity gates, and the pion-hosted majority (short, often < 20 cm,
hosts) are the differences.  A per-Michel recall audit through the stage (which gate loses each Michel) is the next
measurement (kernel `debug_tag` / `debug_out` hook).

**Ending gates (2026-10-02, kernel options `host_dedx_max`, `tip_face_min_cm`; harness `--mnd-host-dedx-max`,
`--mnd-tip-face-cm`): the census's two usable handles applied to candidate generation.**  A host is skipped when its
cluster energy / skeleton extent exceeds the cap (proton-like or shower-like: no Michel) and a free tip is skipped when it
is within 5 cm of the outer detector box (exiting track).  Kernel on the V14A dumps, 63 spills: certified gates alone
108 moves, fixed 1.09 / broke 0.25 GeV, 8.5 s per spill; + dE/dx <= 5 & face >= 5 (13 spills) 23 moves 0.36 / 0.02,
purity 0.94, 6.3 s; + dE/dx <= 4 & face >= 5 (13 spills) 21 moves 0.36 / 0.01, purity 0.98, 5.6 s; dE/dx <= 5 & face >= 5
on all 63 spills: 102 moves, fixed 1.04 / broke 0.25, purity 0.80, delayed class 6.5 -> 9.3 % (same as without the
gates), 6.1 s per spill (perceiver 2.7 s): the gates remove ~30 % of the pieces (and their model compute) and none of
the recovery.  Re-certification arm V14M2 = v1.4 + stage 4.85 with dE/dx <= 4, face >= 5 (`testing/sbatch_v14m_V14M2.sh`).

**Per-Michel recall audit (2026-10-02, `testing/v15_michel_audit.py` through the kernel's `debug_tag` / `debug_out` hooks,
which now record every candidate's status and every scored hypothesis with its geometry and truth; offline gate scan
`testing/v15_michel_gatescan.py`; 63 spills, 671 delayed Michels (> 12 ticks after the host, >= 5 MeV, 13.25 GeV),
`michel_nd_out/michel_audit5_63.{log,json}`, `gatescan_63.log`):** where each delayed Michel is lost.
- Geometry: 40 % of the delayed Michel energy never enters a candidate; 24 % enters only through the whole-cluster
  route, which is useless for a merged cluster (the host dominates its light); a probe piece holds part of the Michel for
  54 %.  The probe is what the scan uses: ~9 MeV, 5 cm, Michel-pure (median purity 1.00) but holding only ~40 % of the
  Michel (Michels are longer than the probe; the refine step extends the piece afterwards).  Reasons for merged Michels
  without a piece: Bragg ball not within 20 cm of a host end 7 %, no Bragg ball within 16 cm of the tip 7 %, tip within
  3 cm of a dense point 5 %, no skeleton tip inside the Michel 4 %, tip within 5 cm of the outer face 3 %, host extent
  < 20 cm 4 % (1 % at 10 cm), none of the recorded reasons 5-10 %.  The cluster-level dE/dx cap (E / extent <= 4) had
  removed 24 % of the delayed Michel energy from coverage: a merged "host" cluster is usually a whole interaction
  (median 145 MeV, extent 42 cm), so E / extent is not a track dE/dx.
- Scan: 43 % of the delayed Michel energy gets a scored probe (the matched-filter curve is negative at the true time
  for 95 % of them, the true time inside the scan range for 88 %); 11 % has a probe but no scan minimum (true time
  outside [t_c + 12, t_c + 300] for 42 %: the cluster already sits AT the Michel time with a wrong host, or elsewhere).
- Gates (20 spills): a hypothesis exists at the true time for 21 % of the delayed Michel energy; 4 % passes (85 %
  correct afterwards), 17 % fails: apc 14 %, mf 12 %, alpha 8 %, unexpl 3 %, cov 2 % (passing / failing values mf 0.88 /
  0.77, apc -1.05 / +0.18, unexpl 0.36 / 0.14).  apc ("the piece's light is missing at the host time") has little power
  for an 8 MeV piece under a 100 MeV host with 17 % model systematics, but removing it collapses the move purity
  (offline scan: right 0.78 -> 0.48); making it conditional on the piece / host light ratio is no better than simply
  loosening it.
- Perceiver threshold: `min_prediction_threshold = 100` ADC had zeroed the faint channels of 5-20 MeV pieces, which
  made the 90 % support set a handful of channels.  With 0 (`amplitudes_only` calls in the harness, evaluator and
  audit): 63-spill kernel 102 -> 81 moves, fixed 1.04 -> 0.96, broke 0.25 -> 0.15 GeV, purity 0.80 -> 0.86.
- 10 % of the delayed Michel energy sits with its merged cluster AT the Michel time (28 clusters in 20 spills, 2.0 GeV
  of host energy wrong): the mirror move (host earlier, Michel stays) is a separate lever, not built.

Kernel sweeps on the V14A dumps (63 spills, un-zeroed predictions, batch 16, certified gates + tip >= 5 cm from the
outer face unless stated; fixed / broke in GeV, purity = fixed / (fixed + broke) on the moved energy, s per spill):

| handle | moves | fixed | broke | purity | delayed class | s |
|---|---|---|---|---|---|---|
| cluster E/extent <= 5 (zeroed predictions, old) | 102 | 1.04 | 0.25 | 0.80 | 6.5 -> 9.3 % | 6.1 |
| cluster E/extent <= 5 | 81 | 0.96 | 0.15 | 0.86 | 9.0 % | 6.7 |
| cluster E/extent <= 8 | 85 | 0.98 | 0.16 | 0.86 | 9.1 % | 7.5 |
| band dE/dx <= 5 (10 cm of host behind the Bragg stop) | 80 | 0.95 | 0.14 | 0.87 | 9.0 % | 6.3 |
| band dE/dx <= 4 | 77 | 0.92 | 0.14 | 0.87 | 8.8 % | 5.9 |
| band 5, host extent >= 10 cm | 97 | 1.08 | 0.24 | 0.82 | 9.3 % | 7.3 |
| band 5, host extent >= 0 | 100 | 1.09 | 0.26 | 0.81 | 9.4 % | 7.6 |

Offline gate scan (probe-level emulation, extent 20, face 5): mf 0.8 / apc -0.5 (certified) net +0.34 GeV at 0.78
right; mf 0.75 / apc -0.25 net +0.43 at 0.72 right with 50 % more Michel energy in the moves; unexpl higher = purer and
smaller; alpha bounds irrelevant.  The light gates sit on a plateau; the recall ceiling is geometry (candidate
generation) and ND pile-up, not the gate values.

Round 4 (kernel, 63 spills, band 5 / extent 20 / face 5 / un-zeroed): host-remainder rule (piece growth limited by
"host remainder >= 10 MeV" instead of "piece < 50 % of the cluster"; a Michel may out-weigh a short pion host) 81 moves,
0.97 / 0.14, purity 0.87, 6.4 s (neutral, cleaner); Bragg-end 40 cm identical to 20; light gates mf 0.75 / apc -0.25 on
everything 150 moves, 1.46 / 0.46, purity 0.76, delayed class 10.2 %: the 35 extra WHOLE-cluster moves are 50 % pure
(net 0.00: a small cluster carries no Michel prior), the 42 extra PIECE moves 69 % pure (net +0.19: the Bragg-attached
geometry is the prior that pays for the looser light test).  Split gates (pieces 0.75 / -0.25, wholes 0.8 / -0.5):
**115 moves (24 whole), fixed 1.27 / broke 0.27 GeV, net +1.00, purity 0.82, 41 / 7 spills, delayed class 6.5 -> 10.1 %,
6.6 s per spill** (`kern_63_split.log`) against the certified V14M kernel 108 moves, 1.09 / 0.25, 8.5 s on the same dumps.

**v1.5 configuration (run_v15.sh, arm V14M2 = `testing/sbatch_v14m_V14M2.sh`, certification launched 2026-10-02 13:20):**
`--michel-nd --mnd-host-extent-cm 20 --mnd-bragg-end-cm 20 --mnd-piece-max-mev 60 --mnd-delay-max 300 --mnd-batch 16
--mnd-tip-face-cm 5 --mnd-set band_dedx_max=5 max_frac=1.0 remainder_min_mev=10 mf_min=0.75 apc_max=-0.25 whole_mf_min=0.8
whole_apc_max=-0.5`, piece predictions with `min_prediction_threshold=0` (harness `_mnd_predict`).

**CERTIFIED (arm V14M2, 63 spills, in-chain, 2026-10-02 13:20-13:55; `testing/v14m_cert_V14M2.log`):**

| | v1.4 (V14A) | v1.4 + stage 4.85 certified gates (V14M) | v1.4 + stage 4.85 revised (V14M2) |
|---|---|---|---|
| NS efficiency | 96.89 | 96.92 (+0.028, 38/8) | 96.92 (+0.033 pp, 38/6) |
| all-TPC accuracy | 95.86 | 95.88 (+0.021, 41/11) | 95.89 (+0.024 pp, 44/9) |
| LE class | 65.55 | 65.57 | 65.57 |
| stage 4.85 moves / fixed / broke | | 122 / 1.14 / 0.34 GeV | 126 / 1.30 / 0.37 GeV |
| probe pieces per spill | | 388 | 209 |
| stage time per spill, median (p90) | | 10.0 s (14.5) | 7.1 s (9.8) |

V14M2 vs V14M directly: NS +0.005 pp (20/15 spills), all-TPC +0.003 (23/16): the same ledger at 71 % of the stage cost,
with 0.16 GeV more fixed and 0.03 GeV more broken; pieces fixed 955 / broke 279 MeV (V14M 791 / 245), whole-cluster moves
identical by construction (34, 345 / 92 MeV).  Wrong energy total 160.63 -> 159.69 GeV (V14M 159.83).  Stage parts:
dense 1.0, geometry 1.4, perceiver 3.1 + 0.4, scans 1.1 s.  Event median 142.9 s (V14M 143.8; the backbone stage ran
5 s slower on this node, untouched code).  run_v15.sh carries this configuration; adoption as v1.5 remains the user's
call (7 s per spill for a Michel class that goes from ~6.5 to ~10 % correctly timed).
Direct Michel metric (`michel_nd_out/michel_metric_v14m2.log`, 916 Michels, 19.7 GeV): energy correctly timed
v1.4 33.3 % / V14M 36.5 % / V14M2 37.0 %; delayed 30-100 ticks 12.8 / 16.0 / 17.6 %, 100-300 ticks 11.1 / 18.6 / 18.8 %;
pion-hosted 17.6 / 20.6 / 21.5 %, muon-hosted 54.7 / 58.4 / 58.4 %; no family within 5 ticks of the Michel flash
3.7 / 8.0 / 8.6 %; Michels >= 50 % right 19 / 23 / 23 %.

**How faint are the delayed Michels the stage never sees?  (2026-10-02, user's question; `testing/v15_michel_waveforms.py`
(GPU: perceiver light of every delayed Michel from its own hits, actual / chain-model amplitude at the Michel time, example
figures), `v15_michel_photons.py` (truth photons), `v15_michel_light_summary.py`; outputs `michel_nd_out/waveforms/`):**
NOT faint.  For the 336 delayed Michels (6.2 GeV) that never become a candidate piece, the Michel's own light on its
brightest channel is median 1,690 ADC (p10 520, p90 8,200; summed over the TPC median 20,600 ADC on ~33 support channels);
72 % exceed 1,000 ADC, 92 % are at least 10 % of what is actually on that channel at that time, 54 % at least 50 %,
14 % brighter than everything else there.  The covered Michels are only slightly brighter (median 2,220 ADC).  131 of the
336 (2.2 GeV) have NO placed family within 15 ticks: an isolated, unexplained flash of ~1,500 ADC that no cluster claims.
The perceiver prediction is calibrated: on the isolated Michels the actual-minus-model residual on the best channel is
1.12x (uncovered) / 1.00x (covered) the prediction.  By exclusion reason (median Michel light / median actual there):
no Bragg ball within 16 cm 141 Michels 2.6 GeV (2,320 / 3,970 ADC), host extent < 20 cm 111, 1.75 GeV (2,470 / 3,980),
Bragg not at a host end 55 (960 / 4,380), tip not free 40 (1,750 / 5,160), no tip in Michel 39 (1,380 / 5,240), outer-face
margin 44 (1,200 / 3,840), host < 10 MeV 46 (1,750 / 2,840).  Example figures (charge view with the Michel hits in red,
full-TPC summed waveform with host / Michel times, zooms on the Michel's six brightest channels with actual, chain model
and model + Michel): `michel_nd_out/waveforms/michel_<spill>_m<id>.png`; population figure `michel_light_population.png`.
Truth photons: produced (segments.n_photons, complete) median 740,000 per delayed Michel (24,340 per MeV of dE; the
Michel deposits 17-47 MeV, charge sees 10-29 MeV).  The flow files' detected-photon truth (`mc_truth/light_module<N>`)
is unusable in this production: the (4, 64) block is four identical copies, only the odd charge TPC of each module
carries photons, and every segment, Michel or host, shows 1-3 detected photons per MeV (1e-4 of production), inconsistent
with the measured 1-20 ADC per photon of the same tables' brighter flashes; not used.  CONCLUSION: the delayed-Michel
ceiling is the geometric candidate generator (Bragg-ball requirement and host-extent gate above all: stopping muons have
no detectable Bragg ball at the 5 MeV / 1.5 cm dense-point definition, short pions fail the 20 cm extent), not the
light.  The light of a typical uncovered Michel is 0.5x the pile-up on its best channel and 1-10x on isolated ones.

#### Stage 4.86 "unexplained flashes at track ends" (2026-10-05, user's request; `testing/michel_flash.py`, evaluator
`v14_michel_nd_eval.py --flash / --flash-only / --fset`, audit `v15_michel_audit.py --flash`)

Design: per light TPC, residual flashes = local maxima of the summed positive residual (actual - chain model, non-vetoed
channels) with sum >= 10,000 ADC, >= 5 channels where the residual is >= 300 ADC and >= 30 % of the actual, residual /
actual >= 0.3 on those channels, a rise (S[t] - S[t-6]) / S[t] >= 0.2, no placed family within 10 ticks, channels above
half the maximum spanning <= 120 cm in y (ND light channels sit on the +-z faces, sorted in y; a whole-wall residual is a
big-flash mismatch), at most 8 per TPC.  Candidates per flash: EVERY skeleton tip of every cluster >= 10 MeV in the TPC
whose current time is >= 12 ticks before the flash and whose tip lies within 30 cm (y) of the flash's lit range, the 8
nearest in y, grown 2-16 cm (piece 5-60 MeV, host remainder >= 10 MeV) - no Bragg, extent or dE/dx rule; plus whole
clusters 5-80 MeV in the window.  One batched perceiver call for the 5 cm probe AND the longest piece of each tip; each
(flash, piece) scored at the flash time with the stage-4.85 tests, the residual measured above its per-channel median
over the 12 ticks before the flash and the amplitude scale fitted inside the flash window (the full-trace product
multiplies the template's slow tail with the decaying residual of earlier flashes and went negative: v2 defect, fixed
in v3); passing pieces refined over lengths; best per flash, one move per cluster.

What the finder sees (10 spills, CPU census): 71 % of residual flashes have NO charge within 5 ticks (light with no hits:
dead regions, between modules), 24 % sit on charge already timed right (model deficit), 5 % on mistimed charge >= 5 MeV;
22 of 23 Michel flashes are in the last class.  Michel flashes: summed residual 17-45k ADC, 6-30 lit channels.  The
first version (sum >= 3,000, 3 channels, rise 0.2) produced 108-198 flashes and up to 6,000 flash-tip pairs per spill
(34 s); the thresholds above give 28 per spill.

Per-Michel audit (20 spills, after the v1.5 piece-first stage): a flash within 5 ticks of the Michel exists for 24 %
of the delayed Michel energy (73 % of the isolated Michels; 76 % have no flash: a placed family within 10 ticks for
most of them, i.e. the flash is explained or shared - the finder is blind there by construction); a piece covers 19 %;
a hypothesis at the true time 17 %; passing 4 % (73 % correct afterwards); failing: mf 10 %, alpha 10 %, apc 9 %.
Michel probes at the right time read mf 0.78 / alpha 1.0 / apc -0.5 (medians); non-Michel probes mf 0.36 / alpha 0.0.
Offline gate scan (20 spills, probes): mf 0.75 / apc -0.25 -> 12 moves, 60 % right, fixed 97 / broke 65 MeV; mf 0.8 /
apc -0.5 -> 9 moves, 89 % right, 97 / 13 MeV (adopted for this stage; the flash prior does not buy a looser light gate).
63 spills, v2 gates (mf 0.75 / apc -0.25, before the alpha fix), run after the v1.5 stage: flash stage 44 moves, fixed
0.43 / broke 0.28 GeV (43 pieces: 24 right 417 MeV, 15 wrong 232 MeV, 13 of them true breaks; wrong ones read mf 0.82 vs
0.89 and apc -0.64 vs -1.02), 28 flashes / 297 candidates / 130 pairs / 222 predictions per spill, 4.5 s per spill;
combined 159 moves, 1.70 / 0.55 GeV, delayed class 6.5 -> 11.6 % (piece-first alone 10.1 %), 11.3 s per spill.
v3 (alpha fitted in the flash window, baseline 12 ticks; mf 0.75 / apc -0.25): flash stage 47 moves, 0.48 / 0.21 GeV,
4.0 s; combined 162 moves, 1.74 / 0.48, delayed class 11.7 %.  **v4 (mf 0.8 / apc -0.5, the stage's defaults): flash
stage 32 moves, fixed 0.40 / broke 0.06 GeV (purity 0.87), 28 flashes / 130 pairs / 220 predictions per spill, 4.2 s;
combined with the v1.5 piece-first stage 147 moves, 1.67 / 0.34 GeV, net +1.33, purity 0.83, 45 / 9 spills, delayed class
6.5 -> 11.3 % (piece-first alone 10.1 %), 10.7 s per spill** (`michel_nd_out/flash4_63_after.log`).  Verdict: the stage
works and is clean, but it adds ~0.4 GeV per 63 spills (+1.2 pp on the delayed Michels) for +4 s per spill, because
it can only see the isolated third of the delayed Michels and the light test then passes a quarter of those.
Harness: `--michel-flash --mfl-set k=v` (stage 4.86, after 4.85, shares its predictor); certification arm V14M3 =
V14M2 + `--michel-flash` (`testing/sbatch_v14m_V14M3.sh`, launched 2026-10-05).

**CERTIFIED (arm V14M3 = V14M2 + stage 4.86, 63 spills, in-chain, 2026-10-05; `testing/v14m_cert_V14M3.log`):**

| | v1.4 (V14A) | + 4.85 revised (V14M2) | + 4.85 + 4.86 (V14M3) |
|---|---|---|---|
| NS efficiency | 96.89 | 96.92 (+0.033, 38/6) | 96.93 (+0.047 pp, 46/6) |
| all-TPC accuracy | 95.86 | 95.89 (+0.024, 44/9) | 95.90 (+0.034 pp, 48/9) |
| stage 4.86 moves, fixed / broke | | | 31, 0.43 / 0.04 GeV (87 % right) |
| stage 4.86 time per spill, median (p90) | | | 4.6 s (7.2) |
| event median | 129 s | 142.9 s | 147.7 s |

V14M3 vs V14M2 directly: NS +0.014 pp (19/3 spills), all-TPC +0.010 (23/3), every file up; wrong energy total
159.69 -> 159.31 GeV.  Stage 4.86 in-chain: 28 flashes, 297 candidates, 130 pairs, 220 predictions per spill.
run_v15.sh now carries `--michel-flash` (one flag to drop it); adoption of v1.5 = user's call.
  Speed levers left:
the perceiver's ~11 ms per piece is per-group overhead (grouping / voxelisation), not GPU compute - a piece-level
voxeliser that reuses the host cluster's voxel grid, or an amplitude-native (no image) prediction path, would take the
stage to ~4 s; restricting hosts to track-type clusters does not pay (37 of 85 true Michel hosts are typed 'cluster').

### v1.4 ADOPTED 2026-10-01 = v1.3 + stage 4.9 light-vote force (20 cm) + stage 4.8 residual-flash filling (tag CLMatching_v1.4)

5-file certification, arm V14A vs V2A5 (v1.3), 63 paired spills, in-chain (`testing/sbatch_v14_V14A.sh`,
evaluation `testing/v14_cert_V14A.log`):

| | v1.3 | **v1.4** | delta | spills better / worse |
|---|---|---|---|---|
| NS efficiency (TPCs <= 1.4 GeV) | 96.77 | **96.89** | +0.114 pp | 63 / 0 |
| all-TPC accuracy | 95.76 | **95.86** | +0.097 pp | 62 / 1 |
| LE class (noise + < 1.5 MeV clusters) | 60.89 | **65.55** | +4.66 pp | |

Per file NS +0.11 to +0.12, all-TPC +0.06 to +0.12, every file up on both.  Wrong energy by class (GeV): noise 23.17 ->
19.55, 1.5-5 MeV 32.57 -> 32.35, 5-20 MeV 20.83 -> 20.59, 20-100 MeV 32.71 -> 32.64, >= 100 MeV 52.37 -> 52.75, total
164.38 -> 160.63.  Stage 4.8 in-chain: 130 residual flashes and 5.0 moves (18 MeV) per spill, fixed 0.70 / broken 0.17
GeV over the 63 spills (offline emulation predicted 0.68 / 0.18), 0.30 s per spill; stage 4.9 light-vote: 554 hits per
spill get the light factor, forced-hit accuracy 60.2 -> 66.1 %, 0.74 s per spill (+3.5 s once per worker for the SiPM
geometry).  New stages together 1.0 s per spill, inside the 3 s budget; total 133.5 s per spill per worker median on
this node (v1.3 measured 146.6 s on another node; the difference is node speed, the per-stage shares are unchanged).
Committed and tagged CLMatching_v1.4 locally (not pushed).  Remaining known levers unchanged: light-prediction engine
retraining for < 20 MeV (the deficit-filling degeneracy caps every light-based mover), delayed-daughter splitter,
apex prior for per-hit precision, and the pending blip re-homing v2 (its noise vote is now part of 4.9; its contact /
re-attach operators would need re-certification on top of v1.4).
*(user: adopt vote x light beyond 20 cm as long as the new stages stay under 3 s per spill; and why does the chain not
place small clusters at isolated flashes a light-only finder sees - try cosine matching on restricted channels, or the
reverse construction: start from the light that needs filling and pick the nearby cluster that matches it best; a finder
peak on the tail of a large flash is not a real t0 and must be ignored)*

**Why the chain leaves isolated small flashes unfilled** (`testing/v14_isolated_flash_diag.py`, 20 spills, log
`testing/flash_discovery/isolated_diag_V2A5.log`): 162 isolated 2-20 MeV truth flashes seen by the loose finder but
without a chain family (8 per spill, 46 MeV per spill = 0.08 % of the energy); 89 % of that energy is in templated
clusters (median 3.1 MeV, one cluster per flash), so the clusters exist and have a prediction.  They sit at a host
family of median 153 MeV (a 50x brighter slot).  With the cluster's own light added back, the Poisson chi2 on its
support channels prefers the TRUE time in only 30 % of the cases (median delta chi2 -35 at the true time vs -135 at the
chosen one): a few per cent of a big family's 17 % model deficit outweighs the cluster's whole flash, the deficit-filling
measured in V200_DESIGN.md.  The pattern cosine on the support channels does separate better than the chi2 but not
cleanly: cos >= 0.9 at the true time in 44 %, cos(true) > cos(chosen) + 0.1 in 45 %.  Among ALL templated 0.5-20 MeV
clusters (20 628, 84 GeV, 81.5 % correct) a naive "move to the best residual flash if cos >= 0.9 and better than now"
rule fires on 13 % of the wrong ones (right in 10 %) and on 10 % of the right ones - a net loss, because right clusters
outnumber wrong ones 4:1.

**What the unexplained light contains** (`testing/v14_resid_flash_census.py`, 20 spills): residual-flash finder (10 sigma
on actual - rendered, away from family times) gives 1556 peaks per spill with 7 % purity (truth flash in the TPC within
5 ticks); purity rises with residual / model at the peak: 3.6 % below 0.2, 10 % at 0.2-0.5, 18 % at 0.5-1, 36 % at
1-3, 51 % above 3 - the tail bumps of big pulses are the fakes, as the user said.  Truth energy at residual flashes
(TPC level) 362 MeV per spill (0.6 % of the energy), 11 % of it currently correct: noise hits 24 %, templated < 20 MeV
whole clusters 34 % (2 % correct: the mover's target, 124 MeV per spill = 0.2 pp if all were fixed), 20-100 MeV whole
9 %, sub-pieces of clusters timed elsewhere (delayed daughters, mixed clusters) 29 %, untemplated 2 %.  So the light
the finder sees and the chain does not fill is a 0.6 % slice, and only a third of it is movable whole clusters.

**Stage 4.8 residual-flash filling** (`testing/resid_fill.py`, flag `--resid-fill`, emulator `resid_fill_offline.py`,
grids `resid_fill_grid{1,2,3}.log`, dev/holdout `resid_fill_devhold.log`): the reverse construction.  Per light TPC:
residual flashes = peaks of the summed residual with prominence >= 10 sigma, not within 5 ticks of a family time, kept
only if the residual at the peak is >= 0.5 x the modelled light there (tail rejection; 0.3 -> 0.5 halves the candidates,
same gain, half the breakage) and >= 600 ADC.  Brightest flash first: among the templated clusters 0.5-100 MeV of the
TPC not already there, score = cosine between the predicted pattern and the residual pattern at the flash on the
cluster's 90 % support channels; accept the best iff cos >= 0.9, cos > cos at the current slot (own light added back)
+ 0.1, residual / template amplitude in [0.5, 3], and the cluster is >= 20 cm from any co-timed host (spray stays with
its parent); move, re-render, repeat up to 4 per flash.  Measured choices: the 8 nearest channels (user's suggestion)
are too few to discriminate (55 % of moves wrong without gates, 32 % with); the 90 % support or the 30 nearest channels
give 16 % wrong; the blip-rehome glare gate (own / others >= 0.17) is NOT used here - it blocks exactly the regime-A
clusters parked at brighter slots (moves 27 -> 6 per spill) and the host-distance gate alone protects the spray;
extending candidates to 100 MeV adds the 20-100 MeV whole clusters at residual flashes.  Offline, force coverage
replayed with the v1.4 light-vote rule in both arms:

| | files 1-3 (39 spills, tuning) | files 4-5 (24 spills, holdout) |
|---|---|---|
| all-TPC | 96.286 -> 96.312 (+0.026 pp) | 95.124 -> 95.144 (+0.021 pp) |
| fixed / broken | 0.47 / 0.13 GeV | 0.21 / 0.05 GeV |
| spills better / worse | 30 / 9 | 16 / 7 |
| moves per spill | 4.6 (19 MeV) | 5.1 (17 MeV) |
| stage time | 0.36 s | 0.40 s |

Every file positive (+0.007 to +0.034 pp).  Moved clusters were wrong 88 % of the time; outcome right 55 %, neutral
(wrong before and after: moved to a residual flash that was not theirs) 33 %, wrong 12 %.  The residual stays
limited by the deficit-filling degeneracy, not by the search: the remaining 84 % of the wrong small-cluster energy sits
at times where the light is already explained by another family (no residual peak to fill).
v1.4 = v1.3 + `--force-light-cm 20` (stage 4.9, 0.84 s/spill, +0.095 pp offline) + `--resid-fill` (stage 4.8, 0.4 s,
+0.024 pp offline); `run_v14.sh`; certification arm V14A (`testing/sbatch_v14_V14A.sh`, reference V2A5) running.

### Flash discovery: chain t0s vs light-only flash finders (2026-10-01, measurement on the v1.3 certification dumps)
*(user: does the residual-based chain find more t0s than flash finding, in particular low-energy light buried under
bigger flashes, and how often are the new t0s right?)*  `testing/v13_flash_discovery.py` + `_plot.py`, log and figure in
`testing/flash_discovery/` (`flash_discovery_V2A5.log`, `flash_discovery_v13.png`).

Setup per light TPC and spill (4409 TPC-events): truth flashes = truth interaction-time groups with >= 0.5 MeV in the TPC
(8.6 per TPC-event); flash finders on the actual light alone: S(k) = summed waveform of the non-vetoed channels, peak
prominence k x pre-trigger sigma (S10 ~ 4 PE summed ... S60 ~ 27 PE), and C(2PE, n) = per-channel 2-PE hits coincident on
>= n channels; chain t0s = family times of the final assignment (8.0 per TPC-event).  Match tolerance 5 ticks; accidental
matches measured with truth shifted by 300 ticks and removed.  Fakes are judged at module level (the two drift volumes
of a module share the light).  The chain has 0.50 fake t0s per TPC-event (purity 0.94 TPC / 0.95 module); the finder
with the same fake rate is S40 (0.41) or C2PE/6 (0.31); S10 has 9.6 fakes per TPC-event.
Discovery efficiency (accidental-corrected) vs energy deposited in the TPC at the flash:

| E in TPC | S10 (loose) | S40 (matched fakes) | C2PE/6 | chain |
|---|---|---|---|---|
| 0.5-2 MeV | 0.66 | 0.20 | 0.05 | 0.23 |
| 2-5 MeV | 0.82 | 0.58 | 0.32 | 0.64 |
| 5-20 MeV | 0.91 | 0.85 | 0.79 | 0.85 |
| 20-100 MeV | 0.96 | 0.95 | 0.97 | 0.97 |
| 100-500 MeV | 0.98 | 0.98 | 0.99 | 0.99 |

Buried flashes (gap to the nearest >= 3x brighter truth flash in the module), 2-20 MeV: within 5 ticks S40 0.80 / S10
0.89 / chain 0.87; 5-20 ticks S40 0.51 / S10 0.66 / chain 0.69; 20-60 ticks S40 0.79 / S10 0.95 / chain 0.75; isolated
S40 0.81 / S10 0.97 / chain 0.77.  20-100 MeV: within 5 ticks S40 0.89 / chain 0.98; 5-20 ticks S40 0.82 / S10 0.84 /
chain 0.94; isolated 0.98 / 0.98 / 0.98.
=> The residual-based chain does resolve light buried within ~20 ticks (320 ns) of a brighter flash better than any
light-only finder, at any threshold, and for 20-100 MeV deposits it is close to complete (94-98 %) where finders stop at
82-89 % at the same fake rate.  For ISOLATED 2-20 MeV flashes the picture inverts: a loose finder sees 97 % of them (the
light is there) but the chain gives only 77 % a t0 - the small cluster was assigned to another flash (the known
small-cluster assignment error), not a light-visibility limit.  Below 2 MeV neither works (chain 0.23).
New t0s (chain t0 with no finder peak within 5 ticks), vs S40: 1.0 per TPC-event, 63 % real in the TPC (70 % in the
module), carrying 37 GeV in 63 spills (1 % of the energy) of which 77 % is correctly placed; by placed energy >= 100 MeV
94 % real (93 % of the energy correct), 20-100 MeV 72 %, < 20 MeV 60-64 % (energy 50-68 % correct).  Vs the loose S10:
0.52 new per TPC-event, 50 % real, 25 GeV, 78 % correct.  The reverse set (finder peaks with no chain t0) carries 11 GeV
of truth energy vs S40 and 20 GeV vs S10 (0.3-0.5 % of all energy): flashes the light shows but the chain merged into a
neighbour (isolated small clusters, Michel daughters).  Implication: a light-only flash finder could SEED candidate
times for small clusters (the isolated 2-20 MeV gap, 0.97 vs 0.77), but the decision problem (deficit filling at
brighter slots) remains the limit measured in V200_DESIGN.md.

### Stage 4.9 light-vote force rule (2026-09-30, TESTED offline on the v1.3 certification dumps; adoption = user's call)
*(user: beyond ~10-20 cm spatial information is useless for an isolated hit; decide it from the residual light on the
nearest channels; keep it ultra fast)*

Anatomy first (`testing/v13_noise_anatomy.py`): the unclustered class (54.9 GeV in 63 spills, 1.4 % of the energy,
58.6 % correct) is exclusively single hits (58 %) and pairs (42 %) that never met the leftover DBSCAN's 3 hits within
4 cm; 97 % of the energy is >= 4 cm from any cluster, 45 % is 10-25 cm away, 13 % is > 25 cm away.  Correctness tracks
the probability that the nearest clustered hit belongs to the same interaction (78 / 55 / 32 % at 4-10 / 10-25 / > 25
cm); only 4-11 % of the energy has an undiscovered truth flash; the oracle "best family time in the TPC" would be
81-87 % correct on the far hits.  Clusters with a light prediction are never left to geometry (~0.4 GeV).

Rule (`testing/force_light_vote.py`, harness `--force-light-cm D`): noise hits within 3 cm of an assigned hit keep
its time; beyond that every assigned cluster within 60 cm votes for the family time within +-6 ticks of its own with
w = exp(-(d_c - d1)/14 cm) (x 0.5 for clusters >= 20 MeV) [the force_noise_vote kernel, one vectorised pass, voxel
representatives 3 cm beyond 15 cm]; for hits farther than D from any assigned hit each candidate's vote is multiplied
by (1 + res/median res), res = positive light residual (actual - everything rendered) at the candidate time, averaged
over +-2 ticks around the template peak on the 6 nearest non-vetoed channels with 1/d^2 weights = the un-rendered
light of the parent's own low-energy deposits near the hit.  Light scans tried and rejected: actual light (worse than
nearest hit: picks the brightest flash), residual/actual, residual over all 120 channels (systematics-dominated; the
best channel count is 3-6), grouping neighbouring far hits before deciding (no gain), largest-family and long-range
energy priors (much worse).  Exact emulation on the 63 certification spills (force is the last assignment stage):

| rule for noise hits | 3-10 cm | 10-15 cm | 15-20 cm | > 20 cm | noise class | all-TPC |
|---|---|---|---|---|---|---|
| nearest assigned hit (v1.3) | 72.7 % | 59.7 % | 51.0 % | 35.8 % | 58.6 % | - |
| cluster-density vote beyond 3 cm | 77.5 % | 66.8 % | 59.1 % | 42.8 % | 64.7 % | +0.088 pp |
| vote x light beyond 20 cm | 77.5 % | 66.8 % | 59.1 % | 44.8 % | 65.2 % | +0.095 pp |
| vote x light beyond 10 cm | 77.5 % | 67.0 % | 59.8 % | 44.7 % | 65.3 % | +0.096 pp |
| vote x light beyond 3 cm | 76.7 % | 67.0 % | 59.8 % | 44.8 % | 65.0 % | +0.092 pp |

Energy per distance bin 1.1 / 21.8 / 11.5 / 8.0 / 12.5 GeV.  The light term helps only where geometry is weak (+2 pp
beyond 20 cm, +0.7 pp at 15-20 cm, nothing below 10 cm, slightly negative at 3-10 cm), so D = 10 cm is the setting.
Most of the gain is the vote itself (already part of the pending blip re-homing v2 as --noise-vote); the light factor
adds +0.008 pp all-TPC (+0.6 pp on the noise class) on both the v1.3 dumps and the BLIP5B dumps (+0.078 -> +0.088 pp
there).  Cost 0.84 s per spill in total, of which 0.25 s is the isolation pass the current stage already pays and
~0.04 s is the light part.  The remaining 20-pp gap to the oracle is not in the local geometry nor in the local
light: isolated 0.3 MeV deposits carry no information about which of the TPC's interactions emitted them.
In-chain smoke (`testing/sbatch_fl10_smoke.sh`, file 1 events 0 and 8, arm FL10smoke): forced-hit accuracy 56.9 -> 64.3 %
and 55.2 -> 63.8 %, LE 57.9 -> 64.0 % and 57.1 -> 63.5 %, all-TPC +0.12 / +0.11 pp vs the v1.3 rows; stage 0.4-1.1 s
(+3.5 s once per worker for the SiPM geometry load).  Matches the offline emulation.

### Stage 4.7 blip re-homing (2026-09-24, TESTED, adoption = user's call)  *(user: "we really need to get these low energy blips to the right place")*

`testing/blip_rehome.py`, flag `--blip-rehome` (off by default; `--blip-no-contact`, `--blip-no-iso`,
`--blip-contact-cm`), runs after 4.5 swap-fix and before 4.9 force; touches only 0.5-20 MeV clusters.
Census that motivated it (v1.3 cert dumps, 63 ev): sub-20 MeV + noise = 79 of 164 GeV wrong; a TPC holding
only small clusters is 0.9 % of TPC-events; templated 0.5-20 MeV wrong splits into clusters whose true time
has no >= 20 MeV deposit in their TPC (20.5 GeV, own light ~0.3 of the light at the true time, wrong ones sit
at 3x / 8x brighter slots) and clusters co-timed with a big object (31.6 GeV, own light ~0.03 = glare).
Every light-driven small-cluster mover tried offline (quiet-hole fill, local-channel cosine of single clusters
or 30-cm groups, peak-centric exclusive filling, cross-TPC parent times) fixes the isolated class well but
breaks correctly placed co-timed spray more (~9x more correct energy sits at bright slots).  The separating
features among proposals are the distance to the current host family (AUC 0.82) and the own light relative to
the host's 17 % systematic band (AUC 0.80), used as gates below (chosen on files 1-3, confirmed on 4-5).
Operators: (1) contact certificate - a small cluster touching (<= 2 cm) placed >= 20 MeV clusters that all share
one time (spread <= 3 ticks) takes it (the joint's connectivity certificate, now also in shower TPCs where the
4.1 matrix has no spatial prior); (2) isolated re-homing - candidates = times of placed >= 20 MeV clusters in
OTHER TPCs within 150 cm at which the own TPC holds no big family; local-channel cosine of the summed template of
small clusters within 30 cm vs the positive residual >= 0.9 and > current + 0.1, amplitude ratio in [0.5, 2],
host distance >= 20 cm, own light >= 0.1 x 0.17 x others' predicted light; static decisions, then commit.
5-file certification (arm BLIP5 vs V2A5, 63 paired events, in-chain): all-TPC 95.76 -> 95.83 (+0.066 pp, 55/8
events better/worse), NS 96.77 -> 96.81 (+0.036, 51/12), all five files up on both (+0.06..+0.07 / +0.03..+0.04),
LE 60.89 -> 60.98; wrong energy 1.5-5 MeV 32.57 -> 31.83 GeV, 5-20 MeV 20.83 -> 19.02, noise and >= 20 MeV
unchanged (total 164.38 -> 161.85 GeV); stage audit contact fixed/broke 2.36/0.49 GeV, isolated 1.46/0.71 GeV;
1.75 s/ev.  Offline emulation (testing/blip_rehome_offline.py, force replayed on the V2A5 dumps) predicted
+0.066 / +0.036 exactly.  Outputs /pscratch/sd/y/yuxuan/v11arms/BLIP5; runner testing/sbatch_blip_BLIP5.sh.
Not addressed: noise hits (23 GeV wrong, force at ~60 %) and most of the isolated class (what separates
'pulled away' from 'belongs here' beyond host distance and glare is still open).

#### Stage 4.7 v2 (2026-09-25, TESTED, adoption = user's call): + operator 3 re-attach + 4.9 noise-hit vote

Four levers explored in parallel on top of stage 4.7 (offline emulator, thresholds on files 1-3, holdout 4-5):
noise-hit vote +0.095/+0.095 pp, re-attach +0.093/+0.082, near-contact +0.054/+0.047 (redundant with re-attach:
+0.003 on top, not used), isolated-operator upgrade +0.006/+0.006 (not used).
- Operator 3 re-attach (`--blip-reattach`, blip_rehome.py `_reattach_moves`, decided after (1)/(2) are committed):
  a templated 0.5-20 MeV cluster whose current slot has no co-timed placed >= 20 MeV cluster within 20 cm in its TPC
  (10 cm in any TPC) takes the time of the placed big cluster of its own TPC (other time, within 40 cm, not in 2-cm
  contact) minimising own(t) x d, own(t) = own light / (0.17 x base light on its support channels at t),
  iff <= 1.25 cm (spray is invisible under its parent's glare; the further away, the more invisible it must be).
- 4.9 noise-hit vote (`--noise-vote`, force_noise_vote.py, with --force-simple): a noise hit within 3 cm of an
  assigned hit keeps the nearest-hit time; otherwise every assigned cluster within 100 cm votes for its time with
  weight exp(-(d_c - d1)/14 cm) x (0.5 if >= 20 MeV), votes summed in +-6-tick windows, heaviest cluster of the
  winning window gives the time (nearest hit belongs to the hit's own interaction only 56-76 % of the time beyond 5 cm).
5-file certification in-chain (arm BLIP5B = v1.3 + --blip-rehome --blip-reattach --noise-vote, vs V2A5, 63 events):
all-TPC 95.76 -> 96.00 (+0.238 pp, 63/0 events), NS 96.77 -> 96.98 (+0.210, 63/0), LE 60.89 -> 66.05, every file
+0.22..+0.25 all-TPC / +0.19..+0.24 NS; wrong energy noise 23.17 -> 19.64 GeV, 1.5-5 MeV 32.57 -> 29.63, 5-20 MeV
20.83 -> 18.04, >= 20 MeV unchanged (total 164.38 -> 155.09); stage audit contact 2.36/0.49, iso 1.46/0.71,
re-attach 5.67/2.48 GeV fixed/broke; median 142 -> 145 s/ev (stage 2.45 s/ev + vote).  Offline predicted +0.241/+0.209.
Outputs /pscratch/sd/y/yuxuan/v11arms/BLIP5B; runner testing/sbatch_blip_BLIP5B.sh (batch wrapper batch_blip_BLIP5B.sbatch).
