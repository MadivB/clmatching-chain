# ND-LAr charge-light matching v1.3: what the output can do, how well, and where it fails

> **Update 2026-10-01, v1.4 certified** (same 63 spills): v1.4 = v1.3 + two late stages that act only on isolated noise
> hits (light-weighted cluster vote, stage 4.9) and on unexplained residual flashes (reverse-construction filling,
> stage 4.8). NS 96.77 -> 96.89 (+0.114 pp, 63/0 spills), all-TPC 95.76 -> 95.86 (+0.097, 62/1), noise-hit class
> 58 -> 66 % correct, LE 60.9 -> 65.6; everything else in this document (interaction recovery, precision, reading keys,
> separation, failure modes) is unchanged to the quoted precision. Details in VERSION_HISTORY.md (v1.4 entry).

Written for a Claude session helping with the thesis. `V130_CHAIN_REPORT.md` (same directory) describes the method
stage by stage; this file describes the OUTPUT as a physics tool: accuracy, precision, reading keys, measured failure
modes, and the physics reach they imply. Every number is measured on the v1.3 certification run
(`/pscratch/sd/y/yuxuan/v11arms/V2A5`: MiniProdN5p1 FHC full-spill ND simulation, flow files 1-5, 63 spills, no truth used
anywhere in the chain; truth is used only to score). The scripts that produced each table are listed in section 9 so
the numbers can be regenerated or extended.

## 1. What the chain delivers, per spill

The input is one FLOW spill: ~177 k charge hits with truth (median), ~1400 clusters, 70 active TPCs, 60 GeV deposited,
~100 distinct interaction times (neutrino vertices in the argon plus rock-muon, rock-neutron and photon entries).
The output is a time for every charge hit and a set of quality and structure tables:

| Field | Meaning | Units, conventions |
|---|---|---|
| `t0` per hit (`ts_final` in the dumps) | the time of the light flash the hit is assigned to, continuous | ticks of 16 ns from the start of the event window; assigned to 99.95 % of the energy; hits isolated by > 25 cm get NaN |
| `hit_t0_conf` | per-cluster confidence, 0-1 | amplitude-weighted fraction of the cluster's light channels whose own best time agrees with the assigned time within 10 ticks; NaN for light-blind clusters (no template) |
| `hit_t0_sigma_p` | per-hit timing uncertainty from the precision stage | ticks in the dumps (x 16 for ns); delivered on 98.7 % of hits, median 0.42 ns |
| family table (`row["phase3"]["families"]`) | one entry per light TPC per flash: apex time, sigma, apex position, energy, channel count | ~520 families per spill; sigma in ns |
| piece table (`row["phase3"]["pieces"]`) | displaced sub-pieces of a family that received their own time | ~25 light-resolved pieces per spill: delay, distance, beta, sigma |

Convention to remember: light-derived times sit a constant -5.6 ns from truth (light path plus template alignment),
stable to 0.25 ns spill to spill. It is one detector constant. Any absolute-time use must subtract it; relative
times (family to family, piece to core) do not see it.

## 2. Assignment accuracy

"Correct" = |t0 - truth t0| <= 10 ticks (160 ns), energy-weighted unless stated. All-TPC 95.76 %, non-shower TPCs
(<= 1.4 GeV in the TPC, 62.5 % of the energy) 96.77 %, hit-count 95.98 %. The spill-to-spill spread is wide: per-spill
all-TPC accuracy min 87.9 %, p10 94.2 %, median 96.6 %, max 98.2 %, and it is driven by activity (correlation with
deposited energy -0.54): spills under 50 GeV average 97.5 %, spills over 65 GeV 94.7 %.

### By cluster class

| Class | Energy in 63 spills | Correct | Note |
|---|---|---|---|
| imaged clusters >= 100 MeV | 3100 GeV (81 %) | 98.3 % | the backbone; tracks 98.7 %, non-track clusters 93.8 % |
| imaged 20-100 MeV | 393 GeV | 91.7 % | v1.3's exact-block re-solve improved this class |
| imaged 5-20 MeV | 162 GeV | 87.2 % | |
| imaged 1.5-5 MeV | 115 GeV | 71.7 % | |
| clusters < 1.5 MeV (no light template) | 8 GeV | 66.4 % | placed by geometry only |
| unclustered (noise) hits | 55 GeV | 58.1 % | 182 k single hits (58 % of the energy) and pairs (42 %) that never met the leftover DBSCAN's 3 hits within 4 cm; 97 % of the energy is >= 4 cm from any cluster; placed by nearest assigned hit (see 6.2) |

By frontend cluster type (energy-weighted correct fraction, cluster energy < 20 / 20-100 / >= 100 MeV):
tracks 95.8 / 97.9 / 98.7 %; non-track clusters (blobs, showers) 85.1 / 90.3 / 93.8 %.

### How wrong is wrong
Of the wrong-but-assigned energy (162 GeV), 17 GeV is off by 10-30 ticks, 34 GeV by 30-100 ticks and 112 GeV by more
than 100 ticks (1.6 us). A wrong hit is on the wrong flash, not slightly mistimed. Only 0.05 % of the energy has its
truth time outside the 885-tick search window.

## 3. Interaction-level recovery (the number that shapes physics goals)

Truth interactions are approximated by truth-time groups (hits whose truth t0 agree within 1 tick, across all TPCs;
about 100 per spill). "Found" = at least half of the interaction's deposited energy carries the right time.

| Deposited energy | n | found (>= 50 %) | >= 90 % of energy right | mean energy fraction right | TPCs touched (median) |
|---|---|---|---|---|---|
| 1-5 MeV | 1243 | 31.7 % | 18.7 % | 0.30 | 1 |
| 5-20 MeV | 636 | 41.7 % | 14.9 % | 0.40 | 2 |
| 20-100 MeV | 656 | 56.9 % | 36.9 % | 0.55 | 2 |
| 100-500 MeV | 935 | 98.8 % | 87.5 % | 0.95 | 5 |
| 0.5-2 GeV | 1800 | 99.8 % | 95.6 % | 0.97 | 10 |
| > 2 GeV | 427 | 99.8 % | 92.0 % | 0.96 | 24 |

Reading: any interaction that deposits >= 100 MeV is found essentially always and 88-96 % of them have >= 90 % of their
energy at the right time. Below 100 MeV the recovery falls to about half, and below 20 MeV to a third. The low-energy
groups include rock neutrons and photons (isolated deep blips), which is exactly the population a low-energy physics
goal would rely on. This curve, not the 95.8 % hit accuracy, is the honest efficiency for a physics selection.

## 4. Timing precision

### Family level (one light TPC, one flash), residual to truth in ns, correctly assigned families with >= 5 hits

| Family energy | n | std | robust sigma |
|---|---|---|---|
| >= 500 MeV | 755 | 2.14 | 1.91 |
| 100-500 MeV | 12 840 | 3.65 | 2.41 |
| 20-100 MeV | 8 924 | 9.22 | 3.17 |
| 5-20 MeV | 5 932 | 25.6 | 6.70 |

Per hit (apex-clock times inside a family): std 4.86 ns, robust 2.80 ns over all correct hits; 2.65-2.69 ns robust for
clusters >= 100 MeV, 3.1 ns for 20-100 MeV, 4.4 ns for 5-20 MeV, 6.2-6.6 ns below 5 MeV. The standard deviations are
tail-dominated (a "correct" hit may be up to 160 ns off); the robust widths describe the core.

### The two reading keys work

`hit_t0_sigma_p` predicts the residual (correctly assigned hits, E-weighted):

| sigma_p | share of energy | robust residual | within 5 ns of the median |
|---|---|---|---|
| < 0.3 ns | 26 % | 2.13 ns | 97 % |
| 0.3-0.6 ns | 48 % | 2.65 ns | 95 % |
| 0.6-1.0 ns | 17 % | 3.38 ns | 86 % |
| 1-2 ns | 7 % | 4.9 ns | 69 % |
| 2-4 ns | 2 % | 8.3 ns | 44 % |
| > 4 ns | 0.4 % | 21 ns | 20 % |

The delivered sigma is not the residual width in absolute terms (the per-channel timing floor of ~2 ns with one average
pulse template is not included), but it orders hits correctly: a cut at sigma_p < 0.6 ns keeps 74 % of the energy at
2.1-2.7 ns.

`hit_t0_conf` predicts the assignment purity:

| conf | share of energy | purity (fraction correct) |
|---|---|---|
| 0.95-1.00 | 59 % | 99.0 % |
| 0.80-0.95 | 29 % | 98.0 % |
| 0.60-0.80 | 5.5 % | 90.1 % |
| 0.40-0.60 | 2.5 % | 77.7 % |
| 0.20-0.40 | 1.4 % | 62.9 % |
| 0.00-0.20 | 1.3 % | 43.7 % |
| NaN (light-blind) | 1.7 % | 62.1 % |

So an analysis can buy purity: conf > 0.8 keeps 88 % of the energy at 98.7 % purity; conf > 0.95 keeps 59 % at 99.0 %.

## 5. Pile-up and separation

- **Two flashes in one TPC.** Pairs of >= 20 MeV deposits from different interactions in the same TPC, each required to
  have its majority within min(10 ticks, 0.45 dt) of its own time: both resolved in 83 % of pairs at dt = 32-48 ns,
  90 % at 48-80 ns, 91-93 % from 80 ns up to the full spill. At least one of the two is right in 99.4-100 % of pairs.
  The chain resolves interactions two to three bunches apart (bunch spacing 19 ns) as well as it resolves distant ones;
  the remaining 7-9 % is not a separation failure but the ordinary per-deposit error rate.
- **Multiplicity is what costs.** Accuracy of a (TPC, interaction) deposit vs the number of interactions with deposits
  in that TPC: 99.8 % for 1, 99.2 % for 2-3, 97.9 % for 4-7, 95.6 % for 8-15, 93.2 % for >= 16. Versus the time gap to
  the nearest other interaction in the TPC it is flat (94-97 % from 0-5 ticks to > 300 ticks).
- **Position does not matter.** Accuracy is flat in x (94.8-96.8 % across seven 100-cm bins) and in y (95.1-96.3 %).
- **Saturation matters.** Accuracy vs the number of saturated (vetoed) channels among the 120 of the hit's own light
  TPC: 97.1 % with none (1680 GeV), 96.1 % with 1-4 (705 GeV), 95.3 % with 5-14 (802 GeV), 92.6 % with 15-39 (597 GeV),
  90.7 % with 40 or more (47 GeV). A third of the energy sits in TPCs with at least 5 saturated channels; the
  saturation-blind pieces of GeV showers ("dominant collapse", section 6) are this dependence. (Corrected 2026-10-01:
  an earlier version of this table used the wrong TPC pairing and showed a weaker effect.)

### Flash discovery compared with light-only flash finders
(`testing/v13_flash_discovery.py`, figure `testing/flash_discovery/flash_discovery_v13.png`.) The chain's family times
are a per-TPC flash list with 0.50 fake t0s per TPC-event (purity 0.94); a summed-waveform peak finder at the same fake
rate (S40, ~18 PE summed) and a loose one (S10, ~4 PE, 9.6 fakes per TPC-event) were run on the same light, with
accidental matches removed. Discovery efficiency of true flashes by energy deposited in the TPC: 2-5 MeV chain 0.64 /
S40 0.58 / S10 0.82; 5-20 MeV 0.85 / 0.85 / 0.91; 20-100 MeV 0.97 / 0.95 / 0.96; >= 100 MeV 0.99 / 0.98 / 0.98.
Where the chain is better: flashes BURIED within 20 ticks (320 ns) of a >= 3x brighter flash in the module: 2-20 MeV
deposits 5-20 ticks after a brighter one, chain 0.69 vs S40 0.51 vs S10 0.66; 20-100 MeV within 5 ticks 0.98 vs 0.89,
5-20 ticks 0.94 vs 0.82. Where it is worse: ISOLATED 2-20 MeV flashes, chain 0.77 vs S10 0.97 (the light is visible;
the small cluster was assigned to another flash). The t0s the chain finds that the matched finder does not (1.0 per
TPC-event) are real 63-70 % of the time and carry 1 % of the energy, 77 % of it correctly placed; above 100 MeV they
are 94 % real. True flashes the finder sees but the chain has no family for hold 0.3-0.5 % of the energy.

## 6. Failure modes, ranked by energy, with what was measured about each

1. **Small imaged clusters (1.5-20 MeV) near several flashes: 53 GeV, 33 % of the wrong energy.** The cluster's own light
   is 1/100-1/1000 of a neighbouring family's; the light chi-square is flat across the nearby slots and geometry decides.
   Measured cause: the light-prediction engine. On isolated families in the low-intensity sample, the per-channel
   prediction error (chi-square per degree of freedom after freeing time and amplitude) is 0.58-2.02 (10-90 %) for
   < 20 MeV families against 0.80-1.20 for >= 100 MeV at equal amplitude. Nine solver regularisations, three variance
   models and a sub-tick solver were all null, so this is not a solver limit. The fix is a retrained engine (relative
   or heteroscedastic loss, low-energy oversampling, per-channel uncertainty), not more assignment logic.
2. **Unclustered noise hits: 23 GeV, 14 %.** Anatomy (`testing/v13_noise_anatomy.py`, log `v13_noise_anatomy_V2A5.log`):
   the class is 182 k hits of 0.3 MeV average, exclusively single hits (31.8 GeV) and pairs (23.1 GeV) that never had
   the 3 hits within 4 cm the frontend's leftover DBSCAN needs; there are no larger unclustered groups. They are not
   cluster fringes: 0.1 % of the energy touches a cluster (< 1 cm), 3 % is within 4 cm, 38 % is 4-10 cm away, 45 % is
   10-25 cm away and 13 % is > 25 cm away (1.4 GeV of that gets no time at all). Correctness follows isolation: 73 % at
   4-10 cm, 54 % at 10-25 cm, 29 % beyond 25 cm. The mechanism is the parent, not the light: the hit takes the nearest
   clustered hit's time in 90-96 % of cases, that neighbour is itself correct 81-85 % of the time, but it belongs to the
   same interaction as the isolated hit only 78 / 55 / 32 % of the time in the three distance bins, which reproduces
   the accuracy almost exactly. Only 4-11 % of the energy (28 % beyond 25 cm) has a truth flash that was never found.
   These are the single-voxel deposits of neutrons and low-energy photons, and their light is invisible; a better
   parent rule ("same time as SOME family in the TPC" weighted by distance, rather than the nearest hit) is the lever.
   Separately, the force stage places 8.6 GeV of CLUSTERED energy per 63 spills (119 hits per spill), of which 8.2 GeV
   is the no-template class (< 1.5 MeV clusters, 66 % correct); imaged clusters that no light stage placed are ~0.4 GeV,
   i.e. every cluster with a light prediction receives a light-based decision.
3. **Delayed daughters inside a correctly timed cluster: 6.3 GeV in the >= 20 MeV classes.** Michel electrons and other
   daughters (delay spectrum follows the muon lifetime) are clustered with their parent and inherit its time; their
   own flash is visible in the light residual in 96-99 % of cases, all pieces are <= 55 MeV. A blind hit-level splitter
   was tested and rejected (3-4 % false splits per segment against an 18:1 host imbalance). Wrong today, but the
   information exists in the light.
4. **Overlapping GeV interactions: mixed clusters and dominant collapse.** When two GeV interactions share a TPC, some
   clusters contain hits of both (a clustering error; one time per cluster cannot be right), and when a GeV flash
   saturates channels, 20-200 MeV pieces of it lose their light information and drift to invented slots. Both live in
   the >= 100 MeV class (52 GeV wrong, 1.7 %).
5. **Contact merges** of 20-35 MeV track segments at crossings and vertices (< 3 % of their host's light).
6. **Activity.** The busiest spills (> 65 GeV, ~100+ interaction times) lose 2-3 pp against quiet ones; the worst spill
   is 87.9 %. Nothing else (position, saturation, time gap) moves the accuracy by more than 2-3 pp.

Event displays of types 1-5: `testing/v12_displays/` (light-vs-prediction, reco 3D, truth 3D, wrong-hit 3D per TPC).

## 7. Beam-timing capability (measured on the same spills)

- Interaction start time = earliest precision family of the interaction. Contained interactions: 3.3 ns robust from
  the true vertex time; 72 % inside the 8.6 ns bunch flat-top (truth 99 %, random 45 %); the correct bunch is identified
  for 97.1 % of contained interactions (96-98 % in every energy class).
- The spill's RF phase is recovered from the light alone to 1.8 ns per spill.
- Rock activity dominates an ND-LAr spill: 73 % of reconstructed interaction times are rock or external. Their arrival is
  late by the flight time (2-88 ns, median 25 ns), so the unselected comb is weak (|R| = 0.16). A truth-free "not
  entering" selection (no family of the interaction within 10 cm of the upstream, side or top faces) plus E > 100 MeV
  reaches 74 % contained purity and |R| = 0.44; adding start sigma < 0.5 ns gives 78 % purity, 80 % in-window, 2.1 ns
  residual.
- Projection of every start time along the beam to the ND front plane, t - ((r - r_front).n)/c, removes the flight-time
  smear without a containment cut: |R| 0.15 -> 0.33 for all interactions, rock residual 28.7 -> 4.7 ns. CAVEAT: the ND MC
  assigns the bunch time AT THE VERTEX for every interaction (rock vertices at -20 m included) and simulates only the
  products' flight, so on this MC the projection had to be evaluated with an emulated per-interaction target-flight
  shift; in data it applies directly with the surveyed beam direction. A comb study on this MC must either stay
  unprojected on contained interactions or apply the emulation.
- Figures: `testing/beam_comb_figs/beam_comb_nd_v13_pi.png`, `beam_comb_nd_v13.png`, `beam_comb_nd_v13_projected.png`,
  `fig1..fig6_*.png`. Arrays: `testing/beam_comb_figs/beam_comb_data.npz`.
- Precision-stage finding: the family apex (chosen among the two axis ends and the centroid by fit loss) is at chance;
  the true start of a family is its upstream end 72-77 % of the time (median 6 cm vs 41 cm for the fitted apex). A
  beam-direction or vertex-anchored apex prior is the next per-hit precision lever; family times are insensitive (<= L/4c).

## 8. What this enables, and what it does not (to shape physics goals)

- **Disentangling interactions in a spill is the main product.** Deposits of different interactions in the same TPC are
  separated down to 2-3 bunches (32-80 ns) with 83-91 % joint success, and any interaction above 100 MeV is found with
  98.8-99.8 % efficiency and 95-97 % of its energy correctly tagged. Analyses of GeV-scale neutrino interactions can treat
  the per-interaction time as available with ~2 ns precision and 97 % bunch identification.
- **The drift coordinate barely depends on this.** At ~1.6 mm/us drift, the whole 10 us spill spans 1.6 cm of drift;
  the 160 ns correctness window is 0.3 mm and the 2-3 ns precision is microns. A wrong t0 within the spill moves x by at
  most ~1.6 cm. The value of t0 is association (which interaction a deposit belongs to) and time-of-flight, not x.
- **Time-of-flight inside an interaction is possible for light-resolvable pieces.** ~25 displaced pieces per spill (>= 5 MeV,
  >= 3 own-dominated channels) receive independent times with 4.5 ns delay resolution over median 48 cm; family-to-family
  delays carry ~2-3 ns each. A 10 MeV neutron (beta 0.145) needs 23 ns per metre, so neutron ToF is a >= 5 sigma
  measurement at 1 m for pieces the light resolves, and unmeasurable for the sub-5 MeV blips that dominate neutron
  deposits (they are placed by geometry, 66-72 % correct). Budget accordingly.
- **Delayed-daughter (Michel) tagging is a potential, not a delivered capability**: the daughter's flash is in the light
  residual 96-99 % of the time, but the chain gives it the parent's time today. A port of the 2x2 Michel stage
  (unexplained late flash -> end piece of a cluster bounded by Bragg points -> perceiver light -> prompt-deficit gate;
  testing/v14_michel_nd.py, 2026-10-01) raises the correctly timed share of that class from 6.4 % to 10.7 % on the 63
  certified spills at 75 % purity (1.44 GeV fixed, 0.49 broken), but costs 100 s per spill offline and is not in the
  chain; 65 % of ND Michel flashes coincide with another family within 5 ticks, which is what limits it (VERSION_HISTORY).
- **Low-energy physics (isolated 1-20 MeV deposits) is the weak point**: interaction-level recovery 32-42 %, hit accuracy
  66-87 %. The measured cause is the light-prediction engine at low energy. A physics goal that rests on isolated
  low-energy blips should plan for the engine retraining first.
- **Quality selection is available and calibrated**: conf > 0.8 keeps 88 % of the energy at 98.7 % purity; sigma_p < 0.6 ns
  keeps 74 % at 2.1-2.7 ns. Both are truth-free and data-driven.
- **Beam bunch structure is recoverable from light alone** on contained or geometrically selected interactions, and, in
  data, on everything after the front-plane projection. This gives an in-situ RF-phase monitor (1.8 ns per spill) and a
  bunch tag per interaction.
- **Rock and external activity** cannot be separated from contained interactions by timing alone once times are
  projected; the separation is geometric (entering faces) and energetic (deep rock blips are low energy).

## 9. How each number was produced (reproducible)

```
# rows and dumps of the certification run
/pscratch/sd/y/yuxuan/v11arms/V2A5/groups_worker*.jsonl      # one row per spill: metrics, stage_t, phase3.families/pieces
/pscratch/sd/y/yuxuan/v11arms/V2A5/full_dumps/*.npz          # per hit: x y z e tpc labels ts_final truth_t0 hit_conf93 hit_sigma_p; lab_rows (per cluster e_ok/e_wrong/type)
# scripts (testing/)
v11_ab.py <ref> <arm> --dumps        # headline ledger vs a reference, per file, wrong energy by class
v13_physics_eval.py <arm>            # sections 3, 5, 6: interaction recovery, pile-up, separation, conf key, type, saturation, position; log v13_physics_eval_V2A5.log
                                     # (second pass, sigma_p key in ns / separation-scaled criterion / family precision: v13_physics_eval2_V2A5.log)
v12_eval.py <arm>                    # per-hit and per-family timing resolution, delivered sigma_p coverage
v12_undiscovered.py <arm>            # delayed-daughter anatomy
v14_beam_comb.py <arm>; v14_beam_comb_nd_figure.py [--pi]; v14_beam_comb_projected.py   # section 7
stage_time_report.py <arm>           # timing per stage (in V130_CHAIN_REPORT.md)
```
Definitions used everywhere: tick = 16 ns; correct = |reco - truth| <= 10 ticks; truth t0 of a hit = time of its best
backtracked segment; energy weights = positive hit energy; non-shower cohort = TPCs with <= 1.4 GeV deposited.
Validation convention agreed with the author: dev on files 1-3, certification on files 1-5 (63 spills); 10 files only
for release ledgers.

## 10. Open items that would change the numbers above

1. Light-prediction engine retraining for < 20 MeV (largest lever; 33 % of the wrong energy).
2. Delayed-daughter splitter using the residual light (6.3 GeV; needs a > 30x better false-split rate than the blind test).
3. Apex prior for the precision stage (per-hit times; family times unaffected).
4. Stage 4.7 blip re-homing (tested on the same spills: all-TPC +0.066 pp, 55/8 spills, every file up; not adopted).
5. Speed: light prediction (29 s) and the shower-TPC error matrix (20 s median, 87 s worst) out of 147 s per spill per
   worker (73-76 s per spill per GPU).
