# ND-LAr charge-light matching, production chain v1.3: stages, accuracy, timing

Certified configuration: `run_v13.sh` (tag `CLMatching_v1.3`, working tree, not pushed).
All numbers below are from the certification run of flow files 1-5 of MiniProdN5p1 FHC full-spill
simulation (arm `/pscratch/sd/y/yuxuan/v11arms/V2A5`, 63 spills), unless stated otherwise.
Reference for the deltas: v1.2.1 (`CERT_P`, same 63 spills).

Scale of one spill (median): 177 k charge hits with truth, 1412 clusters, 70 active TPCs, 60 GeV deposited.
Time unit: 1 tick = 16 ns. A hit is "correct" when |reco t0 - truth t0| <= 10 ticks (160 ns); the truth t0 of a
hit is the time of its best backtracked segment, so a delayed daughter carries its own time.

## 1. Headline accuracy

| Metric | Definition | v1.2.1 | **v1.3** |
|---|---|---|---|
| all-TPC accuracy | energy-weighted fraction of truth-tagged hits within 10 ticks, all TPCs | 95.54 % | **95.76 %** (+0.22 pp, 42/20 spills better/worse) |
| NS efficiency (target metric) | same, restricted to TPCs with <= 1.4 GeV deposited (non-shower cohort, 62.5 % of the energy) | 96.70 % | **96.77 %** (+0.08 pp, 35/25) |
| LE accuracy | among ASSIGNED hits that are unclustered or in clusters < 1.5 MeV (2.0 % of the energy), fraction within 10 ticks | 60.95 % | **60.89 %** |
| coverage | fraction of energy given a t0 | 99.95 % | 99.95 % |
| hit-count accuracy | unweighted fraction of hits within 10 ticks | | 95.98 % |

Per file (NS / all-TPC, v1.2.1 -> v1.3): file 1 96.69 -> 96.73 / 95.95 -> 96.02; file 2 97.02 -> 97.09 / 96.39 -> 96.67;
file 3 96.93 -> 96.94 / 95.81 -> 96.04; file 4 96.48 -> 96.58 / 94.60 -> 94.97; file 5 96.38 -> 96.55 / 95.02 -> 95.15.
Every file improves on both metrics.

### Accuracy by cluster class (63 spills, pooled over hits)

| Class | Energy (GeV) | Correct | Wrong (GeV) | Share of all wrong energy | Unassigned (GeV) |
|---|---|---|---|---|---|
| unclustered (noise) hits | 55.3 | 58.1 % | 23.17 | 14.1 % | 1.43 |
| clusters < 1.5 MeV (no light template) | 8.2 | 66.4 % | 2.74 | 1.7 % | 0.01 |
| imaged clusters 1.5-5 MeV | 114.9 | 71.7 % | 32.57 | 19.8 % | 0.14 |
| imaged 5-20 MeV | 162.3 | 87.2 % | 20.83 | 12.7 % | 0.04 |
| imaged 20-100 MeV | 392.5 | 91.7 % | 32.71 | 19.9 % | 0.21 |
| imaged >= 100 MeV | 3100.2 | 98.3 % | 52.37 | 31.9 % | 0.25 |
| total | 3833 | 95.71 % | 164.38 | | 2.08 |

v1.3's stage 3.99 is where the gain came from: 20-100 MeV wrong energy 33.98 -> 32.71 GeV, >= 100 MeV 59.45 -> 52.37 GeV;
the classes below 20 MeV are unchanged (20.89 -> 20.83, 32.56 -> 32.57, 23.12 -> 23.17).

How wrong is wrong: of the 162 GeV that is assigned but wrong, 17 GeV is off by 10-30 ticks, 34 GeV by 30-100 ticks and
112 GeV by more than 100 ticks. Mistakes are wrong flashes, not timing jitter.

### Anatomy of the remaining mistakes

1. **Small clusters near several coincident flashes** (1.5-20 MeV, 33 % of the wrong energy). Their own light is 1/100 to
   1/1000 of the neighbouring family's, the loss is flat across the nearby slots, and the chain falls back to geometry.
   The limit was measured to be the light-prediction engine, not the solver: on isolated low-intensity families the
   per-channel prediction error is 0.58-2.02 (10-90 %) for < 20 MeV against 0.80-1.20 for >= 100 MeV at equal amplitude.
2. **Unclustered noise hits** (14 %): force-covered to the nearest family; 22 % of them belong to flashes never found.
3. **Delayed daughters inside correctly timed clusters** (the "undiscovered" column of the >= 20 MeV classes): 4.66 GeV in
   the >= 100 MeV class and 1.63 GeV in 20-100 MeV have no reco family within 10 ticks of their own time. 3.65 + 0.96 GeV
   of it are sub-pieces of an otherwise correct cluster, all <= 55 MeV, delayed 10-1000 ticks after the host with the
   muon-lifetime spectrum (Michel electrons and other daughters); their own flash is visible in the light residual in
   96-99 % of the cases. The chain gives one time per cluster; the piece layer scans only +-10 ticks. A blind hit-level
   splitter was tested and is not viable (3-4 % false-split rate per segment against an 18:1 host imbalance).
4. **Mixed clusters where GeV interactions overlap** and **dominant collapse under saturated channels** (>= 100 MeV class):
   clustering-level and light-model-level errors that assignment cannot repair.
5. **Contact merges** of 20-35 MeV track segments at crossings and vertices.

Displays of each type: `testing/v12_displays/` (see `V121_CHAIN_SUMMARY.md`).

## 2. Timing outputs

| Quantity | Value (v1.3, 63 spills) |
|---|---|
| per-hit timing residual, correctly assigned hits, energy-weighted | std 4.86 ns, robust 2.80 ns |
| per-family residual | std 4.41 ns, robust 2.21 ns |
| delivered per-hit sigma (hit_t0_sigma_p) | on 98.7 % of hits, median 0.42 ns |
| sub-tick family time uncertainty (stage E) | median 0.39 ns |
| light-resolved displaced pieces | about 23 per spill, delay resolution 4.5 ns |
| interaction start (earliest family) vs true vertex time, contained | 3.3 ns robust; 72 % inside the 8.6 ns bunch flat-top (truth 99 %, random 45 %) |
| bunch identification of contained interactions | 97.1 % |
| beam RF phase per spill from light alone | 1.8 ns |

Beam-comb figures: `testing/beam_comb_figs/beam_comb_nd_v13_pi.png` (all interactions vs truth),
`beam_comb_nd_v13.png` (selections), `beam_comb_nd_v13_projected.png` (projection to the ND front plane, physical
time convention emulated: the MC assigns the bunch time at the vertex without the target-to-vertex flight).

Confidence output: `hit_t0_conf` per cluster is the amplitude-weighted fraction of the cluster's light channels whose
individual best time agrees with the assigned time within 10 ticks (0-1, NaN for light-blind clusters; median 0.69,
10 % of clusters below the low threshold). Reading key from the 3-file out-of-sample test: purity rises from 33 % at
conf 0 to 99 % at conf 1; AUC 0.78 for small and 0.94 for large clusters.

## 3. Computation time per stage

Per spill, one worker; two workers share one A100, so divide by two for s/spill/GPU. 63 spills.

| Stage | Median (s) | Mean (s) | p90 (s) | Share of mean |
|---|---|---|---|---|
| 1 load + clustering + light prediction + track stage | 51.4 | 52.0 | 68.9 | 34 % |
| of which light prediction (perceiver render) | 28.9 | 30.3 | 45.3 | 20 % |
| of which clustering (frontend) | 11.2 | 11.8 | 16.5 | 8 % |
| of which track stage | 7.7 | 7.8 | 9.8 | 5 % |
| of which event load + flash table | 2.3 | 2.0 | 2.8 | 1 % |
| 2 + 3.1 phase 1 and light-component pool | 13.2 | 13.2 | 18.6 | 9 % |
| 3.2 trigger corrections and rescue | 4.9 | 5.3 | 8.4 | 3.5 % |
| 3.3 + 3.5 P2 scan and police | 2.7 | 2.8 | 3.7 | 2 % |
| 3.9 joint tail solver | 9.2 | 9.9 | 14.7 | 6.5 % |
| 3.95 pass-2 spatial trading | 8.7 | 8.8 | 12.3 | 6 % |
| 3.98 component re-decision | 1.4 | 1.5 | 1.9 | 1 % |
| 3.99 exact-block re-solve (new in v1.3) | 10.1 | 10.5 | 14.5 | 7 % |
| 4.1 error matrix on shower TPCs | 20.2 | 24.8 | 52.2 | 16 % |
| 4.5 swap fix | 0.5 | 0.6 | 0.9 | 0.4 % |
| 4.9 force coverage | 0.3 | 0.3 | 0.4 | 0.2 % |
| E sub-tick refinement | 2.2 | 2.2 | | 1.5 % |
| conf confidence | 1.1 | 1.1 | | 0.7 % |
| P precision timing (3 rounds + pieces) | 15.6 | 17.1 | 25.4 | 11 % |
| I/O, metrics, dumps | | 2.5 | | 1.7 % |
| **total wall time per spill** | **146.6** | **151.4** | 211.5 | = 73-76 s/spill/GPU |

The two large costs are the light prediction (20 %) and the shower-TPC error matrix (16 %, heavy-tailed: up to 87 s on
the busiest spills). Everything from 3.2 to 4.9 together is 46 s. The v1.3 addition costs 10 s.

## 4. The stages

Inputs: one FLOW file event (a full spill): charge hits (x, y, z, E, TPC), light waveforms per channel (16 ns ticks),
MC truth only for evaluation. Outputs per hit: t0 (continuous, tick units), `hit_t0_conf`, `hit_t0_sigma_p`; per
family: apex time, sigma, apex position, energy; per displaced piece: delay, distance, beta, sigma.

**Stage 1: load, clustering, light prediction, tracks.**
The event is loaded and the light waveforms are formatted (baseline subtracted; v1.2.2: the saturation ceiling of each
channel is the int16 rail 32767 minus its own baseline, and samples within 80 counts of it are vetoed).
Clustering runs per TPC in the external frontend (`clmatching-frontend`, with the v0.2.5 merge of tracks emerging from a
common vertex); it yields clusters with a type (track, shower, blob) and leaves unclustered hits as noise.
Every cluster of >= 0.5 MeV (>= 1.5 MeV in TPCs above 1.4 GeV) is rendered by the Charge2Light perceiver into a
predicted peak amplitude per light channel; multiplied by the unit pulse template this gives the cluster's predicted light
image as a function of its time. The support of a cluster is the smallest channel set holding 90 % of its predicted light.
The track stage fits backbone tracks (`--backbone stack2`) and identifies the objects that span >= 3 TPCs.

**Stage 2 and 3.1: phase 1 and the light-component pool.**
Phase 1 places showers and >= 3-TPC tracks first by scanning the light chi-square over the 885-tick window, with the
2x2-ported edge penalty against hiding light past the window end. Then all remaining light-judgeable clusters
(single- and two-TPC backbone tracks and every cluster >= 20 MeV, `--merge-p2 --p2-threshold 20`) are grouped into
light-coupled components: clusters whose predicted light overlaps on shared channels. Each component is solved exactly
over all combinations of its members' candidate times (own-scan minima and the TPC's slot list), the best combination is
kept when its margin is decidable, and undecidable components are deferred (`--pool-mode light`, cos-soft pick, defer
gate restricted to the support channels by `--gate-support`).

**Stage 3.2: trigger corrections and rescue.**
The overflow trigger fires when a placement predicts more light than the waveform holds on the subject's support
channels; fired placements are re-decided (A/B and single-cluster re-scans, adoption of a neighbour's time, pairwise
swaps) and deferred components are rescued at the times that survive.

**Stage 3.3 and 3.5: P2 scan with police.**
Clusters still without a time are scanned one at a time on the light residual of everything placed so far. The placement
police (`--police locked`) resolves any newly placed cluster that fires the overflow gate immediately: keep, rewind or
rescan by the union loss. The V2 spatial-plus-light rescue of earlier versions is off (`--skip-v2`).

**Stage 3.9: joint tail solver.**
The < 20 MeV imaged clusters are assigned together: the light chi-square is exactly pairwise-quadratic in the assignment
variables, so the tail is a sparse Markov random field over the family times of each TPC, solved by annealed mean field
with the geodesic-contact prior of the v0.3 family expansion (`--phase3 joint`).

**Stage 3.95: pass-2 spatial trading.**
Clusters in contact with a family other than their own are offered to it (8 nearest families plus the families whose time
the cluster's own loss curve finds near-optimal, `--pass2-light-nom`); a trade is accepted when the light loss of the
union improves.

**Stage 3.98: component re-decision.**
With the whole event placed, every component of >= 20 MeV clusters in TPCs <= 1.4 GeV is re-solved exactly given
everything else (`--redecide --redecide-emin 20`).

**Stage 3.99: exact-block re-solve (v1.3, from the v2.0 study).**
A large-neighbourhood search on the full-event light chi-square plus a contact prior (clusters in contact prefer one
time): blocks of up to 8 clusters >= 20 MeV are re-solved exactly over their candidate times (own residual minima and TPC
slots), including twin swaps, sweeping until no block improves. A change is accepted only when its margin is >= 25
chi-square units on channels the block dominates (`--v200-min-e 20`, `V200_DOM=3`). Deeper chi-square minima below 20 MeV
were found to fill light-model deficits (17 % per-channel systematics) rather than place clusters better, hence the floor.

**Stage 4.1: error matrix on shower TPCs.**
In TPCs above 1.4 GeV the clusters not fixed by phase 1 are assigned by a per-cluster x candidate-time light-loss matrix
(no spatial prior; showers dominate the light there).

**Stage 4.5: swap fix.**
Pairs of families whose flashes may have been exchanged are tested; the swap is applied when its chi-square is below 0.85
of the current one (`--swap-fix --swap-ratio-max 0.85`).

**Stage 4.9: force coverage.**
Every hit still without a time (noise hits, clusters too small to render) takes the time of the nearest assigned hit
(KD tree, `--force-coverage --force-simple`); hits with no neighbour within 25 cm get no time (`--isolate-cm 25`).

**Stage E: sub-tick refinement.**
Each family's time is refined to a continuous value from the parabolic minimum of its light chi-square on a quarter-tick
grid, with an uncertainty from the curvature (`--subtick`).

**Stage conf: confidence.**
`hit_t0_conf` per cluster (`--conf-cluster`) as defined in section 2: zero tuned parameters, computed from the light
data only.

**Stage P: precision timing (v1.2 / v1.2.1).**
Families = hits sharing a final time within a light TPC. For each family, on its 90 % support channels, the pile-up
residual (actual minus everything else rendered at its own time) is fitted channel by channel with a delay-aware model:
the family's hits are turned into an arrival-time density on the channel (apex clock: a hit at distance d from the family
apex deposits at t0 + d/c, light then travels n d/c), convolved with the template, scanned in 0.25-tick steps and
refined by a parabola. The family time is the weighted median of the channel times (weights sqrt(A) x (1 - pile-up)),
its sigma from the median absolute deviation. Per-channel-index delays and a per-detector-type slew (beta log10 A) are
calibrated in situ on the low-intensity sample (`testing/v120_calib_lowi.json`), no truth used. Three rounds re-render
the other families at their refined times. Per-hit times follow the apex clock. Displaced pieces (3-cm chains >= 5 MeV
with >= 3 channels they dominate) get their own free time within +-10 ticks; their delay, distance, beta and sigma are
written out. Delivers `hit_t0_sigma_p`.

## 5. Known limits and the next levers

- **Low-energy clusters** are limited by the light-prediction engine. Retraining with a relative or heteroscedastic loss,
  low-energy oversampling and a per-channel uncertainty head is the identified path; the solver side is saturated
  (nine regularisations, three variance models and a sub-tick solver were all null).
- **Delayed daughters inside clusters** (about 4.7 GeV of undiscovered energy per 63 spills in the >= 100 MeV class)
  need a hit-level splitter with a false-split rate more than 30x lower than the tested blind version.
- **Per-hit precision**: the apex choice (two principal-axis ends or the centroid, by fit loss) is at chance, while the
  true start of a family is at its upstream end 75 % of the time. A beam-direction or vertex-anchored apex prior is the
  next per-hit timing lever; family times are insensitive to it.
- **Stage 4.7 blip re-homing** (contact certificate plus cross-TPC re-homing of 0.5-20 MeV clusters) is tested on the
  same 63 spills: all-TPC +0.066 pp (55/8 spills), NS +0.036 pp, every file up, 1.75 s per spill. Not yet adopted.
- **Cost**: the perceiver render (30 s) and the shower error matrix (25 s mean, 87 s worst) are the remaining speed
  targets.

## 6. Reproduction

```
bash run_v13.sh "<FLOW files>" <out_dir> [n_workers=8] [n_gpus=4]
python testing/v11_ab.py <ref_dir> <arm_dir> --dumps      # paired ledger + class breakdown
python testing/stage_time_report.py <arm_dir>             # per-stage timing
python testing/v12_eval.py <arm_dir>                      # timing resolution
python testing/v12_undiscovered.py <arm_dir>              # undiscovered-energy anatomy
python testing/v14_beam_comb.py <arm_dir>; python testing/v14_beam_comb_nd_figure.py [--pi]; python testing/v14_beam_comb_projected.py
```
