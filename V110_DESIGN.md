# v1.1 — light-coupled component placement (more combinations, honest loss)

*Development opened 2026-09-21 on top of v1.0 (tag CLMatching_v1.0).  User
direction: better assignment under a constrained compute budget; try several
ideas and measure; data-driven (few thresholds, no MC-truth reverse
engineering, physically sound); the core idea is MORE PERMUTATIONS /
COMBINATIONS of placements to minimize the light loss, possibly with charge
family association relations.*

## 1. What the v1.0 census said (39 events, files 0000001-3, 94.8 GeV wrong of 2286)

1. **No flash finder exists.**  Per-TPC t0 slots start empty and are appended
   only when a cluster places itself.  A new time can only come from an
   imaged cluster's own scan minimum; hits without a template (noise, < 1.5
   MeV clusters) can never land at a time no imaged cluster claimed.
2. **Undiscovered flashes are a small, light-blind pot.**  Wrong energy at a
   truth time where NO reco family exists: 13.0 GeV (13.7 % of wrong), but
   only 2.4 GeV of it is visible in the residual at z > 3 (0.1 pp).
3. **The eye-visible residual peaks sit at EXISTING families.**  1.8 peaks per
   TPC-event; 91 % at a time where a family already exists; the wrong energy
   under those peaks is 49 GeV = 52 % of all wrong energy.  A residual peak
   means a member of that family is placed elsewhere (median 120-175 ticks
   away), and the deficit's channel pattern matches the missing cluster's
   own pattern (cos 0.65 / 0.76 / 0.83 in 5-20 / 20-100 / >= 100 MeV).
4. **Wrong clusters have a light twin from another interaction** (cos > 0.8):
   75 / 79 / 80 / 93 % in 5-20 / 20-30 / 30-100 / >= 100 MeV, while same-TPC
   pairs in general have median cos 0.21 (6 % > 0.9).  The twin's energy ratio
   is inside the pool's energy band only 26-39 % of the time, so the v1.0
   energy-banded permutation pool co-decides the confusable pair in 0 % (< 30
   MeV), 16 % (30-100) and 38 % (>= 100 MeV) of cases.
5. **The piggyback mechanism.**  Correctly placed families are UNDER-predicted
   (actual/pred median 1.02 -> 1.10 from 20 MeV to > 1 GeV, p84 1.2).  Wrong
   clusters sitting at a bigger correct cluster's time fill exactly that
   deficit: family ratio 1.13 / 1.17 / 1.26 without them, 1.03-1.06 with
   them (30 GeV = 32 % of wrong energy).  More permutations alone cannot fix
   this - the loss genuinely prefers the wrong answer - unless the owner's
   amplitude error is profiled out first.
6. vote10 confidence separates: wrong energy with conf < 0.5 = 62 / 69 / 47 %
   while correct energy with conf < 0.5 = 23 / 5 / 0.4 % (5-20 / 20-100 /
   >= 100 MeV).

## 2. Design (testing/v110_components.py, v110_redecide.py)

* **Light-coupling components instead of energy bands.**  The Poisson chi2
  is additive over channels, so two placed clusters interact through the
  loss only where their predicted light overlaps on the same channels.
  Batch = connected component of the graph with an edge when the cosine of
  the two amplitude patterns on a shared light-TPC is >= 0.8 (one number;
  the census's twin definition).  Components are placed biggest-first,
  chunked at 8 members.
* **Exact combination search.**  Within a chunk the joint loss is exactly
  pairwise-quadratic (v0.7): L = const + sum a_i(t_i) + sum C_ij(t_i,t_j).
  Candidates per member = the cosine-softened argmin + the best separated
  local minima of its own isolated curve (k = 6) + the energy-order greedy
  time (so v1.0's path is always enumerated).  All k^n combinations are
  scored by table sums (numpy; torch/GPU above 4e5), the top 8 plus the
  greedy path are re-scored with the exact clipped union loss, the best
  exact one is committed (defer gate per member as before; a rejected member
  is deferred and the rest re-decided on the updated base).  Scans per
  event fall from ~4900 (ordering trees) to ~280 (one isolated curve per
  member + greedy path).
* **Amplitude profiling (`--pool-gain b`).**  Every phase-1 and pool commit
  fits its per-(cluster, TPC) amplitude in [1-b, 1+b] on its own support
  channels against the residual without itself (closed form) and rescales
  the template IN PLACE in image_maps, so rescue, pass2, subtick (sum mode)
  and confidence all see the profiled amplitude.  b = 0.25 covers the
  measured p84 of the under-prediction.
* **Final-state component re-decision (`--redecide`, stage 3.98).**  After
  pass2, components (cos >= 0.8) of PLACED imaged clusters >= 5 MeV in
  non-shower TPCs are re-decided jointly on the leave-component-out base;
  each member's current time is a candidate; the winner must beat the
  incumbent's exact loss.  Singletons are never touched (isolated
  final-state moves were measured harmful three times).
* **Charge-contact prior (`--pool-spatial f`, optional).**  Members whose
  hit clouds come within 3 cm get a same-time bonus of f x (chunk median
  unary spread) - the v0.7 energy-unit convention - in the tables and in
  the exact re-score.

Flags: `--pool-mode light --pool-cos 0.8 --pool-k 6 --pool-nmax 8
--pool-gain 0.25 [--redecide --redecide-emin 5] [--pool-spatial f]`.
Correctness: testing/v110_selftest.py (table loss == exact clipped loss;
undo on rejection exact; re-decision recovers a parked twin).

## 3. Arms (files 0000001-3, 39 events, paired vs v1.0 = v138FS_out)

*(filled from /pscratch/sd/y/yuxuan/v11arms via testing/v11_ab.py)*

| arm | flags | NS | all-TPC | events NS b/w | stage 2+3.1 s/ev |
|---|---|---|---|---|---|
| v1.0 | canonical | 96.59 | 95.89 | - | 18.1 |

## 3b. Results so far (2026-09-22)

| arm | flags | NS | all-TPC | events NS b/w | stage 2+3.1 s/ev |
|---|---|---|---|---|---|
| B | --pool-mode light | 96.62 (+0.03) | 95.87 (-0.03) | 23/16 | 10.5 |
| C | B + --pool-gain 0.25 | 96.42 (-0.17) | 95.58 (-0.31) | 7/32 | 10.7 |

**B (light components + exact combination search): a wash in accuracy, 42 %
less pool time** (scans 4867 -> 319 per event; total 108 -> 96 s/ev/worker).
Flip census on label-identical events: 98 % of moves go to an existing family;
fixed and broke energies balance in every band; the largest breaks are
mirror-image TWIN SWAPS - two look-alike clusters of similar energy exchanged
between two flashes, where the light loss is degenerate at the template-
systematic level and the joint minimum picks the wrong side as often as the
right one.  The joint search beats the greedy path on the loss in ~50 % of
multi-member chunks; the loss margin does not separate good from bad swaps.

**C (amplitude profiling at commit): falsified.**  Gains pile up at the +25 %
edge (63 %), phase-1 gains median 1.16: the sequential fit against the residual
absorbs the light of co-timed clusters that are not yet placed, so those later
see no deficit at their family and go elsewhere.  After C the correct families
are OVER-predicted (actual/pred 0.93-0.98) and the piggyback energy is unchanged
(30.5 GeV).  Only a joint final-state refit could be honest; not pursued.

**Charge-contact prior: physically unjustified for clusters >= 5 MeV.**  In
v1.0's non-shower TPCs the wrong energy in imaged >= 5 MeV clusters is 15 GeV;
3.9 GeV of it has a 3-cm contact partner placed at the truth time (fixable),
but 47-68 GeV of CORRECT energy has contact partners only at other times, and
99 % of those partners are themselves correct - genuine cross-interaction
contacts (pile-up tracks crossing).  The v0.7 certificate (contact -> same
interaction, P ~ 0.99) holds for the disconnected tail only.  Implemented
(`--pool-spatial`, `PlacedContacts`) but left off; no arm run.

**Clustering is not bit-reproducible** between runs (10 of 36 events had
different front-end labels between v138 and arm B); event-level ledgers stay
paired, cluster-level censuses use label-identical events only; the event-
level noise floor is measured by arm A (v1.0 rerun).

| D | B + --pool-gain 0.25 + --p2-threshold 10 | 96.46 (-0.11) | 95.70 (-0.20) | 10/28 | 16.5 |
| E | --pool-gain 0.25 (energy batching) | 96.48 (-0.11) | 95.68 (-0.22) | 8/31 | 15.5 |
| A | v1.0 rerun | 96.59 (0.000) | 95.89 (0.000) | 0/0 | 15.4 |

**A: the chain is deterministic** (bit-identical rerun, same scan counts).  The
label mismatches seen in the flip census come from downstream relabeling
(stage-3.2 detached parts get new ids), not from clustering; the earlier
"not bit-reproducible" reading in this document is withdrawn.  Event-level
deltas of a few hundredths of a pp are therefore real.

**E: the gain is harmful by itself** (energy batching, no component search),
and D (gain + 10 MeV floor) is less negative than C (gain + 30 MeV floor) by
+0.06 / +0.10 pp, which points at the lower floor as a positive lever.

**Chunk audit (arm D, truth-scored joint-vs-greedy within the pool decision):**
net +9.2 GeV per 39 events, ALL of it from chunks of 5-8 members (+9.4 GeV);
pairs (-0.18) and 3-4-member chunks (-0.06) are coin flips; the loss margin
of the joint winner does not separate good from bad decisions (fix / break
medians 0.024 / 0.018 relative).  Joint optimization pays where the exclusion
structure is crowded, which motivates the lower-floor arms D2 (10 MeV) and
D3 (5 MeV) without gain.
| O | --pool-mode light-order (v1.0 ordering rule on component batches) | 96.60 (+0.005) | 95.83 (-0.06) | 19/20 | 13.5 |

**O: the batching rule alone is neutral.**  With v1.0's own decision rule
(all orderings, soft-pick greedy scans) applied to light-coupled components
instead of energy bands, accuracy is a wash (NS +0.005, all-TPC -0.06) and the
pool saving is smaller than with the combination search (13.5 vs 10.5 s/ev).
| F2 | B + --redecide (>= 5 MeV) | 96.57 (-0.03) | 95.83 (-0.06) | 11/28 | 10.2 (+3.9 redecide) |

**F2: the final-state component re-decision splits along the light-information
boundary.**  54 moves per event (317 participants, 47 multi-member components,
3.9 s/ev).  Truth-scored moves: clusters >= 100 MeV +0.88 GeV (5 fixes, 0
breaks); 20-100 MeV +1.64 GeV (65 / 30); 5-20 MeV -2.97 GeV (125 / 461).  The
light-weak band (3-20 MeV, k* < 1 reliable channel) must not be re-decided by
light - the chain's standing lesson - while the light-trustworthy band gains
+2.5 GeV (~ +0.11 pp).  A loss-margin floor helps less cleanly (rel_gain > 0.05:
+1.7 GeV).  Next arm: `--redecide-emin 20`.
| B2 | = B (rerun) | 96.62 (+0.03) | 95.87 (-0.03) | 23/16 | 10.6 |
| D2 | --pool-mode light --p2-threshold 10 | **96.67 (+0.07)** | **95.97 (+0.08)** | 22/17 (all-TPC 26/13) | 15.9 |

**D2: the first arm positive on both metrics.**  Sending the 10-30 MeV
single-TPC clusters through the component search (163 components, 4.2 chunked
per event, 25 M table entries per event) gains +0.07 / +0.08 pp with an all-TPC
split of 26/13, at a pool cost still below v1.0 (15.9 vs 18.1 s/ev).  This is
the lower-floor lever the chunk audit pointed at (joint optimization pays in
crowded components).
| D3 | --pool-mode light --p2-threshold 5 | 96.63 (+0.04) | 95.97 (+0.08) | 18/21 (all-TPC 22/17) | 23.2 |

**D3 (5 MeV floor): same direction, more churn, slower than v1.0.**  Hit-level
flips vs v1.0 (no label pairing): 20-100 MeV +1.89 GeV, >= 100 MeV +1.23, but
5-20 MeV -1.45 (D2: +1.49 / +0.86 / -0.74).  Lower floors make the components
more crowded, which is where the joint decision pays for the light-trustworthy
members, but the light-weak members themselves do worse committed by light
than in v1.0's tail solver (connectivity prior valid there).  Hence the hybrid
`--pool-commit-min 20`: members below the light boundary take part in the joint
decision as light hypotheses but are not committed by it.
| G | --pool-mode light --predict-min-energy 0.5 | **96.74 (+0.15)** | 95.95 (+0.06) | **36/3** (all-TPC 24/15) | 10.1 (stage 1 +10 s, matrix +10 s) |

**G: templates for 0.5-1.5 MeV clusters - the largest gain, and it is the
"small peaks" population.**  LE class 58.1 -> 61.2.  Hit-level flips: clusters
< 1.5 MeV +0.90 GeV, 1.5-5 MeV +0.43, noise +0.45 (these clusters leave
nearest-hit force placement for the tail solver, which sees their light and
applies its connectivity prior); 20-100 MeV -0.74 is arm B's component effect.
Cost: +24 s/ev/worker relative to B (stage 1 47.8 -> 57.8 s; the shower-TPC
matrix 16.5 -> 26.6 s; joint +2.8, pass2 +2.4).  The NS/all-TPC split
(+0.15 / +0.06) says the shower TPCs gain nothing from it while paying the
matrix cost -> next: the 0.5 MeV floor in non-shower TPCs only.
| F20 | B + --redecide --redecide-emin 20 | **96.73 (+0.14)** | 95.94 (+0.05) | 26/13 (all-TPC 20/19) | 10.0 (+1.4 redecide) |

**F20: re-decision restricted to the light-trustworthy band works.**  149
participants and 24 multi-member components per event, 11.8 moves, 1.4 s/ev.
Truth-scored moves: fixed 59 (+2.94 GeV), broke 26 (-0.94 GeV), net +2.0 GeV,
positive in every component size (pairs +0.51, 3-4 +0.39, 5-8 +1.10) and in
both bands (20-100 MeV +1.57, >= 100 MeV +0.43).  End-to-end hit-level
+1.1 GeV, i.e. the re-decision more than repays arm B's -0.6 GeV.
| D20 | --pool-mode light --p2-threshold 20 | 96.66 (+0.07) | 95.89 (+0.00) | 22/17 (all-TPC 19/20) | 12.1 |

**D20 (20 MeV floor) is weaker than D2 (10 MeV).**  The 10-20 MeV members
help the light-trustworthy members through the crowding they create in the
components even though they themselves are placed worse by light than by the
tail solver; hence the hybrid arm H20 (participate down to 5 MeV, commit only
>= 20 MeV).
| H20 | --pool-mode light --p2-threshold 5 --pool-commit-min 20 | 96.56 (-0.04) | 95.85 (-0.04) | 15/24 | 12.6 (total 140 s) |

**H20 (participate-but-do-not-commit hybrid): falsified.**  The light-weak
members still lose (5-20 MeV -1.04 GeV) and the light-trustworthy members no
longer gain (+0.15 / 0.00): decisions made with the small members' light
hypotheses present do not hold once those members are placed elsewhere by the
tail solver, and the tail stages slow down (total 140 s/ev).  The pool-floor
lever stays as D2 (10 MeV, commit everything).
| F20D20 | --pool-mode light --p2-threshold 20 --redecide --redecide-emin 20 | 96.74 (+0.15) | 95.95 (+0.06) | 24/15 (all-TPC 23/16) | 11.0 (+1.2 redecide; total 100 s) |

**F20D20 ~= F20**: with the re-decision in place the 20 MeV pool floor adds
little (re-decision net +1.76 GeV, 55 fixes / 28 breaks).
| H20R | H20 + --redecide --redecide-emin 20 | 96.65 (+0.05) | 95.91 (+0.02) | 16/23 | 12.4 (total 142 s) |

**H20R**: the re-decision recovers part of the hybrid's loss but the arm stays
below F20 and is slow; the participate-but-do-not-commit idea is closed.
| F20D | --pool-mode light --p2-threshold 10 --redecide --redecide-emin 20 | 96.71 (+0.12) | **96.00 (+0.11)** | 23/16 (all-TPC 26/13) | 15.9 (+1.4 redecide; total 106 s) |

**F20D: the two pool/re-decision levers combine.**  Hit-level +2.52 GeV
(20-100 MeV +2.12, >= 100 MeV +0.96, 5-20 MeV -0.74); the re-decision itself
nets +1.09 GeV here (less than in F20 because the 10 MeV floor already fixes
part of what the re-decision would), at v1.0's total time.
| K0 | --pool-mode light --predict-min-energy 0.5 --predict-min-energy-shower 1.5 | 96.74 (+0.15) | 95.94 (+0.05) | 36/3 (all-TPC 24/15) | 10.4 (total 106 s) |

**K0: G's gain at v1.0's cost.**  Keeping the 1.5 MeV floor in shower TPCs
(> 1.4 GeV charge) returns the matrix stage to 16 s (G: 26.6) with stage 1 at
53.8 s; NS is identical to G (+0.150, 36/3), all-TPC within 0.01 pp.
| K1 | K0 + --p2-threshold 20 + --redecide --redecide-emin 20 | **96.88 (+0.29)** | **96.04 (+0.15)** | **33/6** (all-TPC 29/10) | 11.4 (+1.3 redecide; total 109 s) |

**K1: the levers stack.**  Prediction floor (K0, +0.15 NS) + pool floor +
re-decision (F20D20, +0.15 NS) give +0.29 NS / +0.15 all-TPC at v1.0's total
time.  Hit-level +3.37 GeV: < 1.5 MeV +0.91, 1.5-5 +0.55, noise +0.45, 20-100
+1.12, >= 100 +0.48, 5-20 MeV -0.14 (the light-weak band is left alone by
design and stays flat).  Re-decision net +1.94 GeV (58 fixes / 32 breaks).
| K2 | K0 + --p2-threshold 10 + --redecide --redecide-emin 20 | 96.83 (+0.24) | 96.08 (+0.19) | 33/6 (all-TPC 29/10) | 16.2 (total 116 s) |

**K2 vs K1 - the v1.1 candidate is K1.**  K2 gains more all-TPC energy
(hit-level +4.25 vs +3.37 GeV) but pays -0.70 GeV in the light-weak 5-20 MeV
band (K1: -0.14) and 7 s/ev; K1 is higher on the headline NS metric (+0.29 vs
+0.24), runs at v1.0's time, and its 20 MeV pool floor is the light-
information boundary rather than a tuned number.  Certification (files 1-5)
runs with K1's flags = `run_v11.sh`.

## 4. v1.1 definition and certification (2026-09-22)

**v1.1 = v1.0 chain + `--pool-mode light --p2-threshold 20 --redecide
--redecide-emin 20 --predict-min-energy 0.5 --predict-min-energy-shower 1.5`**
(launcher `run_v11.sh`; arm K1).  Confidence output unchanged.

Certification, files 0000001-0000005 (63 events), paired against the v1.0
outputs of the same events (v138FS_out + v141PT_out; the chain is
deterministic, so paired deltas carry no run-to-run noise):

| metric | v1.0 | v1.1 | delta | events better/worse |
|---|---|---|---|---|
| NS total-eff | 96.44 | 96.70 | **+0.26 pp** | **52 / 11** |
| all-TPC | 95.37 | 95.54 | **+0.17 pp** | 46 / 17 |
| LE class | 57.94 | 60.95 | +3.0 pp | |
| per file NS | | | +0.24 / +0.26 / +0.36 / +0.25 / +0.19 | all 5 up |
| per file all-TPC | | | +0.25 / +0.05 / +0.14 / +0.40 / -0.01 | 4 of 5 up |
| time/ev/worker | 110 s | 116 s | +5 % | stage 1 +6 s, pool -5 s, re-decision +1.3 s |

What each lever contributes (39-event arms): templates for 0.5-1.5 MeV
clusters +0.15 NS (the "small peaks" population: they leave nearest-hit force
placement for the tail solver); the 20 MeV re-decision +0.14 NS (+2 GeV
truth-scored, positive in every component size); the 20 MeV pool floor and
the component search themselves are accuracy-neutral but make the pool 30 %
faster, which pays for the other two.

What was learned about the user's two questions:
1. *Small peaks after the backbone*: the light-visible undiscovered flashes
   are worth only ~0.1 pp; the population that mattered is the sub-1.5 MeV
   clusters that had NO light hypothesis at all.  Giving them templates
   (data-only, no thresholds beyond the existing prediction floor) is the
   fix; residual peaks at existing families are dominated by the light
   model's 2-10 % under-prediction, not by missing members.
2. *Who joins the iterative process*: light-coupled components are the right
   unit (and cheaper), but for two look-alikes the light loss is degenerate
   at the template-systematic level - more combinations do not break twin
   swaps.  The information that does break them is not available today:
   contact geometry is unsound for big clusters in a busy spill (crossing
   interactions), and amplitude profiling at commit steals unplaced light.
   The gain came from re-deciding light-trustworthy clusters on the final
   residual, where all families are known.
