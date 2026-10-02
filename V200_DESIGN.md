# v2.0_test: assignment from scratch as a labelled-graph energy minimisation

Status 2026-09-24: offline prototype on chain dumps (`testing/v200/`), first numbers below.
Inputs are the first-stage products only (clusters, per-cluster predicted light, actual light);
the chain's assignment stages 2-4.9 are replaced by one solver.  No MC truth enters anywhere
except the final scoring.

## 1. The loss

One time t_c per cluster c (a cluster spanning several TPCs has one time).  The energy is

    E(t) = sum_TPC sum_{ch in U_TPC} sum_tau w_ch,tau ( A_ch,tau - min( sum_c a_c[ch] P(tau - t_c), rail_ch ) )^2
         + sum_{(c,c') in N} beta_cc' [ |t_c - t_c'| > 3 ticks ]

* Light term: the Poisson chi2 of the whole TPC, weights w = 1/(200 max(A,100)), channel
  set U_TPC = union of the clusters' support channels (90 % of predicted light) minus the
  saturation veto, rail = 2**15-1 minus the dynamic baseline (v1.2.2).  a_c is the per-channel
  predicted peak amplitude, P the unit-peak average pulse.  Unlike the chain, every term of
  every cluster is evaluated on the same TPC-level channel set, so E is one function.
* Prior term: N is the contact graph (minimum hit-hit distance <= 3 cm).  A cluster with no
  contact has no beta term at all (isolated clusters are free); the formula never penalises
  a cluster for sitting at a time nobody else uses, so light-discovered small deposits and new
  slots cost nothing.
* beta is in chi2 units and is measured in situ, not tuned: after a light-only solve, every
  contact pair whose two clusters are both light-decidable (see 2) gives a "same time / not"
  observation; P(same | E_small bucket, E_big bucket, distance bucket) is tabulated over the
  batch and beta = 2 ln P/(1-P).  This is the Bayesian -2 ln prior added to a chi2.  Where the
  data say contacts are usually different interactions (two big clusters touching: 99 %
  different in the v1.1 census) beta comes out negative, i.e. a repulsion; where they are
  usually the same (a fragment touching a shower: 93-96 % same) beta is a positive glue.  The
  two "conflicting" wishes are therefore one table read from the light-decidable population.

## 2. Candidate times, decidability, new slots

* Self-scan: chi2 curve of the cluster alone against the residual (all its support channels,
  all its TPCs, FFT); candidates = the k best local minima separated by > 10 ticks.
* Light-decidable: the second-best minimum is worse than the best by >= 25 (5 sigma) and the
  cluster's own light is significant (sum w a^2 P^2 >= 25).  Decidable clusters may create
  slots; undecidable clusters choose among existing slots of their TPCs plus their own minima
  on the CURRENT residual (which can become decidable once the big families are subtracted:
  this is how light finds small deposits after the fact) and, failing that, beta / proximity.
* Slots per TPC = clustered best times of decidable clusters; recomputed every sweep.

## 3. Search: exact block solves inside large-neighbourhood search

The light term is exactly pairwise-quadratic in the assignment (v0.7): with the residual of
everything outside a block S held fixed,
    E_S = sum_i u_i(t_i) + sum_{i<j} p_ij(t_i, t_j) + beta_ij [..],
    u_i(t)     = sum_tau q_ii(tau) P_t(tau)^2 - 2 sum_tau r_i(tau) P_t(tau),
    p_ij(t,t') = 2 sum_tau q_ij(tau) P_t(tau) P_t'(tau),
    q_ij(tau)  = sum_ch a_i[ch] a_j[ch] w[ch,tau],  r_i(tau) = sum_ch a_i[ch] w[ch,tau] R_ext[ch,tau],
so a k x k table per pair costs k^2 x 1000 flops (no GPU needed) and a block of n clusters
with k candidates is solved EXACTLY by enumerating k^n (capped at 2e6 combinations).
Procedure: initial state = decidable clusters at their best self-time, others at the nearest
slot; then sweeps of: connected blocks (light coupling or contact edges) of size <= 8 solved
exactly, larger components covered by random connected sub-blocks of 6, twin-swap kicks, all
accepted only if the exact clipped E decreases; candidate sets refreshed on the new residual
each sweep; several restarts from different orders keep the best E.  Gradient descent is
never used; local minima are escaped by re-solving whole blocks jointly, which is where the
misplacements (twin swaps, piggybacks) live.  Splitting a cluster into its 3-cm contact
pieces is an optional move (amplitudes shared in proportion to energy; approximate until the
perceiver can be re-run on pieces in-chain).

## 4. Physical sanity (borrowed from the chain)

Coverage (noise / unpredicted clusters go to the nearest placed family in their TPC), the
overflow gate (no placement that exceeds the actual by > 5 sigma on >= 3 unvetoed channels
unless the light margin demands it), one time per cluster, families = clustered times per
TPC; the precision stage (v1.2) and confidence (vote10) run unchanged on the result.

## 5. Evaluation protocol

Offline on the v1.2.1 dev dumps (files 1-3, 39 events) and SAT1: hit-level energy-weighted
efficiency (|t - truth| <= 10 ticks) overall and by cluster class, against the chain's
ts_final in the same dump; and the total light chi2 of both configurations under the SAME
loss, to tell a search failure (chain deeper than us) from a model failure (we are deeper
than the chain but worse against truth).  Development on files 1-3, certification on 1-5.

## 6. Findings, 2026-09-23/24 (offline, 39 dev events, testing/v200/v200_out/)

Search: the block-exact LNS reaches a far deeper light minimum than the chain under the same
loss (sum chi2 33.7M vs 45.7M) in 41 s/event on one CPU; so the permutation machinery works.

Loss vs truth, cold start over all clusters (arm D3): 95.53 % vs chain 96.03 %.  Per class the
solver WINS above 20 MeV (20-100 MeV wrong 6.8 % vs 8.2 %, >= 100 MeV 1.39 % vs 1.45 %) and
LOSES below (5-20: 20.1 vs 11.8 %, 1.5-5: 36 vs 26 %).  The mechanism is measured, not
guessed: the light model's per-channel systematics are 17 % of the light present (in-situ rms of
(A-M)/A on well-fit families, Poisson removed), a small cluster's own light is 5-15 % of the
light around it, and the chi2 rewards filling a big family's 17 % deficit with any small
cluster's light (the misplaced 5-20 MeV clusters sit at slots with a large weighted deficit on
their support channels, margins of 30-1000 chi2 in favour of the wrong slot).  Decidability by
own-dominated channels (D3), support-only channels (S1), Poisson+systematics weights (F1),
per-family gain profiling with a 17 % prior (G17), a light-blind -> nearest-family rule (BL17)
and a residual-pattern cosine gate (C75) all leave the < 20 MeV classes worse than the chain:
below 20 MeV the chi2 is not a faithful likelihood and the chain's staged rules act as the
regulariser.

Prior calibration in situ is selection-biased for exactly the pairs that matter: among
light-decidable contact pairs, small-big P(same) = 0.008 (64 pairs), mid-big 0.15, big-big
0.59 at <= 1 cm and 0.15 at 1-3 cm, because a small cluster touching a big one is decidable
only when its time differs.  Applying those tables (arm D3P) makes things worse (95.47 %).  The
contact prior therefore needs an unbiased source (physics or a truth-validated purity such
as the ND home rules, 93-96 %), not the decidable population.

What transfers: freeze everything below 20 MeV at the chain's times and re-solve the rest.
  W20 (warm from chain)              96.18 %  (+0.15 pp, 23/16 events better/worse, 8 s/ev)
  C20 (cold re-init of >= 20 MeV)    96.21 %  (+0.19 pp, 27/11)   <- stage 3.99 default
  W50                                 96.09 %  (+0.07)
  W10                                 96.00 %  (-0.03; 5-20 MeV 17.8 % vs 11.8 %: the floor is real)
All the gain is in >= 100 MeV (1.29-1.25 % vs 1.45 %) and 20-100 MeV (7.6 % vs 8.2 %).
In-chain version: `--v200-min-e 20` (stage 3.99 after the v1.1 re-decision), testing/v200/v200_stage.py.

## 7. Floor and variance scan (2026-09-24, offline, 39 dev events, cold re-init of the re-solved clusters)

Floor of the re-solve (everything below frozen at the chain's times):
  20 MeV 96.21 | 15 MeV 96.15 | 10 MeV 96.09 | 10 MeV with 10-20 MeV slot-only 95.80 |
  20 MeV + light-decidable 10-20 MeV 96.17          (chain 96.03)
Lowering the floor keeps improving 20-100 MeV (7.60 -> 7.05 % wrong at 10 MeV) but the
10-20 MeV clusters themselves get worse (5-20 class 11.8 -> 16.7 %), even when only the
light-decidable ones are re-solved (12.6 %) and worst when they may only choose among existing
slots (23.5 %).  The 20 MeV floor stays.

Variance model (weights w = 1/var), 20 MeV floor unless noted:
  Poisson 200 A        96.21   Poisson 300 A (single-PE gain)        96.21 (identical: a global
                                                                       scale cancels with beta = 0)
  Poisson + (0.17 A)^2 (model-error term, in-situ 17 %)      95.68  (20-100: 10.6 %, >=100: 1.56 %)
  trained variance predictor (chain checkpoint, maps in
    /pscratch/sd/y/yuxuan/v11arms/P2/var_cache)              94.12  (all classes worse, >=100: 2.63 %)
  predictor + (0.17 A)^2                                     94.67
  full solve (all clusters) with the predictor               92.70  (vs Poisson 95.53)
The predictor's variance is ~0.28 x Poisson-200 where light is present and ~30 x below the
100 ADC floor in quiet samples: it describes waveform noise, not the light-model error that
dominates the residual, so weighting by it lets template mismatch drive the fit.  A per-sample
relative model-error term is also wrong in shape (the 17 % is a per-channel, correlated scale
error, not independent sample noise) and it removes the timing information at the pulse peaks.
Poisson with sigma^2 proportional to A remains the best of the tested weightings; the true
remedy for the model error is a per-channel scale nuisance, which the assignment cannot afford.

## 8. Contact rule for light-blind clusters (2026-09-24) and the delayed-daughter target

Arm BLC (full solve; a light-blind cluster that touches a placed cluster takes the touched
cluster's time, light decides only for blind clusters with no contact): 95.03 % vs 95.53 %
for plain light-among-slots.  With the census (99 % of disagreeing big-big contacts genuine)
and the 2x2 rules (isolated fragment 35 % pure), contact is not a usable prior for the
light-blind population in ND; do not re-try without a topology classifier.

Delayed daughters (testing/v13x_michel_feas.py, 39 events): 675 delayed sub-pieces inside
hosts >= 20 MeV, 14.6 GeV = 16 % of all wrong energy.  Median piece 14.7 MeV (4-31),
compact (rms 2.7 cm, 90 % < 7.8 cm), touching the host (3.3 cm), 52 % within 0.1 of a
host end, delay median 110 ticks (25-373: muon lifetime), own residual flash > 600 ADC in
95 % and > 2000 in 66 %.  A geometric end-blob + own-flash rule reaches 54 % of pieces =
5.7 GeV (about +0.25 pp if selective).  1/d^2 proxy patterns barely separate piece from
host (cos 0.62 vs 0.59) because the daughter sits on the host's end: the discriminant must
be the light TIME plus the geometry, tested by v14_michel_oracle.py.

## 9. Light-model residual decomposition and sub-tick refinement (2026-09-24)

testing/v14_engine_residual.py, 704 isolated families >= 50 MeV (channel-level isolation: other
light < 5 % of own), 11.5k channels with predicted peak >= 2000 ADC, no saturation:
  chi2/sample (Poisson-200): as predicted 0.48 | free sub-tick time 0.25 | free amplitude 0.33 |
  both 0.16;  relative rms residual 18 % -> 8.9 %.
  Per-channel amplitude factor: median 0.996, 10-90 % 0.82-1.18 (~14 % per bright channel),
  per-family normalisation 0.87-1.11, within-family pattern spread 9 % (robust sigma of log
  alpha); no detector-type bias (0.997 / 0.994), no channel-index structure (0.965-1.026).
  Best sub-tick shifts 0 to +0.9 tick (median 0.4): timing is the largest single residual
  component at the integer-tick assignment stage; the single template shape is fine.
Sub-tick refinement inside the solver (V200_SUBTICK=1 between sweeps, 2 after every placement
and accepted move; testing/v200): D3 95.535 -> D3S 95.534 / D3S2 95.550; C20 96.213 ->
C20S 96.209 / C20S2 96.194.  Class by class identical.  Cleaning the big families' timing
residual does not change where the small clusters go: their misplacement is intrinsic to the
light information at their energy, not to the residual of the big ones.

Perceiver-based delayed-daughter oracle (testing/v14_michel_oracle_gpu.py, 501 truth
daughters, 3795 null terminal segments): light confirms 54 % of true daughters at their truth
time (dchi2 < -100; 68 % for 20-40 MeV), the residual scan finds the truth time within 10
ticks for 69 %, amplitude ratio median 0.9; null segments pass the same cut 6.7-8.4 % of the
time (3-4 % with the amplitude gate).  Hosts without a daughter outnumber hosts with one 18:1,
so a blind end-segment splitter would move more energy wrongly than it fixes; a splitter needs
a prior that cuts the null rate by > 30x (stopping-track / end-blob / lifetime window).  The
flash-first idea is dead: 69 % of hosts without a daughter also have a later residual flash
> 2000 ADC on their channels (85 % with one).

## 10. The low-energy limit is the prediction engine (2026-09-24, low-intensity sample, 197 events)

testing/v14_engine_residual.py on /pscratch/sd/y/yuxuan/v11arms/LOWI (isolated families >= 1 MeV,
channels with predicted peak >= 1000 ADC, other light < 5 %): per-channel amplitude factor
alpha = actual / predicted at fixed channel amplitude 2000-5000 ADC:
   families < 20 MeV : alpha 10-90 % = 0.58-2.02, |log alpha| median 0.36, residual floor after
                       free amplitude + time = 0.13 (noise + shape)
   families >= 100 MeV: alpha 10-90 % = 0.80-1.20, |log alpha| median 0.10, same floor 0.13
Same light level, same noise floor, 3.5x larger engine error for small clusters.  Per family
the normalisation error is 0.48-1.65 (1-5 MeV) and 0.67-1.94 (5-20 MeV) at 10-90 %, versus
0.87-1.21 above 100 MeV; the within-family pattern error is 17 % vs 12 %.  So for clusters
below 20 MeV the perceiver's absolute light scale is uncertain by about a factor of two even
when the deposit is isolated, far above photon statistics (13-20 % at these amplitudes).  This,
not the search and not the residual of the big families, is why light cannot place small
clusters and why the support-channel construction was needed.  Lever: retrain the engine for
the low-energy regime (relative / heteroscedastic loss so small clusters count, low-energy
oversampling from the low-intensity single-flash population, a per-channel uncertainty head),
then re-run the v2.0 arms below 20 MeV.
