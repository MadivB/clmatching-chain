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
