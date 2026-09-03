# Confidence-metric research notes (deep-research run, 2026-09-02)

105-agent verified sweep (22 claims confirmed 3-0, 3 refuted 0-3) on turning a
chi2/loss landscape into calibrated per-cluster assignment confidence.
Feeds the v0.9.3 confidence design.

## The recipe the literature converges on

1. **Normalize over competitors, don't read the best fit alone** (PDA/JPDA,
   target tracking): beta_i ∝ exp(−chi2_i/2) over the *gated* set of candidate
   flash-time hypotheses **plus an explicit "none correct" null term**, so the
   probabilities sum to 1 including the no-light outcome. Microseconds per
   object; every input is already in our scan curve.
   [Bar-Shalom & Daum 2009; Fortmann/Bar-Shalom/Scheffe 1983]
2. **Never trust raw Δchi2 → p via a chi-square reference.** Provably
   miscalibrated here for three independent reasons: competing assignments are
   **non-nested** hypotheses (Wilks does not apply — Vuong/Cox); template
   systematics violate the correct-model condition (our chi2/dof >> 1); the
   time-shift scan induces a **look-elsewhere effect** (multiple local minima
   → overconfident p-values). [Algeri/Aalbers/Conrad; Algeri, Conrad & van Dyk
   MNRAS 458:L84; Hardin 2211.06347] A claim that profile-LR asymptotics apply
   directly was refuted 0-3.
3. **So a calibration layer is mandatory and cheap**: fit a temperature /
   isotonic / Platt map on truth-matched data offline, apply as a table read.
   (Fully-empirical Feldman-Cousins-style calibration is CPU-weeks; the
   offline-precomputed lookup form is the viable one. A claim that calibration
   *must* be empirical rather than analytic was also refuted — the
   Gross-Vitells half-chi2 + expected-upcrossings bound handles the continuous
   -scan case analytically after a small toy run.)
4. **Swap-aware/shared-support evidence is validated field practice**: ATLAS
   shared-hit penalties ("shared hits are often an indicator of fakes"), and
   JPDA's *joint* association posteriors give dramatic gains exactly when
   objects contend for the same measurements. Our anchored-covariance metric
   is the same family; the principled generalization is a joint posterior
   over the cluster-flash bipartite graph (loopy BP converges there —
   Williams & Lau 2014 — and is the same mean-field family as the v0.7 MRF).
5. **Landscape-shape features into a tiny learned ranker is precedented**
   (ACTS ambiguity solver: 8 cheap fit summaries → 10/15/10 MLP, margin
   ranking loss, 32x duplicate reduction) — but its authors note the scores
   are *uncalibrated* rankings; calibration is a separate pass. Same posture
   as ours (features → logit/HGB → isotonic).
6. LArTPC baseline (MicroBooNE Wire-Cell: L1-regularized chi2 coefficients
   read as "credibility"; Pandora: pure argmax) stops short of calibrated
   probabilities — a calibrated per-cluster confidence is genuinely novel
   territory for the field.

## What this changed in our implementation (scan_margin.py)

- Scan-curve features now in **Δchi2 units**: depth and competitor margin are
  divided by the measured active-region residual variance (a per-cluster
  empirical temperature — self-fitted, absorbs template systematics).
- **post_null**: PDA posterior of the chosen time over {best, competing local
  minima, null(no own light)} on the normalized curve. The null term is free:
  with the constant dropped, L(no light) = 0 by construction.
- Isotonic on LOO scores stays as the mandated empirical calibration layer.

## Open questions the study flagged

- Effective temperature under chi2/dof >> 1: is one per-cluster variance
  enough, or is a per-energy-band scale needed? (We answer empirically.)
- Discrete competitor set vs continuous-scan Gross-Vitells: unproven
  adaptation → rely on isotonic, not the analytic bound.
- Full bipartite BP posterior subsuming both families: candidate v1.0 work;
  one BP pass per TPC at our multiplicities should be cheap.

## Key sources

- PDA/JPDA: ieeexplore.ieee.org/document/1145560 (Fortmann 1983);
  Bar-Shalom & Daum, IEEE CSM 2009; Meyer et al., Proc. IEEE 2018 (BP);
  Williams & Lau, IEEE TAES 2014 (BP convergence proof).
- Wilks failure modes: arXiv:1911.10237 (Nature Rev. Phys. 2, 245);
  Algeri, Conrad & van Dyk, MNRAS 458:L84 (non-nested + upcrossings bound);
  arXiv:2211.06347 (scan look-elsewhere).
- Annealed soft assignment = template fitting: Strandlie & Fruhwirth,
  Rev. Mod. Phys. 82, 1419 (DAF/elastic-arms EM equivalence).
- ACTS ML ambiguity solver: arXiv:2312.05070 (CHEP 2023).
- ATLAS shared hits: arXiv:2308.09471, 1704.07983, 2412.15090.
- MicroBooNE flash matching: arXiv:2011.01375 (JINST 16 P06043);
  arXiv:1805.04629 (Pandora, proceedings).
- Calibration: arXiv:1706.04599 (temperature scaling); scikit-learn
  calibration guide; arXiv:2107.07511 (conformal).
