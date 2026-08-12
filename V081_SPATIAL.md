# v0.8.1 — spatial-score campaign (user's SPATIAL_SCORE.md design, tested on NERSC)

## ADOPTED (2026-08-07): v0.8.1 = v0.8 canonical + `--pass2-light-nom`

126-event confirmation (both metrics, energy-weighted):

| chain | NS (<=1.4 GeV) | all-TPC |
|---|---|---|
| v0.8 canonical | 96.49 | 95.49 |
| **v0.8.1 (+light-nominated candidates)** | **96.53** | **95.55** |

Event record: NS 97 better / 23 worse; all-TPC **114 / 10**. Attribution is
direct: 1,373 via-light trades (trades to families OUTSIDE the 8-nearest
spatial candidate set, nominated by the cluster's own loss curve), truth
audit **FIXED 740 / BROKE 164** (4.5 : 1), +3.9 GeV net. Cost ≈ +2 s/event.
Tagged `v0.8.1`.

## v0.8.4 side path (2026-08-12): decline-or-repair — NULL, with a physics lesson

Post-force audit of small imaged clusters: r = best-case light delta-loss /
template norm; if r > r_max, try a local permutation pool over
support-channel neighbours (shifts 0/±6/±12), else unassign (-10000).
File 0000001 scan (assigned-hit E-weighted acc / assigned fraction):

| config | NS acc | NS frac | trig | declined | purity |
|---|---|---|---|---|---|
| v0.8.1 | 96.45 | 100.00 | — | — | — |
| r=0.75 | 96.45 | 99.97 | 44 | 41 (180 MeV) | 10 % |
| r=0.50 | 96.44 | 99.96 | 58 | 57 (250 MeV) | 16 % |
| r=0.25 | 96.44 | 99.94 | 76 | 72 (334 MeV) | 21 % |

Verdict: NOT adopted. "Worsens the light everywhere" selects bad TEMPLATES,
not bad placements — 80-90 % of declined clusters were correctly placed
(ML light-prediction systematics on tiny clusters). Mitigation pools found
0-1 real fixes: third independent confirmation that the blocked-spot failure
mode is extinct. The good coverage-for-accuracy trade remains v0.8.3
(spatial+light coincidence, 44-75 % purity). Real lever: better small-
cluster light templates (small Charge2Light retrain).

### The definition of "tiny" (2026-08-12): the light-information boundary

Three independent measurements agree (180k-cluster feature table with
per-energy-bin AUC of the light features vs wrongness; the support-channel
study k*(E) = 27.6·log10 E − 36; the decline-audit exemption replay):

| band | light-regret dz AUC | reliable channels k* | meaning |
|---|---|---|---|
| E < 3 MeV | 0.46–0.52 (≤ chance; 1–2 MeV ANTI-informative) | 0 | **light-blind** |
| 3–20 MeV | 0.55–0.64 | < 1 | weak prior only — never a trigger/veto |
| E ≥ 20 MeV | 0.72+ | ≥ 1 (k*=0 crossing at 20.2 MeV) | **light-judgeable** |

Information boundaries (measured facts, used only to PROTECT small clusters
from light-driven decisions — nothing is discarded by them): light-blind
below ~3–5 MeV, weak prior 3–20 MeV, trustworthy ≥ 20 MeV (k* crossing at
20.2 MeV).  The decline replay confirms the deeper point: no energy cut
rescues a fits-nowhere trigger — above 5 MeV its purity is 0 % because
misplaced clusters by definition fit well SOMEWHERE (low r — the PASS-2
nomination signal); fits-nowhere = broken template, at every energy.

### NEUTRON-PHYSICS CONSTRAINT (user, 2026-08-12): the discard threshold

Small-energy neutron deposits (recoils / capture gammas, ~0.5–6 MeV, e.g.
5 MeV) MUST remain assigned — and they share the exact pruning signature
(small, isolated, light-invisible).  Measured conflict: under the previous
E < 3 MeV rule, **100 % of the pruned energy sat in the 1–10 MeV neutron
band**.  Neutron-safe re-tune (files 1–8 tune / 9–10 test):

| discard gate | hit frac | NS assigned acc | purity | pruned E in 1–10 MeV |
|---|---|---|---|---|
| E < 3 (old) | 0.59 % | 96.68 → 97.18 | 47 % | **100 %** |
| **E < 1 (new default)** | 0.06 % | 96.68 → 96.73 | 57 % | **0 %** |

**Default `v083_prune_cuts.json` is now the E < 1 MeV rule** — deposits
≥ 1 MeV are never unassigned; the aggressive rule survives as
`v083_prune_cuts_e3_nonneutron.json` for analyses without neutron signal.
Consequence accepted: with the neutron band protected, pruning-as-removal
is nearly powerless (+0.05 pp).  The badness score's real product is the
per-cluster QUALITY FLAG shipped alongside the assignment — analyses choose
their own trade; neutron analyses keep everything.

## v0.8.3 (2026-08-11): confidence-based HIT PRUNING — adopted as an output mode

A charge+light "badness" score per assigned cluster (29 v4 spatial features
vs its own reco family + light confidence z_min / z_own / regret dz /
chi2_loc / n_tpc + basics), HGB, leave-one-file-out — every score
out-of-sample.  Pruned clusters get t = −1; headline forced-coverage
metrics unchanged; `--prune-frac F` records row["prune"].

Assigned-hit accuracy (energy-weighted) vs pruned hit fraction:

| substrate | 0 % | 0.5 % | **1 %** | 2 % | OOS AUC |
|---|---|---|---|---|---|
| v0.6 dumps, 126 ev (NS) | 95.99 | 96.40 | **96.81** | 97.47 | 0.855 |
| v0.8 dumps, file 1 (NS, transfer) | 96.46 | 96.83 | **97.10** | 97.59 | 0.828 |

Prune purity at 1 %: 75 % (v0.6) / 60 % (v0.8) of pruned energy truly wrong
— ~20× enrichment over the ~3 % base rate.  Feature importances split
exactly as designed: energy first (logE +0.070 — small clusters carry the
error), then spatial orphan-ness (best_lat, best_gap, d_eu), then the light
block (z_min, z_own, dz, chi2_loc).  Artifacts:
`testing/v083_prune.py` (study + `prune_event`), `testing/v083_prune_model.joblib`
(deploy HGB + calibrated global thresholds 0.25/0.5/1/2 %),
`testing/v083_prune_rows.npz`.

## v0.8.2 exploration (2026-08-07): the loop is at a FIXED POINT — no adoption

Three smarter-loop designs, all measured on file 0000001 vs v0.8.1
(96.45 / 95.54), all flag-gated and kept available:

| design | result |
|---|---|
| `--pass2-iterations 3` (fresh nominations each round from the traded base) | converged: rounds 2-3 add ~10 trades, metrics identical |
| `--loop-p23 2` (full Q1↔Q2 alternation with per-round ledger) | round 2 = 11 trades + 2 closes / 13 ev; P3 ops FIXED 3 / BROKE 2; metrics identical |
| `--pass2-light-prop` (symmetric dual: light proposes, spatial weak veto p≥0.15) | blocked cohort nearly empty: 45 trades, FIXED 13 / BROKE 7, NET NEGATIVE energy (29/35 MeV) |

Conclusion: one pass of light-nominated trading already captures everything
this operator family can reach — iteration, alternation, and symmetric
proposal all find the same fixed point. **v0.8.1 remains canonical.**
Remaining headroom lives OUTSIDE the loop: (1) merged two-interaction
clusters — clustering-level, frontend; (2) saturation-degenerate light
minima — loss shaping / unclipped channels; (3) the 8 % missing-slot
cohort — OPEN operator; (4) sub-5 MeV placement — physics-limited per the
confidence study.

*(2026-08-07. Reimplements the user's local design record
`NDLAr-full/.claude/SPATIAL_SCORE.md` from spec — the trained artifacts never
left the laptop — then extends it (v1.1 topology/reach, v1.2 neighbor-TPC)
and confronts every variant with the OPERATIONAL trading task on file
0000001, out-of-sample for all models.)*

Code: `testing/sptl_v081.py` (features/models), `testing/sptl_v082.py`
(parallel pipeline, all variants), `testing/sptl_v081_h2h.py` (head-to-head
+ replay). Artifacts: `testing/sptl_v081_models.json` (calibrated logistics
incl. deploy copies), `testing/tpc_adjacency.json`, `testing/sptl_v082_hgb.joblib`
(isotonic-calibrated HGB on v1.2 features), `testing/sptl_v082_rows.npz`
(591k-row truth harvest), `testing/sptl_h2h_rows.pkl`.

## 1. Reproduction of the doc design (files 1–8 train / 9–10 test)

| level | mine | doc | note |
|---|---|---|---|
| pair harvest | 103k pairs, base 0.859 | 109k, 0.867 | same population |
| J0 | 0.626 | 0.649 | |
| J1 (+pointing/tpc/triangles) | 0.707 | 0.720 | track-track 0.766 vs 0.864 |
| S_iF v0 | 0.885 | 0.915 | my harvest keeps ≥4-hit clusters (harder) |

Structure reproduces faithfully; consistent −0.02/−0.03 offsets from the
finer cluster population. Calibration honest in every bin (0.99 bin → 0.989).

## 2. Upgrades (truth-side, test files 9–10)

| variant | AUC | crossing | thru-shower | non-contact | fresh-TPC |
|---|---|---|---|---|---|
| S v0 | 0.885 | 0.832 | 0.923 | 0.872 | 0.791 (n=11k) |
| **S v1.1** (+continuous contact topology, shower reach 150 cm) | **0.930** | 0.824 | 0.951 | **0.921** | 0.890 (n=40k) |
| S v1.2 (+neighbor-TPC: tpc_adj, gap-aware axis DOCA, 60 cm adj reach) | 0.929 | 0.816 | 0.951 | 0.919 | 0.862 (n=57k) |
| v1.2-HGB (isotonic) | 0.925 | | | | 0.858 |

- v1.1's +0.045 is real; the doc's §5.3 continuous topology delivers most of it.
- v1.2's adjacency reach grows the fresh-TPC candidate population +40 %
  (the doc's untestable caveat now measured) at comparable discrimination.
- TPC geometry: 70 charge TPCs, 7×5 module grid ×2 drift volumes; module
  gaps 5–6.5 cm in x and z → cross-module contact < 3 cm is IMPOSSIBLE;
  interior TPCs have 5 box-neighbors (cathode partner + ±x + ±z).

## 3. THE OPERATIONAL VERDICT (file 0000001, out-of-sample for all)

14,149 identical (cluster, own-reco-family) rows; production population
(< 30 MeV): 11,452 rows, wrong rate 0.299.

| scorer | AUC small | replay (prod gate, no light veto): net |
|---|---|---|
| **v4 HGB (incumbent)** | 0.718 | **+828 MeV** (328 fixed / 138 broke) |
| S v1.1 | 0.723 | +536 |
| S v1.2 | 0.723 | +533 |
| v1.2-HGB | 0.728 | +564 |

**All four scorers converge to ~0.72 operationally.** The +0.045 truth-side
gain does NOT transfer: the bottleneck is covariate shift (scoring against
half-wrong reco families), not feature design or model class. **v4 stays in
PASS 2.** (v4's overall-energy edge, 0.746, lives in big clusters that
production never trades.)

## 4. The binding constraint is CANDIDATE GENERATION, not scoring

For wrong small clusters (13.5 GeV in file 0000001):

| correct-time family is ... | % of wrong energy |
|---|---|
| within the 8-nearest candidate set | 57 % |
| + adjacency-extended reach (v1.2 rule) | 59 % (+2 only) |
| **somewhere in the event** | **92 %** |
| nowhere (missing slot → OPEN operator) | 8 % |

35 % of wrong energy has its correct family in the event but spatially FAR
(beyond 8-nearest AND beyond adjacent-TPC/150 cm). Geometry cannot nominate
these candidates — but the LIGHT can: the cluster's own a_c(t) curve minima
name the right times, and families exist at those times. PASS 2 already
computes this curve for the veto.

## 5. v0.8.1 conclusions and next steps

1. **Keep v4** as the PASS-2 scorer (measured, twice).
2. **Next in-chain A/B: light-nominated candidates in PASS 2** — candidate
   families = 8-nearest ∪ families whose t0 sits at a strong minimum of the
   cluster's own loss curve. Attacks the 35 % directly with machinery
   already in the pass. (This is the user's Q2-conditioned-on-light
   philosophy, applied to candidate generation.)
3. OPEN operator: only 8 % — low priority (consistent with every earlier
   measurement).
4. −λ·log S into the joint solver: GATED on operational recalibration —
   truth-trained calibration does not survive the shift (this campaign's
   central negative result). Recalibrate isotonic on operational rows first.
5. Neighbor-TPC adjacency (`tpc_adjacency.json`) and the J-graph bonds stay
   available; adjacency-as-reach is measured weak (+2 %), adjacency-as-
   feature is neutral.
