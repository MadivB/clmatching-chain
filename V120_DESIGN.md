# v1.2 — per-hit precision timing for ND (the 2x2 v2 line, ND-adapted)

*Opened 2026-09-22 on top of v1.1 (tag CLMatching_v1.1). Source design:
/pscratch/sd/y/yuxuan/2x2QLMatching/V2Design/ND_HANDOFF.md. User constraints:
accommodate ND — per-family SUPPORT channels rather than a fixed channel count,
explicit light PILE-UP handling, a deliberate FAMILY definition; the single
average pulse shape is acceptable; calibration source = the low-intensity
training sample (MiniProdN5p1 ... lowintensity.sanddrift, 767 files).*

## 1. What limits the v1.1 timing (truth decomposition, testing/v12_decompose.py)

Per-hit residual of v1.1 stage E on correctly assigned core hits (63 events):
std 9.1 ns, robust 6.0 ns.  Decomposed with truth (used only to measure):

| term | size | verdict |
|---|---|---|
| family OFFSET spread (one time per family) | 9.4 ns std | the whole problem |
| within-family spread | 1.4 ns | small; the apex clock t_dep = t_apex + d/c is exact in truth (slope 1.00; 1.44 -> 0.88 ns unexplained) |
| light time of flight across families | onset ToF p10-p90 5.3-7.9 ns | ~1 ns variation, uncorrelated with the offsets |
| per-channel 20 %-edge vs truth (testing/v12_channel_edges.py) | 4.0 ns robust on clean channels, 13 ns with >= 10 % pile-up | raw edges already beat the chain's family time (5.9 ns) |
| family mean of clean channel edges | 2.9 ns robust | the target scale |
| slew (edge vs amplitude) | +5.8 ns / decade, edge later for brighter channels | must be corrected |
| pile-up | 50 % of (family, channel) pairs carry >= 10 % other light | must fit on the residual |
| detector-type delay | period-12 channel pattern, ~10 ns between the two types | hidden in the single average template; calibratable in situ |

## 2. Design (testing/v120_precision.py, stage P after stage E)

* Family = hits sharing one final t0, per light TPC.  Support channels = the
  channels holding 90 % of the family's own predicted light in that TPC.
* Per-channel delay-aware model: the family's light-arrival density on the
  channel from its hits (apex clock + n d/c, c_LAr = 13.4 cm/ns fixed, 1/d^2
  visibility proxy, 0.5 ns bins) convolved with the single average template
  (kept to 3200 ns: the stored template carries ~100 samples of pre-trigger
  padding before its rise at ~1620 ns).
* Pile-up: the fit runs on actual - (all other families rendered at their
  times); channels are down-weighted by their pile-up fraction; the other
  families are re-rendered at their refined arrival-equivalent times and the
  fit repeated (3 rounds).  A convention mismatch here (apex time vs arrival
  time) or subtracting the family's own light at a different time than the
  model holds it makes the iteration diverge — both measured and fixed.
* Per-channel time: 0.25-tick scan (+-3 ticks around the chain time) with the
  amplitude profiled out and a 9-point analytic parabola; per-channel-index
  delay k_idx and per-detector-type slew beta_T removed.
* Family apex time = weighted median over channels (w = sqrt(A) x (1 - pile)),
  robust to pile-up outliers; sigma_t = 1.4826 MAD / sqrt(n).
* Apex = the candidate (two principal-axis ends, centroid) with the lowest
  total per-channel loss (truth-apex ceiling test: only 0.1-0.3 ns better);
  per-hit deposition time = t0 + |r_h - apex| / c.
* Calibration (testing/v120_calibrate.py): alternating medians over the
  families of a dump set — k_idx[120] from per-index residual medians, the
  detector type from the k pattern, beta_T per type — no truth anywhere.
  Converges in 3 passes (|update| median 0.17 ns).

## 3. Measured so far

| sample | metric | stage E (v1.1) | stage P |
|---|---|---|---|
| file 1, 13 events (prototype) | per-hit std / robust | 8.06 / 5.80 ns | 4.55 / 3.26 ns |
| | per-family std / robust | 7.89 / 5.74 ns | 4.09 / 2.99 ns |
| file 1, 4 events, calibrated + 3 rounds | per-hit std / robust | 7.64 / 5.73 ns | 3.65 / 3.04 ns |
| | per-family std / robust | 7.53 / 5.54 ns | 3.17 / 2.82 ns |
| | families with < 5 % pile-up | | 2.1 ns std |
| | families with >= 50 % pile-up (29 % of E) | | 4.1 ns std |

Calibration on 13 events: type pattern exactly 000000111111 repeating, type
offset 9.8 ns, beta_T = 0.33 / 1.23 ns per decade.

Measured and not adopted: fit-window and curvature-weight variants (insensitive);
2x2 topology-corrector home rules (endpoint-at-shower 96 % pure, embedded 93 %,
isolated-fragment 35 % — v1.1 already agrees with the pure ones; the crossing-
track failure mode does not occur in ND, 0 cases / 39 events) — see
testing/v12_home_rules.py.

## 4. Status

In-chain stage wired (`--precision --precision-calib --precision-rounds`),
dumps carry `hit_sigma_p`; evaluator testing/v12_eval.py.  First in-chain run
(P1, files 1-3) and the low-intensity dump run for the official calibration
are in progress.  Open: GPU-path timing, cross-TPC family merging, and the
data-side k_j calibration once a point-like source is identified.

### First in-chain run (P1 = v1.1 + `--precision`, files 1-3, 39 events, calibration from 13 events of file 1)

| metric (core hits, E-weighted) | v1.1 stage E | v1.2 stage P |
|---|---|---|
| per-hit std / robust | 8.79 / 5.92 ns | 6.13 / **2.82 ns** |
| per-family std / robust | 8.41 / 5.80 ns | 5.44 / **2.54 ns** |
| delivered per-hit sigma | | 98.3 % of hits, median 0.43 ns |
| assignment efficiency | 96.88 / 96.04 | identical (+0.000 / +0.005) |
| stage cost | | 13.8 s/ev median (3 rounds), total unchanged in median |

The robust resolution halves; the std improves less because it is carried by
a tail of families where the fit goes wrong (analysed next).

### The delivered per-hit sigma is an honest uncertainty (P1, 39 events)

| sigma_p bin | share of energy | per-hit std (stage P) | same hits, stage E |
|---|---|---|---|
| < 0.3 ns | 24 % | 2.43 ns | 5.38 ns |
| 0.3 - 0.6 ns | 50 % | 3.09 ns | 6.06 ns |
| 0.6 - 1.0 ns | 17 % | 4.19 ns | 8.67 ns |
| 1 - 2 ns | 7 % | 6.7 ns | 13.6 ns |
| 2 - 4 ns | 1.8 % | 12.3 ns | 20.5 ns |
| > 4 ns | 0.6 % | 23 ns | 27 ns |

Stage P beats stage E in every bin, so no fallback is applied; the tail
(0.74 % of core energy beyond 20 ns) is what carries the std, and sigma_p is
the quality flag for it — 74 % of the energy sits below 0.6 ns with a per-hit
std of about 3 ns, 91 % below 1 ns with std at or below 4.2 ns.

### Pile-up-free floor (20 held-out low-intensity events, calibration from file 1)

| metric | stage E | stage P |
|---|---|---|
| per-hit std / robust | 6.78 / 5.10 ns | 4.51 / 2.62 ns |
| per-family std / robust | 6.54 / 5.07 ns | 4.10 / 2.35 ns |
| per-hit std with a 20 ns clip | | 3.02 ns (0.46 % of energy beyond 20 ns) |

Nearly the same as on the standard sample: pile-up is not what limits the
bulk resolution.  The floor is the per-channel timing itself (about 4 ns per
channel) with one average pulse template and a per-channel-index rather than
per-channel delay.  The next levers, in the 2x2 order, are per-channel k_j
from the 767-file low-intensity sample, per-detector-type templates, and a
sub-sample pulse-shape fit.

### Calibration A/B, out of sample (files 2-3, 26 events, seen by neither calibration)

| calibration | per-hit std / robust | clipped std | per-family std / robust |
|---|---|---|---|
| none | 4.69 / 3.18 ns | 3.60 ns | 4.19 / 2.86 ns |
| file 1, 13 events | 4.59 / 2.81 ns | 3.22 ns | 4.08 / 2.54 ns |
| low-intensity, 60 events (delivered) | 4.64 / 2.82 ns | 3.31 ns | 4.13 / 2.52 ns |

The two calibrations agree to 0.67 ns rms (correlation 0.999): the same
period-12 detector-type pattern with a 10 ns offset and per-type slews of
0.4 and 1.4 ns per decade.  The calibration buys ~12 % in robust resolution.

## 5. v1.2 definition and certification (2026-09-23)

**v1.2 = v1.1 + `--precision --precision-calib testing/v120_calib_lowi.json
--precision-rounds 3`** (launcher `run_v12.sh`).  Certified on files
0000001-0000005 (63 events) against the v1.1 outputs of the same events:

| metric (core hits, E-weighted) | v1.1 stage E | v1.2 stage P |
|---|---|---|
| per-hit std / robust | 9.13 / 6.00 ns | 6.43 / **2.85 ns** |
| per-family std / robust | 8.75 / 5.87 ns | 5.74 / **2.56 ns** |
| per-file robust (5 files) | 5.85-6.13 ns | 2.79-2.92 ns |
| per-hit std with a 20 ns clip | | 3.4-3.5 ns; 1.1-1.6 % of energy beyond 20 ns |
| delivered sigma_p | | 98.3 % of hits, median 0.44 ns |
| assignment efficiency | 96.70 / 95.54 | identical within 0.01 pp |
| stage cost | | 9.4 s/ev median on the GPU (p90 16.7, max 28); total 116 -> 128 s/ev/worker |

Adopted 2026-09-23, tag CLMatching_v1.2.

## 6. v1.2.1 — multiple starting times per family (pieces), 2026-09-23

User requirement: time-of-flight analysis needs displaced fragments to carry
their own starting t0 (tracks at v = c from the apex, fragments possibly
slower), with the velocity inferred from the fitted shift.

Census (v1.2 outputs, 13 events, correctly assigned members, pieces = 3-cm
contact chains, robust apex reference = median over core hits of truth - d/c):
28.5 % of family energy is displaced from the main chained piece; true delays
p50 3 ns, p90 13 ns; 17 % of fragment energy has a delay > 10 ns and 7 %
> 30 ns; the slow population (beta < 0.5) is 23 GeV per 13 events, median delay
12 ns at 76 cm; under v1.2's v = c assumption these fragments are mis-timed by
their delay (rms 60-200 ns).  A first version of this census used the earliest
2 % of core truth hits as the apex time and reported delays of 20-100 ns for
pieces that sit exactly at the chain tick — that reference is pulled early by
pathological hits in big families and must not be used.

Piece mode (`--precision-pieces`): the core (biggest chained piece) keeps the
family fit; every other piece of >= 5 MeV gets a FREE time from its own
support channels on the residual after everything else, scanning +-10 ticks,
but only on channels where its own predicted light is at least comparable to
everything else there — with fewer than three such channels a piece is
declared unresolvable by light and keeps the core's apex-clock time (sigma =
NaN).  Each fitted piece is delivered with its delay to the core apex time,
its distance from the core apex, the implied beta and its sigma.

Offline truth test (4 events, correctly assigned pieces): delay resolution
4.6 ns robust (6.1 / 3.8 / 3.8 ns for 5-20 / 20-100 / >= 100 MeV), 94 % within
10 ns; fragment hits improve from 5.2 / 4.0 ns to 3.1 / 2.7 ns (std / robust).
Light-resolvable pieces: 13-19 per event on the standard sample, ~6 per event
on the low-intensity sample; the rest are flagged.

### What the piece study found (in-chain P2, 39 events; debug on file 1)

* Piece fits are correct: for the pieces whose energy-weighted truth appeared
  delayed by 30-160 ns, the light and the median hit truth sit at the chain
  tick within a fraction of a tick; the apparent delay comes from a minority
  of hits with much later truth times inside the same cluster (impure pieces).
  Within the assignment window this sample contains no genuinely delayed,
  light-resolvable fragment population.
* Delay resolution of light-resolvable pieces (751 with a truth reference):
  4.5 ns robust for v = c pieces (5.8 / 4.4 / 4.5 ns for 5-20 / 20-100 /
  >= 100 MeV); 21.5 light-resolvable pieces per event out of ~1000 pieces.
* The genuinely late population (clusters >= 5 MeV with truth > 10 ticks after
  the nearest bigger neighbour within 1 m: 3544 clusters, 207 GeV, median
  delay 2.95 us) is timed by the chain at its OWN flash for 95 % of the energy
  (78 / 92 / 99 % by energy band) — these are separate families, each with its
  own v1.2 apex time (2.6 ns robust), so a time-of-flight delay is the
  difference of two family times.  Below 5 MeV deposits are light-blind and
  carry only their association time.
* The piece layer is an add-on over the certified family fit (bulk resolution
  unchanged) and delivers per piece: delay to the core apex time, distance,
  implied beta, sigma; and per family: apex time, sigma, apex position, energy
  (`row["phase3"]["pieces"]`, `row["phase3"]["families"]`).
