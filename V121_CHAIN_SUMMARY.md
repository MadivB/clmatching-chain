# v1.2.1 production chain: stages, timing, and the mistakes we still make

Measured on the v1.2.1 run of files 1-3 (39 events, dumps in /pscratch/sd/y/yuxuan/v11arms/P2).
Timings are per-event medians for one worker; two workers share one GPU.
A tick is 16 ns; a hit counts as wrong when |reco t0 - truth t0| > 10 ticks.

## Stages

| Stage | Purpose | s/ev/worker |
|---|---|---|
| 1 | load FLOW (3.3), clustering (11.1), perceiver light prediction (32.7), track stage (7.9) | 55.9 |
| 2 + 3.1 | phase-1 showers / >=3-TPC tracks, then the light-component permutation pool (v1.1: light-coupled components, exact k^n search, 20 MeV floor) | 13.4 |
| 3.2 | overflow trigger and rescue | 4.2 |
| 3.3 + 3.5 | P2 scan and placement police | 2.7 |
| 3.9 | joint mean-field tail solver | 8.5 |
| 3.95 | pass-2 spatial trading | 7.9 |
| 3.98 | component re-decision (v1.1, >=20 MeV placed clusters, TPCs <= 1.4 GeV) | 1.3 |
| 4.1 | v11 error matrix on shower TPCs | 17.4 |
| 4.5 | swap fix | 0.5 |
| 4.9 | KD-tree force coverage (everything still unplaced goes to the nearest family) | 0.3 |
| E | sub-tick continuous t0 per family | 2.1 |
| conf | hit_t0_conf = pcu_vote10 | 0.75 |
| P | per-hit precision timing, in-situ channel calibration, 3 pile-up rounds, displaced pieces (v1.2 / v1.2.1) | 11.0 (14.0 with checkpoint) |
| total | | 130.8 (about 65 s/ev/GPU) |

Prediction (stage 1) and the shower error matrix (4.1) are the two big costs; everything between 3.2 and 4.9 is 25 s together.

## Where the wrong energy is (39 events, 91.45 GeV wrong = 4.00 %)

| Class | Wrong E (GeV) | Fraction of class wrong | Share of all wrong | Of which at a flash we never discovered |
|---|---|---|---|---|
| unclustered (noise) hits | 13.73 | 41.6 % | 15.0 % | 22 % |
| clusters < 1.5 MeV | 1.43 | 32.6 % | 1.6 % | - |
| imaged 1.5-5 MeV | 18.77 | 27.4 % | 20.5 % | 17 % |
| 5-20 MeV | 11.31 | 11.8 % | 12.4 % | 11 % |
| 20-100 MeV | 19.11 | 8.2 % | 20.9 % | 7 % |
| >= 100 MeV | 27.10 | 1.5 % | 29.6 % | 10 % |

## Displays

All in /global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing/v12_displays/
Each TPC has four files: `_light_charge.png` (blue actual light, orange predicted from the final assignment),
`_reco3d.png/.html` (hits coloured by reco family), `_truth3d.png/.html` (truth families), `_wrong3d.png/.html` (red = wrong).
Open the `.html` files for rotation.

### A. Two GeV interactions tangled in one TPC (mixed clusters) - demo_0000003_ev12_tpc8
5789 MeV, 35 % wrong (2038 MeV). Truth: 3.5 GeV at t0 458 and 2.0 GeV at t0 200, overlapping in space.
Both flashes are found (458.0 and 200.5) and are 260 ticks apart, so this is not a light ambiguity.
The clusters themselves contain hits of both interactions (reco red 2619 MeV vs truth red 1999; pieces went both ways),
and one t0 per cluster can never be right. Also visible: a 9 MeV blip at t0 414 correctly placed at its own slot,
an 11 MeV blip at t0 438 absorbed by the 458 family (20 ticks off).

### B. Pieces of a 6.8 GeV interaction peeled off to invented slots (dominant collapse) - demo_0000003_ev9_tpc26
7350 MeV, 713 MeV wrong. Truth: 6817 MeV at t0 139.7. Reco keeps 6088 MeV at 139.4 and scatters 20-200 MeV pieces:
130 MeV to t0 13 (a real neighbour flash), 150 MeV to 359 (real), and 49/77/38/39/31/23 MeV to 34/227/207/170/23/0,
none of which exist in truth. The light display shows the big flash clipping many channels (flat tops on L36-L41, L47-L52):
the pieces' light lives inside the clipped region, the loss cannot see them, and they drift to self-made slots.

### C. Small blips among near-coincident flashes (the neutron-like class) - demo_0000003_ev11_tpc4
1968 MeV, 335 MeV wrong. Six flashes at 267/304/348/551/568/587/602. All six large clusters are placed within 0.5 tick.
The wrong energy is entirely the scatter of 1.5-20 MeV blobs (recoils from the 304 and 568 interactions) that ended up
at 586 or 605 instead of 568, or at 348 instead of 304. A 3 MeV blob is 1/200 of the neighbouring hairball's light,
so the loss is flat across the 551-602 slots and the chain falls back to geometry (nearest family). Neutron deposits
are exactly the ones that are not near their parent.

### D. A track of one interaction crossing another's hairball (contact merges) - demo_0000001_ev0_tpc55
2389 MeV, 290 MeV wrong. A 200 MeV EM comb at t0 31 next to a 1.8 GeV hairball at 193. The comb's 50 MeV vertical
track that passes through the hairball was absorbed by the 193 family; a 30 MeV track hanging off the comb (truth 193)
was given to the comb; a 20 MeV curved segment (truth 31) went to a third family at 18. Every piece is < 3 % of its host,
so light cannot arbitrate and contact decides.

### E. Low pile-up, 18 interactions, the good case - demo_0000002_ev3_tpc28
1643 MeV, 146 MeV wrong (91 % right). 15 of 18 families within 0.5 tick. Residual mistakes: a 23 MeV track at 525
merged at its touching point into the 254 track (270 ticks apart); a 19 MeV track at 751 absorbed by the 602 track it
touches; about thirty 1-5 MeV blips of the t0 13.5 interaction handed to the 112/497/525 families. Light display:
R40/R41 carry an actual peak near sample 250 that the prediction does not reproduce (light with no matching charge model,
the undiscovered-flash class).

## Mistake types, ranked by energy

1. Mixed clusters between overlapping GeV interactions (A, D): a clustering-level error; assignment cannot fix a cluster
   that holds two interactions. Needs hit-level splitting of a cluster at inter-family contacts, using both flashes.
2. Dominant collapse (B): clipped channels make the loss blind to 20-200 MeV pieces of a GeV interaction, which then
   float to invented slots. Needs the clipped channels' information used differently (timing of the clip edge, or
   unclipped far channels weighted up), not more permutations.
3. Neutron-like 1.5-20 MeV blips near several coincident flashes (C, E): 33 % of wrong energy, placed by geometry
   because their own light is invisible. Physically they are at their parent's time to within 1-2 ticks
   (a 10 MeV neutron covers 1 m in about 23 ns), so the right prior is "same time as some family in this TPC", not
   "nearest family". Their collective light (ten 3 MeV blips = 30 MeV) is above the 20 MeV light-information boundary
   even though each one is below it, so placing them as a component instead of one at a time is the physically sound lever.
4. Unclustered noise hits (15 %): force-covered to the nearest family; 22 % of them belong to flashes we never discovered.
5. Contact merges of 20-35 MeV track segments at crossings and vertices (D, E).

## What "undiscovered" means for the 20-100 and >= 100 MeV classes (v12_undiscovered.py)

The column is per hit: no reco family in the same TPC within 10 ticks of the hit's own truth time
(truth time = backtracked best segment time, so a delayed daughter carries its own time, not the parent's).
For clusters above 20 MeV it is almost never the cluster's main flash that is missing:

| | 20-100 MeV | >= 100 MeV |
|---|---|---|
| undiscovered wrong E | 1350 MeV (7.1 % of class wrong) | 2611 MeV (9.6 %) |
| sub-piece of an otherwise correct cluster | 707 | 2253 |
| piece later than its host / earlier | 649 / 59 | 2113 / 141 |
| delay 10-30 / 30-100 / 100-300 / >300 ticks | 113 / 190 / 314 / 33 | 336 / 812 / 817 / 148 |
| delayed pieces with E <= 55 MeV | 100 % | 100 % |
| delayed pieces whose own flash is visible in the residual | 99 % | 94 % |
| whole cluster misplaced | 535 (169 of it truth t0 = 0, reco -0.26) | 147 |
| truth t0 beyond the 885-tick window | 108 | 211 |

The delay spectrum follows the muon lifetime (137 ticks): these are Michel electrons and other delayed daughters
sitting inside the parent cluster. The chain gives one time per cluster and the v1.2.1 piece scan is +-10 ticks,
so the daughter inherits the parent's time although its own flash is in the light. Example: 0000002 ev5 TPC42,
33 MeV inside an 844 MeV cluster, delay 133 ticks, residual peak 11.7k at its time.
Genuinely missed flashes with a sizeable cluster are about 0.5 GeV in 39 events (0.6 % of wrong energy).
