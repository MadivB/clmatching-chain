# Charge-Light Matching — Version Log

Every new version records: WHAT changed, the INTUITION behind it, and the BAD
EXAMPLES it removed (file/event/TPC so they can be re-opened in
`FragVsBaseClusters/event_display_compare.ipynb`). Metrics are over the standard
benchmark: MiniProdN5p1 full.sanddrift files 5/6/7, events 0-11 (36 events).

---

## v0.5.5 (2026-07-20) — front-end clustering overhaul

**Two official front-end clusterings; default = `m5p1all`.**

### Front-end A (default): `m5p1all` — fully-fixed M5p1 clustering
`FragVsBaseClusters/save_all.py`; matching inherits it via
`FragVsBaseClusters/upgraded_frontend.py: install()` (drop-in
`run_global_track_clustering`, verified co-membership 1.000 vs saved labels).
Three upgrades on the v0.4 base front-end:

1. **Cross-TPC stitch rebuilt** (`restitch.py`): endpoint tolerance 25→20 cm,
   z-boundary veto (endpoints within 10 cm of detector z-min/max never stitch),
   and a DIRECTIONAL gate — decompose the endpoint gap into longitudinal
   (along-track) vs TRANSVERSE; longitudinal gaps are dead pixels / TPC
   boundaries (OK), transverse gaps are two parallel tracks side by side
   (reject, TRANS_TOL = 7 cm).
   - *Intuition*: distance alone can't distinguish "same track continuing
     through a dead region" from "two nearby parallel tracks". The direction of
     the gap can.
   - *TRANS_TOL = 7 cm because*: at 5 cm, 11.3% of CORRECT stitches broke (their
     transverse offsets cluster at ~5.2 cm from segment-fit noise); 7 cm
     recovers ~90% of them while staying well below the 12.5 cm transverse gap
     of the bad parallel-muon edge → no purity cost.
   - *Bad example removed*: f7 ev8/ev9 tpc60 — two parallel muons crossing 5
     TPCs fused into one 1665 MeV cluster (TPC60 wrong-energy 27% → 0.6%).
2. **Vertex merge tightened**: endpoint-DBSCAN eps 10→5 cm.
   - *Intuition*: at eps 10 the vertex merge fused parallel tracks whose
     endpoints were clearly separable; over-merging is far more costly than
     over-fragmentation at this stage (merging can happen later in matching).
   - *Bad example removed*: f7 ev9 tpc60 stage-4 fusing of separate global
     tracks (stages 1-3 were correct).
3. **Leftover-track peel**: RANSAC-extract clean straight tracks (≥15 hits,
   ≥25 cm, linearity ≥0.92, transverse RMS ≤3.5 cm) out of leftover-DBSCAN
   clusters (vectorised, subsample-scored; ~0.1-2 s/event).
   - *Intuition*: in crowded TPCs the per-TPC RANSAC misses some tracks (budget
     exhausted), and the final leftover-DBSCAN sweeps the misses into single
     blobs — including two PARALLEL missed tracks glued into one.
   - *Bad example removed*: f5 ev5 tpc31 — 11 truth tracks in one TPC, RANSAC
     found 8, two of the 3 missed tracks became one 294 MeV blob (50% pure).
     Peel splits them; TPC foreign-energy 7.8% → 0.6%.
4. **Min-cluster-size floor**: final clusters with <3 hits are demoted to
   unclustered (-1), matching the leftover-DBSCAN `min_samples=3` convention.
   - *Intuition*: the rebuilt Stage-3 relabel strands the old pipeline's
     noise-absorption halo hits (attached to tracks but not part of any RANSAC
     segment) as their own 1-2-hit "clusters" (~40/event) — the old pipeline
     structurally never produced sub-3-hit clusters, and these carry ~0.1 MeV
     each (below any matching threshold).
   - *Bad example removed*: the spurious 0.1 MeV bin in the energy-per-cluster
     figure (cluster_stats_v055.png) — ~880 one-hit + ~600 two-hit clusters
     over 36 events, all gone; purity metrics unchanged.

**Metrics (36-event benchmark)**
| | interaction wrong-E | time-purity (hit-weighted) | clusters/event |
|---|---|---|---|
| v0.4 front-end (m5p1) | 1.21% | 98.55% | 1555 |
| v0.5.5 `m5p1all` | 0.61-0.63% | 99.13% | 1688 |

Runtime: the three upgrades add 0.9-3.3 s/event single-core on top of the
unchanged base toolbox (12-38 s; the toolbox's Python RANSAC + union-find is
the future speed target, not the add-ons).

**Matching-level benchmark (2026-07-20, canonical v0.4 chain UNCHANGED, 39
events files 1-3; SpineFragCompare/v040/v055_table.txt)**: energy eff —
baseline 95.00%, m5p1all 94.48%, stitched SPICE 94.15% (SPINE particles ref
95.43%). The front-end purity gain does NOT yet convert to matching gain.
CORRECTED diagnosis (2026-07-21; the pipeline's ns_total_eff row misleadingly
showed non-shower parity): an independent paired per-(event,TPC) analysis
splits the -0.52 pp deficit ~50/50 — shower-heavy TPCs (15% of E, -1.84 pp
locally) contribute -0.28 pp and non-shower TPCs (-0.28 pp locally) contribute
-0.24 pp. The mechanism is uniform: object SIZE / per-object light leverage,
not purity. Lost energy sits in clean straight backbone tracks that the old
front-end kept inside bigger multi-TPC objects (4-6 TPCs vs our 1-2; stricter
stitch + eps5 vertex), and in peeled tracks whose old host was a 1-2 GeV
leftover blob; each smaller object fits t0 alone and occasionally mis-fits by
100-560 ticks. Backbone E share rose 70.3→77.3% — the anchors exist. Open
opportunity for v0.6: pool light across split-but-collinear objects at SCAN
time while keeping pure labels at ASSIGNMENT time ("scan families"), plus
re-tuned p2/p3 knobs (current ones tuned for the old front-end).

**Assignment-phase tuning (2026-07-21, clustering untouched)**: adopted config
"A5" — peeled tracks keep their pure labels but are placed BELOW split_index
(env V055_PEEL_NONBACKBONE=1 in upgraded_frontend.py) so they inherit t0 via
rescue/expansion instead of anchoring weak solo scans; run with --early-expand.
- *Intuition*: the peel's clean small tracks were promoted to the backbone
  tier, where each anchors its own t0 scan with little light — diluting the
  anchor tier (backbone eff 97.52%). Demoting them restores baseline-like
  anchors without giving up label purity.
- *Nulls (recorded honestly)*: --early-expand alone, emax 50, widened families
  (FAM_DT 4 / TAU 1.5 / D_GEO 30 via env hooks in v030_expansion.py),
  swap-min-energy 8 — no gain on the 13-event arms.
- Full 39-event certification: energy eff 94.48→94.59%, backbone eff 98.20%
  (≈ baseline 98.26%), wall time 370→324 s/event. Remaining −0.41 pp vs the
  v0.4 front-end; next levers are code hooks (batch-band vertex-mate override
  at v02_ordering_pipeline.py:143, force-coverage tiebreak bias for <10 MeV at
  v030_expansion.py:537-556).

**Known residuals (accepted)**: ~108 TPCs with >30 MeV impure clusters remain —
mostly overlapping EM showers / track-in-shower (genuinely ambiguous at the
front end, deferred to matching; e.g. f7 ev2 tpc31, f5 ev4 tpc34, f6 ev2
tpc29). One upgrade-caused regression: f5 ev8 tpc65 (85%→52%) — restitch chains
a foreign collinear segment across TPCs with a purely longitudinal gap, which
the transverse gate cannot see.

### Front-end B (alternative): `spineS` — SPICE-stitched SPINE fragments
`FragVsBaseClusters/save_spice_labels.py` + `spice_stitch.py`; matching runs it
via `SpineFragCompare/v040/spine_frontend_v040.py --spine-variant spice`.
Collinear chaining for track fragments + per-TPC density-peak (Rodriguez-Laio)
aggregation for shower fragments, on the raw Graph-SPICE fragments.
- *Intuition*: raw SPINE fragments are extremely pure (99.95%) but over-split
  (2722/event); aggregate them down to a matchable count while giving back as
  little purity as possible.
- Result: **2722 → 1297 clusters/event at 99.40% time-purity**.
- Labels recorded once at `FragVsBaseClusters/output/f*_ev*_spineS.npz`
  (SPINE GPU inference cached in `SpineFragCompare/spine_rerun/`).

---

## v0.4.0 (baseline, pre-2026-07)
The real production algorithm this log starts from:
`CLMatching_v0.1_backboneOnly/testing/overflow_rescue_pipeline.py` with the
M5p1 toolbox front-end (`global_track_clustering_toolbox_v11_2.py`) — per-TPC
RANSAC lines → DBSCAN rescue → cross-TPC stitch (25 cm, no directional gate) →
vertex merge (eps 10) → leftover DBSCAN. See `SpineFragCompare/v040/V040_ALGORITHM.md`.
