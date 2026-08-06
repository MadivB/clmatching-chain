# CLMatching v0.2 snapshot (frozen) — 2026-07-07

Frozen copy of the v0.2 development state (`CLMatching_v0.2/` is the archive;
`CLMatching_v0.1_backboneOnly/` continues as the working tree for the next
changes). All numbers: 39 MiniProdN5p1 events (FLOW files 0000001-3),
energy-weighted hit accuracy, |reco-truth| <= 10 ticks.

## Headline result

| stage | v0.1 | this snapshot |
|---|---|---|
| backbone (phase 1) | 95.1% | 97.7% |
| final (post Phase2 + V2) | 95.9% | **97.25%** |

Clusters >= 50 MeV: 97.4% at 100% assignment coverage. Quiet TPCs
(< 300 MeV non-track energy < 50 MeV band): > 99.7%.

## What this version contains

1. **v0.2 ordered placement** (`testing/v02_ordering_pipeline.py`): showers ->
   >=3-TPC tracks -> energy-banded permutation batches for the 1-2-TPC pool
   (s = 0.60, batch cap 5; commit N-1, last returns to pool).
2. **stack2 refinements** (`testing/v02_refine_pipeline.py`): soft cosine
   matched filter in every scan (hard gate hurts), defer-once overflow commit
   gate, pairwise swap pass, trigger rescue pass.
3. **Overflow trigger + saving algorithm** (`testing/overflow_rescue_pipeline.py`,
   `--backbone stack2`): locked trigger (>=3 ch excess>=500 z_pois>2 OR
   cos<0.75); Stage A anchored permutations, Stage B per-TPC part detach with
   NEW cluster ids, N=1 unanchored rescan/detach.
4. **Placement police** (`--police locked|top18`): every Phase2/V2 placement or
   move is probed at commit time; on alarm resolve immediately by
   keep/rewind/rescan (rescan always won over rewind). P2-placed cohort
   92.5 -> 94.0.
5. **Saturation veto** tightened option `--max-clip-ticks 4` (helps phase-1
   ~+0.25pp; washes out post-V2 — kept optional).
6. **Displays & analysis** (`testing/v02_displays*`, `v02_failures*`):
   4-plot event displays incl. the wrong-hits 3D view (also added to the
   cl-event-display skill / QLMLOnly viz script); TPC busyness analysis
   (`tpc_energy_vs_wrong.png`, `tpc_nontrack_vs_wrong.png`,
   `tpc_energy_hist_linear.png`).
7. **Presentation**: `testing/v02_presentation/CLMatching v0.2 Update.pptx`
   (SlideKit format) + HTML deck.

## Where it still fails (the motivation for what comes next)

Mistakes are activity-driven: no TPC below ~300 MeV total carries a meaningful
mistake; accuracy falls to ~95% in multi-GeV TPCs. Recurring failure mode:
a coherent multi-prong structure choosing between two real flashes; above
~2 GeV, shower-dominated walls. Remaining wrong energy ~2.7% of assigned.

**Next development (post-snapshot): a dedicated path that recognises
shower-dominated / busy TPCs and handles them separately.** Non-track energy
>= 200 MeV flags 36% of TPC-events covering 86% of the wrong energy
(comparable to a total-energy gate).

## Key output directories

- `testing/v02rescue_out`, `v02police_{l,t}_out` — staged accuracy runs
- `testing/v02dump2_out/dumps` — per-event final states (policed chain)
- `testing/v02_displays`, `v02_failures` — galleries
- scorers: `rescue_pipe_score.py`, `v02_score.py`, `v02s_score.py`,
  `v2tune_score.py`, `police_score.py`
