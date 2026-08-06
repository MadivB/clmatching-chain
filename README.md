# CLMatching — DUNE ND-LAr pre-reconstruction charge-light matching (v0.6)

The matching chain: places charge clusters at light-flash times (t0) before
interaction-level reconstruction. **v0.6** = external front-end clustering
([clmatching-frontend](https://github.com/MadivB/clmatching-frontend)) +
the v0.4.1 matching chain — NS total-efficiency **96.05% at 100% coverage**
(126 MiniProdN5p1 events, energy-weighted, TPCs <= 1.4 GeV, |reco-truth| <= 10
ticks).

- `V060.md` — v0.6 definition, canonical flags, validation, update workflow
- `V040_ALGORITHM.md` — full stage-by-stage algorithm (stages 1 / 2 / 3.x / 4.x)
- `V070_DESIGN.md` — next: joint probabilistic small-cluster assignment
- `testing/overflow_rescue_pipeline.py` — the harness (runs the whole chain)
- `testing/frontend_adapter.py` — seam to the separate clustering repo
- `testing/stage_report.py` — per-stage performance table
- `testing/v040_step_by_step.ipynb` — step-by-step failure-debug notebook
- `testing/tdr_plots_v040.py` — TDR-standard figures
- `CLMatching_AlphaRelease/` — first-stage machinery (clustering wrappers,
  perceiver prediction, track stage, V2, v11 matrix)

Per-run outputs, dumps, and display galleries are not committed (regenerable;
they live on NERSC). Model checkpoints are resolved from NERSC paths at run
time. Frozen stable copies on NERSC: `CLMatching_v0.6` (this release),
`CLMatching_v0.4.1`, `CLMatching_v0.2`.
