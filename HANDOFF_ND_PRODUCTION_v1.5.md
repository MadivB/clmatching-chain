# CLMatching v1.5 — handoff for ND production

Written 2026-10-05 for the ND production agent. Everything below is on the NERSC Perlmutter filesystems
(`/global/cfs`, `/global/homes`, `/pscratch`) under the `dune` project; nothing has been pushed to GitHub.

## 1. What this is

The ND-LAr charge–light matching chain, version 1.5: per calibrated prompt hit it assigns an interaction time t0
(matching ticks of 16 ns, sub-tick resolution), a chain cluster id, a 0–1 confidence and a per-hit timing sigma.
v1.5 = v1.4 (certified 2026-10-01) + two Michel / delayed-daughter re-timing stages (4.85, 4.86) + the speed
campaign of 2026-10-05 (assignment-neutral). Certified on flow files 1–5 (63 spills) against truth; a 10-file run
(files 1–10) is in progress at `/pscratch/sd/y/yuxuan/v15_10files_b` and its report will be appended.

| metric (63 spills, MiniProdN5p1 full sanddrift) | v1.5 |
|---|---|
| NS efficiency (non-shower TPCs, energy-weighted, hit within 10 ticks) | 96.93 % |
| all-TPC accuracy | 95.90 % |
| per-hit timing residual, robust sigma, assigned hits | 2.79 ns |
| Michel-electron energy correctly timed (direct truth metric) | 38.6 % |
| wall time per spill, two worker processes per 40 GB A100 | 122 s median on the MC certification clock, ~110 s without the MC-truth scoring load (p90 160 / 150) |

Ledger provenance: `testing/speed_cert_V15S2d.log` (ledger + stage timing), `testing/v14m_cert_V14M3.log`
(Michel stages), `VERSION_HISTORY.md` (every stage, every certification, including the ones not adopted).

## 2. Where the code is

| component | path | state |
|---|---|---|
| chain repository (harness, late stages, launchers, docs) | `/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly` | git `master`, tag `CLMatching_v1.5`, head `8f3ab4e`+ (remote `https://github.com/MadivB/clmatching-chain.git`, not pushed) |
| alpha-release core (first stage, matrix, light model wrappers) | `.../CLMatching_v0.1_backboneOnly/CLMatching_AlphaRelease` (git submodule) | commit `9afb397` (remote `https://github.com/MadivB/CLMatching_AlphaRelease.git`, not pushed) |
| frontend clustering (external, user's repo) | `/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/clmatching-frontend` | commit `0e398ee`; parameters `params_full.json` |
| light-prediction model checkpoint | `/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/NewMLSection/runs/ndfull_small4x/best_model_arch.pt` (41 MB) | fixed |
| pulse template | `CLMatching_AlphaRelease/assets/avg_pulse.npy` | fixed |
| precision-timing calibration | `testing/v120_calib_lowi.json` | fixed |
| TPC geometry yaml | `CLMatching_AlphaRelease/NewMLSection` (`DEFAULT_TPC_YAML`) | fixed |

To take a frozen copy: `git clone` the chain repository with `--recurse-submodules` from the path above (or copy the
directory), plus the frontend directory, plus the checkpoint file. All three are read-only inputs at run time.

## 3. Environment

```bash
PY=/global/common/software/nersc9/pytorch/2.8.0/bin/python        # NERSC module pytorch/2.8.0: torch 2.8, numpy, scipy, h5py, sklearn
export PYTHONUSERBASE=/global/homes/y/yuxuan/.local/perlmutter/pytorch2.8.0   # user site: plotly/kaleido/awkward/uproot (figures only; not needed to run the chain)
unset PYTHONPATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 V200_DOM=3
```
GPU: one NVIDIA A100 40 GB per two worker processes (the perceiver image batch is 32; a third worker per GPU or a
larger batch runs out of memory). CPU: one thread per worker is enough (the chain is single-threaded numpy/torch).

## 4. How to run

Launcher (all v1.5 flags baked in): `run_v15.sh`.
```bash
cd /global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly
bash run_v15.sh "<FLOW.hdf5> [<FLOW.hdf5> ...]" <out_dir> [n_workers=8] [n_gpus=4]
# e.g. on one 4-GPU node:  bash run_v15.sh "$F1 $F2 ..." /pscratch/.../v15_out 8 4
```
`run_v15_dumps.sh` is the same with per-event dumps (`--dump-dir`, needed to build the .pt files, ~25 MB per spill).
Workers split the events of the given files by stride (`--event-stride n_workers --event-offset w`); each writes
`<out_dir>/groups_worker<w>.jsonl` (one JSON row per spill: per-stage timings, counts, stage diagnostics) and
`<out_dir>/worker<w>.log`. The launcher prints `ALL_CONFIGS_DONE` when every worker has returned.

The harness itself is `testing/overflow_rescue_pipeline.py`; the flag set is the `V15=` string in `run_v15.sh`.
Speed/behaviour switches (defaults are the certified ones): `CLM_FFT_SCAN=0` restores the old loop scan of the
matrix stage; `CLM_IMAGE_BATCH=<n>` overrides the perceiver batch (32); `CLM_PREC_CACHE=0` disables the precision
geometry cache; `--mnd-set k=v` / `--mfl-set k=v` override any Michel-stage parameter.

Input: ndlar-flow `FLOW.hdf5` files with `charge/calib_prompt_hits`, `charge/events` (+ refs), `light/wvfm`,
`geometry_info`. **MC truth is optional**: when `mc_truth` is absent (data) every truth-based metric is NaN and the
assignment is unchanged (verified 2026-10-05 on one spill with the truth deliberately ignored,
`/pscratch/sd/y/yuxuan/v11arms/NOTRUTH`); `CLM_IGNORE_TRUTH=1` forces that path on MC.

## 5. Output for production: one .pt per flow file

```bash
$PY testing/make_pt_v15.py <out_dir_with_full_dumps> <pt_dir>      # -> <pt_dir>/<flow basename>.v1_5.pt
```
`torch.load` gives a dict (prompt-hit arrays indexed like `charge/calib_prompt_hits/data`, length `n_calib_hits`):

| key | dtype | meaning |
|---|---|---|
| `calib_hit_t0_reco` | float32 | t0 in matching ticks (16 ns), sub-tick; `-1` = unassigned |
| `calib_hit_t0_reco_ns` | float32 | the same in ns; `-1` = unassigned |
| `prompt_hit_t_cluster_id` | int32 | chain cluster id, `-1` = noise / unclustered hit |
| `calib_hit_t0_conf` | float32 | per-hit confidence 0–1 (pcu_vote10 / conf93), NaN = no estimate; conf > 0.8 keeps 88 % of the energy at 98.7 % purity |
| `calib_hit_t0_sigma` | float32 | per-hit timing sigma in ticks (precision stage), NaN = none |
| `calib_hit_t0_sigma_ns` | float32 | the same in ns |
| `version`, `algorithm`, `input_file`, `processed_event_ids`, `n_calib_hits`, `n_assigned`, `n_unassigned`, `tick_ns` | | provenance |

This is the v1.0 production schema (`testing/make_pt_v1.py`) plus the ns copies and the precision sigma.

## 6. What the production agent should know

- Time per spill scales with activity (p90 is 1.3x the median); 8 workers on a 4-GPU node process 63 spills in ~35 min.
- The chain is deterministic given the inputs; the only randomness is the frontend RANSAC, which uses a fixed RNG stream.
- Memory: ~6 GB GPU per worker during prediction, up to ~18 GB transiently; ~10 GB host RAM per worker.
- Known limits (physics, documented in `V130_PHYSICS_CAPABILITIES.md`): isolated 1–20 MeV deposits are the weak point
  (interaction-level recovery 32–42 %); delayed Michels are 17–21 % correctly timed (pile-up), prompt ones 95 %.
- Documents: `VERSION_HISTORY.md` (living record), `V130_CHAIN_REPORT.md` (method + ledger + timing),
  `V130_PHYSICS_CAPABILITIES.md` (what the output is good for), `V040_ALGORITHM.md` (stage numbering).
