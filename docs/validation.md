# Validation

freeCAM's claim is that Python owning the workflow changes nothing about the
numbers. That claim is tested one way: the model is run under Python control
and its output compared with the original Fortran model's, byte for byte, on
the full PI-atm configuration -- 512 ranks, or, for a build with other
compile-time options, against that build's own original. This page says what
that comparison covers, which gates exist, and where their evidence lives.

## What bit-for-bit means here

The comparator, `freecam.pi_cam.validation.compare_pi_cam_directories`
(driven by [`tools/verify_pi_cam.py`](../tools/verify_pi_cam.py)), compares
two CAM run directories. It requires identical CAM file inventories,
identical numerical variable inventories, identical dtypes and shapes, and
exactly equal array values in every history and restart file. There is no
numerical tolerance. It does not compare NetCDF compression bytes, path
strings, or non-numerical metadata.

```bash
uv run python tools/verify_pi_cam.py --reference <original run> --candidate <freeCAM run>
```

For the online configuration the coupler boundary is checked as well: every
x2a the provider imports and every a2x CAM exports is compared with the
original run's, step by step.

## The gates

Every gate runs under PBS on Derecho, at 512 ranks unless its row says
otherwise, through
[`validation/jobs/submit.sh`](../validation/jobs/submit.sh), and leaves a
machine-readable record under [`validation/`](../validation/) with the job
id, the source and library hashes, the comparison result, and the first
difference if there was one.

| Gate | What it proves | Record |
| --- | --- | --- |
| Python-controlled CAM, 50 steps, replayed boundary | the control layer, the zero-copy state and the adapters | [`pi_cam_python_zero_copy_state_50step.json`](../validation/pi_cam_python_zero_copy_state_50step.json) |
| Online CESM components and coupler, 50 steps | the live surface components, the coupler and every boundary exchange | [`pi_cam_exact_cesm_online_50step.json`](../validation/pi_cam_exact_cesm_online_50step.json), [`..._bfb.json`](../validation/pi_cam_exact_cesm_online_50step_bfb.json) |
| Online, one model year | the same over every history and restart file of a year | [`pi_cam_exact_cesm_online_1year_bfb.json`](../validation/pi_cam_exact_cesm_online_1year_bfb.json) |
| Online, five model years | the same over five years | [`pi_cam_exact_cesm_online_5year_bfb.json`](../validation/pi_cam_exact_cesm_online_5year_bfb.json) |
| Monthly output against an independent production run, one and five years | the whole lifecycle, against a twenty-year CESM integration this project did not produce | [`pi_cam_monthly_1year_bfb.json`](../validation/pi_cam_monthly_1year_bfb.json), [`pi_cam_monthly_5year_bfb.json`](../validation/pi_cam_monthly_5year_bfb.json) |
| Python-owned fields in CAM history output | a field created from Python reaches the history file; `output=False` reaches none | [`pi_cam_python_history_output_12step.json`](../validation/pi_cam_python_history_output_12step.json) |
| A CAM stage as a Python class, 50 steps, a month, a year, five years | the stage's Fortran run under Python control, whole or paused at a replaced kernel | [`pi_cam_stage7_segmented_original_vs_oracle_50step_bfb.json`](../validation/pi_cam_stage7_segmented_original_vs_oracle_50step_bfb.json), [`pi_cam_python_memory_1year_stage_python_bfb.json`](../validation/pi_cam_python_memory_1year_stage_python_bfb.json), [`pi_cam_python_memory_5year_stage_python_bfb.json`](../validation/pi_cam_python_memory_5year_stage_python_bfb.json) |
| The stage-7 runner paused at `micro_mg_tend`, 50 steps | the microphysics driver run in its verbatim pieces around the substep loop, the original core answered through Python at every pause (100 pauses over 50 steps), bit-for-bit; then both kernels paused in the same run (200 pauses), bit-for-bit | [`pi_cam_stage7_segmented_micro_50step.json`](../validation/pi_cam_stage7_segmented_micro_50step.json), [`..._vs_oracle_50step_bfb.json`](../validation/pi_cam_stage7_segmented_micro_vs_oracle_50step_bfb.json), [`pi_cam_stage7_segmented_both_50step.json`](../validation/pi_cam_stage7_segmented_both_50step.json), [`..._vs_oracle_50step_bfb.json`](../validation/pi_cam_stage7_segmented_both_vs_oracle_50step_bfb.json) |
| Pausable classes for dry adjustment and shallow convection, 50 steps | each class installed with nothing replaced; `dadadj` and `compute_uwshcu_inv` answered by the original through the pause, alone (100 pauses each) and together; the eleven inert actions disabled at once | [`pi_cam_pausable_dadadj-whole_vs_oracle_50step_bfb.json`](../validation/pi_cam_pausable_dadadj-whole_vs_oracle_50step_bfb.json), [`pi_cam_pausable_shcu-whole_vs_oracle_50step_bfb.json`](../validation/pi_cam_pausable_shcu-whole_vs_oracle_50step_bfb.json), [`pi_cam_pausable_dadadj-pause_vs_oracle_50step_bfb.json`](../validation/pi_cam_pausable_dadadj-pause_vs_oracle_50step_bfb.json), [`pi_cam_pausable_shcu-pause_vs_oracle_50step_bfb.json`](../validation/pi_cam_pausable_shcu-pause_vs_oracle_50step_bfb.json), [`pi_cam_pausable_both-pause_vs_oracle_50step_bfb.json`](../validation/pi_cam_pausable_both-pause_vs_oracle_50step_bfb.json), [`pi_cam_pausable_inert_vs_oracle_50step_bfb.json`](../validation/pi_cam_pausable_inert_vs_oracle_50step_bfb.json) |
| The split radiation class over the radt runner, 50 steps | the class installed with nothing replaced (the resume half runs the driver); `rad_rrtmg_sw` and `rad_rrtmg_lw` answered by the original through the pause, alone (50 pauses each) and together; then dry adjustment, shallow convection and radiation all paused in one run | [`pi_cam_pausable_rad-whole_vs_oracle_50step_bfb.json`](../validation/pi_cam_pausable_rad-whole_vs_oracle_50step_bfb.json), [`pi_cam_pausable_rad-sw-pause_vs_oracle_50step_bfb.json`](../validation/pi_cam_pausable_rad-sw-pause_vs_oracle_50step_bfb.json), [`pi_cam_pausable_rad-lw-pause_vs_oracle_50step_bfb.json`](../validation/pi_cam_pausable_rad-lw-pause_vs_oracle_50step_bfb.json), [`pi_cam_pausable_rad-both-pause_vs_oracle_50step_bfb.json`](../validation/pi_cam_pausable_rad-both-pause_vs_oracle_50step_bfb.json), [`pi_cam_pausable_all-pause_vs_oracle_50step_bfb.json`](../validation/pi_cam_pausable_all-pause_vs_oracle_50step_bfb.json) |
| Deep convection, the tracer transport leaf, vertical diffusion and gravity wave drag, 50 steps | each class installed whole; `zm_convr`, `zm_conv_evap`, `momtran`, `convtran`, `compute_tms`, `compute_eddy_diff`, `compute_vdiff` (both sites, 200 pauses) and `gw_drag_prof` answered by the original through the pause, alone and together; deep convection and the leaf paused in one run (the shared module state); then every class installed with all fourteen kernels paused at once | [`pi_cam_pausable_deep-all_vs_oracle_50step_bfb.json`](../validation/pi_cam_pausable_deep-all_vs_oracle_50step_bfb.json), [`pi_cam_pausable_chain-pause_vs_oracle_50step_bfb.json`](../validation/pi_cam_pausable_chain-pause_vs_oracle_50step_bfb.json), [`pi_cam_pausable_vdiff-all_vs_oracle_50step_bfb.json`](../validation/pi_cam_pausable_vdiff-all_vs_oracle_50step_bfb.json), [`pi_cam_pausable_gwd-pause_vs_oracle_50step_bfb.json`](../validation/pi_cam_pausable_gwd-pause_vs_oracle_50step_bfb.json), [`pi_cam_pausable_everything_vs_oracle_50step_bfb.json`](../validation/pi_cam_pausable_everything_vs_oracle_50step_bfb.json), and the single-kernel records beside them |
| The wet deposition, dry deposition and chemistry leaves, and everything at once, 50 steps | each leaf class installed whole; `wetdepa_v2` (both sites, 3000 pauses), `modal_aero_depvel_part` (four sites, 800 pauses) and `gas_phase_chemdr` (100 pauses) answered by the original through the pause, bit-for-bit; the run with every kernel paused at once is in the next row | [`pi_cam_pausable_awet-pause_vs_oracle_50step_bfb.json`](../validation/pi_cam_pausable_awet-pause_vs_oracle_50step_bfb.json), [`pi_cam_pausable_adry-pause_vs_oracle_50step_bfb.json`](../validation/pi_cam_pausable_adry-pause_vs_oracle_50step_bfb.json), [`pi_cam_pausable_chem-pause_vs_oracle_50step_bfb.json`](../validation/pi_cam_pausable_chem-pause_vs_oracle_50step_bfb.json) |
| The kernel-API closure's first entries, 50 steps | `virtem`, an elemental function inside an assignment of the vertical diffusion driver, answered by the original through the pause (100 pauses); `zm_conv_evap` paused through its new three-way dispatch (100 pauses); every class installed with all eighteen kernels paused at once (5400 pauses), bit-for-bit. The pause at `cldfrc_fice` inside the hoisted copy of `zm_conv_evap` (job 7343258) was not bit-for-bit; its first difference and diagnosis are kept in [`pi_cam_pausable_fice-pause_50step_failure.json`](../validation/pi_cam_pausable_fice-pause_50step_failure.json). `virtem` answered by the original at its pause with every call's frame recorded (7343396, bit-for-bit), and all 51200 frames replayed through the standalone function, 691300 elements bit-for-bit. Hooks (link-time redirection of one call inside a compiled kernel, the stage running on a fiber): the image with both hooks linked and nothing replaced (7343707; 53248 and 38152530 calls counted, none paused), `cldfrc_fice` answered by the original at its hook inside `zm_conv_evap` (7343708, 51200 pauses), `fluxbelowinv` at its hook inside `compute_uwshcu` (7343709, 36733580 pauses), both hooks with every class installed (7343813), and `cldfrc_fice` captured at its hook (7343811, 51200 frames) with every frame replayed bit-for-bit through the standalone function, `fluxbelowinv` captured at its hook (7343922, 36733580 whole-profile frames, a 200 GB per node memory request after 7343812 hit the job's 256 GB limit and 7343844 recorded empty profiles), all bit-for-bit. The first replay of those frames (7344048) was not bit-for-bit: the standalone image lacked uwshcu's module constant `g`, which the contract had not declared; kept in [`pi_cam_fluxbelowinv_frame_replay_no_module_state_failure.json`](../validation/pi_cam_fluxbelowinv_frame_replay_no_module_state_failure.json), with the model's value snapshotted in [`pi_cam_fluxbelowinv_module_state.json`](../validation/pi_cam_fluxbelowinv_module_state.json); with it, all 36733580 frames replayed bit-for-bit (7344823, 32 shards on one node); the first `cldfrc_fice` hook gate (7343594) failed on a duplicate global symbol and is kept in [`pi_cam_pausable_fice-hook_50step_failure.json`](../validation/pi_cam_pausable_fice-hook_50step_failure.json) | [`pi_cam_pausable_virtem-pause_vs_oracle_50step_bfb.json`](../validation/pi_cam_pausable_virtem-pause_vs_oracle_50step_bfb.json), [`pi_cam_pausable_deep-evap_vs_oracle_50step_bfb.json`](../validation/pi_cam_pausable_deep-evap_vs_oracle_50step_bfb.json), [`pi_cam_pausable_everything_vs_oracle_50step_bfb.json`](../validation/pi_cam_pausable_everything_vs_oracle_50step_bfb.json), [`pi_cam_pausable_virtem-capture_vs_oracle_50step_bfb.json`](../validation/pi_cam_pausable_virtem-capture_vs_oracle_50step_bfb.json), [`pi_cam_virtem_frame_replay.json`](../validation/pi_cam_virtem_frame_replay.json), [`pi_cam_hooks_unarmed_vs_oracle_50step_bfb.json`](../validation/pi_cam_hooks_unarmed_vs_oracle_50step_bfb.json), [`pi_cam_pausable_fice-hook_vs_oracle_50step_bfb.json`](../validation/pi_cam_pausable_fice-hook_vs_oracle_50step_bfb.json), [`pi_cam_pausable_flux-hook_vs_oracle_50step_bfb.json`](../validation/pi_cam_pausable_flux-hook_vs_oracle_50step_bfb.json), [`pi_cam_pausable_hooks-everything_vs_oracle_50step_bfb.json`](../validation/pi_cam_pausable_hooks-everything_vs_oracle_50step_bfb.json), [`pi_cam_pausable_fice-capture_vs_oracle_50step_bfb.json`](../validation/pi_cam_pausable_fice-capture_vs_oracle_50step_bfb.json), [`pi_cam_cldfrc_fice_frame_replay.json`](../validation/pi_cam_cldfrc_fice_frame_replay.json), [`pi_cam_pausable_flux-capture_vs_oracle_50step_bfb.json`](../validation/pi_cam_pausable_flux-capture_vs_oracle_50step_bfb.json), [`pi_cam_fluxbelowinv_frame_replay.json`](../validation/pi_cam_fluxbelowinv_frame_replay.json) |
| Every class installed, nothing replaced, paired against the original Fortran | months at C/A 0.996 and 0.986, a year at 1.0095, all bit-for-bit; the earlier pairs at 1.05-1.09 measured the export fast path taking the gathering branch every step, fixed in the same record's later pairs | [`pi_cam_faster_than_fortran.json`](../validation/pi_cam_faster_than_fortran.json) |
| A scheme as a function | the standalone image calls the original routine; captured calls replayed through it reproduce the model | `validation/pi_cam_<scheme>_full_chunk_vs_capture.json`, `..._single_column_vs_capture.json`, `..._public_api_vs_capture.json` |
| The Workflow Builder's generated configuration through `freecam.Driver`, 50 steps | the page's path runs the validated default unchanged, and reaches an inserted Python process every step, bit-for-bit | [`pi_cam_workflow_builder_50step.json`](../validation/pi_cam_workflow_builder_50step.json) |
| The kernel decoupling inventory | every action of the step classified once, each exposed kernel followed through contract, capture, replay, in-model replacement and performance; built from the records above and checked current by the unit suite | [`physics_kernel_decoupling.json`](../validation/physics_kernel_decoupling.json) |
| The kernel-API closure inventory | every procedure the configured physics step can reach from `phys_run1` and `phys_run2` in the pinned source, resolved through Fortran scoping with the build's real macros, each call site with its preprocessor condition and namelist guards, classified by reviewed rules; the inventory phase of opening every kernel, checked current by the unit suite | [`pi_cam_kernel_api_closure.json`](../validation/pi_cam_kernel_api_closure.json), [`docs/kernel_api_closure.md`](kernel_api_closure.md) |
| The native image rebuilds | the build pipeline reproduces the image in use, command by command and symbol by symbol | [`pi_cam_native_image_rebuild.json`](../validation/pi_cam_native_image_rebuild.json) |
| The CESM source and cases rebuilt from the repository | the source trees prepared from the pinned submodule with the recipe's patches equal the hand-made ones apart from their recorded, explained differences; the three cases made by `tools/build_pi_cam_cases.py` carry all 346 configured values of the hand-made cases; the rebuilt oracle's and pyCESM case's 50 steps are bit-for-bit with theirs (7665609, 7671374); the image built from the rebuilt oracle passes the stage gates; and the online month on the rebuilt cases, image and coupler library is bit-for-bit, the original's executable (A) and freeCAM (C) each against the original month (7671412) | [`pi_cam_cesm_source_oracle_case.json`](../validation/pi_cam_cesm_source_oracle_case.json), [`pi_cam_cesm_source_state_case.json`](../validation/pi_cam_cesm_source_state_case.json), [`pi_cam_cesm_source_pycesm_case.json`](../validation/pi_cam_cesm_source_pycesm_case.json), [`pi_cam_cesm_source_provider.json`](../validation/pi_cam_cesm_source_provider.json), [`pi_cam_cases_rebuilt_oracle_50step_bfb.json`](../validation/pi_cam_cases_rebuilt_oracle_50step_bfb.json), [`pi_cam_pausable_b1-rebuilt-whole_vs_oracle_50step_bfb.json`](../validation/pi_cam_pausable_b1-rebuilt-whole_vs_oracle_50step_bfb.json), [`pi_cam_pausable_b1-rebuilt-batch-original_vs_oracle_50step_bfb.json`](../validation/pi_cam_pausable_b1-rebuilt-batch-original_vs_oracle_50step_bfb.json), [`pi_cam_cases_rebuilt_online_month_bfb.json`](../validation/pi_cam_cases_rebuilt_online_month_bfb.json), [`pi_cam_online_model_1month_b1-rebuilt-AC.json`](../validation/pi_cam_online_model_1month_b1-rebuilt-AC.json) |
| A build with other compile-time options: `fc.build(pcols=32)`, 50 steps and a month | the build's own CESM cases (CAM configured with `-pcols 32`), its own oracle, the image from that oracle's objects, the coupler library linked to it, and the exact online 50 steps, bit-for-bit against the build's own oracle (7679701-7679706); the online month, freeCAM bit-for-bit with the build's own original in CAM's output and in CLM's, CICE's and the coupler's (7681915). A build is gated against its own original: the pcols=32 and pcols=16 originals are different runs, which part after about six days | [`pi_cam_build_pcols32_online_50step_bfb.json`](../validation/pi_cam_build_pcols32_online_50step_bfb.json), [`pi_cam_build_pcols32_online_month_bfb.json`](../validation/pi_cam_build_pcols32_online_month_bfb.json), [`pi_cam_online_model_1month_pcols32-AC.json`](../validation/pi_cam_online_model_1month_pcols32-AC.json) |
| Another rank count, online, 48 steps | at 300 ranks, with CESM's own uneven layout, freeCAM is bit-for-bit with the original CESM run at the same layout (7649000); the image with the uneven-layout fix stays bit-for-bit at 512. Runs at 128 to 400 ranks are timing studies, compared with the 512-rank oracle and so not bit-for-bit by layout | [`pi_cam_online_model_48step_300rank_uneven-ac_freecam_vs_cesm_bfb.json`](../validation/pi_cam_online_model_48step_300rank_uneven-ac_freecam_vs_cesm_bfb.json), [`pi_cam_pausable_uneven-fix-whole-p39_vs_oracle_50step_bfb.json`](../validation/pi_cam_pausable_uneven-fix-whole-p39_vs_oracle_50step_bfb.json) |
| A rank's chunks answered in one call, and the image grid, 50 steps | the hook batch with nothing bound and with the original answering each gathered chunk (p39); the runner's chunk batch sequential, per chunk and stacked (p38); the image with its column grid (p40); all bit-for-bit. The batch check with the trained model answering is not, by design | [`pi_cam_pausable_hook-whole-p39_vs_oracle_50step_bfb.json`](../validation/pi_cam_pausable_hook-whole-p39_vs_oracle_50step_bfb.json), [`pi_cam_pausable_hook-batch-original-p39_vs_oracle_50step_bfb.json`](../validation/pi_cam_pausable_hook-batch-original-p39_vs_oracle_50step_bfb.json), [`pi_cam_pausable_batch-seq-shcu-p38_vs_oracle_50step_bfb.json`](../validation/pi_cam_pausable_batch-seq-shcu-p38_vs_oracle_50step_bfb.json), [`pi_cam_pausable_batch-shcu-original-p38_vs_oracle_50step_bfb.json`](../validation/pi_cam_pausable_batch-shcu-original-p38_vs_oracle_50step_bfb.json), [`pi_cam_pausable_batch-shcu-stacked-p38_vs_oracle_50step_bfb.json`](../validation/pi_cam_pausable_batch-shcu-stacked-p38_vs_oracle_50step_bfb.json), [`pi_cam_pausable_grid-whole-p40_vs_oracle_50step_bfb.json`](../validation/pi_cam_pausable_grid-whole-p40_vs_oracle_50step_bfb.json), [`pi_cam_pausable_grid-batch-original-p40_vs_oracle_50step_bfb.json`](../validation/pi_cam_pausable_grid-batch-original-p40_vs_oracle_50step_bfb.json) |
| Which kernels ran, counted in place, 50 steps and a month | the counting image's trampolines count every candidate kernel call and attribute it to its process, step and phase, with the run bit-for-bit (the month in 18 files); zero counts on uninstrumented paths are reported as unknown, never as absence | [`pi_cam_kcount_all_vs_oracle_50step_bfb.json`](../validation/pi_cam_kcount_all_vs_oracle_50step_bfb.json), [`pi_cam_kcount_all_vs_fortran_1month_bfb.json`](../validation/pi_cam_kcount_all_vs_fortran_1month_bfb.json), [`pi_cam_kernel_runtime_coverage_50step.json`](../validation/pi_cam_kernel_runtime_coverage_50step.json), [`pi_cam_kernel_runtime_coverage_1month.json`](../validation/pi_cam_kernel_runtime_coverage_1month.json), [`pi_cam_kernel_observability.json`](../validation/pi_cam_kernel_observability.json) |
| The per-rank action timeline, 50 steps | two paired runs with the timeline on and off, each bit-for-bit; the difference of the means (1.22%) is below the spread of the two runs with it off (1.75%) | [`pi_cam_timeline_overhead_50step.json`](../validation/pi_cam_timeline_overhead_50step.json) |
| The state recorder, online 50 steps | `T`, `Q`, `CLDLIQ` and `CLDICE` recorded every step and written every step, as the Workflow Builder's Globe tab records them, and after every action at step 24 (50 step and 49 action snapshots, gathered to rank 0 and written there, one file per kind); bit-for-bit (job 7712576) | [`pi_cam_exact_cesm_online_state-record-live_50step.json`](../validation/pi_cam_exact_cesm_online_state-record-live_50step.json), [`..._bfb.json`](../validation/pi_cam_exact_cesm_online_state-record-live_50step_bfb.json) |
| The state kept in memory, online 50 steps | `T`, `Q`, `CLDLIQ` and `CLDICE` of every step kept in every rank's memory, as the Workflow Builder's Globe tab keeps them (nothing written; 50 step snapshots); bit-for-bit (job 7712854). Advancing 50 steps took 19.80 s, against 19.63 and 19.66 s recording nothing and 21.85 s writing the same snapshots to files every step (jobs 7712853, 7712856, 7712855, run back to back) | [`pi_cam_exact_cesm_online_state-record-memory_50step.json`](../validation/pi_cam_exact_cesm_online_state-record-memory_50step.json), [`..._bfb.json`](../validation/pi_cam_exact_cesm_online_state-record-memory_50step_bfb.json) |

The two 50-step jobs are the gates every change to the numerical runtime has
to pass before it is merged:

```bash
validation/jobs/submit.sh validation/jobs/pi_cam_python_zero_copy_state_50step.pbs
validation/jobs/submit.sh validation/jobs/pi_cam_exact_cesm_online_50step.pbs
```

A wrapper or adapter that compiles is not validated; the gate has to show
that the intended routine executed and that its outputs match. Oracle output
is never overwritten, and a new configuration needs its own gate rather than
reusing the PI-atm evidence. `fc.build` runs that gate for a build with other
compile-time options: the build's own original is its oracle, and the build
counts as validated only when its online 50-step comparison is bit-for-bit
(the record is written under the build's root).

## Performance

The cost of the Python control layer, and of running a stage as a Python
class, is measured against the original Fortran model over months, a year and
five years, and recorded once, in
[`validation/performance_overhead.md`](../validation/performance_overhead.md).
That page explains the method (paired runs of the original executable and
freeCAM in one allocation, timed over the same coupling loop), lists every
run with its job id, and states the caveats. The paired measurements
themselves are in
[`validation/pi_cam_faster_than_fortran.json`](../validation/pi_cam_faster_than_fortran.json),
and a perf profile of where a step's time goes in
[`validation/pi_cam_perf_online_50step.json`](../validation/pi_cam_perf_online_50step.json).
Later timing records: what each kernel costs and what replacing it can return,
in [kernel_replacement_returns.md](kernel_replacement_returns.md); the online
months with a model in a kernel's place, on the host and on the GPUs
(`validation/pi_cam_online_model_1month_*.json`); and the rank sweep, 128 to
512 ranks over five days and a month
(`validation/pi_cam_online_model_{240,1488}step_*rank_*.json`).
The numbers are not repeated here so that they cannot go stale in two places.

## Limits

- The validated configuration is the `ne16` PI-atm case with CAM5 physics,
  SE dynamics and 512 ranks, and the `pcols=32` build of it against its own
  original. Other rank counts run online as explorations
  (`Driver(ntasks=N, exploratory=True)`, at least 128): one, 300 ranks, has
  matched CESM at the same layout over 48 steps. Adding another configuration
  needs a compatible native build context, field bindings, and its own
  validation evidence; PI-atm adapters are not reused for incompatible COSP,
  CARMA or radiation configurations.
- Bit-for-bit holds for the original processes and for the Python stage
  classes with the original kernels in place. A run that installs a
  different model in a kernel's place is, by construction, a different
  model's answer; its output is compared with the original to measure
  drift, not to pass a gate.
- Replay requires the rank layout of the capture. Online runs require the
  provider library, a completed original run to seed the surface components
  from, and the case's input data.
