# Validation

freeCAM's claim is that Python owning the workflow changes nothing about the
numbers. That claim is tested one way: the model is run under Python control
and its output compared with the original Fortran model's, byte for byte, on
the full 512-rank PI-atm configuration. This page says what that comparison
covers, which gates exist, and where their evidence lives.

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

Every gate runs under PBS on Derecho with 512 ranks, through
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
| A scheme as a function | the standalone image calls the original routine; captured calls replayed through it reproduce the model | `validation/pi_cam_<scheme>_full_chunk_vs_capture.json`, `..._single_column_vs_capture.json`, `..._public_api_vs_capture.json` |
| The Workflow Builder's generated configuration through `freecam.Driver`, 50 steps | the page's path runs the validated default unchanged, and reaches an inserted Python process every step, bit-for-bit | [`pi_cam_workflow_builder_50step.json`](../validation/pi_cam_workflow_builder_50step.json) |
| The kernel decoupling inventory | every action of the step classified once, each exposed kernel followed through contract, capture, replay, in-model replacement and performance; built from the records above and checked current by the unit suite | [`physics_kernel_decoupling.json`](../validation/physics_kernel_decoupling.json) |
| The native image rebuilds | the build pipeline reproduces the image in use, command by command and symbol by symbol | [`pi_cam_native_image_rebuild.json`](../validation/pi_cam_native_image_rebuild.json) |

The two 50-step jobs are the gates every change to the numerical runtime has
to pass before it is merged:

```bash
validation/jobs/submit.sh validation/jobs/pi_cam_python_zero_copy_state_50step.pbs
validation/jobs/submit.sh validation/jobs/pi_cam_exact_cesm_online_50step.pbs
```

A wrapper or adapter that compiles is not validated; the gate has to show
that the intended routine executed and that its outputs match. Oracle output
is never overwritten, and a new configuration needs its own gate rather than
reusing the PI-atm evidence.

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
The numbers are not repeated here so that they cannot go stale in two places.

## Limits

- The only validated configuration is the `ne16` PI-atm case with CAM5
  physics, SE dynamics and 512 ranks. Adding another configuration needs a
  compatible native build context, field bindings, and its own validation
  evidence; PI-atm adapters are not reused for incompatible COSP, CARMA or
  radiation configurations.
- Bit-for-bit holds for the original processes and for the Python stage
  classes with the original kernels in place. A run that installs a
  different model in a kernel's place is, by construction, a different
  model's answer; its output is compared with the original to measure
  drift, not to pass a gate.
- Replay requires the rank layout of the capture. Online runs require the
  provider library, a completed original run to seed the surface components
  from, and the case's input data.
