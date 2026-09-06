# Decoupling the physics kernels

The aim is one shape for every active physical process of the PI-atm step: a
Python process class that owns the process's place in the workflow, its
parameters and its kernel slots, with the numerical kernels behind a declared,
replaceable, validated interface. The same kernel can then be called on its
own, sampled into training data, or replaced in the full model by a Python
function or a trained network, under one contract and one replacement
configuration. What runs when nothing is replaced is the original Fortran,
whole; what runs when something is replaced is the original Fortran up to the
replaced call, then Python, then the original Fortran again.

This page says how the pieces fit, what the inventory records, and where the
work stands. The ledger itself is
[`validation/physics_kernel_decoupling.json`](../validation/physics_kernel_decoupling.json),
written by `tools/build_physics_kernel_coverage.py` from the repository's
own records and checked current by the unit suite.

## The shape of a process

```
Python process class
    ├── parameters, lifecycle, place in the workflow
    ├── kernel contracts and kernels[...] slots
    └── execution
          ├── nothing replaced   -> the original Fortran process, called once
          └── something replaced -> Fortran to the replaced call
                                    -> frame to Python -> the replacement
                                    -> write-back, checked -> resume
```

Three pieces make this concrete today.

**The kernel slots.** `stage.kernels[name]` is the only registry of what
computes a kernel. Every way of installing a replacement lands there: assigning
into the mapping, naming a model file with `surrogate=`, or binding over the
method (`stage.mmacro_pcond = MethodType(fn, stage)`). The last used to rebind
the method alone, so a notebook's single-column call saw the new function
while the model, deciding how to run from the slots, ran the original Fortran.
`NativeStage.__setattr__` now puts a bound method into the slot as a
`MethodKernel`, and the walk, the runner's frame and the single-column caller
reach the same function. `OriginalKernel()` in a slot is the validation
replacement: the pause path runs, the original kernel answers through Python,
and bit-for-bit output proves the frame and the write-back.

**The segment-runner manifest.** Where the image can pause is declared in
[`native/pi_cam/segment_runners.yaml`](../native/pi_cam/segment_runners.yaml):
for each stage, the Fortran module, its generator, the descriptor its frames
are decoded with, the kernels it pauses at, and the gate records that
validated each pause. The backend (`freecam.pi_cam.native`) asks the manifest
which stage has a runner instead of knowing one by name; the stages ask it
whether a replacement can run segmented; the Workflow Builder reads it to say
which kernels are bindable and which are validated. A runner is
`ImageSegmentRunner(library, spec)`, one class for every prefix.

**A second pause, without a second transcription.** The stage-7 runner used
to call the microphysics driver whole. `pycam_micro_handles`, which already
held the driver's packer section verbatim for the Python walk, now holds the
whole of `micro_mg_cam_tend` in pieces -- the head before the packer, the
packer's five procedures, the tail after it -- with the routine's locals as
module state and the driver module's private buffer indices resolved by the
same field names. The runner calls those pieces in the source's order around
the substep loop and pauses at every `micro_mg_tend` call when the slot is
filled; the frame is the core's own argument list from the reviewed contract,
served from the packed arrays where the substep left them. Nothing numerical
moved: the pieces are the pinned text, checked line for line, and both modules
compile with the case's own flags before an image is built.

**One generator for the rest.** `tools/pi_cam_pausable.py` turns a spec under
`native/pi_cam/pausable/` into a pausable runner: the action's tphysbc block
and the driver it calls are hoisted verbatim into modules whose locals are
module state, cut into pieces at the kernel calls and at the `if`, `do` and
`select` statements the runner re-expresses, and every paused call's
arguments are served as a frame in the callee's own order, with the callee's
intents and declared shapes (an element or a section passed by sequence
association is served with the shape the callee sees). The runner also runs
the very call on request, so a gate can answer a pause with the original and
still exercise the frame's write-back. The pinned ranges are hashed into the
spec; a source that moves fails `--check`. Dry adjustment (`dadadj`) and
shallow convection (`compute_uwshcu_inv`) are the first two processes made
this way; `PausableStage` in `freecam.physics.pausable` owns each action, runs
it whole when nothing is replaced, and refuses the Python walk it does not
have. The eleven actions whose bodies do no numerical work in this
configuration are `InertStage`s: a class, no kernel, and one gate with all of
them disabled to prove it.

Radiation is the same generator over a split stage. `Radiation` keeps its two
leaves (the stop before `radiation_tend` and the resume after it, control patch
0041); the `pycam_radt` runner hoists `radiation_tend` whole -- the driver's
private variables and helper procedures verbatim beside it -- and pauses at
`rad_rrtmg_sw` and `rad_rrtmg_lw` inside the `dosw` / `dolw` blocks and their
`icall` loops, the RRTMG state served component by component. The runner's
glue is the resume half's driver call with `ptend` and `net_flx` pointed at
the radiation handles' storage, so the resume half takes them exactly as it
took the Python walk's. With nothing replaced the class leaves the step to the
resume half, which calls the driver itself: no runner, no walk. The walk
remains as the `legacy-python` policy.

Deep convection is a chain of three hoisted routines -- the tphysbc block,
`convect_deep_tend`, `zm_conv_tend` -- pausable at the Zhang-McFarlane core
`zm_convr`, the precipitation evaporation `zm_conv_evap` and the momentum
transport `momtran`. The mass fluxes, detrainment and gathering indices the
core writes are zm_conv_intr's own per-chunk module arrays, which control
patch 0044 makes readable (one `public` statement, no executable change), so
the tracer transport leaf a few actions later -- `convect_deep_tend_2`,
`zm_conv_tend_2`, pausable at `convtran` -- transports with exactly what deep
convection left there, never a copy. A frame addresses those arrays through a
TARGET dummy, serves an intent(out) scalar such as the gathered column count
where it lives so a model can answer it, and sizes an automatic array by its
own extents where the callee's would name something only the callee imports.

The two tphysac stages follow. Vertical diffusion hoists the tphysac block and
`vertical_diffusion_tend`, pausing at the turbulent mountain stress
`compute_tms`, the eddy diffusivities `compute_eddy_diff` and the implicit
solver `compute_vdiff` at both of its call sites (the moist and the dry field
lists; a kernel called at several sites pauses at each, and every site serves
the same frame, an optional the site omits as an empty slot). The friction
velocity and Obukhov length it writes are tphysac carries the dry deposition
stage reads (control patch 0043). Gravity wave drag hoists `gw_tend` and
pauses at `gw_drag_prof` inside the orographic block, the only source active
here; the wave band and the pressure coordinates are served component by
component, and the driver's automatic arrays, sized by each chunk's column
count, are module allocatables re-sized when a chunk's shape changes. Both
drivers read their modules' private options, selectors, bands and indices
through control patches 0045 and 0046, accessibility statements like 0044; a
field selector with private components and a procedure argument are passed by
the original and served by the frame as nothing. The frame ABI carries five
extents per slot: the convective transport's water-tracer ratio is rank four.

**The read-only description.** `stage.describe_kernels()` returns one record
per kernel: the owning class, whether the runner pauses at it, whether that
pause has passed a gate, the reviewed contract's inputs and outputs when one
exists, what is in the slot now, and how many times a model answered for it in
this run. The run summary written by the command line carries the same rows
under `stage_execution`, and the Workflow Builder consumes them rather than
keeping a list of its own.

## The inventory

The ledger classifies each of the step's 58 actions once and follows every
exposed kernel through the delivery loop:

```
contract reviewed -> original call captured -> standalone image built
  -> replayed bit for bit (full chunk, single column, public interface)
  -> replaced in the full model with the original kernel answering, bit for bit
  -> performance recorded -> supported
```

An action is one of: `numeric_scheme`, `process_control`, `diagnostics`,
`boundary`, `clock`, `dynamics`, `io`, `host_service`. A disabled action is
recorded as the alternate form of the enabled work it stands for (a stage
whose leaves run, or a leaf whose stage runs whole), never as a hole. An
enabled scheme whose body is expected to do nothing under this configuration
(`rayleigh_friction` without `rayk0`, the CARMA leaves with no CARMA model)
is marked inert-by-configuration and listed as unresolved until a targeted
test confirms it; it is not counted as covered. The inertness gate is that
test: one 50-step run with all eleven such actions disabled, bit-for-bit with
the oracle (`pi_cam_pausable_inert_vs_oracle_50step_bfb.json`), flips them to
inert-confirmed. Execution is evidenced from
the recorded runs at two lengths, 50 steps and a month, not assumed from the
plan.

The record names nothing outside the repository: contract paths, evidence
files and catalog sources are relative, and there is no timestamp, so the
committed file equals a fresh build or the test fails.

## Where it stands

| Kernel | Owner | Contract | Runner pause | In-model gate | Loop |
| --- | --- | --- | --- | --- | --- |
| `mmacro_pcond` | Macrophysics (in the cloud stage) | reviewed | yes, validated | segmented, bit-for-bit; alone and together with `micro_mg_tend` | complete |
| `micro_mg_tend` | Microphysics (in the cloud stage) | reviewed | yes, validated | segmented, bit-for-bit (100 pauses in 50 steps); alone and together with `mmacro_pcond` (200 pauses) | open: no captured calls replayed through its standalone image yet |
| `rad_rrtmg_sw` | Radiation (split, pausable) | frame descriptor | yes, validated | segmented, bit-for-bit (50 pauses in 50 steps); alone, with `rad_rrtmg_lw`, and with every other exposed kernel paused in one run | open: capture and replay |
| `rad_rrtmg_lw` | Radiation (split, pausable) | frame descriptor | yes, validated | segmented, bit-for-bit (50 pauses in 50 steps); alone, with `rad_rrtmg_sw`, and with every other exposed kernel paused in one run | open: capture and replay |
| `dadadj` | DryAdjustment (pausable) | reviewed | yes, validated | segmented, bit-for-bit (100 pauses in 50 steps); alone and together with `compute_uwshcu_inv` | complete |
| `compute_uwshcu_inv` | ShallowConvection (pausable) | reviewed | yes, validated | segmented, bit-for-bit (100 pauses in 50 steps); alone and together with `dadadj` | open: no captured calls replayed through a standalone image yet |
| `zm_convr` | DeepConvection (pausable) | frame descriptor | yes, gate pending | pending | open: the pause gate, capture and replay |
| `zm_conv_evap` | DeepConvection (pausable) | frame descriptor | yes, gate pending | pending | open: the pause gate, capture and replay |
| `momtran` | DeepConvection (pausable) | frame descriptor | yes, gate pending | pending | open: the pause gate, capture and replay |
| `convtran` | ConvectiveTracerTransport (pausable, a leaf) | frame descriptor | yes, gate pending | pending | open: the pause gate, capture and replay |
| `compute_tms` | VerticalDiffusion (pausable) | frame descriptor | yes, gate pending | pending | open: the pause gate, capture and replay |
| `compute_eddy_diff` | VerticalDiffusion (pausable) | frame descriptor | yes, gate pending | pending | open: the pause gate, capture and replay |
| `compute_vdiff` | VerticalDiffusion (pausable, two sites) | frame descriptor | yes, gate pending | pending | open: the pause gate, capture and replay |
| `gw_drag_prof` | GravityWaveDrag (pausable) | frame descriptor | yes, gate pending | pending | open: the pause gate, capture and replay |

The pausable classes were gated on 2026-09-06 in six 512-rank 50-step runs on
one image: each class installed with nothing replaced (dry adjustment, shallow
convection), each kernel answered by the original through the pause, both
paused in the same run, and the eleven inert actions disabled at once. All six
are bit-for-bit with the oracle; the records are the
`validation/pi_cam_pausable_*_50step.json` summaries and their
`_vs_oracle_50step_bfb.json` comparisons. The radiation runner followed the
same day in five runs on its own image: the split class installed with nothing
replaced (the resume half calls the driver; no pause), each core answered by
the original through its pause (50 pauses each, radiation running every other
step), both cores in one run, and one run with dry adjustment, shallow
convection and radiation all paused at once. All five bit-for-bit.

Twelve enabled scheme actions do numerical work in this configuration. Four
have a Python class today, all partial by the loop above: the cloud stage's
two kernels both pause in the runner and both pauses have passed the gate
with the original kernel answering; what the microphysics core still lacks
is the capture-and-replay step of its own standalone image. The other eight are
gaps with their candidate procedures listed from the catalog: vertical
diffusion, gravity-wave drag, the energy fixer, deep convection, and the wet
deposition, dry deposition, convective transport and chemistry leaves. The
energy fixer is deferred on purpose: `check_energy_fix` allocates its tendency
inside and reads a module variable private to `check_energy`, so a frame at
its call cannot serve its outputs; it needs an allocation-aware pause. For the four leaves the catalog's active call
graph lists no procedure yet; the ledger says so rather than choosing kernels
without it.

The radiation cores are next, and the micro pause is their template:
`radiation_tend` (radiation.F90, 577-1320) is one routine with the two calls
at 1034 and 1148, so it hoists into `pycam_rad_handles` as pieces the same
way, with the radiation module's private indices resolved by name. Two things
are different. The stage is today a pair of leaves around the driver, and a
runner owns a whole action, so `Radiation` has to take the whole action back
before it can run segmented. And both cores take an `rrtmg_state_t`, which
no frame can hand a model as one slot: the frame must serve its arrays one
by one, which is what the draft contracts under
`native/pi_cam/functions/drafts/` still mark for review.

The stages that exist are delivered in this order: the generic contract and
runner registration (done, above); the cloud stage's second kernel and the two
radiation kernels through a segment runner, each gated with the original
kernel answering; convection and turbulence; the remaining active schemes in
dependency order; then the combined and long runs. Every step of the loop is
a record under `validation/` before the ledger counts it.
