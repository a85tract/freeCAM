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

The three leaves complete the set. Aerosol wet deposition hoists the leaf's
statements and `aero_model_wetdep`, pausing at the scavenging kernel
`wetdepa_v2` at both of its call sites -- the interstitial and the cloud-borne
forms -- inside the mode, phase and species loops the runner re-expresses;
the driver's `cycle` of a species it skips, and its early `return` when no
species is wet-deposited, are reported by the piece and carried out by the
runner's loop and routine states, as is the chemistry driver's return on a
skipped step. Aerosol dry deposition pauses at `modal_aero_depvel_part` at
its four sites (the cloud droplets before the mode loop, each mode's
interstitial aerosol inside it). Chemistry pauses once, at the whole
gas-phase driver `gas_phase_chemdr`, which is the kernel entire, with its
forty-odd arguments as one wide frame. Control patches 0047 and 0048 name the
chemistry and aerosol modules' private state public, generated with the
others by one module-state generator.

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

### The core inside `mmacro_pcond`

Every output of `mmacro_pcond` is derived from one internal call:
`instratus_condensate`, the saturation adjustment and stratus-fraction closure,
called twice per level (the relaxation iterations) and once more for the final
state.  Its last lines are the routine's own algebra -- `ql = al_st * ql_st`,
`qi = ai_st * qi_st`, `T = T0 - (latvap/cpair)(ql0 - ql) - ((latvap+latice)/cpair)(qi0 - qi)`,
`qv = qv0 + ql0 - ql + qi0 - qi` -- so a model that answers only the free
variables (the two stratus fractions and the two in-stratus condensates, each
bounded by construction) and derives the rest by those lines conserves water
and moist energy exactly and never hands the driver a fraction outside [0, 1].
That was the cut proposed for a surrogate of the macrophysics that the isotope
budget would not see; the outputs `mmacro_pcond` adds on top -- the tendencies,
`qme`, the limiter's adjustments -- stay the original Fortran's.  The first run
with a model in the slot (below) shows the isotope budget does see it.
On 288,000 captured calls (32 ranks of gate 7371232) those four lines hold to
round-off -- water and moist energy to a few 1e-16 relative, the in-stratus
products exactly -- which `validation/pi_cam_instratus_condensate_identities.json`
records.

It is reached the way `fluxbelowinv` is: the definition in `cldwat2m_macro.o`
is weakened and the hook takes its symbol (two PC32 relocations).  The stage-7
runner, which transcribes tphysbc's stage and pauses at `mmacro_pcond`,
`micro_mg_tend` and `cldfrc_fice` in its own state machine, gained the
pausable runners' fiber: with the hooked kernel replaced it arms the hook and
runs the whole stage on the fiber, the hook yields the frame from inside the
compiled routine, and `frame`, `resume` and `original` are served by the hook
table.  This revision refuses to combine the hooked kernel with a runner-level
pause (the manifest's `within` already forbids it with `mmacro_pcond`).  A
pause costs about a tenth of a millisecond: 9000 per rank per fifty steps
added under a second to a run that pauses a hundred times.

Any picklable callable can stand in any exposed slot from the command line:
`--kernel-model NAME=PATH` (or `PYCAM_KERNEL_MODELS` for the gate job) loads a
cloudpickled model into the named kernel of an installed stage class and records
it by file name and content hash; `PYCAM_NO_VERIFY_EXPORTS` lets such a run past
the replay's export check, since a model's answer differs from the oracle's at the
first export.  Every rank loads and re-pickles the model, and the install refuses
a payload that hashes differently across ranks, so the model must pickle the same
everywhere: a set of feature names pickles in per-process hash order and killed
the first surrogate run at step 0 (`pi_cam_pausable_instratus-surrogate_50step_failure.json`);
the gate job fixes `PYTHONHASHSEED` while models are loaded, and a model should
use ordered containers regardless.  Every gate run now also counts the log's water-isotope errors and
QNEG3 resets into `<summary>.health.json`: the original physics counts zero of
each in fifty steps, so a model's number there is entirely its own.

The first model in the slot ran the fifty steps (7371974,
`pi_cam_pausable_instratus-surrogate_50step.json` and its `.health.json`): a
NumPy multilayer perceptron of 72,454 parameters that predicts the two stratus
fractions, the two in-stratus condensates and two gates, trained on eight
million captured columns from 128 ranks of gate 7371232, with the closure's four
lines verbatim after it.  It answered all 9000 calls per rank; the run needed
no QNEG3 reset and reported no isotopic mass error; its step loop took 42 s
against 45 s with the original answering at the same hook and 17 s with nothing
paused, so the pause, not the network, is the cost at this size.  What the model
did trip is the water-tracer check after the macrophysics (`wtrc_apply_rates`
in water_tracers.F90): 6731 `BIG ERROR` lines in fifty steps where the original
prints none -- in about four percent of the chunk-steps the tendency of the
H2O copy tracers, rebuilt from the process rates, differs from the bulk
tendency by 1e-3 to 0.17 relative, and the copies' state has already moved off
the bulk water by up to 2e-3 kg/kg summed over a chunk.  Exact water and moist
energy conservation is therefore not the whole constraint the isotope
bookkeeping puts on this kernel.

Reading the two routines says why.  The tracer path takes only the three bulk
tendencies (macrop_driver.F90, 1128-1133): the phase changes split by sign, and
the net water tendency `qvlat + qcten + qiten` as a vapor self-rate.  A rate is
applied only when positive (water_tracers.F90, 1190), so a level whose net
water tendency is negative has that part dropped from its H2O copy -- the
whole of `diff`.  Net water can only go negative through `positive_moisture`,
which repairs a negative vapor by borrowing from the layer below
(cldwat2m_macro.F90, 2250-2253); the original never triggers it in fifty
steps.  The surrogate does, because the driver's linearisation of the next
relaxation iteration (`QQ`, cldwat2m_macro.F90, 905) takes the routine's
outputs as the saturated equilibrium state and caps the condensation with the
frozen half-step vapor (914-918): a predicted in-stratus condensate that is
not the saturation adjustment's lets that step drive the vapor negative, and
the repair follows.  So the constraint the cut has to keep is saturation, not
only conservation: the next model should predict the fractions and take the
condensates from the saturation adjustment at the predicted fractions, which
`instratus_condensate`'s own iteration defines.

The month said the same thing at length (7373795,
`pi_cam_pausable_instratus-surrogate_1month_failure.json`, run through
`validation/jobs/pi_cam_pausable_1month.pbs`, the fifty-step job's knobs over
the PI-atm month with the drift report added).  The model answered 983 steps,
day 20.5 of 31, at about 0.7 s a step; the water-tracer check fired at a steady
108 lines a step from the first day, the deep-convection tracer check and the
shallow convection's "source air is too dry" warning grew from a handful to a
thousand per three days, twenty-one isotopic precipitation mass errors were
printed, and then every rank took a segmentation fault in the same step -- the
fault itself left no backtrace and is not diagnosed.  Over the ten two-day
history files before it (`pi_cam_pausable_instratus-surrogate_1month_drift.json`)
1828 fields compared with nothing non-finite, the relative RMS difference from
the oracle's month at a median of 0.71 of each field's own spread, the largest
in the isotope precipitation and the aerosol number fields.  The health
counts are the result here, and the health counts of the original are zero.

## Where it stands

| Kernel | Owner | Contract | Runner pause | In-model gate | Loop |
| --- | --- | --- | --- | --- | --- |
| `mmacro_pcond` | Macrophysics (in the cloud stage) | reviewed | yes, validated | segmented, bit-for-bit; alone and together with `micro_mg_tend` | complete |
| `micro_mg_tend` | Microphysics (in the cloud stage) | reviewed | yes, validated | segmented, bit-for-bit (100 pauses in 50 steps); alone and together with `mmacro_pcond` (200 pauses) | open: no captured calls replayed through its standalone image yet |
| `rad_rrtmg_sw` | Radiation (split, pausable) | frame descriptor | yes, validated | segmented, bit-for-bit (50 pauses in 50 steps); alone, with `rad_rrtmg_lw`, and with every other exposed kernel paused in one run | open: capture and replay |
| `rad_rrtmg_lw` | Radiation (split, pausable) | frame descriptor | yes, validated | segmented, bit-for-bit (50 pauses in 50 steps); alone, with `rad_rrtmg_sw`, and with every other exposed kernel paused in one run | open: capture and replay |
| `dadadj` | DryAdjustment (pausable) | reviewed | yes, validated | segmented, bit-for-bit (100 pauses in 50 steps); alone and together with `compute_uwshcu_inv` | complete |
| `compute_uwshcu_inv` | ShallowConvection (pausable) | reviewed | yes, validated | segmented, bit-for-bit (100 pauses in 50 steps); alone and together with `dadadj` | open: no captured calls replayed through a standalone image yet |
| `zm_convr` | DeepConvection (pausable) | frame descriptor | yes, validated | segmented, bit-for-bit (100 pauses in 50 steps); alone, with the stage's other kernels, and with the tracer leaf paused in the same run | open: capture and replay |
| `zm_conv_evap` | DeepConvection (pausable) | frame descriptor | yes, validated | segmented, bit-for-bit (100 pauses); alone and with the stage's other kernels | open: capture and replay |
| `momtran` | DeepConvection (pausable) | frame descriptor | yes, validated | segmented, bit-for-bit (100 pauses); alone and with the stage's other kernels | open: capture and replay |
| `convtran` | ConvectiveTracerTransport (pausable, a leaf) | frame descriptor | yes, validated | segmented, bit-for-bit (100 pauses); alone and with deep convection paused in the same run, transporting with what its runner wrote | open: capture and replay |
| `compute_tms` | VerticalDiffusion (pausable) | frame descriptor | yes, validated | segmented, bit-for-bit (100 pauses); alone and with the stage's other kernels | open: capture and replay |
| `compute_eddy_diff` | VerticalDiffusion (pausable) | frame descriptor | yes, validated | segmented, bit-for-bit (100 pauses); alone and with the stage's other kernels | open: capture and replay |
| `compute_vdiff` | VerticalDiffusion (pausable, two sites) | frame descriptor | yes, validated | segmented, bit-for-bit (200 pauses, both sites); alone and with the stage's other kernels | open: capture and replay |
| `gw_drag_prof` | GravityWaveDrag (pausable) | frame descriptor | yes, validated | segmented, bit-for-bit (100 pauses in 50 steps; the band and coordinates by component) | open: capture and replay |
| `wetdepa_v2` | AerosolWetDeposition (pausable, a leaf, two sites) | frame descriptor | yes, validated | segmented, bit-for-bit (3000 pauses in 50 steps: every mode, phase and species at both sites) | open: capture and replay |
| `modal_aero_depvel_part` | AerosolDryDeposition (pausable, a leaf, four sites) | frame descriptor | yes, validated | segmented, bit-for-bit (800 pauses in 50 steps: the droplets and every mode at all four sites) | open: capture and replay |
| `gas_phase_chemdr` | ChemistryTendencies (pausable, a leaf; the whole driver) | frame descriptor | yes, validated | segmented, bit-for-bit (100 pauses in 50 steps, the whole gas-phase driver answered as one kernel) | open: capture and replay |
| `virtem` | VerticalDiffusion (pausable; a function inside an assignment of the driver) | frame descriptor; standalone contract | yes, validated | segmented, bit-for-bit (100 pauses in 50 steps; gate 7343257, and with every class installed, 7343260); every frame captured at the pause (7343396) replayed bit-for-bit through its standalone image | complete |
| `instratus_condensate` | Macrophysics (in the cloud stage; hooked: a private procedure of `cldwat2m_macro` called inside the compiled `mmacro_pcond`) | reviewed | yes, validated | segmented, bit-for-bit: the stage-7 runner runs on the fiber and the hook yields at every call, 9000 pauses per rank in 50 steps (two chunks by thirty levels by three calls -- the two relaxation iterations and the final state -- by fifty steps; gate 7371011); every frame captured at the hook (4,608,000 over the 512 ranks, gate 7371232, bit-for-bit; the first capture died out of memory at 4 x 64 GB and is kept as a failure record) | open: standalone replay of the captured frames |
| `cldfrc_fice` | DeepConvection (hooked; called inside the compiled `zm_conv_evap`) | function contract; standalone image | yes, validated | a hoisted copy of `zm_conv_evap` was not bit-for-bit (7343258, failure record kept); reached through a hook that redirects `zm_conv.o`'s call, machine code untouched: answered by the original at the hook, bit-for-bit, 51200 pauses in 50 steps (7343708); the first hook gate (7343594) failed on a duplicate fiber-body symbol, failure record kept; every frame captured at the hook (7343811) replayed bit-for-bit through its standalone image | complete |
| `fluxbelowinv` | ShallowConvection (hooked; private, called inside the compiled `compute_uwshcu`) | function contract; standalone image through its own symbol | yes, validated | its weakened definition hands every call site to the hook: answered by the original at the hook, bit-for-bit, 36733580 pauses in 50 steps (7343709); every frame captured at the hook (7343922) replayed bit-for-bit through its standalone image with the model's snapshot of uwshcu's `g` (7344823) -- the first replay, without that module state, was not and is kept as a failure record | complete |

One run installs everything: the nine pausable classes, the split radiation
class and the cloud stage, with all seventeen kernels answered by the original
through their pauses (5300 pauses in 50 steps), bit-for-bit with the oracle
(`pi_cam_pausable_everything_vs_oracle_50step_bfb.json`). Every kernel's
manifest entry names that run beside its own gates.

The energy fixer is the one active numerical action owned without a pause:
`EnergyFixer` runs it whole and exposes no kernel, because `check_energy_fix`
allocates its tendency inside the call and reads the module's private global
heating, so a frame at its call site would serve nothing. The ledger records
that reason on the action. The four leaves' classes declare their kernels
themselves; the physics catalog still lists no procedures under a leaf, so
their candidate counts read zero -- a gap of the catalog's call graph, not of
the classes.

Two items the plan named remain open and are recorded here rather than
dropped: a restart in the middle of a run, which freeCAM cannot do yet (the
driver starts every run as a startup run; continuing from CAM's restart files
is a feature of its own), and the standalone capture-and-replay loop for the
fifteen kernels that have only their pause gates.

**What the default path costs with every class installed.** Paired runs on
one allocation, the original Fortran (A) against freeCAM with all eleven
classes installed and nothing replaced (C), every one bit-for-bit with the
oracle (`validation/pi_cam_faster_than_fortran.json`):

| Pair | C/A | Note |
| --- | --- | --- |
| month, AC 7335688; CA 7335689; AC 7335690 | 1.057; 1.063; 1.054 | before the export fix below |
| year, AC 7335747 | 1.094 | before the export fix |
| month, AC 7336842; CA 7336843 | 0.996; 0.986 | after the fix |
| year, AC 7336930 | 1.0095 | after the fix; the cloud-only year was 1.0125 |

The first pairs were slower than the cloud-only baseline (1.006 for a month)
by five to nine percent. The profiler put the physics one-for-one inside the
class regions and the eleven Python wrappers at about 37 s of a year; the
rest was waiting at the boundary export, whose fast path treated a trusted
process's recorded outcome as something to report and so sent every step
through the pickled allgather once the outcome list was never empty. With
the flag testing for an actual failure, the all-class month runs at the
Fortran's pace and the year within one percent of it, the same as with the
cloud stage alone. The pre-fix pairs stay in the record; 7335690 carries the
commit of the fix because the report reads the tree when the job ends, but
its freeCAM run started before the fix was written.

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
