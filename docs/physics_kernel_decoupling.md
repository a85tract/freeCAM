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

### A model the image runs itself

Every replacement above answered from Python: the hook or the runner stopped,
Python built the frame, the model answered, Python wrote back and resumed.
That round trip is what a replacement costs -- about 3 ms for the 29-argument
instratus frame, 180 times a step, which is why the model run above took
42 s for fifty steps against 16 s with nothing replaced -- and no amount of
Python-side work removes it; the surveyed practice (FTorch in CAM and ICON,
pytorch-fortran and FTorch in E3SM-MMF, Infero in the IFS, the
Fortran-Keras Bridge in SPCAM) is to run the model inside the Fortran process
and keep Python out of the step.

The hook table now allows that.  A hook whose entry in `hooks.yaml` has a
`model` block -- the contract arguments the model's forward takes, in order,
and the outputs it returns -- can be *bound* to a TorchScript file
(`pycam_hooks_bind_model_v1`).  From then on the hook answers every call
itself: it wraps the kernel's arrays as tensors where they live (a scalar such
as `k` travels as a one-element tensor), runs the model through
[FTorch](https://github.com/Cambridge-ICCS/FTorch), and writes the live
columns of the outputs back.  No fiber, no frame, no Python: the stage runs
whole and a step crosses the boundary once, whatever is replaced.  On the
command line the same `--kernel-model NAME=PATH` takes a TorchScript archive
as the model; the stage sees a `NativeModel` in the slot, binds it at the hook
on its first step and runs `native-model`.  A bound model and a Python
replacement cannot share a hook, and a native model cannot stand at a kernel
that is not a hook.  The Python path stays for what it is good at: frame
capture, the original answering at the pause, and quick experiments.

The image links FTorch and the libtorch of the checkout's own `torch`
package (`build_pi_cam_devices.py --ftorch-root`, see the installation
guide), so a rank needs no Python-side torch.  The model must be exported
with its pre- and post-processing inside -- feature scaling, gates, the
closure -- because the hook hands it the kernel's raw arguments; the
instratus surrogate exported that way answers bit-for-bit what its NumPy
form answered on captured frames.  A Fortran program calling it through
FTorch takes about 200 microseconds per 16-column call on a login-node core,
almost all of it TorchScript's per-operator dispatch (the 72,454-parameter
network is a few microseconds of arithmetic): fifteen times cheaper than the
Python pause, still not free for a kernel called 180 times a step, and a
rounding error for the cores that cost 6-10 ms a call.

In the model it measures as follows (50 steps, 512 ranks, the p13 image that
links FTorch; `pi_cam_pausable_p13-whole_50step.json`,
`pi_cam_pausable_instratus-ftorch-excl_50step.json`, both on four exclusive
nodes with one rank a core):

| run | cloud stage, s per rank | step loop, s | instratus hook |
| --- | ---: | ---: | --- |
| nothing replaced (bit-for-bit with the oracle) | 0.89 | 7.70 | 4,792,320 calls, 0 answered by a model |
| the surrogate bound at the hook | 4.59 | 11.49 | 4,608,000 answered by the model, 0 paused |

Every other region is the same to the hundredth of a second; the cost is the
model's 9000 calls per rank at 0.41 ms each, tensor wrapping and TorchScript
dispatch included, against 2.8 ms for the same calls answered from Python
(`pi_cam_pausable_instratus-surrogate_50step.json`, on develop's shared half
nodes, where the FTorch path took 0.65 ms a call and the Python pause 2.8;
`pi_cam_pausable_instratus-ftorch_50step.json`).  The health counts are the
same model's -- 6719 water-tracer lines against 6731 from the NumPy form, the
last bit of a matrix product deciding a gate here and there -- and the image
with nothing replaced counts zero and stays bit-for-bit, so linking FTorch
changed nothing the oracle can see.  What this leaves for a kernel called 180
times a step is a 49 percent step; for the cores called twice a step at 6-10 ms
it leaves the model's own arithmetic.

### The most expensive core, and what the mechanism costs there

The deep GPTL profile of the original Fortran (fifty steps and a month, 512
ranks) ranks the physics by what its timers cost per call: `microp_mg_tend`
first at 10.7 ms on develop's shared half nodes (18 percent of the physics, 6
percent of the step), the UW shallow convection next at 9.4 ms, the two RRTMG
timers at about 6 ms, and `mmacro_pcond` at 2.4 ms -- less than a Python
pause, which is why the macrophysics surrogate could never pay back through
the pause and why the next experiment moved to the microphysics.  Those are
the drivers' timers, though: `microp_mg_tend` (microp_driver.F90) wraps the
whole of `micro_mg_cam_tend` -- the buffer reads, the packing, the core, the
unpacking and about a hundred history fields -- and how much of it the core
`micro_mg_tend` itself costs is measured further down: 1.4 ms a call on
exclusive nodes.

`micro_mg_tend` is called from one compiled site in `micro_mg_cam`, so it is
hooked the same way as `instratus_condensate`, with two differences the hook
table now expresses.  It takes the callee's own kinds -- default logicals, an
assumed-length character, pointer arrays -- so its hook is a plain module
procedure (`binding: fortran`) with no C interface and no frame: it cannot be
paused, only bound to a model, and arming it is refused.  And it receives
*packed* arrays, `(mgncol, nlev)` for the columns that hold cloud, with
`pcols` and `pver` set to those very extents; the first build sized the
hook's arrays by the contract's constants and corrupted memory
(`pi_cam_pausable_micro-ftorch-excl_50step_failure.json`, kept), so a
Fortran-bound hook now sizes every array by the callee's integer dummies.
The model block names 26 inputs and 89 outputs over its 116 arguments.

Measuring the mechanism there needed two more things.  A model that answers
changes the run, so a hook can bind a model *in shadow*: the model runs on
every call for its cost and its answer is discarded while the original
answers, and the run stays bit-for-bit.  And the hooks time themselves per
rank: the model branch, the forward alone, the whole call as the hook sees
it, the first call alone, and the warm-up; the record sums them over the
ranks and keeps the slowest rank's own value beside each sum.  All runs below
are fifty steps on four exclusive nodes with one rank a core; the per-rank
region times come from the run's `timing/freecam_timing_stats`
(`CAM:cloud_macro_microphysics_python`, mean over ranks).

| image, run | step loop, s | cloud stage, s | model branch, ms a call | forward, ms a call | bit-for-bit |
| --- | ---: | ---: | ---: | ---: | --- |
| p15 nothing bound, 7394422 | 7.67 | 0.88 | | | yes |
| p15 null model bound and answering, 7394423 | 8.50 | | | | no, as intended: zeros are the answer |
| p16 (timers) null model answering, 7399102 | 8.51 | | 1.74 | 1.33 | no, as intended |
| p17 null model in shadow, 7399451 | 9.45 | 1.74 | 1.63 | 1.45 | yes |
| p17 instratus surrogate in shadow, 7399452 | 11.82 | 4.80 | 0.35 | 0.33 | yes |
| p18 nothing bound, 7400407 | 7.64 | 0.88 | | | yes |
| p18 null model in shadow, 7400408 | 7.97 | 1.11 | 0.56 | 0.38 | yes |
| p18 instratus surrogate in shadow, 7400409 | 11.20 | 4.24 | 0.34 | 0.32 | yes |

The p17 rows did not add up: the stage grew by 0.85 s a rank with the model
branch at 0.16 s, and by 3.92 s with the branch at 3.19.  The timing tree
placed the difference.  The Fortran region inside the stage grew by exactly
the branch (0.15 s and 3.20 s); the rest sat in the stage's Python around
it, and was the same 0.7 s in both runs whatever the call count: the stage
re-read `hooks.yaml` every step to find the hooks' ids (7 to 10 ms a step),
and the first step paid the model's load and first forward -- rank 0's first
step took 0.23 s against 0.02 s, with 512 ranks faulting the 438 MB libtorch
in from the work filesystem at once.  The p18 image binds once and runs one
forward on zero tensors of the contract's extents at bind, before the step
loop sees the model; the first-call and warm-up timers say what that moved:
the first real call now takes 3.8 ms a rank (4.6 on the slowest) and the
warm-up 83 ms (99), paid at bind.  What is left in the microphysics stage is
the two calls a step at 0.56 ms and a first step 0.16 s longer for the bind.

Standalone, the same null model costs 183 us a call from a Fortran program
on a login-node core, of which the hook's tensor wrapping and write-back are
1.4 us; a column count that changes between calls costs nothing after the
second call, and flushing the data caches between calls adds 55 us.  In the
model it costs three times that: the call comes after forty milliseconds of
other physics with the caches cold, on a node whose 128 ranks share memory
bandwidth.  That 0.56 ms -- with 115 tensors wrapped and 89 outputs written
back -- is the floor for replacing `micro_mg_tend` inside the image.  What
the core itself costs against it, and what trained networks cost in its
place, is the next section.  The placeholder MLP run (7394424, random
weights, the network's real size: 457,155 QNEG3 lines, kept for the timing
only) already said the network's own arithmetic at that size is another
1.4 ms a call standalone.

### A trained surrogate for `micro_mg_tend`, and what it cost

The training data were already there: the capture gate of 2026-09-09
(7359460, bit-for-bit) holds every call of `micro_mg_tend` in fifty steps,
51,200 calls with their 26 inputs and 89 outputs, 20 GB over the 512 ranks.
Looking at them settled the model's shape.  The four `in_place` arrays
(`qc`, `qi`, `nc`, `ni`) come back exactly as they went in, so the model
passes them through; `reff_rain` and `reff_snow` arrive uninitialised and are
outputs only; eight inputs are constants or never set in this configuration
and are dropped.  What is left is a per-column map from 630 numbers to 2,494
(85 outputs, most of them sparse: `tlat` is exactly zero in 56 percent of the
levels, the process rates in 90 to 99 percent).  The surrogate is a two-layer
MLP over standardised features and targets, trained on 448 ranks' columns
(590,815) with the step's own tendencies weighted five times, validated on
the other 64 ranks (83,445 columns), forty epochs each -- about six minutes
on one node per size.  The exported TorchScript carries its own
preprocessing: the standardisation folded into the first and last layers,
the outputs clamped to their observed ranges, and the five tendencies
floored so a step cannot drive water or number below zero.

| width | parameters | weights | `qctend` R² | `tlat` | `nctend` | `prect` | median over the 85 outputs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 1,865,150 | 7.6 MB | 0.89 | 0.72 | 0.65 | 0.97 | 0.62 |
| 256 | 868,286 | 3.6 MB | 0.86 | 0.66 | 0.60 | 0.96 | 0.57 |
| 128 | 419,006 | 1.7 MB | 0.82 | 0.60 | 0.47 | 0.95 | 0.49 |
| 64 | 206,654 | 0.9 MB | 0.73 | 0.52 | 0.21 | 0.94 | 0.44 |

A crude emulator, then, and enough for the question at hand: what does a
network of a realistic size cost in the model, against the core it replaces?
Standalone on one login-node core the answer looked fine -- 1.47 ms a call
for the 512-wide network, 0.38 ms for the 64-wide.  In the model it was not,
and one benchmark on a compute node says why (`bench_concurrent2.pbs`,
7401108): the same model run by 1, 8, 32 and 128 processes at once.

| model | 1 process | 8 | 32 | 128 | in the model (shadow) |
| --- | ---: | ---: | ---: | ---: | ---: |
| null, no weights | 0.20 ms | 0.21 | 0.22 | 0.23 | 0.56 |
| 256 wide, 3.6 MB | 1.33 | 1.40 | 2.17 | 2.30 | 3.71 |
| 512 wide, 7.6 MB | 1.47 | 3.87 | 6.21 | 6.38 | 6.71 |

Eight ranks share one 32 MB slice of L3 on these nodes, so eight copies of a
7.6 MB network already spill it; above 32 processes the memory bandwidth is
the limit.  A network's cost inside the image is set by its weight bytes,
which every call streams from memory again after forty milliseconds of other
physics, not by its arithmetic.

The shadow runs then priced the model and the core on the same calls, with
the p19 image, which times the original too when a shadow model answers
(fifty steps, four exclusive nodes, per-rank means, all bit-for-bit):

| run | model, ms a call | the core `micro_mg_tend`, ms a call | cloud stage, s | step loop, s |
| --- | ---: | ---: | ---: | ---: |
| p19 nothing bound, 7401310 | | | 0.88 | 7.74 |
| null model, 7401312 | 0.56 | 1.33 | 1.13 | 7.93 |
| 64 wide, 7401282 | 1.47 | 1.35 | 1.22 | 8.16 |
| 128 wide, 7401281 | 2.25 | 1.38 | 1.52 | 8.31 |
| 256 wide, 7401311 (7401058 on p18: 3.74) | 3.71 | 1.40 | 1.59 | 8.35 |
| 512 wide, 7400992 on p18 | 6.71 | | 1.98 | 8.77 |

**The core costs 1.4 ms a call** -- 2.8 ms a step, 1.8 percent of the step
loop -- not the 10.7 ms of the driver's timer.  The mechanism alone (the null
model) costs 0.56 ms of that, and the smallest trained network, 64 wide with
a fifth of the skill, costs exactly what the core costs.  No network that
answers this call can make the step faster on these nodes, and a free one
would save 1.8 percent.

The live runs, with the model's answer taken (`--kernel-model`, no
verification of exports), confirm it from the other side: 512 wide 9.02 s
(7400993), 256 wide 8.67 s (7401059), 128 wide 8.55 s (7401232) for the step
loop, against 7.64 s with nothing bound -- every one slower, and slower than
its own shadow run, although the core is skipped.  Two things the surrogate
changes cost more than the core it removes: the state it produces sends the
rest of the physics down other paths (the UW `fluxbelowinv` ran 22 and 30
million times instead of 38), and the water-tracer and QNEG3 checks write
300,000 lines to the log (100,000 BIG ERROR lines in each run, 88 to 10,765
QNEG3).  Those records are kept as they are, not bit-for-bit and not meant to
be.

What this leaves of the microphysics as a target is the driver around the
core, 7 to 9 ms a call on shared half nodes, which is buffer handling and
history output, not arithmetic, and not something a network replaces.  For
the surrogate route the lesson is general: on 128-rank nodes a model pays
only where the core costs several milliseconds a call *and* the network's
weights fit the cache its eight neighbours leave it -- a few hundred
kilobytes -- or where the inference runs on an accelerator the ranks do not
share.

### A kernel written in Python, compiled, called by Fortran

The pause was the wrong tool for one thing the project wants: a kernel
written in Python that runs at the original's speed.  Every pause costs
about 3 ms -- the fiber switch, the frame, the write-back -- against cores
of one or two milliseconds.  The hook now takes a third answerer beside the
original and a TorchScript model: a *compiled plugin*, a C function of the
hook's plugin interface, bound by address (`pycam_hooks_bind_plugin_v1`).
On every call the hook hands it the model block's arguments as two tables --
pointers and extents, the same `float64` arrays it would wrap as tensors for
a model -- and takes its outputs from the same temporaries, writing the live
columns back exactly as the model branch does.  Nothing pauses; a step
crosses the boundary once.

The Python side makes the plugin with Numba.  `compile_kernel(hook, function)`
(`freecam.physics.numba_kernel`) reads the hook's model block and contract,
generates the adapter that unpacks the tables into Fortran-ordered arrays with
the callee's own extents, compiles the user's function with `numba.njit` and
the adapter as a `numba.cfunc`, and returns a `NativePlugin` for the kernel
slot; the stage binds its address like a model's file.  On the command line
`--kernel-plugin NAME=file.py:function` (and `--shadow-kernel-plugin`) does
the same, compiling once per rank.  The user's function takes the inputs then
the outputs, arrays indexed `[column, level]`, scalars as floats, and writes
the outputs in place -- `examples/plugins/numba_kernels/cldfrc_fice.py` is the
ice-fraction kernel written that way, statement for statement the original's
arithmetic.  Offline it answers bit-identically to its NumPy form.

In the model (the p20 image, fifty steps, four exclusive nodes) it answers
bit-identically to the Fortran too: with the Python kernel bound live at the
hook, every one of the 50,176 calls after the bind taken by it, the run is
**bit-for-bit with the oracle** (7402505), at 8 microseconds a call.  In
shadow (7402506) the same.  A zero-writing kernel with
`instratus_condensate`'s 19 inputs and 8 outputs, in shadow with the core
timed on the same calls (7402507), prices the mechanism against the two
other ways of answering the same hook:

| how the hook is answered | a call | for comparison |
| --- | ---: | --- |
| the original `instratus_condensate` | 5 µs | |
| a compiled plugin (pointer tables, the adapter, the call) | 0.4 µs | |
| the surrogate through FTorch, same hook | 350 µs | 7401311 and earlier |
| a Python callable at the pause, same hook | 3 ms | `instratus-surrogate` runs |

The step loop is unchanged by a plugin (7.62 to 7.77 s against 7.66 s with
nothing bound).  For a kernel written in Python this is the path; the pause
remains for frame capture and for the gates that answer with the original
through Python, until a plugin does those too.

The same path carries a trained network without libtorch.  The forward of
the `micro_mg_tend` surrogates above -- the three dense layers with the
standardisation folded into the first and last, the clamps, the floors --
written as loops in `examples/plugins/numba_kernels/micro_mlp.py` over
weights exported from the training checkpoint, and bound in shadow on p20
with the core timed on the same calls (all bit-for-bit):

| width, weights | compiled forward, ms a call | the core, ms a call | the same network through FTorch |
| --- | ---: | ---: | ---: |
| 64, 0.8 MB (7404600) | 0.87 | 1.43 | 1.47 |
| 128, 1.7 MB (7404599) | 1.45 | 1.43 | 2.25 |
| 256, 3.5 MB (7404601) | 2.92 | 1.47 | 3.71 |

Taking libtorch out of the call saves 0.6 to 0.8 ms at every width -- the
tensor objects, the operator dispatch, the output copies -- and leaves the
arithmetic and the weights: the 64-wide network now costs less than the core
it replaces, the 128-wide the same.  Standalone with hot caches the 64-wide
forward takes 0.35 ms; the rest of its 0.87 ms in the model is the weights
read again from memory on every call, which no code change removes.  What
this buys the step is still bounded by the core's 1.8 percent, and the
64-wide network has a fifth of the 512-wide network's skill; the finding is
about the mechanism, not the surrogate: on these nodes a network's inference
belongs in compiled code, and FTorch's in-image path costs 1.3 to 1.7 times
more at these sizes.  Numba compiles the kernel on every rank at start,
about 50 seconds, which the record's initialisation time carries.  The first two attempts
(7402090 to 7402092, 7402200 to 7402202) never stepped: a stage is
cloudpickled into each rank's process registry when it is installed, so the
plugin's compiled code must stay out of the pickle, and then the payload is
compared across ranks, so the pickle may carry no address either -- an
identity that is the same everywhere, resolved to the code in the process
that compiled it.  Both are kept as failure records.

### The radiation process as one replaceable unit

Kernel by kernel, a network cannot make this model faster on these nodes:
the cores are cheap and a network's price is its bytes.  What remains open
for a learned replacement is a *process* that is expensive as a whole, a
column function with a small output set, and free of the water isotopes --
radiation, the classic emulator target and the most expensive physics
process here (7.5 percent of a step, called every other step).  The
`Radiation` stage, which transcribes `radiation_tend` statement for
statement and was gated bit-for-bit that way, now offers the computing
branch of a radiative step -- the optics, the two RRTMG cores and their
diagnostics -- as one *process slot* (`freecam.physics.radiation_process`).

The contract is the driver's, not a model's.  Before the branch the driver
has in hand the state (temperature, pressures, the constituent array), the
buffer's cloud fraction and the optics' inputs (`DEI`, `MU`, `LAMBDAC`,
`ICIWP`, `ICLWP`, `DES`, `ICSWP`, `DGNUMWET`, `QAERWAT`), the surface
albedos and upward longwave from `cam_in`, the cosine of the zenith angle,
and the RRTMG state's gas profiles once it is built; the branch leaves the
two heating rates, in the driver's energy units before it scales them for
storage, and the ten surface and top fluxes the coupler and the energy check
read (`fsns`, `fsnt`, `flns`, `flnt`, `fsds`, `sols`, `soll`, `solsd`,
`solld`, `flwds`).  Everything around the branch stays the driver's: the
quiet-step conversions, `radheat_tend` building the tendency and the net
flux, the copy into `netsw`.  Three things can stand in the slot:

- `RadiationProcessCapture` records inputs and outputs on every radiative
  step of every chunk and writes them per rank (`--radiation-capture DIR`):
  the dataset a process-level emulator is trained on, taken from a run that
  stays bit-for-bit.
- `RadiationReplay` answers the branch with a capture's outputs for the same
  step and chunk (`--radiation-model replay:DIR`): the gate of the write-back
  path, which must itself be bit-for-bit.
- `RadiationProcessModel` answers it with a function, the inputs by name in
  and the outputs by name out (`--radiation-model path.py:function`) -- a
  trained network's forward, compiled with Numba as the kernels above are,
  called once per chunk on radiative steps from the Python that already
  owns the step between the stage's two halves.

The slot was gated on 2026-09-12 (fifty steps, develop's half nodes, the
p20 image).  With a capture in the slot the run is bit-for-bit with the
oracle and every rank recorded its fifty radiative-step calls: 25,600
records, 9.1 GB, forty inputs and twelve outputs (7417373).  With the replay
of that capture in the slot -- the branch never computed, its outputs taken
from the record -- every restart file and the history file are identical to
the oracle's, and the history restart's accumulators differ in exactly the
26 fields the branch writes to history that a model has no source for: the
aerosol optical depths and burdens the aerosol optics writes inside the
branch, the clear-sky and top-of-atmosphere fluxes, the cloud forcings and
the incoming solar (7417486; `tools/compare_pi_cam_variables.py` lists them,
file by file, in
`pi_cam_pausable_rad-process-replay3_vs_oracle_50step_variables.json`).  The
heating rates and the ten fluxes a model does produce are written to history
by the slot as the driver writes them.  The first replay (7417422) also
differed in those, before the slot wrote them; the one before it (7417389)
never stepped, its replay pickled per rank -- both kept as records.  The
model state, then, is exact through the slot; what a model owes the history
is stated by that list.

The first emulator went into the slot the same afternoon, to prove the
whole loop rather than the model: a per-column network over 1,029 features
-- the 30-level profiles of temperature, pressure, water vapour, cloud
fraction, in-cloud water and ice paths, the optics' effective sizes and
snow, the fifteen modal aerosol mixing ratios with their wet diameters and
aerosol water, the 31-level ozone, and the zenith angle, albedos, upward
longwave and latitude -- to the 70 targets, trained on the day-1 capture's
448 ranks (303,700 columns) and validated on the other 64
(`examples/plugins/numba_kernels/train_rad.py`).  Positive heavy-tailed
inputs enter as logarithms; shortwave targets are learned on lit columns and
set to zero in the dark, where the physics has them exactly zero.  Training
is cheap: 70 s for forty epochs of a 256-wide network on one node, 7 minutes
for two hundred of a 512-wide one.

| network, data | `qrs` R² | `qrl` | `fsnt` | `flnt` | `flwds` | `fsnt` RMSE, W/m² |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 wide, 40 epochs, one day | 0.929 | 0.814 | 0.976 | 0.973 | 0.988 | 53 |
| 256 wide, 200 epochs | 0.943 | 0.832 | 0.982 | 0.978 | 0.990 | 46 |
| 512 wide, 200 epochs | 0.947 | 0.834 | 0.984 | 0.981 | 0.991 | 44 |

The longwave heating near the surface is the weak point (RMSE up to 1.1
K/day against 0.1-0.5 aloft); the shortwave error is the clouds' (39 W/m²
RMSE in the top-of-atmosphere net flux, a +3 W/m² bias).  Two hundred epochs
gain a little over forty and the training loss falls far below the
validation loss: one day of one January is the limit, and a month's capture,
every eighth radiative step, is the next dataset.  In the model
(`examples/plugins/numba_kernels/rad_mlp.py`: the features built in Python,
the forward compiled by Numba with the weights as arguments, 0.6 ms a chunk)
the first network answered every radiative step of fifty steps (7417501): no
fault, every health count zero, and a state a day later 0.11 K from the
oracle's in temperature, 0.08 g/kg in water vapour, 0.07 hPa in surface
pressure.  Not bit-for-bit by design; the record's variable comparison lists
what moved.

A month of captures followed (7417573: the transcription with a capture in
the slot ran the 1,488 steps bit-for-bit with the monthly oracle, recording
every eighth radiative step of every rank -- 95,232 chunk records, 34 GB),
and the Python that assembled the 1,029 features, which had cost most of the
emulator's 4.7 ms a call in the model, became one function generated from
the feature layout and compiled by Numba: the same matrix, bit-identically,
in 0.1 ms.  Two networks trained on the month (1,129,764 columns; validated
on 64 held-out ranks) answered fifty steps each (7417764, 7417763): no fault,
every health count zero, the state a day later 0.14 and 0.12 K from the
oracle's in temperature.

| network, data | `qrs` R² | `qrl` | `fsnt` | `flnt` | `flwds` | `fsnt` RMSE, W/m² |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 wide, 60 epochs, one month | 0.957 | 0.873 | 0.986 | 0.982 | 0.992 | 41 |
| 512 wide, 60 epochs, one month | 0.965 | 0.891 | 0.988 | 0.985 | 0.993 | 39 |
| 512 wide, 200 epochs, one month | 0.968 | 0.901 | 0.988 | 0.986 | 0.994 | 38 |

In the model the 256-wide network cost 1.5 ms a chunk call and the 512-wide
4.5 (its 3.3 MB of weights streamed by 128 ranks a node), against the
driver's branch at 12 ms and more on the same nodes.  The gain did not reach
the step: the slot lived in the Python transcription of the driver, whose
own cost -- about 45 ms a step and rank for the walk -- exceeded what the
network saved.  The emulator had to answer the branch from inside Fortran.

#### The slot inside the image

A control patch on `radiation.F90` was written and withdrawn: the image
links the oracle's numerical objects unchanged, and the driver is one of
them.  The slot went instead into the one copy of `radiation_tend` this
repository owns -- the pausable runner's hoisted driver, generated from
`native/pi_cam/pausable/radiation.yaml` -- as a *skeleton slot*: the spec
marks the branch's `if (dosw .or. dolw)` node (`if: 875`, `slot:
pycam_rad_process_answer(...)`), and the runner, when the condition holds,
asks the slot first; a bound plugin answers the whole branch and the runner
continues after it, otherwise the original pieces run as before.  A new
support module, `pycam_rad_process`, hands the plugin what the driver has in
hand as pointer and extent tables in a fixed order of 46 inputs -- the
state's fields and constituents, the buffer's cloud fraction and the optics'
inputs, the albedos and upward longwave, the zenith angle, the RRTMG state's
gas profiles, which it builds itself -- through the same C interface the
kernel hooks use, takes the two heating rates and the ten fluxes back,
writes them where the driver writes them and their history as the driver
does, and times its calls per rank; in shadow it runs the plugin for its
cost and lets the branch answer.  Python's `TABLE_INPUTS` and
`TABLE_OUTPUTS` mirror the order (a test pins them to the module), a table
kernel compiles for it with Numba (`compile_radiation_plugin`), the
`Radiation` stage binds it and runs the runner whole with no pause armed,
and the command line takes `--radiation-plugin` and
`--shadow-radiation-plugin`.  The emulator's table kernel
(`examples/plugins/numba_kernels/rad_mlp.py`, the weights frozen into the
compiled code) reproduces its Python path bit-identically offline.

The p21 image carries the runner with the slot and the module.  Gated on
2026-09-12 and 13 (fifty steps, 512 ranks, develop):

- The image is bit-for-bit with the oracle with the cloud class installed
  (7418221), with the `Radiation` class installed and nothing bound
  (7418222), and with the runner running the driver through the slot,
  unbound, the shortwave core answered by the original at its pause
  (7418303: 50 starts, 50 pauses).
- The first two plugin runs (7418223 shadow, 7418224 live) completed one
  step and were killed for memory: every rank's Numba compilation and frozen
  weights add about 0.35 GB, 268 GB over four half nodes' 256 GB.  Kept as
  failure records; plugin runs take 235 GB a node (`-l select=4:ncpus=64:
  mpiprocs=128:ompthreads=1:mem=235GB`).
- The next two (7418304, 7418305) ran bit-for-bit with the plugin compiled
  and never bound: `Radiation` defined `prepare_segmented` twice and the
  later definition, which binds the runner's hosts, silently replaced the
  one that bound the plugin.  The record shows it -- `radiation_process`
  counts zero calls -- and a live plugin run can never be bit-for-bit, so
  both were set aside as no evidence; the definitions are merged and a unit
  test now forbids a class in `freecam` from defining a method twice.
- In shadow, the month-trained 256-wide network bound at the slot ran on
  every radiative step of every chunk -- 25,600 calls, 50 a rank -- while
  the branch answered, and the run stays bit-for-bit (7418443).  The
  plugin costs 1.4 ms a call inside the image (the slowest rank 1.8), the
  price the Python path had measured, now without the Python.
- Live, the same network answered all 25,600 calls (7418444).  The
  radiation stage's region fell from 1.40 s a rank per fifty steps to 0.23
  (its share of the step loop from 8.7 to 1.5 percent), and the step loop
  from 16.07 s (7418303: the same image and runner with the branch computed
  and one pause a step) to 14.72: 8 percent, on develop's shared nodes.  The state it produced is
  bit-for-bit with the Python path's run of the same weights (7417764):
  the two slots -- one in the transcription, one in Fortran -- compute the
  same emulator on the same inputs.  Against the oracle it drifts as that
  run did: a day later 0.14 K rms in temperature (5.5 K at worst), 0.08
  g/kg in water vapour, 0.28 m/s in zonal wind, 0.08 hPa in surface
  pressure; the 26 branch diagnostics are absent from the history restart
  as before.  Every rank held 1.09 GB at most; the job used 328 GB.

The month followed on the main queue, full nodes, the replayed boundary
(the surface prescribed, so no surface feedback), the same image and class
in both runs:

| run | step loop, 1,488 steps | radiation a rank | against the monthly oracle |
| --- | ---: | ---: | --- |
| `Radiation` installed, nothing bound (7418517) | 401.3 s | 34.2 s | bit-for-bit; every health count zero |
| the 256-wide network at the slot (7418518) | 363.9 s | 2.9 s | not bit-for-bit by design |

Nine percent of the month's step loop.  The plugin answered all 761,856
calls (512 ranks, 744 radiative steps, two chunks) at 1.31 ms a call, the
slowest rank 1.6; every rank held 1.09 GB at most.  The run stayed finite
and the water-isotope checks fired thirteen times in the month -- two
"BIG ERROR" lines at steps 825 and 1314, eleven stratiform isotopic mass
errors on two ranks around step 1100 -- against none in the baseline and
106,487 in the earlier instratus surrogate's month.  Against the oracle's
history (`pi_cam_pausable_rad-month-slot_1month_drift.json`; median
relative rms over 2,590 fields 0.64) the emulator's own error shows on day
3, before the weather has diverged: net shortwave at the top of the
atmosphere 0.4 W/m² low in the global mean with 9.4 W/m² rms over the
columns, outgoing longwave 0.2 W/m² low with 4.7 rms, cloud fraction
unmoved.  By day 30 the global means have drifted -- column water vapour
1.5 kg/m² lower, net shortwave 4.8 W/m² lower, outgoing longwave 4.2 W/m²
higher, the lowest level 0.9 K colder -- while the column-wise rms (surface
pressure 13.9 hPa, temperature 6.6 K at the lowest level) is the scale at
which any perturbed January diverges from the oracle in a month.  The
state after the month is 5.3 K rms from the oracle's in temperature.

What the loop now shows: a process emulator can be trained on the model's
own captures, compiled, and bound inside the image at the top of the branch
it replaces, and it then saves what it saves -- nine percent of a month's
step loop for a network whose science is not good enough to use: its
radiation budget drifts by several W/m² in a month and it produces none of
the clear-sky and top-of-atmosphere diagnostics.  The mechanism is closed;
the network (more data, the longwave near the surface, the diagnostics) is
the open problem it exposes cleanly.

#### Any model at the slot: a transformer through FTorch

The slot answers with a C function; a compiled Python kernel is one way to
make one, and it suits small dense networks because Numba has no BLAS here.
For anything larger the slot took a second answerer on 2026-09-13: a
TorchScript file loaded through FTorch (`pycam_rad_process_bind_model_v1`,
`--radiation-torch-model`), which sees the same 46 inputs as tensors -- a
Fortran `(pcols, pver)` array is a `(pcols, pver)` tensor, FTorch's default
layout -- and fills the same 12 outputs, the timers and the shadow mode
shared with the plugin path.  The image p22 carries it and is bit-for-bit
with the slot unbound (7431583 with the cloud class, 7431584 with the
runner and the shortwave core at its pause).  To exercise it with a model
the MLP path cannot express, `train_rad_tf.py` trains a level-token
transformer: the thirty levels are the tokens, each carrying 34 features
(the twelve profiles, ozone, the fifteen aerosol mixing ratios, their wet
diameters and water) and eight column scalars, a learned position per
level, two heads -- the heating rates per level, the ten fluxes from the
mean token -- and the export wraps the network in the slot's 46-tensor
signature, float64 in and out, frozen and optimised for inference.

| network | parameters | training | `qrs` R² | `qrl` | `fsnt` | `flnt` | alone, 1 thread | in the image, steady | first call a rank |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 wide, 2 layers | 72k | 12 epochs, 26 min | 0.956 | 0.872 | 0.974 | 0.975 | 2.1 ms | 7.1 ms | 0.28 s |
| 96 wide, 3 layers | 233k | 10 epochs, 44 min | 0.965 | 0.892 | 0.982 | 0.985 | 4.6 ms | 18.7 ms | 0.36 s |
| 256-wide MLP (the plugin) | 347k | 60 epochs | 0.957 | 0.873 | 0.986 | 0.982 | 0.4 ms | 1.3 ms | -- |

The transformers match the MLPs' accuracy with a fifth of the parameters
(the 64-wide one equals the 256-wide MLP; the 96-wide equals the 512-wide),
and both were still improving when their epochs ran out.  Their price in
the image is another matter.  Both ran in shadow on every radiative step
of fifty steps, bit-for-bit (7431802, 7431990): 7.1 and 18.7 ms a call
once warm, 3.4 and 4.1 times their single-thread cost alone -- the same
factor the compiled MLP pays (0.4 to 1.3 ms) on a node running two ranks a
core.  Live, the 64-wide network answered all 25,600 calls (7431803) and
brought the radiation stage from 1.30 s a rank to 0.72, two percent of the
fifty-step loop against the plugin's eight; the 96-wide one (7431991) costs
more than the branch it replaces, 1.66 s against 1.30, and the loop gets
slower.  Both drift from the oracle as the MLP does (0.17 and 0.14 K rms in
temperature after a day) with every health count zero.  The lesson is not
that transformers are out: the slot takes any TorchScript model, and the
first call's 0.3 s and the per-operator overhead of a 30-token attention on
sixteen columns are what a leaner export (fused preprocessing, one layer,
an ONNX runtime) would attack.  On this node layout only the small one
pays for itself, and the compiled MLP pays four times better.

#### Python answering the slot: the runner pauses at the branch

Both answerers above are called by Fortran.  The other way round -- the
driver's preparation and write-back in Fortran, the model call itself in
Python, with as few crossings as the structure allows -- was built the same
day as a third form of the slot: the runner *pauses* at the top of the
radiative branch.  The radiation spec declares a `process_slot` beside the
`slot` on its `if` node; when the slot is armed the runner asks
`pycam_rad_process_prepare` to build the slot's tables and the RRTMG state,
parks its program counter and returns to Python as a kernel pause does, with
a frame of the 46 inputs and 12 outputs over the same storage
(`radiation_process`, numbered after the runner's two kernels in the
manifest's `process_slot`).  Python answers the frame -- a process model on
the live-lane views, or the original branch on request -- and the resume
runs `pycam_rad_process_finish`, which writes the outputs where the driver
writes them and destroys the state, or, after the original was asked for,
continues into the branch's own pieces.  Three crossings a chunk; the
Python that owns the step is the same `Radiation` class, whose `process`
slot now takes a Python model on any image whose runner pauses there
(`--radiation-model path.py:function`, or `original` for the gate), and
still falls back to the transcription on an older image.

The p23 image carries the pause.  Its gates (fifty steps, 512 ranks,
develop, whole-node memory, one run at a time; the first pair died at the
resume on a Python-side numbering slip and the second Python run on a
one-element scalar, both kept as failure records, and one bit-for-bit run
shared its nodes with a neighbour and timed everything twice as slow):

| run | what answered the slot | `rad_tend` a rank, 50 steps | step loop | against the oracle |
| --- | --- | ---: | ---: | --- |
| 7434627 | nothing (the cloud class installed) | -- | 16.00 s | bit-for-bit |
| 7434783 | the original branch, asked for at the pause | 1.73 s | 16.40 s | bit-for-bit, 25,600 pauses |
| 7434965 | the compiled MLP called from Python at the pause | 0.82 s (once compiled) | 15.5 s (once compiled) | drift as the plugin's: 0.137 K rms |
| 7418444 (p21) | the same MLP as a plugin, called by Fortran | 0.23 s | 14.72 s | the same state, bit-for-bit with this run |
| 7431584 (p22) | the branch, computed | 1.30 s | 15.96 s | bit-for-bit |
| 7439535 (p24) | the 64-wide transformer as a compiled plugin, called by Fortran (12.5 ms a call) | 0.84 s | 16.03 s | 0.17 K rms |

The pause itself costs about 9 ms a chunk on top of the shortwave pause
path (7434783 against 7431584), and with the model in Python 16 ms a chunk
all told -- the model's own arithmetic is 1.3 ms of that, the same as in
the plugin: the rest is the frame's 58 slots decoded into views, the
write-back's checks and the resume, run by an interpreter sharing a core
with another rank.  Handing the model views instead of copies of the 46
inputs changed nothing measurable.  The Python path also compiles the
model on its first call inside the step loop, 9 s a rank (the plugin
compiles at install); the fifty-step loop reads 24.9 s for that reason and
15.5 s without it.  The state the two paths produce from the same weights
is identical, so what the pause buys is flexibility -- any Python model,
no compilation step, a notebook can hand one in -- and what it costs is
about two thirds of the saving: 0.5 s a rank of the branch's 1.3 against
the plugin's 1.1.

#### The Python driver: the model replaces the whole block, Python does the rest from memory

The user's own picture of the process was sharper than the slot: split
`radiation_tend` into what is read from memory before any arithmetic, one
compute block, and what is written to memory after; keep the block in
Fortran or hand it to a model; let Python do the reading and the writing.
Drawn on the driver, the block is everything numerical -- the zenith angle,
the optics, the two cores, the heating-to-tendency step, the energy scaling
-- and the model that replaces it must take only what the driver had in
memory: the state and its constituents, the buffer's cloud fraction, optics
inputs and ozone mass mixing ratio, the surface albedos and upward longwave,
the calendar day and the column's latitude and longitude
(`radiation_process.BLOCK_INPUTS`).  The zenith angle and the RRTMG gas
profiles are learned; a thirteenth output, a lit-column logit, gates the
shortwave to exact zeros in the dark.  The `Radiation` class gained the
mode `python-driver` (`--radiation-block-model`): per chunk it reads its
views -- taken once and kept, the state's arrays, the surface, the fluxes,
the buffer's plain fields, the column geometry; the two time-sliced cloud
fields per step -- decides the radiative step by the driver's own rule
(`radiation_steps`, radiation.F90:240-246), calls the model, writes the
heating rates and fluxes where the driver leaves them, and then does what
the driver does after the branch: the tendency and net flux through the
one Fortran call `radheat_tend` (which also allocates the tendency the
resume half takes), the heating-rate diagnostic in NumPy, the step's
history in one new call (`pycam_rad_process_history_v1`, image p24), the
storage scaling by the layer mass, the copy into `netsw`; on a
non-radiative step the same without the model.  Four small Fortran calls
a chunk for bookkeeping; the compute is the model's alone.

The gates (fifty steps, whole develop nodes):

| run | what answered the block | `rad_tend` a rank | step loop | against the oracle |
| --- | --- | ---: | ---: | --- |
| 7435709 | nothing (the cloud class installed) | -- | 15.85 s | bit-for-bit |
| 7436003 | the day-1 capture's recorded outputs, through the Python driver | 0.30 s | 14.77 s | **`cam.r`, `cam.rs`, `h0` bit-for-bit**; `rh0` differs in the same 26 diagnostics as through the transcription (7417486), HR included among the identical ones |
| 7436082 | the block emulator (256 wide, trained on the month) | 0.38 s once compiled | 15.0 s once compiled (22.5 with the 7.5 s compile) | drift: 1.5 K rms in temperature after a day |
| 7418444 (p21) | the slot emulator as a plugin, for comparison | 0.23 s | 14.72 s | 0.14 K rms |
| 7439603 | the 64-wide block transformer, TorchScript on one torch thread a rank, through the Python driver | 1.20 s | 15.75 s | drift: 1.27 K rms after a day |

The replay is the proof of the driver: with the block's outputs given, the
Python around them reproduces the state exactly, so the bookkeeping --
which step radiates, the unscaling and rescaling by the layer mass, the
tendency, the net flux, the surface copy -- is the driver's own.  Its cost
is 0.30 s a rank per fifty steps, about 3 ms a chunk, and with the
compiled network the stage runs at 0.38 s: a shade above the plugin's
0.23, with the step loop within noise of it.  The first pair of runs
(7435764, 7435791) died at the first chunk on a view of the tendency taken
before `radheat_tend` had allocated it, a third (7435825) on the node
fabric, and a fourth (7436004) on the constituent array the handles do not
carry; all kept as failure records, and the driver now reports a chunk's
failure before Fortran aborts on the missing tendency.

The price is in the model.  Asked to learn the zenith angle and the gas
conversions instead of being given them, the 256-wide network validates
at R² 0.945 for the shortwave heating (0.957 with them) and 0.975 for the
net shortwave at the top (0.986), and its lit-column gate is right on
99.88 percent of columns -- and the run drifts ten times faster than the
slot emulator's from the same weights and data: 1.5 K rms in temperature
after a day against 0.14, 0.66 hPa in surface pressure against 0.08.  The
geometry the physics computes in twenty lines is what the network learns
worst, and the columns it gets wrong at the terminator get a full day's
shortwave or none.  Giving the model the zenith angle -- twenty lines of
trigonometry in Python, or one more Fortran call -- is the obvious repair;
the driver does not change.

The notebook's way in is the same path.  `driver.processes["radiation"]` is
the `Radiation` stage bound to a run; put a model in its `process` slot (or
a callable in a kernel's) and the next `advance` attaches the stage where
its action runs, set the slot back to `None` and the Fortran runs again
(`docs/usage.md`).  Gated on 2026-09-13 through `fc.Driver` on 512 ranks
(`tools/run_process_table_gate.py`, `validation/jobs/pi_cam_process_table_50step.pbs`):
the day-1 capture replayed through the block slot set from the table
produced every file byte for byte the same as the command line's replay
of it (7438656 against 7436003; `pi_cam_process-table-replay_50step.json`),
and so is bit-for-bit in state with the oracle as that run is.  One path,
two ways in.

#### The same transformer on the other two paths

The transformer had been priced on one path only, through FTorch.  The
question that remained was what the compiled-plugin path and the Python
driver pay for it, so on 2026-09-13 the 64-wide, 2-layer network was
trained for both contracts on the same month of captured calls (12 epochs
each, 72k parameters; the slot contract is given the zenith angle, the
block contract learns it and a lit-column logit, and validates a little
lower: R² 0.931 against 0.956 for the shortwave heating, the gate right on
99.90 percent of columns) and gated on the p24 image, one run at a time on
whole 235 GB nodes.  For the plugin path the network is written out in
Numba (`examples/plugins/numba_kernels/rad_tf_plugin.py`): tokens,
embedding and position, the pre-norm layers with their attention and GELU
feed-forward, the heads, de-standardisation, clamps and day/night gate,
the weights frozen into the compiled kernel from an export of the
checkpoint.  There is no BLAS behind Numba in this environment, so every
product is a loop; the compiled kernel agrees with PyTorch to 1e-6
relative on a captured chunk and costs 3.6 ms a chunk on one login-node
thread against 2.1 for the TorchScript module.  For the driver path the
block-contract TorchScript is loaded in the rank's Python
(`rad_block_tf.py`, one torch thread) and given the block's inputs as
tensors.

| run | path | who runs the arithmetic | model, ms a call | first call a rank | `rad_tend` a rank | step loop | against the oracle |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 7431584 (p22) | the branch, computed | RRTMG | -- | -- | 1.30 s | 15.96 s | bit-for-bit |
| 7431803 (p22) | TorchScript at the slot through FTorch | libtorch, called by Fortran | 7.1 | 0.28 s | 0.72 s | 15.65 s | 0.17 K rms |
| 7439472 (p24) | the Numba plugin at the slot, in shadow | compiled loops, called by Fortran | 12.6 | 14 ms | 1.93 s | 17.45 s | bit-for-bit, 25,600 calls |
| 7439535 (p24) | the Numba plugin at the slot, live | compiled loops, called by Fortran | 12.5 | 14 ms | 0.84 s | 16.03 s | 0.17 K rms |
| 7439603 (p24) | the block transformer through the Python driver | libtorch, called by Python | 15.5 | 0.38 s | 1.20 s | 15.75 s | 1.27 K rms |
| 7418444 (p21) | the 256-wide MLP as a plugin, for scale | compiled loops, called by Fortran | 1.3 | -- | 0.23 s | 14.72 s | 0.14 K rms |

The three paths land within a factor of two of each other, and the order
is set by what runs the arithmetic and what sits around it, not by which
side makes the call.  libtorch inside Fortran is the cheapest at 7.1 ms.
The Numba loops take 12.5: the linear layers run at about 20 GMAC/s, the
30-token attention at a tenth of that, and no loop order tried on the login
node closed the gap to a tensor library (the plugin compiles for 7 s at
install, and its first call in the loop costs nothing more).  libtorch
from Python takes 15.5 with the forward itself about 7 of it: the rest is
turning thirty-three Fortran-ordered views into contiguous float64 tensors
and back, on an interpreter sharing its core -- a conversion the FTorch path
does not do, since it hands the arrays over as they lie, and one that
`torch.from_numpy` over the views would mostly remove.  TorchScript's first
call costs 0.38 s a rank for its profiling passes, once, against the 7.5 s
the Numba block model spent compiling inside the loop.  Around the model
the driver's own bookkeeping is the 0.42 s a rank measured with the replay
(0.30) and the MLP (0.38).  The step-loop column is the noisier measure:
the runs scatter by 0.3 s for reasons outside radiation -- the boundary
export waits 0.88 s a rank in the baseline and 0.57 in the others, the
cloud stage costs 0.17 s more on a drifted state -- so `rad_tend` is the
number to read.  On it, none of the three transformer paths saves more than
the FTorch one's 0.6 s a rank of the branch's 1.3, and the block model's
drift (1.27 K rms against 0.17 for the same network given the zenith
angle) repeats the MLP's lesson: what the driver path needs next is not a
faster call but the twenty lines of geometry handed to the model.

#### A month on each path: the model's speed and its drift

Fifty steps price a call; a month (1,488 steps, the replayed boundary,
p24 image, whole 235 GB nodes) shows what the run keeps of it and how far
the model's own error carries the state.  Five months were run on
2026-09-14: the Radiation class installed with nothing bound, the 256-wide
MLP at the slot as a Numba plugin and, trained for the block contract, as
the block model through the Python driver; and the 64-wide transformer the
same two ways.  Drift is the last day's global mean against the oracle's
month; the plugin months repeat the p21 month pair's numbers (401.3 and
363.9 s there) within a few seconds.

| run | radiation | step loop | against the original | model, a rank over the month | first call | day 30: net shortwave at the top, column water vapour, lowest-level temperature |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 7452662 | nothing bound | 400.2 s | -- | -- | -- | bit-for-bit |
| 7452663 | MLP as a Numba plugin | 366.2 s | -8.5% | 1.95 s (1.3 ms a call) | 3 ms | -4.8 W/m², -1.5 kg/m², -0.9 K |
| 7452664 | MLP through the Python driver | 371.9 s | -7.1% | 7.6 s, 6.0 of it compiling at the first call | 6.0 s | -46 W/m², -7.7 kg/m², -12.3 K |
| 7452666 | transformer as a Numba plugin | 385.4 s | -3.7% | 18.8 s (12.6 ms a call) | 15 ms | -4.4 W/m², -1.2 kg/m², -1.0 K |
| 7452667 | transformer through the Python driver | 374.3 s | -6.5% | 10.3 s (6.9 ms a call) | 0.47 s | -243 W/m², -4.1 kg/m², -1.2 K; 45,308 "BIG ERROR" lines |
| 7454279 | the same MLP through the Python driver, the forward in NumPy | **359.5 s** | **-10.2%** | 2.2 s (1.5 ms a call) | 2 ms | -55 W/m², -7.9 kg/m², -11.8 K (the same model) |
| 7455413 | the same transformer through the Python driver, written out in NumPy | 373.9 s | -6.6% | 11.7 s (7.9 ms a call) | 10 ms | -243 W/m²; 45,069 "BIG ERROR" lines (the same model) |

The two paths are a percent and a half apart on the loop with the same
network when both run it in Numba: the driver's month costs its 6 s
compile on the first call and about 3 ms a chunk of bookkeeping, and
nothing else.  The Python driver has no need of Numba -- it calls any
Python callable; the compiled kernel is what the plugin path's C function
pointer needs -- and with the forward written as three NumPy matrix
products through the single-threaded BLAS (7454279) the first call costs
2 ms instead of 6 s and the month runs in 359.5 s: the fastest of the
five, ten percent under the original and seven seconds under the plugin.
Over fifty steps the same run (7454278) puts the radiation stage at 0.32 s
a rank against the plugin's 0.23 and the loop at 14.71 s against 14.72.
The transformer written out in NumPy (`rad_block_tf_np.py`: batched matrix
products for the attention, NumPy's softmax and layer norm, an exact-erf
GELU; 2e-6 from the scripted module) gains nothing of the kind: 7.9 ms a
call against libtorch's 6.9, the month 373.9 s against 374.3 (7455413
against 7452667; over fifty steps 7455411 against 7439603, 15.06 s against
15.75 with the conversion's first-call cost gone).  What libtorch spends on
operator dispatch and the copy of the inputs, NumPy spends on its own
per-call overhead over some forty small array operations; the six-fold
arithmetic of the transformer against the MLP is what both pay for.  The
NumPy version keeps libtorch off the ranks (0.91 GB a rank against 1.03)
and its first call costs 10 ms instead of 0.47 s; on speed the two are the
same.  The drift column is
the model's, not the path's: the slot models (given the zenith angle)
drift as the month pair did; the block models, made to learn it, lose
118 W/m² of net shortwave in the global mean by day 3 and 12 K at the
lowest level by day 30 -- a model that cannot be used, on a path that
replays a capture bit-for-bit.  The health counters agree: two "BIG
ERROR" lines and eleven isotopic mass errors in the plugin's month
(7452663), twenty and six in the driver's (7452664), six in the
transformer plugin's (7452666), none in the baseline -- and 45,308 "BIG
ERROR" lines in the transformer block model's month (7452667), whose lit
gate leaves the shortwave out over most of the globe: 243 W/m² of net
shortwave missing from day 3 on.  The runtime numbers of that month stand
(6.9 ms a call, the loop 6.5 percent under the original); its physics does
not.

#### The cloud stage as the Python driver: two blocks, the memory around them in Python

The stage `mmacro_pcond` lives in -- tphysbc's stage 7, the cloud
macro/microphysics action -- got the same form on 2026-09-14.  Under the
admitted configuration (one macro/micro substep) the stage is two compute
blocks with bookkeeping between and after them: the macrophysics driver
(`macrop_driver_tend`, with `mmacro_pcond` inside it), then the flux terms,
the tendency scaling and application and the energy check; the
microphysics driver with the aerosol activation that feeds it and the sum
of their tendencies (`microp_aero_run`, `micro_mg_cam_tend`,
`physics_ptend_sum`), then scaling, application, energy check, the
precipitation means and the water-tracer mass fixer.  The split is drawn
in `freecam.physics.cloud_block`: each block's contract names what the
driver has in memory before its arithmetic and what it leaves behind.
The macrophysics block takes 60 inputs -- the step, the state's thirteen
components, the five surface fields, the six convection carries, its
thirty buffer fields -- and leaves 38 outputs: the tendency object (its
`s`, its `q` over every constituent, and the `ls`/`lq` flags
`physics_update` reads), the two detrainment integrals, and the thirty
buffer fields again, because the driver writes into most of them (the
cloud fractions, the in-cloud water, the old-time-sample copies of the
state), and the two convective cloud fractions `cldfrc` computes inside
it.  The microphysics block takes 84 and leaves 71 over the two drivers'
67 buffer fields -- plus, resolved from the image at run time because
they are registered per constituent or tracer, the cloud-borne aerosol
fields the activation rewrites (`modal_aero_data`'s `qqcw`, 15 here) and
the water tracers' surface precipitation (`wtrc_srfpcp_indices`).  Everything numerical is inside the blocks;
what is between them is the glue's own calls, made from Python through the
stage's handles as the transliteration has made them since Gate M-1.

`CloudMacroMicrophysics` gained two slots for the two blocks -- `process`
(the macrophysics block) and `micro_process` -- a `block_capture`, and the
mode `python-driver` they switch on.  Per chunk the driver takes its views
once and keeps them (the state pool's arrays, the surface, the carries,
every registered buffer field of both contracts; a field this
configuration never registers, UNICON's detrainment, is left out as the
driver's own pointer stays unassociated), then: the macrophysics block --
the original driver in place, or the slot's answer written back: the
tendency object allocated and flagged as the driver would have left it
(`pycam_mm_ptend_init_v1`, one of two entries added to the mm handles; the
other reads the flags for a capture), its arrays and the detrainment
filled through the views, the buffer fields assigned -- then the four
bookkeeping calls; the microphysics block the same way; the last five
calls.  A capture around the original drivers records each block's inputs
in single precision and its outputs exactly (the tendency's constituent
array for the flagged constituents only), one file per block per rank; a
replay writes them back; a model is any callable over the contract.  The
command line takes `--cloud-block-model`, `--cloud-micro-block-model` and
`--cloud-block-capture` beside `--cloud-macro-micro-python`; the notebook
sets `driver.processes["cloud_macro_microphysics"].process` or
`.micro_process`, and the process table arms the stage on either.

The gates (p26 image, fifty steps, whole 235 GB nodes, one run at a time
for the timed ones), and the road to them: the contract was drawn from the
field tables and completed by the runs, each of which named what it missed.

| run | in the slots | against the oracle | stage, a rank | step loop |
| --- | --- | --- | ---: | ---: |
| 7450698 | the class installed, nothing armed (native-whole) | bit-for-bit | 1.96 s | 15.95 s |
| 7451930 | both originals, through the Python driver | bit-for-bit | 2.33 s | 16.52 s |
| 7453135 | both originals, the capture around them (51,200 calls a block, 19 GB compressed) | bit-for-bit | 2.7 s | 16.66 s |
| 7452498 | the microphysics block replayed, the macrophysics original verifying itself | `cam.r`, `cam.rs`, `h0` bit-for-bit; `rh0` differs in the microphysics driver's 68 diagnostics | -- | 15.3 s |
| 7453151, 7453313 | the macrophysics block replayed, the microphysics original verifying itself | every file bit-for-bit, `rh0` included; the verified microphysics matched the capture's record on all 51,200 calls, inputs and outputs | 1.96 s | 15.77 s |
| 7453150 | **both blocks replayed** | **`cam.r`, `cam.rs`, `h0` bit-for-bit**; `rh0` differs in the two drivers' diagnostics | 0.59 s | 14.37 s |
| 7452704, 7452705 | each original with the census around it | bit-for-bit; the macrophysics block wrote `DP_FRAC` and `SH_FRAC` beyond its contract, the microphysics block nothing | -- | -- |

Seven replays failed before the last one passed, and each failure is a
record: the first capture asked the buffer for a field this configuration
never registers (7450699); the first replay left the oracle at step 3
because the aerosol activation inside the microphysics block rewrites the
cloud-borne aerosol fields through a registry no field table sees
(7451924); the second because the views of the time-rotated fields were
taken once and CAM moves the older plane every step (7452063); the third
and its half-replays because `wtrc_output_precip` fills the water
tracers' surface precipitation through an index array (7452220, 7452222);
the macrophysics half-replays because `cldfrc` writes the convective cloud
fractions the wet deposition reads later in the step (7452221, 7452497),
found not by reading but by the census; and one round fell over reporting
its own finding (7452422-7452425).  Two tools came out of it and stay:
the capture keeps an exact digest of every input, so a replay names the
first input and step at which its run has left the captured one; and
`verify:DIR` or `census` in a slot runs the original in place and names
the outputs it produced differently, or every buffer field it changed
that its contract does not list.  The driver costs 0.37 s a rank per
fifty steps over native-whole (about 3.7 ms a chunk: the views, nine small
Fortran calls of bookkeeping); with both blocks replayed the stage is
0.59 s.

#### The first networks for the macrophysics block

With the contract complete, the capture is a training set (19 GB, 51,200
calls a block), and `train_cloud_block.py` fits an MLP to a block from it;
`cloud_block_mlp.py` answers the block with it from a slot.  Two shapes
were tried on the macrophysics block on 2026-09-14, and the first taught
two lessons that have nothing to do with physics.

| run | the network | parameters | model, ms a call (alone) | stage, a rank | step loop | after 50 steps |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 7450698 | the original | -- | 6.6 (GPTL `macrop_tend`) | 1.96 s | 15.95 s | bit-for-bit |
| 7453811 | everything the contract names, 2322 in, 1682 out, Numba, columns outside the weight loop | 2.3M | 36 (2.3) | 10.3 s | 26.6 s | 1.53 K rms in T |
| 7454183 | the same, weight rows outside the column loop | 2.3M | 8.3 (2.1) | 7.2 s | 24.1 s | 1.53 K |
| 7453875 | the microphysics block the same way, 3963 in, 3257 out | 4.0M | 85 | 12.9 s | 30.4 s | 1.82 K |
| 7454182 | the physics only (`--core`): 969 in, 632 learned, seven fields derived, Numba | 0.48M | 2.2 (0.57) | 6.6 s | 23.3 s | 5.78 K |
| 7454277 | the same core network, the forward in NumPy | 0.48M | 2.7 (0.58) | 2.34 s | 19.27 s | 5.78 K |

The first lesson is the loop order: the first network's forward took 36 ms
a call in the image against 2.3 alone because its first-layer weights (4.75
MB) were streamed once per column, and 128 ranks sharing their caches turned
that into memory traffic; with the weight rows outside the column loop the
same network costs 8.3 ms, and the core network 2.2.  The second is Numba
itself: in the Numba runs the stage carries 4.3 s a rank of compilation at
the first call inside the loop; with the forward as NumPy matrix products
the first call costs 3 ms and the stage is 2.34 s a rank -- the original's
1.96 plus the driver's 0.37 of bookkeeping, the network's 0.27 s replacing
the original macrophysics' 0.66.  The core network answers the block in
2.7 ms against `macrop_tend`'s 6.6; the step loop is still 3 s over the
original because the model's negative water triggers 446,000 `qneg3`
warnings that Fortran writes to the log.  What the model must learn is
still open: the bulk tendencies validate at R² 0.72 (`ptend_s`) and 0.46
(`ptend_q`), the cloud fractions at 0.84 to 0.94, the in-cloud water at
0.3, and the run drifts 5.8 K rms in temperature over fifty steps (the
tracer tendencies it leaves zero cost the isotopes their mass); the full
network, which learned the tracer tendencies too, drifts 1.5 K.  The block
contract's inputs and outputs are what the driver reads and writes; the
network's own inputs and outputs should be the physics -- the bulk
tendencies, the cloud fractions -- with the copies, the integrals and the
tracer tendencies formed from them, as `cldwat2m_macro` forms them.

What the driver does not yet have is the radiation driver's history entry:
the two drivers write 141 history fields from inside their arithmetic, and
a replay or a model does not, so `h0` differs in those diagnostics while
the state and the restart buffer are what they are.  The drivers' own cost
is unchanged: the blocks are the original routines called whole, as
`whole_drivers` called them.  The bookkeeping was nine small Fortran calls
a chunk here where radiation's is four; the next round made it two.

#### Both blocks as networks, and the bookkeeping as two calls

Three things were done to the cloud driver on 2026-09-14 afternoon, on a
new image (p27), and then measured.  The bookkeeping between and after the
blocks -- the flux terms, the scaling and application of the tendency, the
energy check, the precipitation means, the water-tracer fixer: nine small
Fortran calls a chunk -- became two entries of the mm handles,
`pycam_mm_finish_macro_v1` and `pycam_mm_finish_micro_v1`, each the glue's
own lines for the one substep this configuration runs, so a chunk of the
driver is now four Fortran calls when both blocks are the originals and two
when both are answered from Python (the driver still makes the nine calls
on an image without the entries).  The microphysics block got a core
network of its own the way the macrophysics block had (`--core`: 2053
features in, 1479 targets out, one field derived -- the old cloud
fraction the activation copies -- the cloud-borne aerosols and the water
tracers' precipitation left to the originals' record, 128 wide, 0.47M
parameters), and the emulator module holds one network per block, named by
`FREECAM_CLOUD_BLOCK_MACRO` and `FREECAM_CLOUD_BLOCK_MICRO`.  And a network's
answer is clamped before it is written: no constituent it tends may be
taken below zero over the step (`dq >= -q/dt`), which is where `qneg3`
would have clipped it a moment later, with a warning line each.

| run | in the slots | stage, a rank | step loop | model, ms a call | after 50 steps |
| --- | --- | ---: | ---: | ---: | --- |
| 7450698 | nothing armed (native-whole, p26) | 1.96 s | 15.95 s | -- | bit-for-bit |
| 7451930 | both originals through the driver, nine bookkeeping calls (p26) | 2.33 s | 16.52 s | -- | bit-for-bit |
| 7456732 | both originals, the two finish entries | 2.30 s | 16.16 s | -- | bit-for-bit |
| 7456998 | the same, the rotated views kept per plane | 2.28 s | 16.17 s | -- | bit-for-bit |
| 7457385 | the same, the runtime cache keyed on the pool again | 2.28 s | 16.29 s | -- | bit-for-bit |
| 7456791 | both blocks replayed, the finish entries | 0.51 s | 14.35 s | -- | state bit-for-bit; `rh0` differs in the diagnostics |
| 7456895 | the macrophysics core network (NumPy, clamped), the microphysics original | 2.25 s | 18.83 s | 2.8 | 3.07 K rms in T; 402,000 `qneg3` lines, none from the macrophysics |
| 7457487 | the same, on the round's final code | 2.30 s | 19.06 s | 2.9 | 3.07 K rms in T |
| 7457386 | **both core networks** (macrophysics 256 wide, microphysics 128 wide), NumPy, clamped | **1.24 s** | 15.47 s | 3.1 and 3.3 | 6.12 K rms in T; one `qneg3` line |

The driver's own cost is now measured rather than inferred.  A profile of
rank 0 over the fifty steps with both originals in place (7457073, the
`FREECAM_CPROFILE_RANKS` knob) puts the stage at 2.40 s: 1.89 s inside the
five Fortran calls a chunk (the microphysics driver 0.97, the macrophysics
driver 0.60, the activation 0.25, the two finish entries 0.07), 0.36 s
building the stage's runtime once at the first step (the handles, the
buffer tables, the reviewed descriptors read from their YAML), and 0.15 s
of Python across the hundred chunks -- the views, the input dictionaries,
the write-back: 1.5 ms a chunk.  That profile was first misread as a runtime
rebuilt every step, and the cache re-keyed on the access object the
driver hands the stage; that object is made anew each call, so the change
did what the misreading had feared and cost a second a rank (7457242,
7457244, 7457245 -- kept as records of the wrong turn); the key is the pool
again.

With both blocks answered by the core networks the stage costs 1.24 s a
rank against the original's 1.96 -- the two networks 0.74 s of it,
3.1 and 3.3 ms a call against `macrop_tend`'s 6.6 and the microphysics'
19.2 (`microp_aero_run` 3.7 and `microp_tend` 15.5 in the oracle's GPTL) --
and the step loop is 15.47 s against 15.95: the loop's gain is less
than the stage's because the drifted state costs the other processes time,
as it did in the radiation months.  The health counters are clean (one
`qneg3` line in fifty steps against 446,000 before the clamp) but the
physics is not: the run drifts 6.1 K rms in temperature over fifty steps,
the macrophysics network alone 3.1 K (5.78 before the clamp, 7454277).  The
networks are the open problem, as they are for radiation: the bulk
tendencies validate at R² 0.72 (macrophysics `ptend_s`) and 0.55
(microphysics), the constituent tendencies at 0.46 and 0.23, the
precipitation fluxes at 0.94, the effective radii at 0.1 to 0.3.  What a
network for either block should learn and how -- the physics only, with
the copies and integrals formed from it; per-level normalisation;
a loss in the units the state feels -- is the next work, and the
mechanism around it is complete: capture, replay, model, verify, census,
two Fortran calls a chunk.

#### The walk, bound once: what the fine-grained path costs when nothing is rebuilt per call

The statement-by-statement walk of the cloud stage -- the transliteration
of tphysbc's stage 7 with the macrophysics, aerosol activation and
microphysics drivers each walked in turn, the `legacy-python` policy -- was
the path this document's earlier sections priced at a third more than the
Fortran step and set aside.  On 2026-09-15 it was measured again and taken
apart, on the premise that the cost was not the form but the work Python
did around each of its fifty-one kernel calls a chunk.  The baseline on
the p27 image (7473911) put the stage at 4.03 s a rank against
native-whole's 1.96 and the step loop at 18.33 s against 16.3; the region
profile of the same walk (7473829, every rank) split the 4.03 into 2.45 s
inside the Fortran regions, 0.52 s of copies between views and scratch,
and 1.05 s of Python between the regions, and a profile of rank 0 named the
Python.

| run | change | stage, a rank | step loop | call sites re-resolved, 50 steps |
| --- | --- | ---: | ---: | ---: |
| 7473911 | the walk as it was | 4.03 s | 18.33 s | -- |
| 7474018 | call sites resolved once per set of objects handed; probe arguments and history addresses kept | 3.95 s | 18.09 s | -- |
| 7474195 | the bound call's post-call check compares shapes, not addresses read through ctypes | 3.81 s | 17.86 s | -- |
| 7474311 | the macrophysics tracer rates written in place (a 1.3 MB copy in and out, eleven times a chunk) | 3.33 s | 17.61 s | -- |
| 7474414 | every output written in place, in all four walks | 3.22 s | 17.32 s | -- |
| 7474655 | constituent lanes and the tracer sum kept per array; the bound-call table sized to the rate kernel | 3.22 s | 17.43 s | 830 |
| 7474686 | the drivers' state copies kept between calls (image p28) | 3.09 s | 17.01 s | 486 |
| 7475341 | the step's inverse and the tracer index slices kept as objects | 3.08 s | 17.06 s | 440 |
| 7475430 | a buffer view kept per time plane, so the alternating plane is not a new object | 3.06 s | 16.98 s | 455 |
| 7475549 | the step's scalar kept as an object at its two call sites; the record names what churns | 3.06 s | 17.20 s | 395 |

Every run is bit-for-bit with the oracle.  What each change removed:

- **The call site, resolved once.**  A kernel call decided on every call
  which of its arguments could be read in place, copied the others into
  scratch, keyed the bound call by every argument's address read through
  `ndarray.ctypes` (a microsecond each, forty arguments), and copied the
  outputs back.  The walk hands the same view objects on every call of a
  chunk -- the runtime's view caches keep an array while its storage
  stays -- so the decisions are made once per set of objects and kept on
  the plan (`_PreparedCall`); a repeat call is the copies, the bound
  invocation and the copies back.  The probe a view is fetched through and
  the encoded name and address a history call hands over are kept too.
- **The post-call check.**  The bound call verified after every call that
  no array's address, shape or dtype had changed, reading each address
  through ctypes: forty arguments, sixty microseconds, on a kernel whose
  arithmetic takes five.  A view's address cannot move underneath it; its
  shape can be reassigned, and that is what is checked now.  This was the
  largest single item inside what the region profile had counted as
  Fortran time.
- **Outputs in place.**  Every output of a carved kernel went through
  scratch and was copied back to its target, live lanes only, so that a
  kernel writing every lane could not touch a view's padding.  But a
  transliteration's output targets are, by construction, the storage the
  source statements write -- the driver's locals through the handles, the
  buffer fields it points into -- and the original writes every lane of
  them that the kernel writes.  Handed the storage itself (`in_place`,
  `ALL_OUTPUTS` in the four walks), the kernel leaves in every lane what
  the original left; the copies fell from 0.52 s to 0.05, and the
  macrophysics tracer rates alone, a six-dimensional array of 1.3 MB copied
  in and out around each of six calls a chunk, were half a second.
- **Objects that were new every call.**  A slice of a constituent out of a
  kept view, a sum formed into a fresh array, and the state copy the
  drivers allocate on entry and free on exit: each hands the kernel a new
  object and has its call site resolved again (`prepared` in the record's
  `call_sites` counts them).  The slices and the sum are kept per array;
  the state copies are kept in the handles modules from call to call,
  their live columns rewritten by the copy's own statements without the
  allocation (`pycam_state_copy`, generated from `physics_state_copy`).

What is left, and what the walk's floor is.  The region profile of the
walk with every output in place (7474415) puts the Fortran regions at
1.90 s a rank -- the original stage's own 1.96 -- and the copies at 0.05;
everything above that is Python around fifty-one kernel calls and about
two hundred and forty view probes a chunk: the call sites still resolved
again (the tendency object the driver allocates and frees around each
kernel, and two sites handed the step counter and the surface lanes,
which the record's `call_sites.churn` now names: about 400 of the 5,100
calls a rank per fifty steps, a few hundredths of a second), the probe a view costs even when the storage
has not moved (about 0.2 s), the driver's timer and trace record around
each bound kernel (about 0.1 s), the walks' own statements (about 0.1 s).
The walk stands at 3.06 s a rank for the stage against 1.96 and
16.98 to 17.20 s for the step loop in two runs against 16.00 on the same
image (7474687) --
half of the fine-grained path's overhead removed (2.07 s to 1.10), the
loop within six percent of the Fortran's where it was fourteen -- with every kernel call site still a slot a model can take.
Two levers remain and are not free: one Fortran entry that answers every
view of a chunk in one crossing (an image change), and a compiled glue
(Cython, ahead of time) for the walk's own statements and the driver's
timers, which is a policy decision rather than an engineering one.

**The compiled glue, tried.**  The second lever was measured the same
afternoon rather than argued.  `tools/build_glue_trial.py` compiles the
modules the walk runs through -- the stage runtime, the Fortran adapter,
the physics buffer, the four walks -- as they are, in Cython's pure-Python
mode, and builds `freecam/core/_glue.pyx`: direct callers for the image's
hottest entries (the bound kernel call, the two view probes, the buffer's
two accessors, the history call), which the Python modules use when the
extension is importable and fall back from otherwise.  The compiled
objects sit beside the sources, ignored by git; nothing in the install
changes, and every unit test passes either way.  With both layers built
the walk ran bit-for-bit at 3.04 s a rank for the stage and 17.17 s for
the step loop (7476883) against 3.06 and 16.98 to 17.20 uncompiled: the
compiled glue changes nothing measurable.  On the login node the same
paths had said as much -- a resolved kernel call site costs 2.3 µs
compiled or not, because its time is already inside NumPy's and ctypes'
C code; the direct callers took a forty-argument bound call from 4.5 to
2.6 µs, which over 5,100 calls is a hundredth of a second.  The 1.1 s
above the Fortran is therefore not the interpreter running these modules.
The trial also cost sixteen dead gates (7475712 to 7476822, kept as
failure records): built through the site's compiler wrappers, the
extensions linked a second MPI and a libfabric into the process before
mpi4py initialised its own, and MPI_Init failed on scattered ranks -- a
pattern that looked, for an afternoon, exactly like broken nodes.  The
build script now uses the bare compiler and refuses a module that needs
more than libc.

## Where it stands

| Kernel | Owner | Contract | Runner pause | In-model gate | Loop |
| --- | --- | --- | --- | --- | --- |
| `mmacro_pcond` | Macrophysics (in the cloud stage) | reviewed | yes, validated | segmented, bit-for-bit; alone and together with `micro_mg_tend`; the whole macrophysics driver is a block of the cloud stage's Python driver (`cloud_block.MACRO_BLOCK`): captured (7453135) and replayed bit-for-bit in state (7453150, 7453151); with the bookkeeping as one finish entry a block (p27) the originals through the driver stay bit-for-bit at 2.28 s a rank for the stage (7456732, 7457385) and the replay at 0.51 s (7456791); the block answered by a core NumPy network at 3.1 ms a call (7457386, with the microphysics network; 7456895 alone; not bit-for-bit by design) | complete |
| `micro_mg_tend` | Microphysics (in the cloud stage; hooked: a Fortran-bound hook over its packed arrays, bindable to a model, not pausable) | reviewed | yes, validated (the runner's pause) | segmented, bit-for-bit (100 pauses in 50 steps); alone and together with `mmacro_pcond` (200 pauses); a null model bound at the hook answered all 51,200 calls inside the image (7394423), and in shadow the p18 image stays bit-for-bit at 0.56 ms a call (7400408); four trained MLP surrogates (64 to 512 wide) run in shadow on p19, bit-for-bit, pricing the core at 1.4 ms a call and every network at 1.5 to 6.7 ms (7401282, 7401281, 7401311, 7400992), and live (7400993, 7401059, 7401232, not bit-for-bit, slower than the original); the driver with its activation and tendency sum is a block of the cloud stage's Python driver (`cloud_block.MICRO_BLOCK`): captured and replayed bit-for-bit in state (7452498, 7453150, 7456791); the block answered by a core NumPy network at 3.3 ms a call against the two drivers' 19.2, both blocks as networks running the stage in 1.24 s a rank against the original's 1.96 (7457386, not bit-for-bit by design) | open: no captured calls replayed through its standalone image yet; as a speed target, closed: the core costs 1.4 ms a call |
| `rad_rrtmg_sw` | Radiation (split, pausable) | frame descriptor | yes, validated | segmented, bit-for-bit (50 pauses in 50 steps); alone, with `rad_rrtmg_lw`, and with every other exposed kernel paused in one run; the whole radiative branch around it is a process slot: captured (7417373) and replayed (7417486) bit-for-bit in state through the transcription, and answered inside the image by a compiled emulator at the runner's skeleton slot (shadow 7418443 bit-for-bit; live 7418444, not bit-for-bit by design) and by TorchScript transformers through FTorch (shadow 7431802 and 7431990 bit-for-bit; live 7431803, 7431991), and by the 64-wide transformer written out in Numba as a plugin (shadow 7439472 bit-for-bit; live 7439535) and, for its block contract, through the Python driver (7439603) | open: capture and replay of the core itself |
| `rad_rrtmg_lw` | Radiation (split, pausable) | frame descriptor | yes, validated | segmented, bit-for-bit (50 pauses in 50 steps); alone, with `rad_rrtmg_sw`, and with every other exposed kernel paused in one run; the whole radiative branch around it is a process slot: captured (7417373) and replayed (7417486) bit-for-bit in state through the transcription, and answered inside the image by a compiled emulator at the runner's skeleton slot (shadow 7418443 bit-for-bit; live 7418444, not bit-for-bit by design) and by TorchScript transformers through FTorch (shadow 7431802 and 7431990 bit-for-bit; live 7431803, 7431991), and by the 64-wide transformer written out in Numba as a plugin (shadow 7439472 bit-for-bit; live 7439535) and, for its block contract, through the Python driver (7439603) | open: capture and replay of the core itself |
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
| `cldfrc_fice` | Macrophysics in the cloud stage (hooked; called inside the compiled `zm_conv_evap`; a model block: `t` in, `fice` and `fsnow` out) | function contract; standalone image | yes, validated | a hoisted copy of `zm_conv_evap` was not bit-for-bit (7343258, failure record kept); reached through a hook that redirects `zm_conv.o`'s call, machine code untouched: answered by the original at the hook, bit-for-bit, 51200 pauses in 50 steps (7343708); the first hook gate (7343594) failed on a duplicate fiber-body symbol, failure record kept; every frame captured at the hook (7343811) replayed bit-for-bit through its standalone image; the kernel written in Python, compiled with Numba and bound at the hook answers every call bit-for-bit with the oracle (7402505; shadow 7402506) | complete |
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
