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

## Where it stands

| Kernel | Owner | Contract | Runner pause | In-model gate | Loop |
| --- | --- | --- | --- | --- | --- |
| `mmacro_pcond` | Macrophysics (in the cloud stage) | reviewed | yes, validated | segmented, bit-for-bit; alone and together with `micro_mg_tend` | complete |
| `micro_mg_tend` | Microphysics (in the cloud stage; hooked: a Fortran-bound hook over its packed arrays, bindable to a model, not pausable) | reviewed | yes, validated (the runner's pause) | segmented, bit-for-bit (100 pauses in 50 steps); alone and together with `mmacro_pcond` (200 pauses); a null model bound at the hook answered all 51,200 calls inside the image (7394423), and in shadow the p18 image stays bit-for-bit at 0.56 ms a call (7400408); four trained MLP surrogates (64 to 512 wide) run in shadow on p19, bit-for-bit, pricing the core at 1.4 ms a call and every network at 1.5 to 6.7 ms (7401282, 7401281, 7401311, 7400992), and live (7400993, 7401059, 7401232, not bit-for-bit, slower than the original) | open: no captured calls replayed through its standalone image yet; as a speed target, closed: the core costs 1.4 ms a call |
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
