# Decoupling the physics kernels

The aim is one shape for every active physical process of the PI-atm step: a
Python process class that owns the process's place in the workflow, its
parameters and its kernel slots, with the numerical kernels behind a declared,
replaceable, validated interface. The same kernel can then be called on its
own, sampled into training data, or replaced in the full model by a Python
function or a trained network, under one contract and one replacement
configuration. What runs when nothing is replaced is the original Fortran,
whole; what runs when something is replaced is the original Fortran up to the
replaced call, then the replacement, then the original Fortran again.

This page says how the pieces fit, what each way of answering a slot costs,
what the first surrogates taught, and where the work stands. The ledger is
[`validation/physics_kernel_decoupling.json`](../validation/physics_kernel_decoupling.json),
written by `tools/build_physics_kernel_coverage.py` from the repository's own
records and checked current by the unit suite; the
[progress dashboard](https://a85tract.github.io/freeCAM/progress/) publishes
it. How to replace a process or a kernel from a script or a notebook is in
[usage.md](usage.md#replacing-a-process-the-model-already-owns-as-a-class);
every contract is listed in the generated [contracts.md](contracts.md); what
replacing each kernel can give back is in
[kernel_replacement_returns.md](kernel_replacement_returns.md).

## The shape of a process

```
Python process class
    ├── parameters, lifecycle, place in the workflow
    ├── kernel contracts and kernels[...] slots
    └── execution
          ├── nothing replaced   -> the original Fortran process, called once
          └── something replaced -> Fortran to the replaced call
                                    -> the replacement: inside the image at a hook,
                                       or a frame to Python at a pause
                                    -> write-back, checked -> resume
```

**The kernel slots.** `stage.kernels[name]` is the only registry of what
computes a kernel. Assigning into the mapping, naming a model file with
`surrogate=`, and binding over the method (`stage.mmacro_pcond =
MethodType(fn, stage)`, which `NativeStage.__setattr__` turns into a
`MethodKernel`) all land there, so the walk, the runner's frame and the
single-column caller reach the same function. `OriginalKernel()` in a slot is
the validation replacement: the replacement path runs, the original kernel
answers through it, and bit-for-bit output proves the frame and the
write-back.

Who can answer a slot, and where:

| answerer | where it runs | crossings a call | used for |
| --- | --- | --- | --- |
| nothing (`None`) | the original stage, whole | none | the default |
| `OriginalKernel()` | the original kernel, through the replacement path | as the path | the gate of that path |
| a Python callable over the frame | Python, at the runner's or the hook's pause | three | frame capture, quick experiments |
| a function over the kernel's arrays, or `compile_kernel(...)` | compiled by Numba on every rank, called by Fortran at the hook | none | a kernel written in Python |
| `fc.NativeModel("model.pt")` | TorchScript through FTorch inside the image, at the hook (`device="cuda"` on a CUDA image) | none | a trained network |
| any of the last two in shadow | runs on every call for its cost while the original answers | none | timing a model on the model's own calls, bit-for-bit |

Which path the stage takes each step is its execution policy
([usage.md](usage.md#replacing-one-kernel-inside-a-process) has the table):
`auto` runs the original whole while nothing is replaced, the image's
answerer where a hook is bound, segmented where the runner pauses at every
replaced kernel, and the Python transliteration (`legacy-python`) otherwise.

**The read-only description.** `stage.describe_kernels()` returns one record
per kernel: the owning class, whether the runner pauses at it, whether that
pause has passed a gate, the reviewed contract's inputs and outputs when one
exists, what is in the slot now, and how many times a model answered for it in
this run. The run summary written by the command line carries the same rows
under `stage_execution`, and the Workflow Builder consumes them rather than
keeping a list of its own.

## Where the image stops

### Segment runners

Where the image can pause is declared in
[`native/pi_cam/segment_runners.yaml`](../native/pi_cam/segment_runners.yaml):
for each stage, the Fortran module, its generator, the descriptor its frames
are decoded with, the kernels it pauses at, and the gate records that
validated each pause. The backend (`freecam.pi_cam.native`) asks the manifest
which stage has a runner instead of knowing one by name; the stages ask it
whether a replacement can run segmented; the Workflow Builder reads it to say
which kernels are bindable and which are validated. A runner is
`ImageSegmentRunner(library, spec)`, one class for every prefix.

`tools/pi_cam_pausable.py` makes the runners from specs under
`native/pi_cam/pausable/`. The action's tphysbc or tphysac block and the
drivers it calls are hoisted verbatim into modules whose locals are module
state, cut into pieces at the kernel calls and at the `if`, `do` and `select`
statements the runner re-expresses, and every paused call's arguments are
served as a frame in the callee's own order, with its intents and declared
shapes (an element or a section passed by sequence association is served
with the shape the callee sees). The runner also runs the very call on
request, so a gate can answer a pause with the original and still exercise
the frame's write-back. The pinned source ranges are hashed into the spec, so
a source that moves fails `--check`. Nothing numerical moves: the pieces are
the pinned text, checked line for line, compiled with the case's own flags.
`PausableStage` (`freecam.physics.pausable`) owns each action, runs it whole
when nothing is replaced, and refuses a Python walk it does not have.

| stage (class) | pauses at | how it is built |
| --- | --- | --- |
| cloud macro/microphysics (`CloudMacroMicrophysics`) | `mmacro_pcond`, `micro_mg_tend`, `cldfrc_fice` | the stage-7 runner; `micro_mg_cam_tend` held whole in pieces (the head, the packer's five procedures, the tail) in `pycam_micro_handles`, called around the substep loop |
| dry adjustment (`DryAdjustment`) | `dadadj` | the generator |
| shallow convection (`ShallowConvection`) | `compute_uwshcu_inv` | the generator |
| radiation (`Radiation`, split) | `rad_rrtmg_sw`, `rad_rrtmg_lw` | `pycam_radt` hoists `radiation_tend` whole; pauses inside the `dosw`/`dolw` blocks and their `icall` loops, the RRTMG state served component by component; the stage keeps its two leaves around the driver (control patch 0041) |
| deep convection (`DeepConvection`) | `zm_convr`, `zm_conv_evap`, `momtran` | a chain of three hoisted routines (the tphysbc block, `convect_deep_tend`, `zm_conv_tend`); control patch 0044 makes zm_conv_intr's per-chunk arrays readable |
| convective tracer transport (`ConvectiveTracerTransport`, a leaf) | `convtran` | transports with exactly what deep convection left in those arrays, never a copy |
| vertical diffusion (`VerticalDiffusion`) | `compute_tms`, `compute_eddy_diff`, `compute_vdiff` (two sites), `virtem` | the tphysac block and `vertical_diffusion_tend`; patches 0043 (the friction velocity and Obukhov length dry deposition reads) and 0045 |
| gravity wave drag (`GravityWaveDrag`) | `gw_drag_prof` | `gw_tend`'s orographic block, the only source active here; the wave band and pressure coordinates by component; patch 0046 |
| aerosol wet deposition (`AerosolWetDeposition`, a leaf) | `wetdepa_v2` (two sites) | inside the mode, phase and species loops; the driver's `cycle` and early `return` carried out by the runner |
| aerosol dry deposition (`AerosolDryDeposition`, a leaf) | `modal_aero_depvel_part` (four sites) | the cloud droplets before the mode loop, each mode inside it |
| chemistry (`ChemistryTendencies`, a leaf) | `gas_phase_chemdr` | the whole gas-phase driver as one kernel, a forty-odd-argument frame; patches 0047 and 0048 |

The frame's rules, each learned on one of these: a kernel called at several
sites pauses at each and every site serves the same frame, an optional the
site omits as an empty slot; module arrays are addressed through a TARGET
dummy; an intent(out) scalar (a gathered column count) is served where it
lives so a model can answer it; an automatic array is sized by the runner's
own extents where the callee's would name something only the callee imports,
and a driver's automatic arrays sized by the chunk are module allocatables
re-sized when the chunk changes; a field selector with private components and
a procedure argument are passed by the original and served as nothing; the
frame ABI carries five extents per slot (the transport's water-tracer ratio
is rank four). Control patches 0043-0048 are accessibility statements only,
generated by one module-state generator.

The eleven actions whose bodies do no numerical work in this configuration
are `InertStage`s: a class, no kernel, and one gate with all of them disabled
(below).

### Hooks inside compiled routines

A kernel called from inside a compiled routine cannot be reached by hoisting.
For those the image redirects the call: the definition in the oracle's object
is weakened and a hook takes its symbol (`fluxbelowinv`,
`instratus_condensate`: two PC32 relocations), or the caller's relocation is
pointed at the hook with its machine code untouched (`cldfrc_fice` inside
`zm_conv.o`). [kernel_api_closure.md](kernel_api_closure.md) describes the
mechanism and which calls it can reach. Eleven kernels have hooks today
([`native/pi_cam/hooks.yaml`](../native/pi_cam/hooks.yaml), each listed in
[contracts.md](contracts.md)): `cldfrc_fice`, `fluxbelowinv`,
`instratus_condensate`, `micro_mg_tend`, `compute_tms`, `compute_uwshcu_inv`,
`mmacro_pcond`, `zm_conv_evap`, `momtran`, `zm_convr` and
`compute_eddy_diff`.

A hook with the C binding can pause: the stage runs whole on a fiber, the
hook yields the frame from inside the compiled routine, and `frame`,
`resume` and `original` are served by the hook table. A pause costs about a
tenth of a millisecond (9000 per rank in fifty steps added under a second).
A hook cannot be combined with a runner-level pause in the same run (the
manifest's `within`). A hook with the Fortran binding (`micro_mg_tend`: the
callee's own kinds, default logicals, an assumed-length character, pointer
arrays) has no C interface and no frame: it cannot pause, only be bound to a
model, and arming it is refused. It receives packed arrays, `(mgncol, nlev)`
for the columns that hold cloud, so it sizes every array by the callee's
integer dummies; a first build that used the contract's constants corrupted
memory (`pi_cam_pausable_micro-ftorch-excl_50step_failure.json`).

## Answering inside the image

A replacement answered from Python costs the round trip: about 3 ms for the
29-argument `instratus_condensate` frame. The surveyed practice (FTorch in CAM
and ICON, pytorch-fortran and FTorch in E3SM-MMF, Infero in the IFS, the
Fortran-Keras Bridge in SPCAM) is to run the model inside the Fortran process
and keep Python out of the step; the hooks allow two ways.

**A TorchScript model through FTorch.** A hook whose `hooks.yaml` entry has a
`model` block -- the contract arguments the forward takes, in order, and the
outputs it returns -- can be bound to a TorchScript file
(`pycam_hooks_bind_model_v1`). The hook wraps the kernel's arrays as tensors
where they live, runs the model through
[FTorch](https://github.com/Cambridge-ICCS/FTorch), and writes the live
columns of the outputs back: no fiber, no frame, no Python, one crossing a
step whatever is replaced. The image links FTorch and the libtorch of the
checkout's own `torch` (`build_pi_cam_devices.py --ftorch-root`, see
[installation.md](installation.md)). The model must carry its pre- and
post-processing (feature scaling, gates, the closure) because the hook hands
it the kernel's raw arguments. At bind the hook runs one forward on zero
tensors of the contract's extents, so the first call in the step loop is not
the warm-up (image p18: the warm-up 83 ms a rank, paid at bind).

**A compiled plugin.** A C function of the hook's plugin interface, bound by
address (`pycam_hooks_bind_plugin_v1`), is handed the model block's arguments
as pointer and extent tables and writes its outputs into the same
temporaries. `compile_kernel(hook, function)` (`freecam.physics.numba_kernel`)
generates the adapter from the hook's model block and contract, compiles the
user's function with `numba.njit` and the adapter as a `numba.cfunc`, and
returns a `NativePlugin` for the slot; `--kernel-plugin NAME=file.py:function`
does the same from the command line.
`examples/plugins/numba_kernels/cldfrc_fice.py`, the ice-fraction kernel
written statement for statement in Python, answered every one of the 50,176
calls after the bind bit-for-bit with the oracle (7402505), at 8 µs a call.

**Shadow and the hook timers.** A bound model or plugin can run in shadow:
it runs on every call for its cost and its answer is discarded while the
original answers, so the run stays bit-for-bit and prices the model on the
model's own calls. The hooks time themselves per rank (the model branch, the
forward alone, the whole call, the first call, the warm-up); the record sums
them over the ranks and keeps the slowest rank's value beside each sum.

The same hook answered four ways (`instratus_condensate`, 50 steps, 512 ranks):

| how the hook is answered | a call | record |
| --- | ---: | --- |
| the original `instratus_condensate` | 5 µs | 7402507 |
| a compiled plugin (pointer tables, the adapter, the call) | 0.4 µs | 7402507 |
| a TorchScript surrogate through FTorch | 350-410 µs | 7401311, `instratus-ftorch-excl` |
| a Python callable at the pause | 3 ms | `instratus-surrogate` runs |

**Rules that came out of the gates.**

- `--kernel-model NAME=PATH` (or `PYCAM_KERNEL_MODELS` in the gate job) puts
  a cloudpickled model or a TorchScript archive into the named slot and
  records it by file name and content hash. Every rank loads and re-pickles
  it, and the install refuses a payload that hashes differently across ranks:
  a model must pickle the same everywhere (use ordered containers; the gate
  job fixes `PYTHONHASHSEED`;
  `pi_cam_pausable_instratus-surrogate_50step_failure.json`). A compiled
  plugin stays out of the pickle and carries no address, only an identity
  resolved in each process (failure records 7402090-7402092, 7402200-7402202).
- `PYCAM_NO_VERIFY_EXPORTS` lets a run with a model in a slot past the
  replay's export check, since a model's answer differs from the oracle's at
  the first export.
- Every gate run counts the log's water-isotope errors and QNEG3 resets into
  `<summary>.health.json`; the original counts zero of each, so a model's
  number there is entirely its own.
- A bound model and a Python replacement cannot share a hook, and a native
  model cannot stand at a kernel that is not a hook.
- Numba compiles a plugin on every rank at start (about 50 s, in the record's
  initialisation time), and compilation and frozen weights add about 0.35 GB a
  rank: plugin runs on four nodes ask for `mem=235GB` a node (7418223 and
  7418224 were killed for memory).
- A class in `freecam` may not define a method twice (a unit test); a second
  `prepare_segmented` once silently unbound a plugin, and the run was
  bit-for-bit for that reason (7418304, 7418305).

## Process slots: a whole process as one replaceable unit

Kernel by kernel, a network cannot make this model faster on these nodes (see
[What the surrogates taught](#what-the-surrogates-taught)): the cores are
cheap and a network's price is its bytes. What remains open for a learned
replacement is a process that is expensive as a whole. Two stages offer one.

### Radiation

`Radiation` offers the computing branch of a radiative step -- the optics, the
two RRTMG cores and their diagnostics -- as a process slot
(`freecam.physics.radiation_process`). The contract is the driver's: before
the branch it has the state, the buffer's cloud fraction and the optics'
inputs, the surface albedos and upward longwave, the cosine of the zenith
angle and the RRTMG gas profiles; the branch leaves the two heating rates and
the ten surface and top fluxes the coupler and the energy check read
(`fsns`, `fsnt`, `flns`, `flnt`, `fsds`, `sols`, `soll`, `solsd`, `solld`,
`flwds`). [contracts.md](contracts.md#radiation-branch) lists them. The slot
can be answered four ways:

- **From Python, between the stage's two halves.**
  `RadiationProcessCapture` records inputs and outputs on every radiative
  step (`--radiation-capture DIR`); `RadiationReplay` answers with a capture
  (`--radiation-model replay:DIR`); `RadiationProcessModel` answers with a
  function over the inputs by name (`--radiation-model path.py:function`).
  The capture is bit-for-bit (7417373: 25,600 records, 9.1 GB). The replay
  is bit-for-bit in every restart file and the history file (7417486); the
  history restart differs in exactly the 26 fields the branch writes to
  history that a model has no source for -- aerosol optical depths and
  burdens, clear-sky and top-of-atmosphere fluxes, cloud forcings, incoming
  solar -- listed in
  `pi_cam_pausable_rad-process-replay3_vs_oracle_50step_variables.json`.
- **Inside the image, at a skeleton slot.** The pausable runner's hoisted
  `radiation_tend` marks the branch's `if (dosw .or. dolw)` node (`if: 875`,
  `slot: pycam_rad_process_answer(...)` in
  `native/pi_cam/pausable/radiation.yaml`). The support module
  `pycam_rad_process` hands a bound answerer 46 inputs and takes 12 outputs
  as pointer and extent tables (Python's `TABLE_INPUTS` and `TABLE_OUTPUTS`
  mirror the order; a test pins them), writes them and their history where
  the driver does, and times its calls. A Numba plugin binds with
  `--radiation-plugin` (`compile_radiation_plugin`), a TorchScript model
  through FTorch with `--radiation-torch-model`; both run in shadow too. A
  control patch on `radiation.F90` was written and withdrawn: the image links
  the oracle's numerical objects unchanged, and the hoisted driver is the one
  copy this repository owns.
- **From Python, at a pause at the branch** (image p23). The spec's
  `process_slot` parks the runner at the top of the branch and hands Python a
  frame of the same 46 inputs and 12 outputs; three crossings a chunk, about
  9 ms a chunk on top of the shortwave pause path (7434783, bit-for-bit with
  the original asked for, 25,600 pauses).
- **The Python driver** (`--radiation-block-model`, mode `python-driver`).
  The block is everything numerical -- the zenith angle, the optics, the two
  cores, the heating-to-tendency step -- and the model takes only what the
  driver had in memory (`radiation_process.BLOCK_INPUTS`); Python does the
  reading and writing, with `radheat_tend` and one history entry
  (`pycam_rad_process_history_v1`, image p24) as the only Fortran calls. A
  replay through it (7436003) leaves `cam.r`, `cam.rs` and `h0` bit-for-bit
  and `rh0` different in the same 26 diagnostics. The notebook route is the
  same path: `driver.processes["radiation"].process = model`; a replay set
  from the process table produced every file byte for byte the same as the
  command line's (7438656 against 7436003; `pi_cam_process-table-replay_50step.json`).

### The cloud stage

The cloud macro/microphysics stage, under one macro/micro substep, is two
compute blocks with bookkeeping between and after them: the macrophysics
driver (`macrop_driver_tend`, with `mmacro_pcond` inside), and the
microphysics driver with the aerosol activation that feeds it and the sum of
their tendencies (`microp_aero_run`, `micro_mg_cam_tend`,
`physics_ptend_sum`). `freecam.physics.cloud_block` draws each block's
contract: what the driver has in memory before its arithmetic and what it
leaves behind, buffer fields included, plus the cloud-borne aerosol fields
and the water tracers' surface precipitation, which are registered per
constituent and resolved at run time
([contracts.md](contracts.md#cloud-block-macro)).

`CloudMacroMicrophysics` has a slot per block -- `process` (macrophysics) and
`micro_process` -- a `block_capture`, and the mode `python-driver` they switch
on (`--cloud-block-model`, `--cloud-micro-block-model`,
`--cloud-block-capture` beside `--cloud-macro-micro-python`; in a notebook,
`driver.processes["cloud_macro_microphysics"].process`). Per chunk the driver
takes its views once, runs each block -- the original driver in place, or the
slot's answer written back through a tendency object allocated as the driver
would have left it (`pycam_mm_ptend_init_v1`) -- and the bookkeeping as two
entries, `pycam_mm_finish_macro_v1` and `pycam_mm_finish_micro_v1` (image
p27). A capture records each block's inputs in single precision and its
outputs exactly, with an exact digest of every input, so a replay names the
first input and step at which its run left the captured one; `verify:DIR`
or `census` in a slot runs the original in place and names the outputs it
produced differently, or every buffer field it changed that the contract
does not list. Seven replays failed before the last passed, each on a field
the contract had missed (the cloud-borne aerosol registry, the time-rotated
buffer planes, the water tracers' precipitation index array, the convective
cloud fractions `cldfrc` writes); each is a failure record.

| run (50 steps) | in the slots | stage, a rank | step loop | against the oracle |
| --- | --- | ---: | ---: | --- |
| 7450698 | the class installed, nothing armed (native-whole) | 1.96 s | 15.95 s | bit-for-bit |
| 7456732 | both originals through the Python driver, two finish entries | 2.30 s | 16.16 s | bit-for-bit |
| 7453135 | both originals with the capture around them (51,200 calls a block, 19 GB) | 2.7 s | 16.66 s | bit-for-bit |
| 7456791 | both blocks replayed | 0.51 s | 14.35 s | state bit-for-bit; `rh0` differs in the drivers' diagnostics |
| 7457386 | both blocks answered by core networks (NumPy, clamped) | 1.24 s | 15.47 s | not by design: 6.1 K rms in T after 50 steps |

**What the process slots still lack.** History written from inside the
replaced arithmetic. Radiation's slot writes the heating rates and fluxes to
history as the driver does, but a model has no source for the 26 branch
diagnostics. The two cloud drivers write 141 history fields from inside their
arithmetic, which a replay or a model does not, and there is no history entry
for them yet ([plans/interface_completion.md](plans/interface_completion.md)).

## What each path costs

**Radiation, a month** (1,488 steps, replayed boundary, image p24, whole
235 GB nodes; drift is the last day's global mean against the oracle's
month):

| run | answerer | step loop | against the original | model, a call | day 30: net shortwave at the top, column water vapour, lowest-level T |
| --- | --- | ---: | ---: | ---: | --- |
| 7452662 | nothing bound | 400.2 s | -- | -- | bit-for-bit |
| 7452663 | 256-wide MLP as a Numba plugin at the slot | 366.2 s | -8.5% | 1.3 ms | -4.8 W/m², -1.5 kg/m², -0.9 K |
| 7454279 | the MLP through the Python driver, forward in NumPy | **359.5 s** | **-10.2%** | 1.5 ms | -55 W/m², -7.9 kg/m², -11.8 K |
| 7452664 | the MLP through the Python driver, Numba | 371.9 s | -7.1% | (6 s compiling at the first call) | as above |
| 7452666 | 64-wide transformer as a Numba plugin | 385.4 s | -3.7% | 12.6 ms | -4.4 W/m², -1.2 kg/m², -1.0 K |
| 7452667 | the transformer through the Python driver, libtorch | 374.3 s | -6.5% | 6.9 ms | -243 W/m²; 45,308 "BIG ERROR" lines |

The paths are within a few percent of each other with the same network: what
runs the arithmetic and what sits around it decides the order, not which side
makes the call. The Python driver needs no compilation step -- any callable,
and three NumPy matrix products through the single-threaded BLAS made it the
fastest. Through FTorch at the slot the transformer cost 7.1 ms a call over
fifty steps (7431803); libtorch from Python spends most of its extra time
turning Fortran-ordered views into contiguous tensors. The drift column is
the model's, not the path's: the block models, made to learn the zenith angle
instead of being given it, lose the terminator and the state with it.

**The cloud stage's statement-by-statement walk** (`legacy-python`, the
transliteration of the stage with every kernel call a slot) was taken apart on
2026-09-15: from 4.03 s a rank for the stage and 18.33 s for the fifty-step
loop (7473911, image p27) to 3.06 s and 16.98-17.20 s (image p28), against
the original's 1.96 s and 16.00 s on the same image (7474687), every run
bit-for-bit. What was removed: call sites decided on every call (now resolved
once per set of objects handed, `_PreparedCall`); a post-call check that read
forty addresses through ctypes (now it compares shapes); outputs copied
through scratch (now written in place, `in_place`/`ALL_OUTPUTS`, the copies
from 0.52 s to 0.05); and objects made anew every call (slices, sums, and the
state copies the drivers allocate, now kept, `pycam_state_copy`). The Fortran
regions are then 1.90 s, the original's own; the rest, about 1.1 s, is Python
around 51 kernel calls and about 240 view probes a chunk. Compiling the glue
(`tools/build_glue_trial.py`, Cython) changed nothing measurable (7476883):
the time is already inside NumPy's and ctypes' C code. The command line
freezes the heap out of the cyclic collector after initialisation
(`FREECAM_GC_FREEZE=0` leaves it alone), a few hundredths of a second.

**Each kernel, priced.** A run with every exposed kernel paused and answered
by the original prices each kernel by its own call, inside its
`FORTRAN:ORIGINAL:<kernel>` region (fifty steps 7479753 and the month 7479754,
image p28, both bit-for-bit). The seventeen paused kernels sum to 68 s a rank
of the 400 s month, seventeen percent of the step; three carry half of it
(`compute_uwshcu_inv` 6.5 ms a call, `rad_rrtmg_sw` 6.5 ms and `rad_rrtmg_lw`
5.5 ms, the two every other step). `macro_microphysics`, the most expensive
action at fourteen percent of the step, holds two kernels worth 2.5 percent;
the rest is the drivers' packing, buffer handling and history. The table, and
what a free replacement of each kernel would return after its path cost, is
[kernel_replacement_returns.md](kernel_replacement_returns.md).

## What the surrogates taught

### `instratus_condensate`: conservation is not the whole constraint

Every output of `mmacro_pcond` is derived from one internal call,
`instratus_condensate`, the saturation adjustment and stratus-fraction
closure, called twice per level (the relaxation iterations) and once for the
final state. Its last lines are the routine's own algebra -- `ql = al_st *
ql_st`, `qi = ai_st * qi_st`, `T = T0 - (latvap/cpair)(ql0 - ql) -
((latvap+latice)/cpair)(qi0 - qi)`, `qv = qv0 + ql0 - ql + qi0 - qi` -- so a
model that answers only the two stratus fractions and the two in-stratus
condensates and derives the rest by those lines conserves water and moist
energy exactly. On 288,000 captured calls (32 ranks of gate 7371232) the four
lines hold to round-off (`validation/pi_cam_instratus_condensate_identities.json`).

The first model in the slot (7371974; a NumPy MLP of 72,454 parameters
trained on eight million captured columns, the closure verbatim after it)
answered all 9000 calls per rank with no QNEG3 reset and no isotopic mass
error, but tripped the water-tracer check after the macrophysics
(`wtrc_apply_rates` in water_tracers.F90): 6731 `BIG ERROR` lines in fifty
steps where the original prints none.

Reading the two routines says why. The tracer path takes only the three bulk
tendencies (macrop_driver.F90, 1128-1133): the phase changes split by sign,
and the net water tendency `qvlat + qcten + qiten` as a vapor self-rate. A
rate is applied only when positive (water_tracers.F90, 1190), so a level whose
net water tendency is negative has that part dropped from its H2O copy -- the
whole of `diff`. Net water can only go negative through `positive_moisture`,
which repairs a negative vapor by borrowing from the layer below
(cldwat2m_macro.F90, 2250-2253); the original never triggers it in fifty
steps. The surrogate does, because the driver's linearisation of the next
relaxation iteration (`QQ`, cldwat2m_macro.F90, 905) takes the routine's
outputs as the saturated equilibrium state and caps the condensation with the
frozen half-step vapor (914-918): a predicted in-stratus condensate that is
not the saturation adjustment's lets that step drive the vapor negative, and
the repair follows. So the constraint the cut has to keep is saturation, not
only conservation: the next model should predict the fractions and take the
condensates from the saturation adjustment at the predicted fractions, which
`instratus_condensate`'s own iteration defines.

The month said the same at length (7373795,
`pi_cam_pausable_instratus-surrogate_1month_failure.json`): 983 steps, the
water-tracer check firing a steady 108 lines a step from the first day, the
other water checks growing from a handful to a thousand per three days, then
a segmentation fault on every rank in the same step, not diagnosed. The ten
two-day history files before it compare finite, at a median relative RMS of
0.71 of each field's own spread (`..._1month_drift.json`).

Bound through FTorch instead of answered from Python, the same network took
the cloud stage from 0.89 to 4.59 s a rank per fifty steps and the step loop
from 7.70 to 11.49 s (image p13, exclusive nodes;
`pi_cam_pausable_p13-whole_50step.json`,
`pi_cam_pausable_instratus-ftorch-excl_50step.json`): 0.41 ms a call against
2.8 ms at the pause, and still a 49 percent longer step for a kernel called
180 times a step.

### `micro_mg_tend`: the price of a network is its bytes

The deep GPTL profile ranks `microp_mg_tend` first at 10.7 ms a call, but that
timer wraps the whole of `micro_mg_cam_tend` -- buffer reads, packing,
unpacking and about a hundred history fields. The core itself, timed on the
same calls in shadow (image p19, exclusive nodes, all bit-for-bit), costs
**1.4 ms a call** -- 1.8 percent of the step loop. A null model at the
Fortran-bound hook costs 0.56 ms (7400408): the floor of replacing it inside
the image, with 115 tensors wrapped and 89 outputs written back.

Trained MLPs (over the model block's 26 input and 89 output arguments, from
the capture gate 7359460; 64 to 512 wide, R² of the median output 0.44 to 0.62) cost, in
shadow on the same calls:

| width, weights | through FTorch, ms a call | compiled forward (Numba), ms a call | the core, ms a call |
| --- | ---: | ---: | ---: |
| 64, 0.9 MB | 1.47 (7401282) | 0.87 (7404600) | 1.35-1.43 |
| 128, 1.7 MB | 2.25 (7401281) | 1.45 (7404599) | 1.38-1.43 |
| 256, 3.6 MB | 3.71 (7401311) | 2.92 (7404601) | 1.40-1.47 |
| 512, 7.6 MB | 6.71 (7400992) | -- | -- |

Standalone on one core the 512-wide network takes 1.47 ms; run by 1, 8, 32
and 128 processes at once on a compute node (job 7401108) it takes 1.47,
3.87, 6.21 and 6.38 ms, and the null model 0.20 to 0.23. Eight ranks share a
32 MB slice of L3, so eight copies of a 7.6 MB network spill it, and above 32
processes memory bandwidth is the limit: inside the image a network's cost is
set by its weight bytes, streamed from memory again after forty milliseconds
of other physics, not by its arithmetic. Live (`--kernel-model`), every
network made the step slower than the original although the core was
skipped (7400993, 7401059, 7401232): the state it produced sent the rest of
the physics down other paths and the water-tracer and QNEG3 checks wrote
300,000 lines to the log. On 128-rank nodes a model pays only where the core
costs several milliseconds a call and the weights fit the cache its
neighbours leave it, or where the inference runs on an accelerator the ranks
do not share. Taking libtorch out of the call saves 0.6 to 0.8 ms at every
width; the inference belongs in compiled code.

### Radiation: the mechanism works, the network is the open problem

A per-column MLP over 1,029 features (`examples/plugins/numba_kernels/train_rad.py`,
trained on a month of captures, 1,129,764 columns) validates at R² 0.96-0.97
for the shortwave heating, 0.87-0.90 for the longwave and 0.98-0.99 for the
top-of-atmosphere fluxes. Bound at the skeleton slot it answered all 761,856
calls of a month (7418518) at 1.31 ms a call and took nine percent off the
month's step loop (401.3 to 363.9 s against 7418517), with the isotope checks
firing thirteen times. Its science is not good enough to use: by day 30 the
global means drift (net shortwave 4.8 W/m² lower, outgoing longwave 4.2 W/m²
higher, the lowest level 0.9 K colder; `pi_cam_pausable_rad-month-slot_1month_drift.json`),
and it produces none of the 26 branch diagnostics. A level-token transformer
(`train_rad_tf.py`) matches the MLPs' accuracy with a fifth of the parameters
but costs 7.1 to 18.7 ms a call in the image. Asked to learn the zenith angle
instead of being given it, the same network drifts ten times faster (1.5 K
rms in temperature after a day against 0.14): the geometry the physics
computes in twenty lines is what a network learns worst, and the driver path
needs it handed to the model.

### The cloud blocks: loop order, compilation, and the clamp

With the contract complete, the capture is a training set
(`train_cloud_block.py`; `cloud_block_mlp.py` answers a block from a slot).
Three lessons, none about physics. A forward that streamed its first-layer
weights (4.75 MB) once per column cost 36 ms a call in the image against 2.3
alone; with the weight rows outside the column loop, 8.3 ms (7453811,
7454183). Numba carried 4.3 s a rank of compilation at the first call inside
the loop; the forward as NumPy matrix products costs 3 ms at its first call
(7454277). And a network's answer is clamped before it is written, so no
constituent it tends is taken below zero over the step (`dq >= -q/dt`), where
`qneg3` would have clipped it with a warning line each: 446,000 lines before,
one after. With both blocks answered by core networks (the physics only, the
copies and integrals derived) the stage costs 1.24 s a rank against 1.96, the
networks 3.1 and 3.3 ms a call against the drivers' 6.6 and 19.2 (7457386);
the run drifts 6.1 K rms in temperature over fifty steps. The bulk tendencies
validate at R² 0.72 and 0.55, the constituent tendencies at 0.46 and 0.23: what
a network for either block should learn, and in what units, is the open work.

## The inventory and where it stands

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
is marked inert-by-configuration and counted as covered only once a targeted
test confirms it. That test is one 50-step run with all eleven such actions
disabled, bit-for-bit with the oracle
(`pi_cam_pausable_inert_vs_oracle_50step_bfb.json`). Execution is evidenced
from the recorded runs at two lengths, 50 steps and a month, not assumed from
the plan.

The record names nothing outside the repository: contract paths, evidence
files and catalog sources are relative, and there is no timestamp, so the
committed file equals a fresh build or the test fails.

**Where it stands** (the ledger's `summary`; the dashboard shows each kernel):

- Twelve enabled scheme actions do numerical work here, and all twelve have a
  Python class. Three are complete by the loop above (dry adjustment, shallow
  convection, the cloud stage), eight partial, and one a gap: the energy
  fixer, which `EnergyFixer` runs whole with no kernel exposed, because
  `check_energy_fix` allocates its tendency inside the call and reads the
  module's private global heating, so a frame at its call site would serve
  nothing. The ledger records that reason on the action.
- Twenty-two kernel entries (`cldfrc_fice` is tracked under the cloud stage
  and under deep convection), every one paused or hooked and validated with
  the original answering. Twelve are complete; ten are open, nine of them
  lacking the capture-and-replay half of the loop (`rad_rrtmg_sw`,
  `rad_rrtmg_lw`, `convtran`, `compute_eddy_diff`, `compute_vdiff`,
  `gw_drag_prof`, `wetdepa_v2`, `modal_aero_depvel_part`,
  `gas_phase_chemdr`) and `zm_conv_evap` its single-column and public-interface
  replays (its lane-mode frame replay differs in the last bit of `tend_s`; the
  chunk mode is bit-for-bit).
- One run installs everything: the nine pausable classes, the split radiation
  class and the cloud stage, with all eighteen runner kernels (`virtem`
  included) answered by the original through their pauses, 5400 pauses in 50
  steps, bit-for-bit with the oracle (7343260,
  `pi_cam_pausable_everything_vs_oracle_50step_bfb.json`); with the kernel
  timers, the same over the month (7479754,
  `pi_cam_pausable_p28-everything-timers-1month_vs_oracle_1month_bfb.json`).
- The four leaves' classes declare their kernels themselves; the physics
  catalog still lists no procedures under a leaf, so their candidate counts
  read zero -- a gap of the catalog's call graph, not of the classes.

**Open.** The capture-and-replay loop for the ten open kernels; hooks for the
rest of batch A and for batch B, whose kernels answer only at the runner's
pause; the cloud drivers' history entry; coarse contracts for the other
processes ([plans/interface_completion.md](plans/interface_completion.md));
and a restart in the middle of a run, which freeCAM cannot do yet (every run
starts as a startup run; continuing from CAM's restart files is a feature of
its own).
