# Using freeCAM

This guide covers the public Python interface: the model handle, its state,
its workflow, parameters, output and timing, and the single-column function
interface that needs no model at all. The maintained notebook walkthrough is
[`examples/try_pi_cam.ipynb`](../examples/try_pi_cam.ipynb).

## The model

```python
import freecam as fc

with fc.Driver(case="PI-atm", nsteps=2) as driver:
    driver.initialize()
    print(driver.cam.state.T.stats(rank="global"))
    result = driver.run(progress=True)
    print(result)
```

Constructing `Driver` does not submit PBS or start MPI. The first live model
operation prepares a private run directory and starts one persistent MPI
session; later calls reuse the same ranks and arrays until `driver.close()` or
the context manager exits. `driver.run()` executes the steps the case
declares; `driver.advance(n)` executes `n` of them.

The default case runs the original CESM surface components and coupler
(CLM-SP, CICE%PRES, DOCN%DOM, RTM and the CESM mapping and flux kernels) live
in the same MPI processes as CAM. The rank-local MCT x2a/a2x arrays are exposed
as zero-copy NumPy views; there is no shadow atmosphere and no
Fortran-to-Python callback.

### Other rank counts

The admitted case runs on 512 MPI ranks. The online case can run on another
count without a rebuild -- CAM decomposes its grid when it starts, and CESM
reads its layout from `drv_in` -- but only as an exploration:

```python
with fc.Driver(case="PI-atm", nsteps=48, ntasks=256, exploratory=True) as driver:
    result = driver.run()
```

CAM and the coupler run on every rank, CLM on `min(N/2, N-160)`, CICE on its
compiled 128 tasks after it and DOCN on 32 after that
([`freecam.pi_cam.layout`](../src/freecam/pi_cam/layout.py)); the provider's
copy of `drv_in` is rewritten, its seed is not. The answers are not
bit-for-bit with the 512-rank oracle (CESM's own are not either), which is why
a count other than 512 needs `exploratory=True`, and `driver.status` and
`driver.diagnose()` say so. A replay case keeps the ranks it was captured on.

### Other compile-time options

CAM's columns a chunk (`pcols`) and the device FTorch is linked for are
compiled in, so changing one means building the model again. `fc.build`
does that:

```python
build = fc.build(pcols=32)       # submits the build's CPU jobs and returns at once
build.status()                   # each stage: pending, queued, running, done or failed
build.wait()
with fc.Driver(case="PI-atm", build=build, nsteps=48) as driver:
    result = driver.run()
```

The default options are the installed model: `fc.build()` returns it and
builds nothing. Any other options get a root of their own under
`$FREECAM_SCRATCH/freeCAM/builds/` and six jobs, each waiting on the one
before ([`freecam.pi_cam.build`](../src/freecam/pi_cam/build.py)):

1. the CESM sources and cases from the recipe, with `-pcols` in CAM's configure;
2. `case.build` of the three cases;
3. the oracle's 50 steps and the original coupled model's;
4. the native image from that oracle's own objects;
5. the online coupler library linked to that image;
6. the exact online 50 steps, bit for bit against the build's own oracle.

Calling `fc.build` with the same options again resumes: a stage whose product
exists is skipped, and a failed one is submitted again with everything after
it. A build is validated once its online gate is bit-for-bit;
`Driver(build=...)` runs an unvalidated one only with `exploratory=True`. It
uses the build's image, oracle case and run and coupler library, refuses a
`FREECAM_NATIVE_MANIFEST` that names another image, and refuses a coupler
library linked for another grid. The replay cases keep the installed build.
From a shell: `freecam build --pcols 32 [--plan | --status | --wait]`.

The `pcols=32` build has been made this way and passed its online 50-step
gate and an online month against its own original
(`validation/pi_cam_build_pcols32_online_{50step,month}_bfb.json`). No
`device="cuda"` build has been gated yet.

### Offline replay

Select a replay case when x2a should come from a captured boundary dataset
instead of live CESM components:

```python
with fc.Driver(
    case="PI-atm-replay",
    nsteps=50,
    verify_boundary_exports=True,
) as driver:
    driver.initialize()
    result = driver.run()
```

`PI-atm-replay` contains 50 complete CAM steps and `PI-atm-1month` a month of
them. Replay requires the same 512-rank layout used by the capture.
`verify_boundary_exports=True` checks each generated a2x against the captured
reference; use `False` for experiments that intentionally change CAM output.

## State

State fields behave like distributed NumPy arrays:

```python
import numpy as np

state = driver.cam.state

state.T += 1.0
state.q[:] = np.maximum(state.q, 0.0)
state.create("tracer", like="T", units="kg kg-1")

rank_zero = state.T.get(rank=0)
global_mean = state.T.mean()
```

Each MPI rank owns its local arrays; `get`, `stats` and `mean` gather from
the ranks on request, and in-place arithmetic is shipped to every rank
collectively.

### Python-owned fields in CAM's history output

A field created from Python lands in the model's own history files, beside
`T` and `PS`, exactly as a newly registered CAM field would:

```python
driver.cam.state.create("heating_rate", like="T", units="K s-1")
result = driver.run()

driver.cam.history.latest()   # the usual case.cam.h0.*.nc, now with heating_rate
```

No configuration is required. The field is accumulated over the window the
case's `nhtfrq` selects and written at the same time samples CAM writes; the
run's final sample is completed when the model closes. Pass `output=False`
when creating a variable to keep a scratch field out of history, or construct
the model with `default_history_stream=False` to disable the behaviour. A run
that defines no Python-owned fields writes exactly the files the original
model writes.

### The state on a globe

CAM's history files hold fields at the end of an output interval. freeCAM can
also keep the state as the model steps, and as every process leaves it inside
a step: it owns the order of the actions, and the state pool is the model's
own memory. The recorder is off unless asked for:

```bash
# the rank command line (the online 50-step job's knobs are PYCAM_STATE=1, PYCAM_STATE_*)
mpiexec -n 512 python -m freecam.pi_cam.cli ... --state-dir <run>/state \
    --state-fields T,Q,CLDLIQ,CLDICE --state-every 1 --state-action-steps 24
```

```python
fc.Driver(case="PI-atm", record_state=True)      # T, Q, CLDLIQ and CLDICE every step, into <run>/state
fc.Driver(case="PI-atm", record_state={"fields": ["T", "CLDLIQ"], "every": 6, "action_steps": [24]})
```

Every rank copies the chosen fields of its own columns (read-only: a float32
copy of each real column at the end of every `--state-every`-th step) and
appends them to its own file; at the `--state-action-steps` it also copies the
state as the step begins and after every plan action, in float64, since what
one process changes in one step can be smaller than float32 resolves (2e-5 K at
280 K). Nothing is exchanged while the model steps. `T`, `Q`, `CLDLIQ`,
`CLDICE`, `U`, `V`, `OMEGA` and `PS` are named directly; any
`phys_state.<field>` or `phys_state.q:<constituent>` can be named too. A 3-D
field costs 13,826 columns x 30 levels x 4 bytes = 1.7 MB a snapshot at ne16,
so a month of every step is 2.5 GB a field: record every few steps for long
runs.

```bash
freecam globe <run>/state                   # serve the viewer on loopback; it follows a running model
freecam globe <run>/state --html globe.html --fields T,CLDLIQ --levels 14,20,23 --step-stride 2
```

The page shows one field at one level on a globe with the continents
(Natural Earth 1:110m coastlines, public domain), each column painted over the
part of the sphere nearest to it:

- **Steps**: the slider moves through the recorded steps; *Change* shows what
  the step changed since the recorded step before.
- **Processes in a step**: at an action step, the slider moves through the
  plan actions; *Value* is the state after that action, *Change* what that
  action alone changed. The strip above the slider is coloured by how much
  each action changed the field (root-mean-square over every column and
  level); grey actions do not change it. Each action's box is as wide as its
  share of the step's time: the driver times every plan action at an action
  step on every rank (not the copy after it), and the width is the mean over
  the ranks; the label names the slowest rank's time.
- A click picks a column: its vertical profile (now and before) and the
  latitude-pressure section along its meridian, on their own colour scale.
  Under *Processes in a step* a third panel traces the column through the
  step (below).

A self-contained page embeds the fields, levels and steps it is given, 8-bit
over each level's range (a colour map has no more), and each action's change
8-bit over its own scale; the server reads the recorded values themselves.
Differences between steps are left to the server: 8-bit steps would difference
to quantization noise.

The recorder only reads the state, so a run with it on stays bit-for-bit with
the original: the online 50-step gate recording the four default fields every
step, written every step as the Workflow Builder records them, and after every
action at step 24 -- 50 step and 49 action snapshots -- is bit-for-bit with the
oracle (`validation/pi_cam_exact_cesm_online_state-record-live_50step*.json`,
job 7711982).

#### In the Workflow Builder

The page started by `driver.ui()` or `freecam ui` has a *Globe* tab with the
same viewer, served by the page's own service under `/globe/` and following
the run the page started: each step appears as the model records it, and the
slider stays on the newest step until it is moved back. For that the page has
a model that is not yet running record `T`, `Q`, `CLDLIQ` and `CLDICE` every
step and write them every step (`record_state={"flush_every": 1}`); a
`record_state` the caller chose is kept, with its fields and interval.
`driver.ui(globe=False)` or `freecam ui --no-globe` leaves the model as it is,
and a model already running keeps what it was started with. *Open in a new
tab* gives the viewer a window of its own.

Every rank writes at the same steps, and rank 0 names a step in the manifest
only once every rank has written it, so a page following the run never reads a
step some rank has not written yet. A rank that fails keeps what it recorded
and waits for no other rank; the run is then not marked complete.

#### Who changed a column

At an action step the recorder also keeps, beside each action's snapshot, what
computed that action on this run: the original Fortran, the coupler's
exchange, output, a Python process, or a Python stage class with the way it
ran and what stood in each replaced kernel slot (an ML model with its file and
device, a compiled plugin, the original called through Python, or a Python
function). The driver reads this from the plan and the installed stages; it
does not look inside the numbers. In the default step 36 actions read
"original Fortran", six output, three the state service, two the coupler
exchange and one the clock. With the shallow-convection stage installed and its
kernel answered by a TorchScript model on the host, the trace names it:

```
        ↓ cam_run1.shallow_convection_python   (original Fortran, except compute_uwshcu_inv by an ML model (uwshcu_h256_notr_2cap_sub.pt, cpu), 12.0 ms)
  T = 294.28 K   Q = 15.332 g/kg   CLDLIQ = 16.077 mg/kg
      → ΔT +0.122 K   ΔQ -0.125 g/kg   ΔCLDLIQ -3.91 mg/kg
```

The label says what stood in the slot, not that every call of the step went
to it: the image's hook counters do (51,200 of the kernel's 53,248 calls in
that run went to the model; the other 2,048 are four a rank, the same four a
rank as in a month of 1488 steps, so they do not recur step by step). A trace then follows one column through one
step, one field set at one level:

```bash
freecam trace <run>/state --lat -31 --lon -50.6 --step 24 --hpa 925
freecam trace <run>/state --lat 10 --lon 120 --step 24 --level 20 --fields T,Q --all --json
```

From a 50-step online run recorded before the owners were kept, so they read
"not recorded" (`--fields T,Q,CLDLIQ`):

```
step 24, the column at 31.0°S, 50.6°W (sea), level 27, about 923 hPa

  T = 298.43 K   Q = 13.251 g/kg   CLDLIQ = 0.0000 mg/kg

        ↓ cam_run1.dynamics_to_physics   (not recorded, 4.6 ms)
  T = 298.27 K   Q = 13.291 g/kg   CLDLIQ = 0.0000 mg/kg
      → ΔT -0.159 K   ΔQ +0.0394 g/kg   ΔCLDLIQ ≈0 mg/kg

        ↓ cam_run1.deep_convection   (not recorded, 8.8 ms)
  T = 294.90 K   Q = 14.626 g/kg   CLDLIQ = 0.0695 mg/kg
      → ΔT -3.36 K   ΔQ +1.34 g/kg   ΔCLDLIQ +0.0695 mg/kg

        ↓ cam_run1.cloud_macro_microphysics   (not recorded, 38.9 ms)
  T = 294.89 K   Q = 14.631 g/kg   CLDLIQ = 0.0000 mg/kg
      → ΔT -0.0116 K   ΔQ +0.00479 g/kg   ΔCLDLIQ -0.0695 mg/kg

        ↓ cam_run1.radiation   (not recorded, 46.3 ms)
  T = 294.94 K   Q = 14.631 g/kg   CLDLIQ = 0.0000 mg/kg
      → ΔT +0.0437 K   ΔQ 0 g/kg   ΔCLDLIQ 0 mg/kg

(4 more changed this level by less than 0.1% of its largest change; 40 left it unchanged)
over the step: ΔT -3.49 K   ΔQ +1.38 g/kg   ΔCLDLIQ ≈0 mg/kg
```

It lists the actions that changed a field at that level by at least 0.1% of
the largest change any action made there (`--relative`, `--all` for every
action), with each action's mean time over the ranks, and says how many more
changed it less and how many left it alone; a change below 1e-12 of the
column's largest value (round-off, a denormal) counts as none. With no level it takes the level
where the first field changed most over the step. On the page the same trace
appears when a column is picked under *Processes in a step*; a row moves the
globe to that action. A self-contained page traces from the levels it
embeds, and its changes are 8-bit over each action's largest change on the
globe: one column's change is then only within half a code of it (the page
prints the ± beside each), and the errors add up over the step. At one column
of a 50-step run the page gave -2.60 ± 1.30 mg/kg of liquid for vertical
diffusion (one code) where the recorded change is -1.51, and +3.42 ± 4.43 over
the step where it is +4.14; use the served viewer or `freecam trace` for
numbers.

The trace names the action, not the line of Fortran inside it: a change inside
`cloud_macro_microphysics` is that whole action's, and a run recorded before
the owners were kept shows them as not recorded.

#### Finding what went wrong

The viewer also marks anomalies: a column where a recorded field is not finite
(NaN or Inf), where water or any constituent (`Q`, `CLDLIQ`, `CLDICE`,
`phys_state.q:<name>`) is negative, or where temperature is outside 100 to
400 K. The original physics leaves none of these between its actions: at an
action step of the bit-for-bit gate no recorded value is negative, and the
default run's check reports nothing. *Anomalies* on the page paints the
anomalous columns in their kind's colour over the faded field, says for the
recorded steps where each field first went wrong, and at an action step names
the first action that made each kind of anomaly, with the columns it made so;
a band over the process strip marks those actions. Clicking an entry goes to
the action and picks one of its columns, and the column's trace marks the
action after which it became anomalous. From the command line:

```bash
freecam anomalies <run>/state                 # every recorded step and action step
freecam anomalies <run>/state --step 2 --json
```

On a 50-step run with a shallow-convection model whose cloud fraction came out
slightly negative (H2O2 and SO2 recorded as `phys_state.q:H2O2`,
`phys_state.q:SO2`):

```
steps: phys_state.q:H2O2 non-finite first at the end of step 2 (13282 columns, every column from step 3)
action step 1: no action made a column anomalous
action step 2: phys_state.q:H2O2 became non-finite after cam_run2.chemistry_tendencies_leaf (original Fortran): 5579 columns, e.g. 10 (-28.0, 84.4), 24 (-32.1, 85.9), 26 (11.3, 90.0)
action steps 3-10: phys_state.q:H2O2 non-finite already as the step begins (up to 13826 columns); no action made a new kind of anomaly
```

The action named is where a recorded field first went wrong, which is not
always where the error began: here the chemistry is the original Fortran, and
the cause was upstream (the model's shallow cloud fraction, which chemistry's
photolysis raises to the power 1.5). Record the fields the suspect process
writes -- a cloud fraction, a tendency -- to follow the chain further back.
A value that is non-finite before and after an action is shown as "still
non-finite" and does not list the action in a trace. JSON has no NaN, so the
server and the page carry a non-finite value as null.

## The workflow

Scientific processes are exposed through one ordered workflow:

```python
workflow = driver.cam.workflow

workflow["radiation"].disable()
workflow["radiation"].enable()
workflow["dry_adjustment"].run()
workflow["radiation"].move(before="vertical_diffusion")
```

The workflow is a list, and the list is what runs. Assigning one leaves one
scientific process in the step; control, clock and I/O actions keep their
slots, so the step still writes CAM's history file at its end:

```python
workflow[:] = [workflow["macro_microphysics"]]
driver.cam.state.T += 2.0
driver.run()                   # one step, one process, one history sample
driver.cam.history.latest()
```

A process left out of the list stops running: an original CAM process is
disabled and can be enabled again; a notebook process is uninstalled, the same
as `workflow.pop()` and `workflow.remove()`.
[`examples/macro_microphysics.ipynb`](../examples/macro_microphysics.ipynb)
does this for CAM5's cloud macro/microphysics stage.

### Python physics

Notebook-defined physics is inserted without rebuilding CAM:

```python
class Heating(fc.Physics):
    name = "notebook_heating"
    after = "dry_adjustment"

    def run(self, state):
        state.T += 0.01


driver.cam.workflow.insert(Heating())
```

A `fc.Property` declares a tunable parameter of a Python process. Assigning to
it on a live model ships the value to every MPI rank collectively and takes
effect at the process's next invocation:

```python
class TunableHeating(fc.Physics):
    name = "notebook_heating"
    after = "dry_adjustment"
    rate = fc.Property(0.01)

    def run(self, state, context):
        state.T += self.rate * context.timestep_seconds


heating = TunableHeating()
driver.cam.workflow.insert(heating)

heating.rate = 0.02                                     # live update
driver.cam.workflow["notebook_heating"].properties      # authoritative view
driver.cam.workflow["notebook_heating"].properties["rate"] = 0.03
```

Values must be JSON-compatible scalars or small containers; large arrays
belong in state fields.

### Replacing a process the model already owns as a class

`driver.processes` holds every physics process freeCAM owns as a Python
class -- `radiation`, `cloud_macro_microphysics`, `dry_adjustment`,
`shallow_convection`, `deep_convection`, `vertical_diffusion`, ... -- bound
to this run.  Looking one up changes nothing; filling a slot on it does:

```python
rad = driver.processes["radiation"]        # the Radiation stage of this run

def my_radiation(inputs):                  # the block contract: inputs by name in, the 12 outputs out
    ...
    return {"qrs": ..., "qrl": ..., "fsns": ..., "fsnt": ..., "flns": ..., "flnt": ..., "fsds": ...,
            "sols": ..., "soll": ..., "solsd": ..., "solld": ..., "flwds": ...}

rad.process = RadiationBlockModel(my_radiation, label="notebook")
driver.advance(48)                         # the radiation block is my_radiation from here on

rad.kernels["rad_rrtmg_sw"] = network      # or one kernel inside the driver instead
rad.process = None                         # back to the original Fortran
```

The next `advance` or `run` attaches the stage where its action runs -- for
radiation, between the two halves of the split stage; for a whole action,
in its place -- and detaches it again when every slot is empty, so the
Fortran path is exactly the original whenever nothing is replaced.  A
changed slot re-attaches.  `driver.status["processes"]` says who computes
each process asked for.  The block contract's inputs are
`freecam.physics.radiation_process.BLOCK_INPUTS`; a capture's outputs replay
through the same slot with `load_block_model("replay:DIR")`, which is how
the path is gated (see `docs/physics_kernel_decoupling.md`).

The cloud macro/microphysics stage has two such slots, one per compute
block (`freecam.physics.cloud_block`): `process` is the macrophysics
driver's block, the one `mmacro_pcond` lives in, and `micro_process` the
microphysics driver's with its aerosol activation.  Either slot turns the
stage into its Python driver, which reads the block's inputs from memory,
writes the answer -- the tendency object with its flags, the detrainment,
the buffer fields -- where the driver leaves it, and makes the glue's
bookkeeping calls around it:

```python
cloud = driver.processes["cloud_macro_microphysics"]
cloud.process = load_block_model("replay:DIR", block=MACRO_BLOCK, rank=rank)   # or BlockModel(f, label=..., block=MACRO_BLOCK)
cloud.micro_process = None                 # the microphysics driver stays the original, in place
driver.advance(48)
```

`examples/replace_process.ipynb` walks through both stages this way on a live
run: the table, a replay in the radiation slot, a network in it, the original
back, and the cloud stage's two slots.

### Replacing one kernel inside a process

Each owned process also names the numerical kernels its driver calls, and
each is a slot (`stage.kernels[name]`, `None` for the original):

```python
from freecam.physics.segments import OriginalKernel
from freecam.physics.numba_kernel import compile_kernel

vdiff = driver.processes["vertical_diffusion"]
vdiff.kernels["compute_tms"] = OriginalKernel()      # the original, through the pause: the gate
deep = driver.processes["deep_convection"]
deep.kernels["cldfrc_fice"] = my_ice_fraction        # a function over the kernel's arrays: compiled, called by Fortran at the hook
deep.kernels["cldfrc_fice"] = compile_kernel("cldfrc_fice", my_numba_kernel)   # compiled, called by Fortran at the hook
deep.kernels["cldfrc_fice"] = fc.NativeModel("ice.pt")                          # TorchScript, run by the image through FTorch
deep.kernels["cldfrc_fice"] = fc.NativeModel("ice.pt", device="cuda")           # the same, on this rank's GPU (a CUDA-linked image)
```

A function in a slot runs where the stage can run it.  Written over the
kernel's arrays, one positional argument per input then per output of the
kernel's model block (`docs/contracts.md` lists them; `float64` arrays indexed
`[column, level]`, scalars as floats, outputs written in place, an output the
block returns at a subset of constituents over the subset's slots), and put in
the slot of a kernel that has a hook, the stage compiles it with Numba on every
rank and Fortran calls the compiled code at the hook: the stage runs whole and
the interpreter never runs inside a Fortran call.  A function that Numba cannot
compile there is refused, not run from the interpreter; a network goes in as a
TorchScript `NativeModel`.  Written over one batch dict (inputs by dummy name in,
outputs by name out), or under `stage.execution_policy = "segmented"`, or at a
kernel without a hook, the function answers at the kernel's pause: the runner
stops at the call, hands Python the live columns, resumes after the write-back
(three crossings a call and a per-step cost per stage).  `compile_kernel` is
the same compilation by hand, with `shadow=True` to run the compiled function on
every call while the original answers, for a bit-for-bit cost measurement; on
the command line `--kernel-plugin NAME=file.py:function` and
`--shadow-kernel-plugin` do the same.  `stage.describe_kernels()` lists each
kernel's contract and binding; `docs/contracts.md` is the generated reference of
every contract.  `examples/replace_kernel.ipynb` walks through the ways on a
live run.

Which path a stage takes each step is `stage.execution_policy`
(`--stage-execution` on the command line); the run record says which one ran
(`stage.execution.mode`):

| policy | nothing replaced | kernels replaced |
| --- | --- | --- |
| `auto` (default) | the original Fortran stage whole (`native-whole`) | a compiled function or `NativeModel` at a hook: the stage whole, the image answering the hook (`native-model`); otherwise `segmented` where the image's runner pauses at every replaced kernel, and the Python transliteration (`legacy-python`) with a warning where it does not |
| `native-whole` | the original stage whole | refused, except a model at a hook |
| `segmented` | refused: there is no kernel to pause at | the runner stops at each replaced kernel |
| `legacy-python` | the transliteration, statement by statement | the same, calling the slot |

Native models and Python functions cannot share a stage in one step.  Every
path runs the same Fortran arithmetic where nothing is replaced; they differ
in cost (see [validation/performance_overhead.md](../validation/performance_overhead.md)).

A model is called once per chunk: a rank of 512 has two chunks of at most 16
columns, and a network pays its per-call cost twice a step.  A TorchScript model
at a hook whose table entry has a `batch` block (`native/pi_cam/hooks.yaml`;
`compute_uwshcu_inv` today) answers all of the rank's chunks in one forward
instead, with `stage.batch_chunks = True` (`--batch-chunks shallow_convection`
beside `--kernel-model`): before the stage runs, the image gathers every chunk's
inputs for the kernel -- `convect_shallow_tend` computes none of them, it hands
on the chunk's state and physics-buffer fields, and `pycam_shcu_batch` fetches
the same fields the same way -- the model runs once over their live columns,
and the stage then runs whole, each call taking its chunk's rows.  Every call
first checks its inputs against what was gathered for its chunk, bit for bit,
and stops the run if they differ.  No Python in the step beyond the one call
that starts the gathering.  `--batched-original` (test only) answers the calls
from the original run on each gathered chunk: the gate of the gathering, which
must be bit-for-bit.  `--batch-check` also runs the model on every call's own
chunk and records the largest difference from the batch's answer
(`hooks.<kernel>.batch.check_max_abs_diff` in the run record).

A Python model answering at a pause can be batched too, at a cost the hook batch
does not pay: a crossing into Python at every pause.  A stage whose runner keeps
every chunk in a slot of its own (`batch_chunks: true` in its spec under
`native/pi_cam/pausable/`; shallow convection today) runs every chunk from its
entry to its pause first; a model with `takes_chunk_batches = True` is handed one
`ChunkBatch` -- the live columns of every waiting chunk stacked in chunk order
(`rows`, `lchnks`), an input with no column axis once when every chunk has the
same and otherwise in `per_chunk` -- and answers every output stacked the same
way; the answer is split back by chunk and the chunks resume, each from where it
stopped.  Any other model is still called once per chunk.  `ByChunk(f)` wraps a
per-chunk function as a chunk-batch model.  `OriginalByChunk()`
(`--segmented-original-by-chunk`) is the path's gate: one call for every waiting
chunk, each chunk's original run at its own pause after its stacked inputs are
checked against its frame bit for bit.  At 512 ranks over 50 steps it answers 100
pauses in 50 calls, bit-for-bit (`validation/pi_cam_pausable_batch-shcu-stacked-p38_*`),
as are the batched run answering each chunk on its own frame
(`batch-shcu-original-p38`, every parked chunk's inputs checked when it is live
again: `FREECAM_BATCH_VERIFY=1`) and the sequential run on the same image
(`batch-seq-shcu-p38`).

## Parameters

### Namelist

CAM's physics tunables live in the run directory's `atm_in` namelist and are
read once, at initialization. Pass overrides when constructing the model and
they are applied to that file before CAM sees it:

```python
driver = fc.Driver(
    case="PI-atm",
    nsteps=50,
    namelist={"cldfrc_rhminl": 0.9, "zmconv_c0_lnd": 0.0075},
)
driver.cam.namelist["cldfrc_rhminl"]   # current file value
driver.cam.namelist.overrides           # what this run changed
```

Every name and value is validated against the pinned iCESM source's own
namelist definition before anything launches: unknown variables (with
spelling suggestions), Fortran type mismatches, and variables whose namelist
group this configuration never reads are rejected, because CAM itself either
aborts without naming the variable or ignores the setting silently. With no
overrides the file is not touched. `fc.CaseConfig` accepts the same
`namelist=` mapping for reusable case declarations, and the MPI command line
accepts repeatable `--namelist NAME=VALUE` flags.

### Runtime parameters

A hand-audited subset of the tunables can be changed while the model is
running. CAM copies namelist values into Fortran module variables at
initialization; for parameters proven to be re-read on every timestep, freeCAM
binds that module storage directly and a write takes effect at the owning
routine's next call:

```python
driver.cam.parameters["zmconv_c0_lnd"] = 0.0075   # all 512 ranks, next step
driver.cam.parameters.overrides                    # {'zmconv_c0_lnd': (0.0059, 0.0075)}

driver.cam.workflow["deep_convection"].properties  # the same tunables, per process
```

The admitted set lives in
[`native/pi_cam/runtime_parameters.yaml`](../native/pi_cam/runtime_parameters.yaml),
one audited entry per parameter. Every binding verifies at initialization
that the value read through the symbol equals the value in `atm_in`, and
refuses to bind otherwise. Where initialization copied a value into a second
module, a write updates every copy together. These values are not part of any
restart file, so runtime changes must be re-applied after a restart.

## Timing reports

freeCAM profiles its Python control regions, boundary operations, complete
steps, individually dispatched processes and Fortran calls by default. When
the model closes it writes three CESM-style text reports under the run
directory:

```text
timing/freecam_timing.0000        rank-0 hierarchical call timing
timing/freecam_timing_stats       aggregate statistics across all MPI ranks
timing/cesm_timing.<case>.<lid>   CIME-format performance profile
```

The performance profile carries the summary CIME writes for a CESM case
(Model Cost, Model Throughput, and Init/Run/Final times), derived from the
gathered `FREECAM:INITIALIZE`/`STEP`/`FINALIZE` totals. Because freeCAM
advances the CAM atmosphere as one timed unit, the component breakdown reads
like a standalone `atm`-only compset. Timing uses `MPI_Wtime`; process
execution adds no barriers, and rank-local records are gathered once, at
finalization. The online provider writes its own `cesm_timing.*` files into
its separate CESM run directory, never freeCAM's.

The in-memory action trace is bounded to the most recent 4,096 records per
rank by default. Run results always report exact action counts and state
whether the trace was truncated; pass `trace_limit=None` to `fc.Driver` only
when a complete in-memory trace is explicitly needed.

### The action timeline

The reports above say how long each region took in total. The timeline says
when, on every rank, step by step: which action ran, how long it took, and how
long the rank then waited for the others. It is off unless asked for:

```bash
# the rank command line (the job knob is PYCAM_TIMELINE=1)
mpiexec -n 512 python -m freecam.pi_cam.cli ... --timeline-dir <run>/timeline
```

```python
fc.Driver(case="PI-atm", timeline=True)   # into the run directory's timeline/
```

Each rank keeps one record per action in memory (two clock readings and an
append), plus the time spent inside the driver's collective agreement calls,
and appends them to its own file every `--timeline-flush-every` steps
(default 100). Nothing is exchanged between ranks while the model steps; the
only collectives are a barrier that sets a common time origin and one gather,
after initialization, of each rank's host and column coordinates.

```bash
freecam timeline <run>/timeline                  # serve the viewer on loopback; it follows a running model
freecam timeline <run>/timeline --html view.html # one self-contained page instead
```

The server prints an address with a token; from a login node, forward the
port (`ssh -L`, or the editor's port forwarding). The page has three views:

- **Where the time goes**: actions by steps, each cell the slowest rank's
  time in that action (or the mean, the imbalance between the two, or the
  time spent waiting), with each step's duration above. The step waits for
  its slowest rank, so that is the default colour.
- **One step, every rank**: the ranks, grouped by node, against time within
  the chosen step. Load imbalance shows as a ragged edge; waiting in a
  collective is grey.
- **Where on Earth**: the chosen action's time on each rank, painted on the
  columns that rank computes. CAM's physics load balancing gives each rank
  pairs of columns spread over a wide area (at ne16, a rank's 28 columns span
  some 80 degrees of latitude), so the globe shows which ranks are slow and
  where their columns lie, not a map of cost by place; the `rank` and `node`
  colourings show that decomposition itself. A per-place cost would need
  timing per column, which the chunked kernels do not have.

A snapshot embeds the overview, the globe for every action, and the timelines
of the first, slowest, a typical and the last step (`--steps` chooses others).

## The Workflow Builder

A browser page edits the step -- add, remove, replace, move, enable and
disable processes, set their parameters, write Python processes, put a
trained network in a kernel's slot -- and generates the freeCAM code that
runs it. It runs in two modes.

**Locally, beside a model.** From a Python session on the machine that has
the model:

```python
import freecam as fc

ui = fc.Driver(case="PI-atm", nsteps=2).ui()
ui.url        # open it; the address carries a session token
ui.close()    # stops the page; the model, if started, stays as it is
```

or from a shell, `freecam ui --case PI-atm --port 8765`. Opening the page
starts nothing: no PBS, no MPI. The first Run confirms the resources the
case needs, initializes the model, applies the workflow and runs the
declared steps; later Runs apply only what changed and continue from the
current step. A change the live model cannot take -- the case, a namelist
override, a kernel binding already attached -- is refused with a message to
close the model and start again. Stop ends a run at the next complete step;
closing the browser tab does not touch the model; Close model releases it.
The service listens on loopback with a session token and refuses cross-origin
requests; reach one on a remote machine through SSH port forwarding.

**As a preview, on GitHub Pages.** The same page, published from the
repository, edits, checks, generates and downloads with no model behind it,
and says so. Download `workflow.json` there and Import it into a local page
to run it.

What the page shows comes from the model's own records: the default order is
the current step plan, the library is the physics catalog with a reason for
every process that cannot be added, the tunables are the audited runtime
parameters, and a kernel is offered for replacement only where the image's
segment runner pauses at it -- today `mmacro_pcond` -- and labelled
separately for whether that path has passed a bit-for-bit gate. Control,
clock and output actions run every step and are shown read-only under
"Full step".

The canvas groups the step by what each phase does rather than by CAM's
routine names: *Physics after surface coupling* (`cam_run2`, `tphysac`),
*Dynamics* (`cam_run3`), and *Physics before surface coupling* (`cam_run1`,
`tphysbc`), in the order CESM runs them within one coupling interval. A
process's About tab gives its science -- what it represents, how it is set up
in this case (and whether it does anything here: Rayleigh friction, QBO
relaxation, ion drag and the CARMA hooks do not), its governing equations and
its literature with DOI links. That text is the one hand-written record the
page reads, `src/freecam/pi_cam/data/pi_cam_process_science.yaml`, read from
the pinned source and the reference case's namelist; the catalog refuses an
entry for an action the step plan does not have, and a test typesets every
formula. A catalogued sub-process shows the description of the stage it
belongs to. After editing the record, run
`uv run python tools/export_workflow_catalog.py` (and
`tools/export_progress_snapshot.py`, which hashes the catalog).

The check runs at two levels: the page checks names, duplicates, the control
skeleton, parent/leaf exclusivity, bindings and parameter types; the local
service adds Python syntax, model files and the catalog version. Changing
the scientific order or the set of physical processes needs Experimental.
A passing check says the declared constraints hold; only a gate says the
result is bit-for-bit.

Generate freezes the draft and produces a complete script -- save it on the
machine with the model and run `uv run python <file>` from the checkout -- a
notebook of the same cells, a setup-only snippet for a session that already
has a driver (`configure(driver)` after `driver.initialize()`), and
`workflow.json`, all through the ordinary interface described above --
`fc.Driver`, the workflow list, `fc.Physics`, `state.create`,
`driver.cam.parameters`, a stage class attached to the model. The default
workflow generates a run that configures nothing.
Model files are referred to by path and never embedded. The service applies
a document with the same calls in the same order as the generated script,
and a test holds the two to that.

## A scheme as a function

Besides running the model, freeCAM can hand you one physics routine as an
ordinary numerical function, `y = f(x, p)` on a single vertical column, with
no `Driver`, no MPI session and no model state. The routine is linked from the
oracle build's own objects into a small standalone image and runs in a worker
process beside your Python:

```python
import freecam as fc

scheme = fc.physics.load_function("mmacro_pcond")   # CAM5 cloud macrophysics condensation
print(scheme.describe())                             # inputs, in/outs, outputs, parameters

column = scheme.example_input("captured-anchor")     # a real column, shipped with the package
result = scheme.run(inputs=column, parameters={"cldfrc_rhminl": 0.85})
result.outputs["cld"]                                # one column's cloud fraction, (lev,)
```

| Interface | What it does |
| --- | --- |
| `driver.cam.workflow[...]` | runs a process on the full model field, inside a timestep |
| `fc.physics.load_function(...)` | calls the scheme on one column, with no model |

Inputs are `(lev,)` profiles and scalars; parameters are the routine's own
namelist tunables. An input the Fortran refuses raises `FortranAbortError`
with the routine's diagnostic (`try_run` returns the status instead). Every
function has a reviewed specification under
[`native/pi_cam/functions/`](../native/pi_cam/functions/), and its image is
proven before use: replaying calls captured from a real 512-rank run through
the image reproduces the model bit for bit (see
[validation.md](validation.md)).

### Datasets

The same function samples its own input space into a training dataset, with
parameters as extra dimensions:

```python
space = scheme.sampling_space(
    base=column,
    inputs={"t0": fc.physics.Anchored(column["t0"], absolute_scale=1.0),
            "p": fc.physics.HybridPressure.from_column(column, fc.physics.Uniform(9.0e4, 1.0e5))},
    parameters={"cldfrc_rhminl": fc.physics.Uniform(0.80, 0.95)},
)
dataset = scheme.generate_dataset(n_samples=10_000, space=space, seed=42)
dataset.to_netcdf("mmacro_pcond_training.nc")        # inputs, parameters, outputs, status, provenance
fc.physics.open_dataset("mmacro_pcond_training.nc").verify_sample(scheme).assert_equal()
```

A sample the Fortran refuses keeps its status and is never written as data.
`examples/generate_mmacro_pcond_dataset.py` is that route for `mmacro_pcond`
with every knob drawn, and `examples/generate_compute_uwshcu_inv_dataset.py`
for the UW shallow cumulus kernel: every one of its 20 inputs drawn per sample
around a real column, the 57-constituent tracer array rebuilt so its water is
the drawn water and its isotopes keep the column's ratios, the static energy
following the temperature (`CapturedColumns(derived=...)`).  The anchors come
from frames captured at the kernel's hook in a run of the model
(`PYCAM_CAPTURE_KERNELS`, `PYCAM_CAPTURE_EVERY`;
`tools/extract_pi_cam_anchor_columns.py --frame-capture`), and
`examples/generate_compute_uwshcu_inv_training_data.ipynb` runs the whole
route as `generate_training_data.ipynb` does for `mmacro_pcond`.
[`examples/physics_function.ipynb`](../examples/physics_function.ipynb) walks
through the function interface,
[`examples/generate_training_data.ipynb`](../examples/generate_training_data.ipynb)
through dataset generation, and
[`examples/kernel_surrogate.ipynb`](../examples/kernel_surrogate.ipynb) through
putting a trained network in a kernel's place inside the running model.
