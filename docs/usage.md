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
[`examples/physics_function.ipynb`](../examples/physics_function.ipynb) walks
through the function interface,
[`examples/generate_training_data.ipynb`](../examples/generate_training_data.ipynb)
through dataset generation, and
[`examples/kernel_surrogate.ipynb`](../examples/kernel_surrogate.ipynb) through
putting a trained network in a kernel's place inside the running model.
