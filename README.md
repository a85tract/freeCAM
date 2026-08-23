# freeCAM

freeCAM runs the CAM atmosphere component from iCESM1.3.1 under a Python
control layer. Python owns the model workflow, clock, coupling decisions, and
rank-local state. Original Fortran remains the numerical source of truth and
is called through generated C-interoperable adapters.

The current scientific target is the `ne16` PI-atm configuration with CAM5
physics, SE dynamics, and 512 MPI ranks on NCAR Derecho.

## Highlights

- Python-controlled CAM initialization, timestep workflow, and finalization.
- Persistent MPI execution for interactive Python and Jupyter workflows.
- Rank-local NumPy StatePool with distributed inspection and mutation.
- Runtime reordering, enabling, and disabling of physical processes.
- Dynamic Python processes and StatePool variables.
- Online execution of the original CESM surface components and coupler.
- Explicit offline replay mode for captured x2a/a2x boundaries.
- Generated adapters for all 276 catalogued physical processes; 240 are
  loadable in the admitted PI-atm executable.
- Exact 50-step and one-year validation against the original Fortran model.

## Architecture

```text
Python / Jupyter
    |
    |  Driver, workflow, clock, StatePool
    v
512 persistent MPI Python ranks
    |
    +-- CAM numerical kernels ---------> original iCESM Fortran .so
    |
    +-- online coupling provider ------> CLM-SP, CICE%PRES, DOCN%DOM, RTM
                                        and CESM mapping/flux kernels
```

Each MPI rank owns its local NumPy arrays and StatePool. Python chooses which
operation runs next; Fortran performs the admitted numerical kernels. Online
coupling exposes the live rank-local MCT x2a/a2x arrays as zero-copy NumPy
views. There is no shadow CAM and no Fortran-to-Python callback path.

## Installation

```bash
git clone git@github.com:a85tract/freeCAM.git
cd freeCAM
git submodule update --init external/iCESM1.3.1_fzhu
uv sync --extra notebook --extra test
```

The supplied runtime and PBS jobs target NCAR Derecho. A configured iCESM
reference case, its machine environment, and the required input data must be
available before launching the 512-rank scientific configuration.

## Quick start

Online CESM coupling is the default:

```python
import freecam as fc

with fc.Driver(case="PI-atm", nsteps=2) as driver:
    driver.initialize()
    print(driver.cam.state.T.stats(rank="global"))

    result = driver.run(progress=True)
    print(result)
```

Constructing `Driver` does not submit PBS or start MPI. The first live model
operation starts one persistent MPI model; later calls reuse the same ranks
and arrays until `driver.close()` or the context manager exits.

### Timing reports

FreeCAM profiles its Python control regions, boundary operations, complete
steps, individually dispatched processes, and Fortran calls by default. When
the model closes it writes three CESM-style text reports under the run
directory:

```text
timing/freecam_timing.0000        rank-0 hierarchical call timing
timing/freecam_timing_stats       aggregate statistics across all MPI ranks
timing/cesm_timing.<case>.<lid>   CIME-format performance profile
```

The performance profile carries the same high-level summary CIME writes for a
CESM case — Model Cost (pe-hrs/simulated-year), Model Throughput
(simulated-years/day), and Init/Run/Final times — derived from the gathered
`FREECAM:INITIALIZE`/`STEP`/`FINALIZE` totals. Because freeCAM advances the CAM
atmosphere as one timed unit, its component breakdown reads like a standalone
`atm`-only compset: ATM carries the whole run cost and every other component
reads zero. Model Cost bills whole nodes, matching CIME's node-granular
accounting.

Timing uses `MPI_Wtime`. Process execution adds no timing barriers; rank-local
records are gathered only once during finalization. The online surface/coupler
provider writes its own original `cesm_timing.*` files into its separate,
private CESM run directory (never freeCAM's). Those files profile different
code and are intentionally retained.

The in-memory action trace is bounded to the most recent 4,096 records per
rank by default, so long simulations do not accumulate one Python object per
process call. Run results always report exact action counts and state whether
their trace was truncated. Pass `trace_limit=None` to `freecam.Driver` (or
`PICAMNotebookSession`) only when a complete in-memory debug trace is
explicitly needed.

### History output for Python-owned fields

The original CAM writer only knows the fields CAM registered at build time,
so Notebook-defined StatePool variables never reached an output file. They now
land in the model's own history files, beside `T` and `PS`, exactly as a newly
registered CAM field would:

```python
driver.cam.state.create("heating_rate", like="T", units="K s-1")
result = driver.run()

driver.cam.history.latest()   # the usual case.cam.h0.*.nc, now with heating_rate
```

No configuration is required. A Python-owned field joins the default output
automatically, accumulated over the same window the case's `nhtfrq` selects and
written at the same time samples CAM wrote. The run's final sample is completed
when the model closes, over the part of its window the run actually reached.
Pass `output=False` when creating a variable to keep a scratch field out of
history, or construct the model with `default_history_stream=False` to disable
the behaviour entirely.

The run directory therefore stays indistinguishable from the original model's:
a run with no Python-owned fields writes exactly the files the original writes,
bit for bit, and a run that defines them adds those variables to those same
files rather than creating new ones.

The maintained Jupyter walkthrough is
[`examples/try_pi_cam.ipynb`](examples/try_pi_cam.ipynb).

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

`PI-atm-replay` contains 50 complete CAM steps. `PI-atm-1month` contains 1,488
steps. Replay requires the same 512-rank layout used by the capture.
`verify_boundary_exports=True` checks each generated a2x against the captured
reference; use `False` for experiments that intentionally change CAM output.

## Python interface

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

Scientific processes are exposed through one ordered workflow:

```python
workflow = driver.cam.workflow

workflow["radiation"].disable()
workflow["radiation"].enable()
workflow["dry_adjustment"].run()
workflow["radiation"].move(before="vertical_diffusion")
```

The workflow is a list, and the list is what runs. Assigning one leaves one
scientific process in the step; control, clock, and I/O actions keep their
slots, so the step still writes CAM's history file at its end:

```python
workflow[:] = [workflow["macro_microphysics"]]
driver.cam.state.T += 2.0
driver.run()                   # one step, one process, one history sample
driver.cam.history.latest()
```

A process left out of the list stops running: an original CAM process is
disabled and can be enabled again, a notebook process is uninstalled, the
same as `workflow.pop()` and `workflow.remove()`.

[`examples/macro_microphysics.ipynb`](examples/macro_microphysics.ipynb) is
that in one cell for CAM5's cloud macro/microphysics stage.

Notebook-defined Python physics can be inserted without rebuilding CAM:

```python
class Heating(fc.Physics):
    name = "notebook_heating"
    after = "dry_adjustment"

    def run(self, state):
        state.T += 0.01


driver.cam.workflow.insert(Heating())
```

### CAM namelist parameters

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
group this configuration never reads are all rejected outright, because CAM
itself either aborts without naming the variable or ignores the setting
silently. With no overrides the file is not touched at all, byte for byte.
`fc.CaseConfig` accepts the same `namelist=` mapping for reusable case
declarations, and the MPI command line accepts repeatable
`--namelist NAME=VALUE` flags.

A hand-audited subset of these tunables can also be changed **while the
model is running**. CAM copies namelist values into Fortran module
variables at initialization; for parameters proven to be re-read on every
timestep, freeCAM binds that module storage directly and a write takes
effect at the owning routine's next call:

```python
driver.cam.parameters["zmconv_c0_lnd"] = 0.0075   # all 512 ranks, next step
driver.cam.parameters.overrides                    # {'zmconv_c0_lnd': (0.0059, 0.0075)}

driver.cam.workflow["deep_convection"].properties  # the same tunables, per process
```

The admitted set lives in `native/pi_cam/runtime_parameters.yaml`, one
audited entry per parameter; every binding verifies at initialization that
the value read through the symbol equals the value in `atm_in`, and refuses
to bind otherwise. Where initialization copied a value into a second module
(the CAM5 macrophysics keeps shadow copies of the `rhminl` family), a write
updates every copy together. These values are not part of any restart
file: a run restarted from CAM restart files reverts to its namelist
values, so runtime changes must be re-applied after a restart.

### Runtime Physics properties

A `fc.Property` declares a tunable parameter of a Python process. Assigning
to it on a live model ships the value to every MPI rank collectively and
takes effect at the process's next invocation:

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
belong in StatePool fields.

See the Notebook for field aliases, plotting, workflow construction, runtime
process replacement, asynchronous execution, and Xarray history access.

## Calling a CAM scheme as a function

Besides running the model, freeCAM can hand you one physics routine as an
ordinary numerical function -- `y = f(x, p)` on a single vertical column --
with no `Driver`, no MPI session and no Driver-managed model state.  The routine is linked from the
oracle build's own objects into a small standalone image and runs in a worker
process beside your Python:

```python
import freecam as fc

scheme = fc.physics.load_function("mmacro_pcond")   # the condensation routine of CAM5 cloud macrophysics
print(scheme.describe())                             # inputs, in/outs, outputs, parameters

column = scheme.example_input("captured-anchor")     # a real column, shipped with the package
result = scheme.run(inputs=column, parameters={"cldfrc_rhminl": 0.85})
result.outputs["cld"]                                # one column's cloud fraction, (lev,)

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

| Interface | What it does |
| --- | --- |
| `driver.cam.workflow[...]` | runs a process on the full model field, inside a timestep |
| `fc.physics.load_function(...)` | calls the scheme on one column, Driver-free |

Inputs are `(lev,)` profiles and scalars; parameters are the routine's own
namelist tunables and join the sampling space as extra dimensions.  An input
the Fortran refuses raises `FortranAbortError` with the routine's diagnostic
(`try_run` returns the status instead); during dataset generation such a
sample keeps its status and is never written as data.  Every function has a reviewed specification under
`native/pi_cam/functions/`, and its image is proven before use: the wrapper
demonstrably calls the original routine, and replaying calls captured from a
real 512-rank run through the image reproduces the model bit for bit
(`validation/pi_cam_*_full_chunk_vs_capture.json`,
`..._single_column_vs_capture.json`, `..._public_api_vs_capture.json`).
`examples/physics_function.ipynb` walks through it.

## Validation

The current validated results are:

| Gate | MPI ranks | Result |
| --- | ---: | --- |
| Python-controlled PI-CAM, 50 steps | 512 | BFB with the pinned Fortran reference |
| Exact online CESM provider, 50 steps | 512 | 53/53 x2a, 53/53 a2x, and 4/4 CAM output files match |
| Exact online CESM provider, one year | 512 | 180/180 CAM history and restart files match |
| Exact online CESM provider, five years | 512 | 884/884 CAM history and restart files match |
| Monthly output vs. an independent production run, one year | 512 | 12/12 monthly files, 215 variables each, bit identical |
| Monthly output vs. an independent production run, five years | 512 | 60/60 monthly files, 215 variables each, bit identical |
| Python-owned fields in CAM history output, twelve steps | 512 | 6/6 hourly samples carry the field, `output=False` reaches none |

The two monthly-output gates compare against a separately produced twenty-year
CESM integration of the same case rather than against a reference this project
generated, so they test the whole lifecycle end to end.

Measured overhead against the original Fortran lifecycle is +8.7% run time and
+8.2% memory over five model years, and does not grow with integration length.
[`validation/performance_overhead.md`](validation/performance_overhead.md)
records the method, the per-run numbers, and their caveats.

Primary evidence:

- [`validation/pi_cam_exact_cesm_online_50step.json`](validation/pi_cam_exact_cesm_online_50step.json)
- [`validation/pi_cam_exact_cesm_online_1year.json`](validation/pi_cam_exact_cesm_online_1year.json)
- [`validation/pi_cam_exact_cesm_online_1year_bfb.json`](validation/pi_cam_exact_cesm_online_1year_bfb.json)
- [`validation/pi_cam_exact_cesm_online_5year.json`](validation/pi_cam_exact_cesm_online_5year.json)
- [`validation/pi_cam_exact_cesm_online_5year_bfb.json`](validation/pi_cam_exact_cesm_online_5year_bfb.json)
- [`validation/pi_cam_monthly_1year_bfb.json`](validation/pi_cam_monthly_1year_bfb.json)
- [`validation/pi_cam_monthly_5year_bfb.json`](validation/pi_cam_monthly_5year_bfb.json)
- [`validation/pi_cam_python_history_output_12step.json`](validation/pi_cam_python_history_output_12step.json)
- [`validation/pi_cam_process_support.json`](validation/pi_cam_process_support.json)

The directory comparator requires identical CAM file inventories, numerical
variable inventories, dtypes, shapes, and exact array values without a
tolerance. It does not compare NetCDF compression bytes, path strings, or
non-numerical metadata.

## Repository layout

```text
src/freecam/pi_cam/       Python driver, StatePool, workflow, and public API
native/pi_cam/            adapters, support code, and source patches
external/iCESM1.3.1_fzhu pinned upstream iCESM source
configs/                  admitted PI-CAM configurations
examples/                 maintained Jupyter walkthrough
tools/                    build, capture, audit, and validation utilities
tests/unit/               local Python test suite
validation/               PBS jobs and machine-readable scientific evidence
```

Generated libraries and compiler products belong under `build/`. PBS output
belongs under `logs/`; neither should be committed.

## Development

```bash
uv sync --extra notebook --extra test
uv run pytest -q
uv run freecam --help
git diff --check
```

The 512-rank 50-step scientific gate is submitted with:

```bash
qsub validation/jobs/pi_cam_exact_cesm_online_50step.pbs
```

Adding another CAM configuration requires a compatible native build context,
field bindings, and independent numerical validation. freeCAM does not
silently reuse PI-atm adapters for incompatible COSP, CARMA, or radiation
configurations.

## License

See [`LICENSE.txt`](LICENSE.txt), [`LICENSES/`](LICENSES/), and
[`NOTICE`](NOTICE) for project and third-party terms.
