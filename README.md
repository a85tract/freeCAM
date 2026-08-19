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
the model closes it writes two CESM-style text reports under the run directory:

```text
timing/freecam_timing.0000   rank-0 hierarchical call timing
timing/freecam_timing_stats  aggregate statistics across all MPI ranks
```

Timing uses `MPI_Wtime`. Process execution adds no timing barriers; rank-local
records are gathered only once during finalization. The online surface/coupler
provider may also write its original `cesm_timing.*` files in its own run
directory. Those files profile different code and are intentionally retained.

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
written at the same time samples CAM wrote. Pass `output=False` when creating a
variable to keep a scratch field out of history, or construct the model with
`default_history_stream=False` to disable the behaviour entirely.

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

Notebook-defined Python physics can be inserted without rebuilding CAM:

```python
class Heating(fc.Physics):
    name = "notebook_heating"
    after = "dry_adjustment"

    def run(self, state):
        state.T += 0.01


driver.cam.workflow.insert(Heating())
```

See the Notebook for field aliases, plotting, workflow construction, runtime
process replacement, asynchronous execution, and Xarray history access.

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

The last two gates compare against a separately produced twenty-year CESM
integration of the same case rather than against a reference this project
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
