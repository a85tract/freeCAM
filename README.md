# freeCAM

freeCAM runs the CAM atmosphere component from iCESM1.3.1 with a Python
control layer. Python owns the execution order, boundary/restart decisions,
clock, and rank-local StatePool. Original Fortran code is retained only as a
numerical kernel or a low-level state, PIO, or time-manager service called by
Python through generated C-interoperable adapters.

The current scientific target is the `ne16` PI-atm CAM configuration: CAM5
physics, SE dynamics, replayed coupler input, and 512 MPI ranks.

## Current status

| Capability | Status |
| --- | --- |
| Complete Python-controlled CAM step | Available |
| Persistent MPI session | Available |
| Rank-local NumPy StatePool | Available |
| Dynamic Python and Fortran processes | Available |
| Dynamic StatePool variables | Available |
| Source physics catalog | 276 physical processes and 95 helper routines |
| Generated process adapters | 276/276 compile in their owning CAM configurations |
| Former catalog-only interfaces | 262/262 now have compiled StatePool pointer adapters |
| Loadable in the current PI-CAM image | 240/276 |
| PI-CAM 50-step reference | Bitwise identical |

The 36 remaining configuration-specific devices belong to COSP, CARMA, or
legacy-radiation builds that are not active in this PI-CAM executable. They
have compiled adapters, but freeCAM does not label them runnable in the
current case.

Validation records are under [`validation/`](validation/). The primary BFB
record is
[`pi_cam_python_control_vs_oracle_50step_bfb.json`](validation/pi_cam_python_control_vs_oracle_50step_bfb.json),
produced by the 512-rank, 50-step Python-controlled leaf workflow in
[`pi_cam_python_control_50step.json`](validation/pi_cam_python_control_50step.json).
The per-process support table is
[`pi_cam_process_support.json`](validation/pi_cam_process_support.json).
The declarative `CaseConfig` startup path, including two independently named
instances of the same Python process, is recorded in
[`pi_cam_declarative_workflow_50step.json`](validation/pi_cam_declarative_workflow_50step.json).
The complete Pythonic UI gate—including distributed NumPy expressions,
StatePool mapping, declarative workflow, and Xarray output access—is
[`pi_cam_pythonic_ui_complete_50step.json`](validation/pi_cam_pythonic_ui_complete_50step.json).
It ran 50 steps on 512 ranks and remained BFB for all four CAM
history/restart files.

## Install

```bash
git clone git@github.com:a85tract/freeCAM.git
cd freeCAM
git submodule update --init external/iCESM1.3.1_fzhu
uv sync --extra notebook --extra test
```

The supplied runtime and PBS jobs target NCAR Derecho. `Driver` reads the PBS
account from the configured CESM reference case; no personal project is
embedded in the package.

## Quick start

```python
import freecam as fc

with fc.Driver(
    case="PI-atm",
    nsteps=2,
    history_every=1,
    restart_every="end",
) as driver:
    print(driver.case)
    print(driver.preview())       # no PBS/MPI launch
    print(driver.cam.state.summary(rank=0))

    print(driver.cam.state.T.mean())
    driver.run(steps=2)

    temperature = driver.cam.state.T
    print(temperature.stats(rank="global"))
```

Constructing `Driver` does not submit a job. The first operation requiring
live state starts one persistent MPI model; later notebook cells reuse the
same ranks and arrays.

See [`examples/try_pi_cam.ipynb`](examples/try_pi_cam.ipynb) for the complete
walkthrough.

## StatePool

Every MPI rank owns the arrays for its local CAM domain. Select one rank or
request global statistics without copying the complete model to Jupyter:

```python
rank_zero_temperature = driver.cam.state.T.get(rank=0)
global_temperature = driver.cam.state.T.stats(rank="global")
global_mean = driver.cam.state.T.mean()
```

Scalar edits use ordinary augmented assignment. One compact command is sent
to the persistent model, then every MPI rank edits its own local array:

```python
driver.cam.state.T += 1.0
driver.cam.state.q *= 0.95
driver.cam.state.v.fill(0.0)

# NumPy-style slices are evaluated independently on every MPI rank.
driver.cam.state.T[:, 0, :] += 0.25
top_level_mean = driver.cam.state.T[:, 0, :].mean()
```

Slice edits never copy the full distributed field through Jupyter. The index
keeps the field's rank-local NumPy axis order, and freeCAM automatically skips
inactive `pcols` padding.

NumPy expressions are also lazy and distributed. The Notebook sends the small
expression tree; every MPI rank evaluates it against its own StatePool arrays:

```python
state = driver.cam.state
state.T = np.minimum(state.T + state.heating_rate * 1800.0, 320.0)
state.q[:] = np.maximum(state.q, 0.0)

# Only an explicit compute() copies one selected rank's result to Jupyter.
rank_zero_celsius = (state.T - 273.15).compute(rank=0)
```

The StatePool can also be inspected as a mapping:

```python
print(state.keys())
print(state.describe())
temperature = state["phys_state.t"]
```

For a small rank-independent array, use ordinary NumPy assignment. The same
values are copied into an independent array on every MPI rank:

```python
NLEV = driver.cam.state.T.metadata["shape"][1]
driver.cam.state.rh = np.zeros(NLEV)
```

For a grid-distributed variable whose local shape depends on each rank, declare
its CAM dimensions:

```python
driver.cam.state.experiment_tracer = fc.Variable(
    dims=("pcols", "pver", "chunks"),
    units="kg kg-1",
    initial=0.0,
    aliases=("tracer",),
    standard_name="experiment_tracer",
)
```

Both forms create Fortran-contiguous arrays and register them in each rank's
StatePool. `np.ndarray` means “copy this same-shaped value to every rank”; a
`Variable` means “resolve these dimensions against each rank's local CAM
partition.” Delete unused dynamic fields with:

```python
del driver.cam.state.experiment_tracer
del driver.cam.state.rh
```

Deletion is rejected while an installed process still uses the field.

## Physics processes

The public interface is flat. Users select processes by scientific or
original routine name; CAM source phases are an internal implementation
detail.

```python
physics = driver.cam.physics
print(physics.coverage)

physics.dry_adjustment.run()
physics.deep_convection.disable()
physics.deep_convection.enable()
```

The original flat API contained 36 workflow processes plus 262 source-catalog
interfaces. All 262 former catalog-only interfaces are runtime-process
templates with generated pointer adapters. In the admitted PI-CAM executable,
226 are loadable; the remaining 36 belong to inactive CARMA, COSP, or legacy
radiation configurations.

Bind caller-local arguments to StatePool fields, then place the process anywhere
in the complete workflow:

```python
process = physics.cloud_fraction_fice.bind()
driver.cam.workflow.insert(process, after="dry_adjustment")

process.run()                 # run only this process
process.disable()             # skip it in complete steps
process.enable()
process.move(before="radiation")
process.remove()              # remove the plan node and owned temporary fields
```

Common CAM derived state (`physics_state`, `physics_tend`, `cam_in`, and
`cam_out`) is bound to the live Python-owned StatePool without copying.
Literal values can also be passed to `bind(...)`; rank-local arrays are created
in StatePool when a caller variable has no existing field. Processes belonging
to a disabled physics configuration remain visible with metadata, but insertion
fails explicitly because their `.so` is not loaded by this case.

## Workflow

`driver.cam.workflow` is an ordered Python sequence of the processes used by a
complete model step:

```python
workflow = driver.cam.workflow
source_order = workflow[:]

radiation = workflow["radiation"]
vertical_diffusion = workflow["vertical_diffusion"]
custom_order = source_order.copy()
custom_order.remove(radiation)
custom_order.insert(custom_order.index(vertical_diffusion), radiation)

workflow[:] = custom_order
workflow[:] = source_order

# Familiar list operations are also available for a Physics object or handle.
workflow.append(custom_process)  # inserted before the required final export
removed = workflow.pop(-2)      # removes runtime processes; disables source ones
```

The three required control boundaries (`boundary_import`, `advance_timestep`,
and `boundary_export`) cannot be popped or removed.

The default workflow already uses every validated leaf boundary. Composite
Fortran stages such as the old run2 `finish` and run4 `wrapup` remain visible
as disabled compatibility entries, but they are not executed. Reordering or
disabling scientific processes is experimental; freeCAM keeps the granular
default in original source order.

Every workflow row reports both ownership and implementation:

- `control_owner="python"` means Python decides ordering and conditions.
- `fortran-numerical-kernel` is a scheme, dynamics, boundary mapping, or
  explicitly admitted numerical kernel.
- `fortran-state-service`, `fortran-io-service`, and
  `fortran-clock-mirror` are primitive services. They do not choose the next
  action or own the workflow.

No enabled default action has `kind="control"`. In particular, Python decides
whether an import is fresh, whether restart is due, when the public clock
advances, and when each service is invoked.

### Define a case with its workflow

A case can declare its complete atmosphere workflow before the model starts.
The factory receives the validated default order, so it can insert new
processes without copying dozens of required CAM actions into the notebook:

```python
class VolcanicAerosol(fc.Physics):
    name = "volcanic_aerosol"
    writes = ("phys_state.t",)

    def run(self, state, context):
        state.T += 1.0e-4


def volcanic_workflow(default):
    workflow = default.copy()
    workflow.insert_after("dadadj", VolcanicAerosol())
    workflow.insert_before("radiation_tend", VolcanicAerosol())
    return workflow


volcanic_case = fc.CaseConfig(
    name="PI-atm-volcanic",
    description="PI-atm with two volcanic aerosol tendencies",
    forcing="1850 prescribed SST and sea ice plus volcanic aerosol",
    make_atm=lambda: fc.FreeCAM(workflow=volcanic_workflow),
)

volcanic_case.workflow.describe()  # no job submission

fc.CASES.register(volcanic_case)

with fc.Driver(case="PI-atm-volcanic", nsteps=2) as driver:
    driver.cam.advance(steps=2)
```

The two `VolcanicAerosol()` entries become independent runtime processes named
`volcanic_aerosol_1` and `volcanic_aerosol_2`. Omitting an ordinary physics
process from the returned list disables it. Required boundary, clock, and
export actions are checked before the new order is applied.

`Driver` replays the captured coupler imports but does not require an
experiment's exports to match the unmodified oracle. For a strict BFB gate,
use `verify_boundary_exports=True`; an export difference is then reported
collectively by all MPI ranks before another CAM step can begin.

## Add a Python process

```python
class Heating(fc.Physics):
    name = "notebook_heating"
    after = "dry_adjustment"
    writes = ("phys_state.t",)

    def run(self, state, context):
        state.T += 0.01

process = driver.cam.workflow.insert(Heating())
process.run()
process.disable()
process.enable()
process.move(after="radiation")
process.remove()
```

The neighbor name determines placement; users do not specify a CAM phase.
`state.T` is a direct NumPy view of that MPI rank's StatePool array. The
callback is serialized with `cloudpickle`, broadcast to every rank, and runs
locally without a socket round trip. The older `tendency(fields, context)`
mapping interface remains supported.

## Output cadence

History and restart alarms are ordinary `Driver` options:

```python
driver = fc.Driver(
    case="PI-atm",
    nsteps=50,
    history_every=5,       # history every five model steps; None disables it
    restart_every="end",  # "end", a step interval, or None
)
```

Python decides whether the corresponding Fortran PIO service is called. The
defaults, `history_every=1` and `restart_every="end"`, preserve the validated
PI-CAM execution path.

CAM output is exposed lazily through Xarray after the run:

```python
print(driver.cam.history.files)
print(driver.cam.history.streams)

with driver.cam.history.open("h0") as history:
    history["T"].isel(time=-1).plot()

with driver.cam.restart.open("r") as restart:
    print(restart)
```

## Build and test

```bash
uv run pytest -q tests/unit
```

The source inventory, adapter generation, compile validation, runtime loading,
and 512-rank BFB jobs are implemented by the scripts in [`tools/`](tools/) and
[`validation/jobs/`](validation/jobs/).

## Scope

freeCAM currently supports one admitted PI-CAM configuration. Adding another
CAM configuration requires its own build context and numerical validation;
the runtime does not silently reuse incompatible COSP, CARMA, or radiation
state.
