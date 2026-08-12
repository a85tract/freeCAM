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

with fc.Driver(case="PI-atm", nsteps=2) as driver:
    print(driver.case)
    print(driver.cam.state.summary(rank=0))

    driver.cam.advance(steps=2)

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
```

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

## Add a Python process

```python
class Heating(fc.Physics):
    name = "notebook_heating"
    after = "dry_adjustment"
    writes = ("phys_state.t",)

    def tendency(self, fields, context):
        fields["phys_state.t"][...] += 0.01

process = driver.cam.workflow.insert(Heating())
process.run()
process.disable()
process.enable()
process.move(after="radiation")
process.remove()
```

The neighbor name determines placement; users do not specify a CAM phase.
The callback is serialized with `cloudpickle`, broadcast to every MPI rank,
and runs against that rank's local StatePool views.

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
