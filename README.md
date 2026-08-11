# freeCAM

freeCAM runs the CAM atmosphere component from iCESM1.3.1 with a Python
control layer. Python owns the execution order and rank-local StatePool;
original Fortran routines remain the numerical kernels and are called through
generated C-interoperable adapters.

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
[`pi_cam_python_zero_copy_state_vs_oracle_50step_bfb.json`](validation/pi_cam_python_zero_copy_state_vs_oracle_50step_bfb.json).
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

Add a distributed variable with normal attribute assignment:

```python
driver.cam.state.experiment_tracer = fc.Variable(
    dims=("pcols", "pver", "chunks"),
    units="kg kg-1",
    initial=0.0,
    aliases=("tracer",),
    standard_name="experiment_tracer",
)
```

Each rank resolves the named dimensions locally, allocates a
Fortran-contiguous NumPy array, and registers it in that rank's StatePool.
Delete an unused dynamic field with:

```python
del driver.cam.state.experiment_tracer
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
interfaces. All 262 former catalog-only interfaces now have compiled pointer
adapters. A standalone call uses keyword arguments for caller-local inputs and
returns named output StatePool fields:

```python
result = physics.calc_hltalt(t=250.0)
print(result.hltalt.stats(rank="global"))
result.remove()
```

Common CAM derived state (`physics_state`, `physics_tend`, `cam_in`, and
`cam_out`) is bound to the live Python-owned StatePool without copying.
Processes belonging to a disabled physics configuration remain visible with
metadata but are not presented as runnable in the PI-CAM image.

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

`workflow.expand()` exposes the validated leaf boundaries. Reordering or
disabling scientific processes is an experimental operation; freeCAM keeps
the safe default in original source order.

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
