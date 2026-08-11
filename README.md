# freeCAM

freeCAM is a Python-controlled runtime for the CAM atmosphere component from
iCESM1.3.1. Python owns the model workflow and observable state. Original
Fortran routines remain responsible for numerical calculations and are called
through generated C-interoperable adapters.

The current release supports one configuration: the `ne16` PI-atm CAM case
with CAM5 physics, SE dynamics, 512 MPI ranks, and replayed atmosphere coupling
inputs.

## Status

| Capability | Status |
| --- | --- |
| Python-owned CAM workflow | Available |
| Python-owned rank-local StatePool | Available |
| Complete CAM step | Available |
| Individual phase and process calls | 36 workflow boundaries; 21 additional StatePool-promotable processes |
| Runtime Python process | Available |
| Runtime Fortran process | Available |
| PI-atm 50-step comparison | Bitwise identical |
| Other cases and resolutions | Not supported yet |

The committed comparison record is
[`validation/pi_cam_python_zero_copy_state_vs_oracle_50step_bfb.json`](validation/pi_cam_python_zero_copy_state_vs_oracle_50step_bfb.json).

## Requirements

- Python 3.11
- `uv`
- MPI with `mpi4py`
- A Fortran compiler compatible with the iCESM build
- NCAR Derecho for the supplied PBS jobs and reference paths

## Installation

```bash
git clone --branch main git@github.com:a85tract/freeCAM.git
cd freeCAM
git submodule update --init external/iCESM1.3.1_fzhu
uv sync --extra notebook --extra test
```

The default `Driver` paths expect the existing PI-atm reference case and
boundary capture under the current user's NCAR work and scratch directories.
They can be overridden through constructor arguments when needed.

For PBS launches, `Driver` reads `CHARGE_ACCOUNT` or `PROJECT` from the
reference case's `env_batch.xml`. Pass `account="..."` only when an explicit
override is needed; freeCAM does not embed a personal allocation in source.

## Quick start

```python
import freecam as fc

driver = fc.Driver(case="PI-atm", nsteps=2)
print(driver.case)

fig, axes = driver.cam.state.plot(rank=0, label="initial")
trace = driver.execute()

print(driver.cam.state.summary(rank=0))
print(f"executed {len(trace)} CAM actions")

driver.close()
```

Creating `Driver` does not immediately start MPI. The first operation that
needs live CAM state prepares an isolated run directory and starts one
persistent CAM worker. Later calls reuse the same MPI ranks and StatePools.

The complete walkthrough is
[`examples/try_pi_cam.ipynb`](examples/try_pi_cam.ipynb).

## State

Every MPI rank owns a StatePool containing NumPy arrays for that rank's local
CAM domain. Read one field or calculate statistics without copying the full
model state into Jupyter:

```python
temperature = driver.cam.state.T

rank_zero_values = temperature.get(rank=0)
global_statistics = temperature.stats(rank="global")
```

Add a distributed variable with attribute assignment:

```python
driver.cam.state.experiment_tracer = fc.Variable(
    dims=("pcols", "pver", "chunks"),
    units="kg kg-1",
    initial=0.0,
    aliases=("tracer",),
    standard_name="experiment_tracer",
)
```

The named dimensions are resolved independently on every rank. Each rank
allocates a Fortran-contiguous array with its local shape and registers it in
its StatePool.

Remove an unused dynamic variable with:

```python
del driver.cam.state.experiment_tracer
```

The runtime rejects deletion while an installed process still depends on the
field.

## Workflow

`driver.cam.workflow` is the live process order. It is iterable and renders as
a table in Jupyter.

```python
for action in driver.cam.workflow:
    print(action.phase, action.name)
```

A complete step follows the validated CAM order and advances model time:

```python
driver.cam.advance(steps=1)
```

Individual phases and processes can be called for controlled experiments:

```python
print(driver.cam.physics.coverage)
driver.cam.physics                 # rich table in Jupyter

driver.cam.physics.dadadj.run()
driver.cam.physics.deep_convection.disable()
driver.cam.physics.deep_convection.enable()

for process in driver.cam.physics.by_phase("cam_run1"):
    print(process.name, process.runnable, process.capability)
```

The main API is deliberately flat. Original module names and caller/callee
structure are metadata, so a user writes:

```python
driver.cam.physics.deep_convection
driver.cam.physics.cloud_fraction_fice
driver.cam.physics.zm_conv_evap
```

and never needs a path such as `physics.kernels.zm_conv.zm_conv_evap`.

The admitted PI-atm call graph contains 372 reachable Fortran procedures. They
are not automatically 372 physical processes: deterministic signature rules
classify 276 state-updating routines as physical processes, 95 read-only math
or lookup routines as helpers, and one lifecycle routine as non-physics. The
36 independently runnable workflow boundaries overlap 14 source processes, so
`driver.cam.physics` currently contains 298 unique flat handles. Helpers remain
searchable in `driver.cam.physics.catalog`, but do not pollute the main physics
API. The `coverage` mapping reports every count separately.

Every handle is honest about its capability:

```python
deep = driver.cam.physics.deep_convection
print(deep.runnable)          # True: complete runtime boundary

evap = driver.cam.physics.zm_conv_evap
print(evap.runnable)          # False: source kernel discovered by the call graph
print(evap.metadata)          # source, parents, arguments, adapter status, blockers
```

Calling `.run()`, `.disable()`, or `.move()` on a catalog-only entry fails with
the missing StatePool/context bindings instead of silently running the wrong
caller context. Fine-grained calls do not automatically execute prerequisites
and do not advance model time.

Twenty-one generated-adapter processes can turn caller-local arguments into
explicit rank-local StatePool fields. The normal interface hides that plumbing:
it infers unambiguous model inputs such as `phys_state.t`, supplies model values
such as `ncol`, and allocates output-only arguments automatically.

```python
result = driver.cam.physics.cloud_fraction_fice()

print(result.fice.stats(rank="global"))
print(result.fsnow.stats(rank="global"))
result.remove()
```

Only an experiment that replaces the model temperature needs an explicit
keyword argument:

```python
result = driver.cam.physics.cloud_fraction_fice(t=my_temperature)
```

Fourteen use named dimensions that can be resolved automatically; seven use
assumed/expression shapes and therefore require explicit field bindings or an
aggregate initial array. The current PI-atm native image contains 18 of these;
three COSP/MODIS processes have no module in this case build and remain
StatePool-bound but non-runnable. The 18 generated adapters live in a lazy
add-on `.so`; the validated main CAM image is unchanged and only becomes
globally visible when the user explicitly runs a promoted process. Promotion
creates a standalone experimental call; it does not silently insert a
duplicate call into the validated full timestep. Routines with derived types,
pointers, allocatables, optional arguments, or missing native symbols remain
catalog-only with an explicit blocker. Regenerate the combined direct-adapter
descriptor with:

```bash
uv run python tools/build_pi_cam_promoted_kernels.py
```

The catalog is reproducible, not a handwritten Python list. AST reachability
comes from the committed source inventory, while the small reviewed naming
layer lives in
[`native/pi_cam/physics_process_rules.yaml`](native/pi_cam/physics_process_rules.yaml).
Regenerate the packaged catalog with:

```bash
uv run python tools/build_pi_cam_physics_catalog.py
```

## Python processes

Subclass `freecam.Physics` and insert the object into the workflow:

```python
class TracerSource(fc.Physics):
    name = "tracer_source"
    phase = "cam_run1"
    after = "dadadj"
    writes = ("tracer",)

    def tendency(self, fields, context):
        fields["tracer"][...] += 1.0e-6 * context.timestep_seconds


process = driver.cam.workflow.insert(TracerSource())
```

The callback runs on every MPI rank against that rank's declared fields.
Writable fields are restored collectively if the callback fails.

```python
process.run()
process.move(before="deep_convection")
process.disable()
process.enable()
process.remove()
```

Notebook callbacks are serialized with `cloudpickle`. They are trusted code,
not a security sandbox.

## Fortran processes

A runtime Fortran process consists of original source, metadata, and a small
device descriptor. freeCAM generates the `bind(C)` adapter, compiles a shared
library, and loads it on every target rank.

```python
process = driver.cam.physics.install_fortran(
    driver.repo / "examples/plugins/runtime_temperature_offset/device.yaml",
    project_root=driver.repo,
    process="runtime_temperature_offset",
    phase="cam_run1",
    after="dadadj",
    unsafe=True,
)

process.run()
```

See
[`examples/plugins/runtime_temperature_offset`](examples/plugins/runtime_temperature_offset)
for a complete minimal device.

## Architecture

```text
Jupyter / Python
    Driver, state, workflow
             |
             | commands and selected results
             v
persistent MPI worker
    one Python CAMDriver and StatePool per rank
             |
             | array pointers, shapes, and scalar arguments
             v
generated bind(C) adapters
             |
             v
original iCESM CAM numerical routines
```

The control socket carries commands, status, traces, and requested field
results. The complete StatePool remains in the MPI ranks.

## Command-line execution

The low-level MPI entry point is useful for batch validation:

```bash
mpiexec -n 512 .venv/bin/freecam \
  --config configs/pi_cam_icesm131.yaml \
  --boundary /path/to/boundary/replay \
  --run-dir /path/to/isolated/run \
  --steps 50
```

For interactive work, prefer `freecam.Driver`; it manages the persistent
worker and machine-specific setup.

## Validation

Run the Python tests:

```bash
uv run pytest -q
```

Run the 512-rank PI-atm gate on Derecho:

```bash
qsub validation/jobs/pi_cam_python_zero_copy_state_50step.pbs
```

Build and validate the StatePool-promoted original processes with:

```bash
qsub validation/jobs/pi_cam_promoted_statepool_build.pbs
qsub validation/jobs/pi_cam_promoted_statepool_50step.pbs
```

The gate compares CAM history output with the original iCESM reference without
a tolerance. Job IDs and paths are excluded from the numerical comparison.

Other PI-CAM jobs under [`validation/jobs`](validation/jobs) cover boundary
capture, direct-kernel execution, runtime extensions, and the persistent
Notebook session.

## Rebuilding native devices

Prepare the pinned source and build the PI-CAM libraries with:

```bash
uv run python tools/prepare_pi_cam_source.py
uv run python tools/build_pi_cam_devices.py --help
```

Generated libraries and build products are written below `build/` and are not
committed.

## Repository layout

```text
configs/                 supported PI-CAM configuration
examples/                Jupyter walkthrough and runtime plugin example
external/                pinned iCESM source submodule
native/pi_cam/           source patches, adapter rules, and native support
src/freecam/pi_cam/      PI-CAM control, state, adapters, and Notebook API
tests/unit/              unit tests for the current PI-CAM implementation
tools/                   PI-CAM source, build, capture, and validation tools
validation/              PI-CAM PBS jobs and machine-readable results
```

`src/freecam/model` and `src/freecam/core` contain a small set of internal ABI
and runtime helpers used by PI-CAM. They are not public APIs.

## Current limitations

- Only the supplied PI-atm CAM configuration is supported.
- Surface coupling inputs are replayed from the reference PI-atm run.
- A Fortran kernel cannot be paused in the middle of a call.
- Fine-grained process reordering is experimental unless separately validated.
- Opaque Fortran module state is not treated as Python-owned StatePool memory.

Unsupported configurations fail explicitly rather than silently changing the
scientific path.

## License

freeCAM is licensed under Apache-2.0. The pinned iCESM source, generated source
patches, and derived interfaces retain their upstream licenses. See
[`NOTICE`](NOTICE) and [`LICENSES`](LICENSES).
