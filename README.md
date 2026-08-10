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
| Individual phase and process calls | Available for controlled experiments |
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
driver.cam.physics.dadadj.run()
driver.cam.phases.cam_run1.run()
```

These fine-grained calls do not automatically execute missing prerequisites
and do not advance model time.

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
