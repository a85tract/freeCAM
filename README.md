# pycam-sima

`pycam-sima` is a Python-owned driver for one validated CAM-SIMA target:
FKESSLER, `ne3np4.pg3`, L30, 24 MPI ranks, a 1800-second timestep, and the
DCMIP2016 moist baroclinic-wave initial condition.

Python owns the model lifecycle, clock, grid/decomposition metadata, persistent
NumPy state, mpi4py communication, phase and CCPP-scheme ordering, and NetCDF
history output.
Every MPI worker constructs the CSLAM/FVM geometry directly in Python from the
cubed-sphere topology and hybrid coordinate; no pre-generated grid file is
loaded. Dimensions and runtime controls are passed from Python through ABI v2.
FP-sensitive SE layout values are validated in a separate ABI call immediately
before the numerical call, so a control argument cannot change the kernel's
floating-point instruction order.
The shared library contains stateless numerical kernels only. It does not
export or call `cam_init`, `cam_run*`, ESMF, PIO, or CAM's native control loop.
Its clean build does not load the CAM machine environment and has no MPI,
ESMF, PIO, NetCDF, or HDF5 dynamic dependency.

The current architecture is also available as an interactive
[online diagram](https://pycam-sima-architecture-2026.bubblehuntr.chatgpt.site)
and as the repository-local
[`docs/pycam_sima_architecture.html`](docs/pycam_sima_architecture.html).

```text
pycam_sima/
  core/       MPI loader and remote-field utilities
  model/      complete Python-owned CAM driver
  notebook/   Jupyter/PBS controller and MPI worker
```

## Run the model

```bash
uv sync --extra test
uv run pycam-sima build-kernels
uv run python tools/validate_kessler_kernel.py
readelf -d build/libpycam_sima_kernels.so

RUN_DIR=/path/containing/atm_in \
HISTORY_DIR=/new/history/directory \
STEPS=50 qsub -V jobs/fkessler_model_24x50.pbs

uv run pycam-sima compare-history \
  /path/to/oracle/history /new/history/directory \
  --files 51 --numeric-variables 26
```

The maintained PBS jobs merge standard output and error and write their job
logs under `logs/`, rather than into the repository root.

The history gate compares filenames, timestamps, dtype, shape, and float64 bit
patterns for all 51 output times and 26 diagnostic variables. The upstream
CAM-SIMA executable is used only to produce an external test oracle; it is not
a selectable pycam-sima backend.

## Python API

```python
from pycam_sima import (
    CAMDriver,
    ModelConfig,
    PHYSICS_AFTER_COUPLER,
)

config = ModelConfig.from_yaml("configs/fkessler_model.yaml")
model = CAMDriver(config, run_dir=run_dir, history_dir=history_dir).start()
assert model.backend.call_count == 0  # initialization is pure Python

temperature = model.get_field("air_temperature")
model.prepare_initial_step()          # writes nstep=0
model.run_scheme_group(PHYSICS_AFTER_COUPLER)
model.step()
model.finalize()
```

The fixed Kessler suite exposes all 19 `physics_before_coupler` schemes and all
5 `physics_after_coupler` schemes individually. `model.scheme_plan.describe()`
shows their exact pinned-XML order, and `run_scheme()` pauses after one scheme:

```python
model.run_scheme("kessler", group="physics_before_coupler")

# Explicit control experiments are marked unsafe because they are not BFB gates.
model.scheme_plan.disable("kessler_diagnostics", unsafe=True)
model.scheme_plan.move(
    "kessler", after="kessler_update", unsafe=True,
)
# Move a before-coupler scheme to the end of the after-coupler group.
model.scheme_plan.move(
    "kessler", to_group="physics_after_coupler", unsafe=True,
)
model.scheme_plan.reset()
```

`step()` executes only enabled schemes, in the editable plan order and current
execution group. Schemes may move within a group or between the before/after
groups. Their source-qualified identity remains stable because
`check_energy_scaling` occurs in both groups. The default unmodified plan is
the only scientifically validated order.

A cross-group move changes the execution stage: moving a scheme from before to
after removes it from nstep=0/end-of-step preparation and places it at the
beginning of the following model step.

All persistent fields have `owner="python"`. Prognostic, tendency, and process
arrays are writable at phase boundaries. Static grid/topology arrays require
`unsafe=True`; kernel calls must preserve every NumPy address.

`FVMKernelConfig.from_pool(model.pool)` derives `nc`, `nlev`, tracer count,
halo widths, reconstruction order, quadrature count, jet-level range, active
level range, and the large-Courant switch in Python. The resulting C-compatible
configuration is supplied to both FVM kernel calls; the Fortran wrappers do not
define case dimensions or timestep controls.

To preserve CAM's BFB floating-point instruction order, `build-kernels` also
generates a compile-time specialization module from
`configs/fkessler_model.yaml`. ABI v2 checks the Python values against that
specialization on every FVM call. A different shape therefore requires a new
Python configuration and rebuild; it never silently reuses the wrong layout.

## Jupyter

`NotebookSession` controls 24 MPI workers from a normal one-process Notebook:

```python
from pycam_sima import NotebookSession

with NotebookSession(
    "configs/fkessler_model.yaml",
    run_dir=run_dir,
) as model:
    model.prepare_initial_step()
    model.scheme_plan.describe("physics_before_coupler")
    model.run_scheme("kessler", group="physics_before_coupler")
    model.step()
    field = model.parameters.air_temperature.get(rank=0)
```

On a Derecho login node, `start()` submits the worker through PBS; inside an
allocation it launches locally. The only maintained Notebook is
`examples/try_notebook_session.ipynb`. See `docs/JUPYTER.md` for details.

## Development

```bash
uv run pytest
uv run python tools/validate_kessler_kernel.py
git diff --check
```

The CAM-SIMA source is pinned at commit
`f8daa568eae2696b7c4ebff7768f02f5d097d9df`.
