# pycam-sima

`pycam-sima` is a Python-owned, CCPP-like coupling framework for numerical
model components. Python supplies the lifecycle, process scheduler, common
field bus, MPI communication, and checkpoint semantics; independently built
Fortran devices supply numerical schemes through a generated C ABI.

The first fully validated model target is CAM-SIMA FKESSLER,
`ne3np4.pg3`, L30, 24 MPI ranks, a 1800-second timestep, and the DCMIP2016
moist baroclinic-wave initial condition.

Python owns the model lifecycle, clock, grid/decomposition metadata, persistent
NumPy state, mpi4py communication, phase and CCPP-scheme ordering, and NetCDF
history output.
Every MPI worker constructs the CSLAM/FVM geometry directly in Python from the
cubed-sphere topology and hybrid coordinate; no pre-generated grid file is
loaded. Dimensions and runtime controls are passed from Python through ABI v2.
FP-sensitive SE layout values are validated in a separate ABI call immediately
before the numerical call, so a control argument cannot change the kernel's
floating-point instruction order.
The main shared library contains numerical dycore/mapping kernels only.
CCPP schemes such as Kessler are compiled from the pinned, unmodified upstream
Fortran source into separate device libraries. A generated adapter translates
explicit NumPy pointers and scalar values to the scheme's original interface.
No device exports or calls `cam_init`, `cam_run*`, ESMF, PIO, or CAM's native
control loop. Clean builds do not load the CAM machine environment and reject
MPI, ESMF, PIO, NetCDF, HDF5, and RPATH dependencies.

The current architecture is also available as an interactive
[online diagram](https://pycam-sima-architecture-2026.bubblehuntr.chatgpt.site)
and as the repository-local
[`docs/pycam_sima_architecture.html`](docs/pycam_sima_architecture.html).

Runnable examples are split by execution mode:

- [`examples/try_notebook_session.ipynb`](examples/try_notebook_session.ipynb)
  keeps a 24-rank MPI worker alive for interactive phase, scheme, step, and
  field control through the authenticated socket bridge.
- [`examples/try_dask_fanout.ipynb`](examples/try_dask_fanout.ipynb) submits a
  restartable base PBS/MPI task, fans out independent Dask branches, and reads
  their final checkpoints and per-step history fields.

```text
pycam_sima/
  core/       MPI loader and remote-field utilities
  model/      Python driver, StatePool, DeviceRegistry, device runtime
  notebook/   Jupyter/PBS controller and MPI worker
devices/      source/metadata descriptors for pluggable Fortran schemes
native/
  devices/    portable dependency providers used by generated adapters
  kernels/    non-CCPP dycore/mapping kernels and clean build
build/devices/
  <name>/     generated adapter, manifest, and one independent device .so
```

## Run the model

```bash
uv sync --extra test --extra notebook
uv run pycam-sima build-kernels
uv run python tools/validate_kessler_kernel.py
readelf -d build/libpycam_sima_kernels.so
readelf -d build/devices/kessler/libpycam_device_kessler.so

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
a selectable pycam-sima backend. The source-device build and full-run evidence
is recorded in
[`validation/source_preserving_devices.json`](validation/source_preserving_devices.json).

## Source-preserving Fortran devices

A device is defined by a small YAML description, the original CCPP `.meta`
file, and the original Fortran sources. `build-kernels` runs CAM-SIMA's own
CCPP parser to verify that metadata and source signatures agree, scans
Fortran module dependencies, generates the C ABI adapter and JSON manifest,
and compiles one isolated `.so`:

```text
devices/kessler/device.yaml
       ├── kessler.meta
       ├── original kessler.F90
       └── ccpp_kinds provider
                   │
                   ▼
            CCPP parser/verifier
                   │
                   ├── generated kessler_adapter.F90
                   ├── generated device.json
                   └── libpycam_device_kessler.so
```

The generated adapter contains calls to `kessler_init` and `kessler_run`; it
does not contain Kessler's formulas or loops. The former handwritten
`native/kernels/kessler_kernel.F90` algorithm copy has been removed.

Build one descriptor independently with:

```bash
uv run pycam-sima build-device devices/kessler/device.yaml
```

At runtime, `DeviceRegistry` reads `device.json`. Every ABI argument declares
its CCPP `standard_name`, dtype, rank, dimensions, units, intent, and binding.
`StatePool` resolves that standard name to its canonical array or a zero-copy
constituent slice, verifies the complete contract, and passes the existing
Fortran-contiguous NumPy address:

```python
from pycam_sima import DeviceRegistry

registry = DeviceRegistry("build/devices")
assert registry.process_names == {"kessler", "kessler_update"}
print(registry.describe())
registry.invoke("kessler", model.pool)
```

Users normally keep calling `model.run_scheme("kessler", ...)`; the driver
routes that process through the registry. Adding a compatible CCPP scheme does
not require adding a hard-coded ctypes signature or StatePool argument list to
`backend.py`.

Fortran module state is explicit in the device policy. Kessler is
`reinitialize_each_run`: Python passes `lv`, `pref`, and `rhoqr` through the
generated initialize adapter immediately before each original run, so Python
remains the authoritative and checkpointed state. `stateless` and
`initialize_once` policies are also supported. The latter is marked as
persistent native state and must be treated explicitly by future restart
contracts.

Dependency handling is fail-closed. Intrinsic modules, source-local modules,
and declared portable providers are accepted. An undeclared module or a
host/framework dependency such as MPI, ESMF, PIO, or `cam_history` stops device
generation instead of silently linking the complete CAM runtime.
See [`docs/DEVICE_AUTHORING.md`](docs/DEVICE_AUTHORING.md) for the descriptor,
ABI-v1 support matrix, state policies, and new-scheme checklist.

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

## Dask experiment fan-out

Dask is an optional experiment-level scheduler; it does not replace mpi4py
inside one model run. A base task runs a restartable 24-rank MPI segment and
returns a `Future` whose result contains an immutable serialized checkpoint for
all ranks. Multiple branch tasks may depend on that same Future. Each branch
creates new MPI processes, restores private Fortran-contiguous NumPy arrays,
applies its edits, and continues independently without a persistent socket.

```python
from dask.distributed import Client
from pycam_sima import (
    BranchSpec,
    DaskExperimentClient,
    FieldEdit,
)

client = Client(processes=False, n_workers=2, threads_per_worker=1)
experiments = DaskExperimentClient(
    client,
    config=repo / "configs/fkessler_model.yaml",
    initial_run_dir=reference_run,
    run_root=scratch / "pycam-sima/dask-experiments",
    python_executable=repo / ".venv/bin/python",
)

base = experiments.submit_base(BranchSpec("base", steps=10))
branches = experiments.fork(
    base,
    (
        BranchSpec("control", steps=5),
        BranchSpec("no-kessler", steps=5, disable_schemes=("kessler",)),
        BranchSpec(
            "warm",
            steps=5,
            field_edits=(FieldEdit("air_temperature", "add", 1.0),),
        ),
    ),
)
summaries = experiments.summaries(branches)
```

`BranchSpec` remains the compact interface for edit-then-step experiments.
For phase/scheme boundaries, submit a serializable `SegmentPlan`:

```python
from pycam_sima import (
    ObserveFields,
    RunPhase,
    RunScheme,
    SegmentPlan,
)

granular = experiments.submit_plan(
    base,
    SegmentPlan(
        "kessler-then-map",
        (
            RunScheme(
                "kessler",
                group="physics_before_coupler",
            ),
            ObserveFields((
                "potential_temperature",
                "large_scale_precipitation_rate",
            )),
            RunPhase("physics_to_dynamics"),
        ),
        unsafe=True,
    ),
)
summary = experiments.summary(granular).result()
temperature = experiments.field(
    granular,
    "potential_temperature",
    rank=0,
).result()
```

All actions in one plan share one live StatePool and one 24-rank MPI launch.
`submit_action(parent, name=..., action=...)` creates a separate checkpointed
Dask task when an action boundary must become a Future or fork point. A
separate task starts new MPI processes and restores private arrays from the
parent checkpoint; it does not inherit process memory. Standalone phase,
scheme, and scheme-group actions require `unsafe=True`. They do not advance the
clock or write history; `RunSteps` retains the complete validated model
semantics.

The version-1 action vocabulary is `PrepareInitialStep`, `RunPhase`,
`RunScheme`, `RunSchemeGroup`, `RunSteps`, `SetSchemeEnabled`, `MoveScheme`,
`FieldEdit`, and `ObserveFields`. A plan is completely validated before its
first action mutates model state.

The default `execution_mode="pbs"` submits a PBS job for every segment. The
single-allocation mode reserves one node once, starts the Dask scheduler and
worker inside it, and lets every Dask task call `mpiexec` directly:

```bash
qsub jobs/dask_allocation_fanout_24x50.pbs
```

Its Python controller uses `execution_mode="allocation"` and one Dask worker.
All base/branch summaries therefore have the same outer `PBS_JOBID`, and no
branch invokes `qsub`. Full-node 24-rank segments are serialized; concurrent
branches require explicit multi-node resource partitioning.

`summaries()` returns only small metadata to the Notebook, including each
branch's run, history, checkpoint, log paths, segment plan, and action trace.
`ObserveFields` adds rank-local and global statistics to that trace.
`experiments.field()` extracts one rank-local array on a Dask worker without
downloading the complete snapshot. `gather()` is also available, but it
downloads each complete checkpoint bundle. Dask keeps a
bundle in distributed memory while its Future is referenced; the durable
checkpoint directories allow restart after a Dask worker or PBS allocation has
ended. `run-segment` is the non-socket MPI entry point used by these tasks.

The committed validation in `validation/dask_checkpoint_fanout.json` records a
real 24-rank base Future and two PBS branches. The control branch changed no
fields; the edited branch changed only `air_temperature` by exactly `+1.0`.
The separate 25+25 restart gate produced all 51 history files BFB against the
external CAM-SIMA oracle.

`validation/dask_granular_actions.json` records the action-level gates: real
PBS-mode parent and phase segments, one-allocation batch and chained action
execution, all-rank StatePool bitwise comparison, direct
`CAMDriver.run_phase()` comparison, one-call Kessler proof, field extraction,
and the unchanged 24-rank/50-step history BFB result.

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
