# pycam-sima

`pycam-sima` is a Python control layer for the fixed CAM-SIMA configuration
`FKESSLER + ne3pg3 + L30 + moist_baroclinic_wave_dcmip2016`.

It has two execution modes:

- `run` is the small Python-owned Kessler kernel path. Every numeric buffer is
  allocated by NumPy and individual CCPP scheme wrappers borrow those buffers.
- `run-full` is the complete CAM-SIMA path. Python and Taskflow schedule
  `cam_init`, `cam_run1`, `cam_run2`, `cam_run3`, timestep finalization and time
  advancement. The shared library contains the real SE dycore, analytic IC,
  dynamics/physics mappings and Kessler suite. Long-lived CAM allocations are
  writable zero-copy NumPy views registered in the Python `StatePool`.

The complete model exposes the same Python control concepts through the
Notebook MPI controller:

```python
from pycam_sima import (
    FullCAMRuntimeOptions,
    FullCAMStepPlan,
    NotebookSession,
)

options = FullCAMRuntimeOptions(
    timestep_seconds=1800,
    physics_profile="kessler",
    mediator_present=False,
)
model = NotebookSession(
    config,
    run_dir=run_dir,
    options=options,
    step_plan=FullCAMStepPlan.default(),
)
model.start()

print(model.step_plan.describe())
temperature = model.parameters.air_temperature.get(rank=0)
model.step()
```

The runtime options are fixed at `cam_init`. Live fields may be changed at any
Python boundary. Changing or disabling a required complete-CAM phase requires
an explicit `unsafe=True`; use the FADIAB profile for a scientifically valid
SE dynamics-only run.

The small Kessler path also exposes a declarative Python step plan and mutable
runtime controls:

```python
from pycam_sima import FKesslerDriver, RuntimeOptions, StepPlan
from pycam_sima.config import CaseConfig

config = CaseConfig.from_yaml("configs/fkessler_ne3pg3.yaml")
options = RuntimeOptions(
    timestep_seconds=1800,
    physics_before=True,
    physics_after=True,
    dynamics=True,
)
model = FKesslerDriver(config, options=options, step_plan=StepPlan.default())
model.initialize()

print(model.step_plan.describe(model.options))
model.parameters.surface_reference_pressure = 98_500.0
model.step()
temperature = model.pool["air_temperature"]
```

Optional physics/dynamics sections can be switched between steps. Required
lifecycle phases are protected, and changing CAM's default order requires an
explicit `unsafe=True`. See `docs/KERNEL_STEP_CONTROL.md` for parameter edits,
phase-boundary observers, and order experiments.

The source is pinned at `external/CAM-SIMA` commit
`f8daa568eae2696b7c4ebff7768f02f5d097d9df`.

## Full-model workflow

Create the 24-rank, 50-step native reference case:

```bash
uv run python tools/create_reference_case.py --build --submit
```

Build the complete position-independent CAM/SE shared library from that case:

```bash
uv run python tools/build_full_native.py \
  --case-root reference/cases/FKESSLER_ne3pg3_gnu_24x50

uv run python tools/build_full_native.py \
  --case-root reference/cases/FADIAB_ne3pg3_gnu_24x50 \
  --output build/libpycam_sima_adiabatic_full.so
```

Run the Python driver through PBS:

```bash
qsub jobs/fkessler_full_24x50.pbs
```

Compare the five prognostic history fields, failing closed on a missing file,
extra file, shape/dtype change, or one-bit numerical difference:

```bash
uv run pycam-sima compare-history \
  /glade/derecho/scratch/ruitong/pycam-sima/FKESSLER_ne3pg3_gnu_24x50/FKESSLER_ne3pg3_gnu_24x50/run \
  /glade/derecho/scratch/ruitong/pycam-sima/pyfull_bfb/FKESSLER_ne3pg3_24x50/run
```

The validated run compared 51 timestamps (the nstep-0 send cycle plus 50
requested steps). `T`, `Q`, `U`, `V`, and `PS` were bitwise identical; a
second pass also verified all 26 numeric variables in the history files.
Evidence is recorded in `validation/fkessler_full_bfb.json`.

## Inspecting state in Python

Interactive sessions can also execute one top-level CAM phase at a time:

```python
model.run_phase("cam_run2")
temperature_after_physics = model.get_field("air_temperature", rank=0)
model.run_phase("cam_run3")
temperature_after_dynamics = model.get_field("air_temperature", rank=0)
```

The safe default order is checked by a Python state machine. See
`docs/PHASE_CONTROL.md` for phase status, explicit unsafe-order experiments,
and the `FADIAB` no-physics-forcing dynamics configuration.

Use `--watch` at any Python phase boundary:

```bash
mpiexec -n 24 .venv/bin/python -m pycam_sima.cli run-full \
  configs/fkessler_ne3pg3.yaml \
  --run-dir /path/to/runtime-directory \
  --steps 50 \
  --watch air_temperature \
  --watch-event step_end
```

The full state registry currently exposes temperature, winds, wet/dry surface
and layer pressures, interface pressures, geopotential/geopotential height,
omega, Exner function, dry static energy, total physics tendencies, and the
complete CCPP constituent array. Interactive observers may edit these arrays;
validation observers are read-only. `--snapshot-dir` writes selected fields to
per-rank NPZ snapshots.

## Jupyter Notebook

An ordinary single-process Notebook can control a separate 24-rank MPI worker
and receive live fields without writing an intermediate snapshot:

```python
from pycam_sima import NotebookSession

with NotebookSession(
    "configs/fkessler_ne3pg3.yaml",
    run_dir="/path/to/fresh/runtime-directory",
    env_script="reference/cases/FKESSLER_ne3pg3_gnu_24x50/.env_mach_specific.sh",
) as model:
    model.step()
    temperature = model.parameters.air_temperature.get(rank=0)
    print(temperature.min(), temperature.max())
```

From a Derecho login-node Notebook, `start()` automatically submits the
24-rank worker through PBS; inside an existing compute allocation it launches
locally. See `docs/JUPYTER.md` for field metadata, all-rank statistics, live
field modification, cleanup, and global-layout limitations.

The single maintained interactive Notebook is:

```text
examples/try_notebook_session.ipynb
```

It keeps the model alive between cells. `examples/try_notebook_session.py` is
the non-interactive command-line companion.

## Development checks

```bash
uv sync --extra build --extra test
uv run pytest
uv run pycam-sima build-native
uv run pycam-sima inspect-contract
qsub jobs/fkessler_kernel_smoke_24.pbs
```
