# Jupyter Notebook interface

`NotebookSession` lets one ordinary Jupyter kernel control the validated
24-rank CAM-SIMA configuration. It starts a separate MPI worker and exposes a
synchronous Python API over an authenticated socket connection.

The Notebook can run on a Derecho login node. In that environment,
`NotebookSession.start()` automatically submits a 24-rank PBS job and waits for
the compute worker to connect back to the kernel. Inside an existing compute
allocation it launches the worker locally. Prepare a fresh run directory
containing `atm_in`; do not point a new session at a directory containing
results that must be preserved.

Open the single maintained Notebook for the complete-model step plan,
phase-by-phase execution, runtime options, and typed live-field controls:

```text
/glade/work/ruitong/pycam-sima/examples/try_notebook_session.ipynb
```

It keeps the MPI session open between cells, so `model.step()` and
`model.get_field()` are genuinely interactive. This is the only maintained
example; the CAM controls are demonstrated directly in its cells.

```python
from pathlib import Path

from pycam_sima import FullCAMRuntimeOptions, FullCAMStepPlan, NotebookSession

repo = Path("/glade/work/ruitong/pycam-sima")
run_dir = Path(
    "/glade/derecho/scratch/ruitong/pycam-sima/experiments/notebook01/run"
)
case = repo / "reference/cases/FKESSLER_ne3pg3_gnu_24x50"

run_dir.mkdir(parents=True, exist_ok=True)
(run_dir / "atm_in").write_bytes(
    Path(
        "/glade/derecho/scratch/ruitong/pycam-sima/"
        "FKESSLER_ne3pg3_gnu_24x50/FKESSLER_ne3pg3_gnu_24x50/run/atm_in"
    ).read_bytes()
)

model = NotebookSession(
    repo / "configs/fkessler_ne3pg3.yaml",
    run_dir=run_dir,
    env_script=case / ".env_mach_specific.sh",
    options=FullCAMRuntimeOptions(
        timestep_seconds=1800,
        physics_profile="kessler",
        mediator_present=False,
    ),
    step_plan=FullCAMStepPlan.default(),
)
model.start()
```

`model.options` contains settings consumed by `cam_init`. They can be edited
before `start()`, but changing them afterward is rejected because CAM's time
manager, physics suite, and dycore are already initialized. Select
`configs/adiabatic_ne3pg3.yaml` with `physics_profile="adiabatic"` for a real
SE dynamics-only run. The reference BFB configuration uses 1800 seconds;
changing the timestep intentionally selects a different experiment.

Inspect the exact plan used by `model.step()`:

```python
model.step_plan.describe()

# Explicitly unsafe experiments only:
# model.step_plan.disable("cam_run3", unsafe=True)
# model.step_plan.move("cam_run3", before="cam_run2", unsafe=True)
```

Inspect the available rank-zero fields before or after a step:

```python
model.parameters.describe()
temperature_field = model.parameters.air_temperature
temperature_field.info

model.step()
temperature = temperature_field.get(rank=0)
temperature.shape, temperature.min(), temperature.max()
```

Get one value without transferring every rank-local array:

```python
statistics = temperature_field.stats(rank="all")
statistics[0]
```

`rank="all"` returns one local result per MPI rank. It does not reconstruct SE
global-column order. History NetCDF remains the authoritative globally ordered
output.

Interactive sessions may modify live CAM memory. The following change is
applied on rank zero and is consumed by the next model step:

```python
temperature = temperature_field.get(rank=0)
temperature[0, 0] += 0.01
temperature_field.set(temperature, rank=0)
model.step()
```

Such changes intentionally break BFB. Close the worker to run CAM finalization
and release all MPI processes:

```python
model.close()
```

A context manager closes the model if a Notebook cell raises an exception:

```python
with NotebookSession(
    repo / "configs/fkessler_ne3pg3.yaml",
    run_dir=run_dir,
    env_script=case / ".env_mach_specific.sh",
) as model:
    for _ in range(10):
        model.step()
        print(model.get_field_stats("air_temperature", rank=0))
```
