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
    temperature = model.get_field("air_temperature", rank=0)
    print(temperature.min(), temperature.max())
```

The Notebook must be running inside an allocation that can launch 24 MPI
processes. See `docs/JUPYTER.md` for field metadata, all-rank statistics,
live field modification, cleanup, and global-layout limitations.

The shortest Notebook smoke test is:

```python
%run /glade/work/ruitong/pycam-sima/examples/try_notebook_session.py --steps 2
```

## Development checks

```bash
uv sync --extra build --extra test
uv run pytest
uv run pycam-sima build-native
uv run pycam-sima inspect-contract
qsub jobs/fkessler_kernel_smoke_24.pbs
```
