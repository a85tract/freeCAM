# Repository Guidelines

## Project Structure & Module Organization

Python sources live in `src/pycam_sima/`. The complete Python-owned CAM driver,
branch edits, and bit-preserving checkpoints are in `model/`; shared MPI-loader
and remote-field utilities are in `core/`; and `notebook/` contains the
Jupyter/PBS controller, Dask experiment client, and MPI worker. Stateless
Fortran kernels live in `native/kernels/`, with generated libraries written to
`build/`. Tests are in `tests/unit/`, with Fortran source adapters in
`tests/fortran/`. Configuration, PBS scripts, tools, documentation, and durable
evidence belong in `configs/`, `jobs/`, `tools/`, `docs/`, and `validation/`.
Maintain `examples/try_notebook_session.ipynb` for interactive socket control
and `examples/try_dask_fanout.ipynb` for restartable Dask task fan-out. Route
PBS standard output and error to `logs/`; do not leave scheduler logs in the
repository root.

## Build, Test, and Development Commands

- `uv sync --extra test --extra notebook`: install the Python 3.11/Jupyter environment.
- `uv run pycam-sima build-kernels`: build `libpycam_sima_kernels.so`.
- `uv run python tools/validate_kessler_kernel.py`: compare the Kessler kernel
  bit-for-bit with pinned CAM-SIMA source.
- `uv run pytest`: run the complete local test suite.
- `uv run python tools/dask_branch_smoke.py --run-root ... --initial-run-dir ...`:
  submit a common checkpoint and two Dask-controlled PBS branches.
- `uv run python tools/validate_dask_branch.py RUN_ROOT`: verify branch-state
  isolation and the exact requested field edit.
- `RUN_DIR=... HISTORY_DIR=... STEPS=50 qsub -V jobs/fkessler_model_24x50.pbs`:
  run the required 24-rank model gate.
- `uv run pycam-sima compare-history ORACLE CANDIDATE`: verify 51 files and 26
  numeric variables.

## Coding Style & Naming Conventions

Use four-space indentation, type hints, concise docstrings, `snake_case` for
functions/modules, `PascalCase` for classes, and `UPPER_CASE` for constants.
Keep arrays Fortran-contiguous where the ABI requires it. Native kernels must
not retain pointers or mutable model state. Run `git diff --check`; no formatter
is currently enforced.

## Testing Guidelines

Name pytest files and functions `test_*.py` and `test_*`. Add focused tests for
contracts and state transitions, plus PBS smoke evidence for MPI changes.
Runtime work is incomplete until the fixed 24-rank, 50-step run produces 51
timestamps and all 26 numeric variables are BFB. Record evidence in
`validation/`.

## Commit & Pull Request Guidelines

Use concise imperative subjects such as `Add ...`, `Implement ...`, or
`Remove ...`. Keep commits focused. Pull requests must describe the affected
model phases, list commands run, include PBS job IDs and BFB evidence for
numeric changes, and identify any intentional unsafe state edits.

## Configuration & Safety

Keep `external/CAM-SIMA` pinned to
`f8daa568eae2696b7c4ebff7768f02f5d097d9df`. Never overwrite oracle output.
The package has one model backend: do not reintroduce `cam_init`/`cam_run*`
wrappers or silently fall back to native CAM control.
