# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`AGENTS.md` contains the maintainer's repository guidelines and is authoritative; the essentials are summarized here.

## What this is

freeCAM runs the CAM atmosphere component of iCESM1.3.1 under a Python control layer. Python owns the workflow, clock, coupling decisions, and rank-local state; the original Fortran (pinned submodule `external/iCESM1.3.1_fzhu`) remains the numerical source of truth, called through generated C-interoperable adapters. The only admitted scientific configuration is the `ne16` PI-atm case with CAM5 physics, SE dynamics, and 512 MPI ranks on NCAR Derecho (`configs/pi_cam_icesm131.yaml` and variants).

## Commands

```bash
uv sync --extra notebook --extra test    # install (requires Python >=3.11,<3.12)
cp site.env.example site.env             # then set FREECAM_ACCOUNT; not committed
uv run python -m freecam.site            # what this checkout resolves to, and lacks
uv run pytest -q                         # full local unit suite
uv run pytest tests/unit/test_pi_cam_state.py -q     # one test file
uv run pytest tests/unit/test_pi_cam_state.py -k name -q   # one test
uv run freecam --help                    # CLI entry point
git diff --check                         # run before committing
```

Scientific gates run under PBS on Derecho (512 ranks):

```bash
validation/jobs/submit.sh validation/jobs/pi_cam_python_zero_copy_state_50step.pbs
validation/jobs/submit.sh validation/jobs/pi_cam_exact_cesm_online_50step.pbs
```

`submit.sh` supplies `-A $FREECAM_ACCOUNT` from `site.env`; jobs carry no
`#PBS -A` directive and take every path from `validation/jobs/common.sh`.

`tools/verify_pi_cam.py --reference <dir> --candidate <dir>` compares CAM output directories bit-for-bit (no numerical tolerance) via `freecam.pi_cam.validation.compare_pi_cam_directories`.

Native build pipeline lives in `tools/`, driven by `validation/jobs/pi_cam_promoted_statepool_build.pbs`: `prepare_pi_cam_source.py` rebuilds the patched tree under `build/iCESM1.3.1_PI_cam_only` from the pinned submodule (rejecting any revision mismatch, applying the 12 patches and 10 support modules `apply_pi_cam_source_patches.py` reports, and recording them in `.pycam-source.json`); `build_pi_cam_promoted_kernels.py` regenerates the direct-kernel descriptor; `build_pi_cam_devices.py` links the fixed-address image from the oracle's own objects and writes `native_cam_manifest.json`. Other `build_pi_cam_*.py` build the standalone functions, the online coupler, and the capture executable. See the README's *Building the native image*.

## Architecture

Layered control path, from user API down to Fortran:

1. **`src/freecam/pi_cam/facade.py`** — `Driver` / `FreeCAM`, the public user interface (re-exported from `freecam`). Prepares an isolated run directory; constructing it starts nothing — the first live model operation lazily launches one persistent MPI session that later calls reuse.
2. **`src/freecam/pi_cam/session.py`** — `PICAMNotebookSession`, the interactive controller in the user's Python/Jupyter process. Launches the MPI job and talks to rank workers over `multiprocessing.connection`.
3. **`src/freecam/pi_cam/session_worker.py`** — long-lived worker running on each MPI rank; receives cloudpickled commands from the session.
4. **`src/freecam/pi_cam/driver.py`** — `PICAMDriver`, the per-rank control plane: initialization, timestep workflow, action trace (bounded to 4096 records by default), finalization, timing.
5. **Native layer** — generated adapters (`kernel_codegen.py`, `process_codegen.py`, `state_codegen.py`, `in_module_adapter.py` plus `native/pi_cam/` rules and patches) call the original iCESM Fortran `.so` through `src/freecam/core/` (ABI, Fortran runtime environment).

Key subsystems around that spine:

- **StatePool** (`pi_cam/state.py`): rank-local NumPy arrays exposed as zero-copy views of live Fortran state, with distributed get/stats/mutation through the facade.
- **Workflow/plan** (`pi_cam/plan.py`, `runtime_processes.py`, `model/python_processes.py`): one ordered list of process actions supporting enable/disable/move/run and insertion of notebook-defined `fc.Physics` without rebuilding CAM.
- **Boundary providers** (`pi_cam/boundary.py`): online CESM coupling (default — live CLM/CICE/DOCN/RTM plus coupler kernels, exposing rank-local MCT x2a/a2x arrays as zero-copy views) versus offline replay of captured boundary datasets (`PI-atm-replay`, `PI-atm-1month` cases).
- **Catalogs** (`pi_cam/physics_catalog.py`, `source_catalog.py`): the 276 catalogued physical processes and their source/adapter metadata.
- **`src/freecam/model/`**: clock, collective error handling, NetCDF service, device codegen.
- **`validation/`**: machine-readable JSON evidence and the PBS jobs that produced it; **`tests/unit/`**: local API/control-semantics tests; **`tests/integration/`**: MPI smoke tests.

## Hard rules

- **Numeric runtime changes need the 512-rank 50-step PI-atm gate** in addition to unit tests; the result must be bit-for-bit with the pinned iCESM reference and recorded under `validation/`. Never overwrite oracle output. A wrapper or adapter that compiles is not validated — prove the intended routine executed and its outputs match.
- **Keep floating-point algorithms in the original iCESM source.** Generated adapters may convert pointers, shapes, scalar values, and communicator handles, but must not copy numerical scheme bodies. Fail closed when a type, dependency, or process state cannot be represented safely.
- Keep ABI arrays Fortran-contiguous. Native code must not retain Python-owned pointers beyond a declared call boundary.
- Do not reintroduce retired generic runtimes into the public API. New cases need their own configuration and independent validation evidence — PI-atm adapters are not silently reused for incompatible configurations.
- Generated libraries and compiler products belong under `build/`; scheduler output under `logs/`. Neither is committed. Do not commit `error.json`, runtime output, or unrelated validation records.
- No file may name a user or an allocation. Site facts live in `site.env` (not committed) and are declared in `freecam.site.SETTINGS` and `site.env.example`; Python reads them through `freecam.site`, bash through `validation/jobs/common.sh`.
- Style: four-space indentation, type hints, concise docstrings, `snake_case` functions/modules, `PascalCase` classes.
- Commits: concise imperative subject (e.g. `Add PI-CAM process control`); stage only files related to the current task and leave unrelated dirty-tree work alone.
