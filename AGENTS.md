# Repository Guidelines

## Scope

freeCAM currently supports the iCESM1.3.1 PI-atm CAM configuration described
by `configs/pi_cam_icesm131.yaml`. Python owns the observable rank-local state
and CAM workflow. Original iCESM Fortran routines remain the numerical source
of truth and are called through generated C-interoperable adapters.

Do not reintroduce retired generic runtimes into the public API. New cases
should be added deliberately with their own configuration and validation
evidence.

## Repository layout

- `src/freecam/pi_cam/`: PI-CAM driver, StatePool, workflow, runtime processes,
  persistent session, and public facade.
- `src/freecam/core/` and `src/freecam/model/`: internal ABI and runtime helpers.
- `native/pi_cam/`: source patches, adapter rules, and native support code.
- `external/iCESM1.3.1_fzhu/`: pinned upstream iCESM source submodule.
- `examples/try_pi_cam.ipynb`: maintained user-facing Notebook.
- `examples/macro_microphysics.ipynb`: one-cell single-process example.
- `tests/unit/`: local Python tests.
- `tools/`: PI-CAM preparation, build, capture, and validation tools.
- `validation/`: PI-CAM PBS jobs and machine-readable evidence.

Generated libraries and compiler products belong under `build/` and must not
be committed. Scheduler output belongs under `logs/`.

## Development

```bash
uv sync --extra notebook --extra test
cp site.env.example site.env    # then set FREECAM_ACCOUNT
uv run python -m freecam.site   # what this checkout resolves to, and lacks
uv run pytest -q
uv run freecam --help
```

No file in this repository may name a user or an allocation. A site declares
its own once, in `site.env` at the root, which is not committed; both readers
use it, `freecam.site` from Python and `validation/jobs/common.sh` from every
PBS job. Add a new site fact to `site.SETTINGS` and to `site.env.example`
rather than writing a path into a script, a notebook, or a job.

Use four-space indentation, type hints, concise docstrings, `snake_case` for
functions and modules, and `PascalCase` for classes. Keep ABI arrays
Fortran-contiguous. Native code must not retain Python-owned pointers beyond a
declared call boundary.

Run `git diff --check` before committing. Stage only files related to the
current task and preserve unrelated work in a dirty tree.

## Scientific validation

Local tests check API and control semantics. Numeric runtime changes also need
the 512-rank, 50-step PI-atm gate:

```bash
validation/jobs/submit.sh validation/jobs/pi_cam_python_zero_copy_state_50step.pbs
```

`submit.sh` passes `-A $FREECAM_ACCOUNT` on the command line. Jobs carry no
`#PBS -A` directive: `qsub` does not expand variables in directives, so a
working one would name a project in a shared file. A new job sources
`validation/jobs/common.sh` after its `set -euo pipefail` and takes every path
from there.

The result must compare bit-for-bit with the pinned iCESM reference and be
recorded under `validation/`. Never overwrite oracle output. A wrapper or
adapter is not considered validated merely because it compiles; prove that the
intended routine executed and that its outputs match.

## Native interfaces

Keep floating-point algorithms in the original iCESM source. Generated
adapters may convert pointers, shapes, scalar values, and communicator handles,
but must not copy numerical scheme bodies. Fail closed when a type, dependency,
or process state cannot be represented safely.

## Commits

Use a concise imperative subject such as `Remove retired runtime code` or `Add
PI-CAM process control`. Include the tests and PBS evidence relevant to
the change. Do not commit `error.json`, runtime output, or unrelated validation
records.
