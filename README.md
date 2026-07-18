# pycam-sima

`pycam-sima` is a Python-owned FKESSLER runtime for CAM-SIMA. NumPy arrays are
the authoritative model state; native Fortran kernels borrow those arrays
through a zero-copy C ABI. Python schedules every suite function and exposes
the state before and after every model step and native call.

The pinned CAM-SIMA source is the `external/CAM-SIMA` submodule at commit
`f8daa568eae2696b7c4ebff7768f02f5d097d9df`.

## Development commands

```bash
uv sync --extra build --extra test
uv run pytest
uv run pycam-sima inspect-contract
uv run pycam-sima run configs/fkessler_ne3pg3.yaml --backend recording --steps 2 --allow-rank-mismatch
uv run pycam-sima run configs/fkessler_ne3pg3.yaml --backend native --steps 2 \
  --allow-rank-mismatch --watch air_temperature --watch-event 'after:kessler'
```

The production native and 50-step validation commands are:

```bash
uv run pycam-sima build-native
qsub jobs/fkessler_kernel_smoke_24.pbs
```

This milestone executes the complete Kessler CCPP suite around an explicit
identity dynamics adapter. It is a physics-kernel/control-flow implementation,
not yet a replacement for the SE dynamical core. Consequently the 24-rank job
is an MPI/ABI/50-step kernel smoke and is not a CAM-SIMA BFB claim. A future
validation command must refuse to claim BFB until the SE adapter and a native
CAM-SIMA reference capture are both present.

Use `--watch FIELD --watch-event EVENT` to inspect Python-owned variables at a
function or step boundary. `--snapshot-dir PATH --snapshot-event EVENT` writes
all fields (or repeated `--snapshot-field FIELD` selections) to per-rank NPZ
files.
