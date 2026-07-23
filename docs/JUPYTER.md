# Jupyter interface

`NotebookSession` lets one ordinary Jupyter kernel control a separate 24-rank
MPI model over an authenticated socket. On a Derecho login node `start()`
submits the worker through PBS; inside an allocation it uses `mpiexec`
directly.

The maintained Notebooks are `examples/try_notebook_session.ipynb` for a
persistent interactive worker and `examples/try_dask_fanout.ipynb` for
checkpoint/restart experiments.

There are now two control modes:

- `NotebookSession` keeps one 24-rank model alive for low-latency phase,
  scheme, field, and step interaction.
- `DaskExperimentClient` submits restartable MPI tasks. A common base `Future`
  retains an immutable checkpoint bundle and can fan out into independent
  model branches without a persistent socket worker. It can either submit one
  PBS job per segment or launch segments directly inside one existing PBS
  allocation.

## Start a session

Create a fresh run directory containing `atm_in`, then construct the model:

```python
from pathlib import Path
import shutil

from pycam_sima import NotebookSession

repo = Path("/glade/work/ruitong/pycam-sima")
run_dir = Path("/path/to/fresh/run")
run_dir.mkdir(parents=True)
shutil.copy2(oracle_dir / "atm_in", run_dir / "atm_in")

model = NotebookSession(
    repo / "configs/fkessler_model.yaml",
    run_dir=run_dir,
    python_executable=repo / ".venv/bin/python",
)
model.start()
assert model.initialized_native_calls == 0
assert model.initialized_abi_checked is False
```

There is no runtime selector or fallback backend. A failed Python
initialization is reported directly.

The worker discovers generated Fortran devices beside the main kernel
library. `driver.stats()["devices"]` reports their source hashes, libraries,
entrypoints, processes, and state policies.

For Kessler, `run_scheme()` does not enter a handwritten replacement
algorithm. It resolves CCPP standard names from StatePool, zero-copy passes
their NumPy addresses through the generated adapter, and calls the original
CAM-SIMA `kessler_init` plus `kessler_run`. Every MPI rank loads its own
device `.so`; device code does not create or replace the communicator.

## Inspect and edit fields

```python
temperature = model.parameters.air_temperature
print(temperature.info)
print(temperature.stats(rank=0))

rank_zero = temperature.get(rank=0)
rank_zero[0, 0, 0, 0, 0] += 1.0e-6
temperature.set(rank_zero, rank=0)
```

`rank="all"` returns one local array or statistics record per MPI rank. Static
grid, topology, and communication fields reject writes unless `unsafe=True`
is passed explicitly.

## Advance and pause

```python
model.prepare_initial_step()  # prime nstep=0 without advancing the clock
model.scheme_plan.describe("physics_after_coupler")
model.run_scheme_group("physics_after_coupler")
after_phase = model.parameters.physics_air_temperature.get(rank=0)
model.step()
```

Calling `step()` from `INITIALIZED` primes nstep=0 automatically. A normal
50-step run creates 51 history files. Prefer `step()` for scientific runs;
`run_phase()` is intended for dynamics-level boundaries, while `run_scheme()`
and `run_scheme_group()` expose the individual CCPP boundaries inside the two
coupler groups.

## Inspect, disable, or reorder schemes

The default plan exactly follows the pinned `suite_kessler.xml`: 19 schemes in
`physics_before_coupler` and 5 in `physics_after_coupler`. The same plan is
installed on every MPI rank at worker startup.

```python
before = model.scheme_plan.describe("physics_before_coupler")
model.run_scheme("calc_exner", group="physics_before_coupler")
exner = model.parameters.field("exner_function").get(rank=0)

# An intentional non-validated experiment:
model.scheme_plan.disable("kessler", unsafe=True)
model.step()

# Restore the BFB-validated order and on/off state.
model.scheme_plan.reset()
```

Reordering also requires explicit acknowledgement:

```python
model.scheme_plan.move(
    "kessler", after="kessler_update", unsafe=True,
)

# Append kessler to physics_after_coupler instead.
model.scheme_plan.move(
    "kessler", to_group="physics_after_coupler", unsafe=True,
)
```

When `to_group` is supplied without `before` or `after`, the scheme is appended
to that group. An anchor in the other group also moves it across the boundary,
for example `after="sima_tend_diagnostics"`. `describe(group)` reports both its
current `group` and immutable `source_group`, so duplicate names remain
unambiguous after a move.

Moving groups changes the model-time location, not just the printed order. A
scheme moved out of `physics_before_coupler` no longer runs during nstep=0 or
end-of-step preparation; it instead runs when the next step enters
`physics_after_coupler`.

`unsafe=True` is required because disabling or moving a required scheme makes
the sequence scientifically different from the validated default. It does not
bypass array shape, pointer-stability, MPI, or ABI checks.

Always close the worker, or use a context manager:

```python
with NotebookSession(repo / "configs/fkessler_model.yaml", run_dir=run_dir) as model:
    model.step(50)
```

## Fork independent Dask tasks

Install the optional scheduler dependencies with
`uv sync --extra test --extra notebook`. The Dask client runs only orchestration
tasks; every CAM segment still creates a new 24-rank MPI world.

```python
from dask.distributed import Client
from pycam_sima import BranchSpec, DaskExperimentClient, FieldEdit

client = Client(processes=False, n_workers=3, threads_per_worker=1)
experiments = DaskExperimentClient(
    client,
    config=repo / "configs/fkessler_model.yaml",
    initial_run_dir=reference_run,
    run_root=scratch / "pycam-sima/dask-branches",
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

For finer control, submit a serializable action plan. All actions in one plan
share the same in-memory `StatePool` and one 24-rank MPI launch:

```python
from pycam_sima import ObserveFields, RunPhase, RunScheme, SegmentPlan

plan = SegmentPlan(
    name="custom-kessler-path",
    unsafe=True,
    actions=(
        RunScheme("kessler", group="physics_before_coupler"),
        ObserveFields(
            ("physics_air_temperature", "large_scale_precipitation_rate")
        ),
        RunPhase("physics_to_dynamics"),
        ObserveFields(("air_temperature",)),
    ),
)
result = experiments.submit_plan(base, plan)
summary = experiments.summaries({"custom": result})["custom"]
temperature = experiments.field(
    result, "air_temperature", rank=0
).result()
```

Use `submit_action()` when the action boundary itself must become a restartable
or forkable checkpoint:

```python
after_kessler = experiments.submit_action(
    base,
    name="after-kessler",
    action=RunScheme("kessler", group="physics_before_coupler"),
)
```

Independent phase and scheme calls require `SegmentPlan(unsafe=True)`. They do
not advance the model clock, write a history sample, or insert omitted
prerequisite phases. `RunSteps(count)` remains the validated complete-step
operation. `summaries()` includes the compact action trace, while `field()`
loads and returns only one requested rank-local array from the final
checkpoint.

The example above uses the default `execution_mode="pbs"`: each Dask task
calls blocking `qsub` for its own segment. To reserve one node once and keep
all segments in that allocation, submit:

```bash
qsub jobs/dask_allocation_fanout_24x50.pbs
```

Inside that job the controller starts one in-process Dask scheduler and one
worker, then constructs the experiment with:

```python
client = Client(processes=False, n_workers=1, threads_per_worker=1)
experiments = DaskExperimentClient(
    client,
    config=repo / "configs/fkessler_model.yaml",
    initial_run_dir=reference_run,
    run_root=run_root,
    python_executable=repo / ".venv/bin/python",
    execution_mode="allocation",
)
```

Every task then runs `mpiexec -n 24 ... run-segment` directly. No task writes
`job.pbs` or invokes `qsub`; all result summaries carry the same outer
`PBS_JOBID`. The current one-node implementation serializes full-node MPI
segments. Running branches simultaneously requires separately partitioned
nodes/ranks and is not enabled by this mode.

The base Future is a dependency of every branch, so Dask computes it once.
The value retained by Dask is a serialized copy of every rank's Python-owned
state, not a live MPI communicator. Each branch restores new
Fortran-contiguous arrays and creates a new `MPI.COMM_WORLD`. `summaries()`
returns only metadata, including `run_dir`, `history_dir`, and
`checkpoint_dir`; it also reports `execution_mode` and `pbs_job_id`.
`gather()` downloads the complete state bundles.
