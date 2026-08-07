# Jupyter interface

`NotebookSession` lets one ordinary Jupyter kernel control a separate 24-rank
MPI model over an authenticated socket. On a Derecho login node `start()`
submits the worker through PBS; inside an allocation it uses `mpiexec`
directly.

The maintained Notebooks are:

- `examples/try_notebook_session.ipynb` for direct socket control;
- `examples/try_dask_fanout.ipynb` for checkpoint/restart Dask tasks;
- `examples/try_persistent_dask.ipynb` for the dynamic, single-MPI-world model
  pool.

The primary Dask control surface is `DaskExperimentClient.pool()`. A launcher
Actor starts one dynamically sized MPI world and divides it into reusable
model slots. Each live model has a separate ModelActor on a separate Dask
worker. `base.fork(...)` copies rank-local StatePool data directly to child
slots inside that world.

The older control surfaces remain available:

- `NotebookSession` keeps one 24-rank model alive for low-latency phase,
  scheme, field, and step interaction.
- `DaskExperimentClient.model()` pins an Actor to one Dask worker and returns
  a blocking context-managed model.
  The Actor owns a `NotebookSession`, starts MPI once, and exposes the same
  live StatePool through asynchronous Dask method calls.
- `DaskExperimentClient.fork_models()` keeps a base snapshot in Dask
  distributed memory and restores multiple independent, long-lived child MPI
  models without checkpoint files.
- `DaskExperimentClient` submits restartable MPI tasks. A common base `Future`
  retains an immutable checkpoint bundle and can fan out into independent
  model branches without a persistent socket worker. It can either submit one
  PBS job per segment or launch segments directly inside one existing PBS
  allocation.

## Use the dynamic persistent pool

Run the Notebook inside a PBS allocation, then let the resource planner use
the allocation limits:

```python
from dask.distributed import Client

client = Client(
    processes=False,
    n_workers=5,  # launcher plus four model workers
    threads_per_worker=1,
)
resource_plan = experiments.plan_pool(
    max_concurrent_models=4,
    ranks_per_model=None,
    memory_per_model="auto",
)

with experiments.pool("cam-pool", resource_plan=resource_plan) as pool:
    with pool.model("base") as base:
        base.advance(steps=2)
        with base.fork("control", "no_kessler", "warm") as branches:
            branches.no_kessler.physics.kessler.enabled = False
            branches.warm.fields.air_temperature += 1.0
            branches.advance(steps=1)
```

This launches MPI once. All models report the same outer PBS job and pool
launch. The socket carries commands and small results only; fork state moves
directly between corresponding MPI ranks. If there are not enough idle slots
for all requested live children, the current API reports the required and
available capacity so the pool can be planned larger.

Use `model.submit` to build a Scheduler-visible Future graph:

```python
advanced = base.submit.advance(steps=10)
stats = base.submit.fields.air_temperature.stats(
    rank=0,
    depends_on=advanced,
)
branches = base.fork("control", "warm", depends_on=stats)
control = branches.control.submit.advance(steps=5)
warm = branches.warm.submit.advance(steps=5)
client.gather((control, warm))
```

Commands for one model are automatically chained in submission order.
Commands ready on distinct model workers are batched by the launcher and
executed concurrently by their MPI slot communicators.

## Control one legacy persistent model through Dask

Inside a one-node allocation, create one Dask worker and one persistent Actor:

```python
from dask.distributed import Client
from freecam import DaskExperimentClient

client = Client(processes=False, n_workers=1, threads_per_worker=1)
experiments = DaskExperimentClient(
    client,
    config=repo / "configs/fkessler_model.yaml",
    initial_run_dir=reference_run,
    run_root=scratch / "freecam/persistent-dask",
    python_executable=repo / ".venv/bin/python",
    execution_mode="allocation",
)

with experiments.model("live") as model:
    started = model.status
    model.advance(steps=1)
    stats = model.fields.air_temperature.stats(rank=0)
    values = model.fields.air_temperature.get(rank=0)
    checkpoint = model.save()
    finished = model.status

assert finished.step == started.step + 1
assert finished.mpi_launch_count == 1
```

`model()` returns the blocking Notebook façade. `status` and `save()` return
typed objects, and fields with valid Python identifiers support attribute
access. Use `start_persistent()` for the Dask-native Future interface, or use
`model.submit` when asynchronous scheduling is explicitly required.

The Actor owns the full-node allocation lock for its lifetime. Always call
`model.close()` (or leave its `with` block) before launching checkpoint
segments in the same allocation. Actor loss also loses uncheckpointed memory,
so use `model.save()` at important boundaries.

## Fork independent persistent models from memory

This path uses separate PBS jobs, not the one-node allocation mode. Create at
least one Dask worker per child:

```python
from dask.distributed import Client
client = Client(
    processes=True,
    n_workers=3,
    threads_per_worker=1,
)
experiments = DaskExperimentClient(
    client,
    config=repo / "configs/fkessler_model.yaml",
    initial_run_dir=reference_run,
    run_root=scratch / "freecam/persistent-fork",
    python_executable=repo / ".venv/bin/python",
    execution_mode="pbs",
)

control = experiments.plan("control")
warm = experiments.plan("warm")
warm.fields.edit("air_temperature", "add", 1.0)

with experiments.model("base") as base:
    base.advance(steps=10)
    children = experiments.fork_models(
        base, (control, warm), close_parent=True
    )
    with children:
        children.advance(steps=1)
        statuses = children.statuses
```

The data path is:

```text
base MPI ranks
    -> rank-local serialized StatePools
    -> base Actor
    -> one immutable Dask Future
    -> child Actors
    -> child MPI rank-local restores
```

The child receives new arrays with the exact parent bit patterns; it does not
share writable pointers with the parent or siblings. The transfer does not
write `rank-*.npz` or a checkpoint manifest, but it is not zero-copy: bytes
move over the socket/Dask network and are deserialized inside every child.
`close_parent=True` waits until Dask owns the snapshot, then closes the base
MPI job before launching children.

Use `fork_models()` when all branches should remain interactive and the
Dask cluster will stay alive. Use ordinary `fork()` when checkpoint durability,
later restart, or allocation-mode execution matters.

## Start a session

Create a fresh run directory containing `atm_in`, then construct the model:

```python
from pathlib import Path
import shutil

from freecam import NotebookSession

repo = Path("/glade/work/ruitong/freeCAM")
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

The default plan is compiled from the suite XML selected by `ModelConfig` and
the same serialized tree is installed on every MPI rank at worker startup.
For the maintained FKESSLER profile this is 19 schemes in
`physics_before_coupler` and 5 in `physics_after_coupler`.

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

## Add variables and physics at runtime

Every command below is broadcast to all MPI ranks and commits only after their
execution cursor and schema hashes agree:

```python
model.fields.create(
    "experiment_tracer",
    dims=("column", "level"),
    units="kg kg-1",
    initial=0.0,
)

plugin = model.physics.install(
    "/shared/plugin/device.json",
    after="kessler",
    inputs={"required_plugin_input": 1.0},
)

model.physics["experiment_microphysics"].run()
field = model.fields["experiment_tracer"].get(rank=0)

# Only succeeds when no installed plugin/device still references this field.
model.fields.delete("experiment_tracer")
```

Pass a source `device.yaml` instead of `device.json` to build its adapter and
`.so` in the shared plugin cache. `effective="next_step"` defers activation;
`activate_physics()` and `deactivate_physics()` are explicit collective
controls. For Dask, the same operations are available as `DefineVariable`,
`InstallPhysics`, `ActivatePhysics`, and `DeactivatePhysics` actions inside a
`SegmentPlan`.

Deletion is collective and intentionally conservative: only a Python-owned
dynamic field can be removed, all ranks must be at the same action boundary,
and the operation is rejected for history fields or live device/plugin
dependencies. A failure rolls back the schema on every rank.

The explicit `model.physics.install()` call opts into adding an experimental
process. The serializable low-level `install_physics()` and `SegmentPlan`
interfaces retain their `unsafe=True` guard. Neither interface bypasses array
shape, pointer-stability, MPI, or ABI checks.

Always close the worker, or use a context manager:

```python
with NotebookSession(repo / "configs/fkessler_model.yaml", run_dir=run_dir) as model:
    model.step(50)
```

## Fork independent Dask tasks

Install the optional scheduler dependencies with
`uv sync --extra test --extra notebook`. The checkpoint/restart API below is
different from the persistent Actor: every CAM segment creates a new 24-rank
MPI world.

```python
from dask.distributed import Client
from freecam import BranchSpec, DaskExperimentClient, FieldEdit

client = Client(processes=False, n_workers=3, threads_per_worker=1)
experiments = DaskExperimentClient(
    client,
    config=repo / "configs/fkessler_model.yaml",
    initial_run_dir=reference_run,
    run_root=scratch / "freecam/dask-branches",
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
from freecam import ObserveFields, RunPhase, RunScheme, SegmentPlan

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
