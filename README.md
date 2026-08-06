# pycam-sima

`pycam-sima` is a Python-owned, CCPP-like coupling framework for numerical
model components. Python supplies the lifecycle, process scheduler, common
field bus, MPI communication, and checkpoint semantics; independently built
Fortran devices supply numerical schemes through a generated C ABI.

The scientific gate covers all seven pinned CAM-SIMA suites at
`ne3np4.pg3`, 24 MPI ranks, 50 model steps, and a 1800-second timestep.
CAM4 uses its native L26 coordinate; the other six suites use L30. Each suite
is compared bit-for-bit with an independent CAM-SIMA executable oracle for
the complete `h1i` numerical payload (`T`, `Q`, `U`, `V`, and `PS`) at all
51 output times.

That reference profile is a validation target, not the model definition.
`ModelConfig` accepts the selected suite, `ne`, spectral order, FVM cell
count, vertical levels, constituent registry, timestep, run length, calendar,
start date, startup provider or restart checkpoint, case name, source tree,
and input path. The same Python runtime generates the corresponding SFC
decomposition, grid, FVM geometry, StatePool, clock, and history metadata.
`CAM_SE_FVM_V1` now rejects only real implementation bounds such as empty
partitions or unsupported threading; it no longer treats the reference values
as capability constraints.

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
CCPP schemes such as Kessler are compiled from the pinned upstream Fortran
source into separate device libraries. A generated adapter translates explicit
NumPy pointers and scalar values to the scheme's original interface. Small,
reviewable CAM-SIMA fixes required for deterministic analytic initialization
and shared-library/executable floating-point parity live in
`validation/patches/` and are applied idempotently by
`tools/apply_cam_sima_patches.py`.
No device exports or calls `cam_init`, `cam_run*`, ESMF, PIO, or CAM's native
control loop. Clean builds do not load the CAM machine environment and reject
MPI, ESMF, PIO, NetCDF, HDF5, and RPATH dependencies.

The current architecture is available as an interactive
[online diagram](https://pycam-statepool-mpi-fork.bubblehuntr.chatgpt.site).

## iCESM1.3.1 PI-CAM-only runtime

The `pi-cam-only` branch also contains a deliberately separate port of the
CAM component used by Feng Zhu's iCESM1.3.1 PI-atm case.  The coupled PI-atm
executable is used only once to capture authoritative rank-local `x2a` inputs
and `a2x` outputs.  The production runtime starts only CAM: Python owns the
CAM action order, public clock, state contracts and boundary-provider API;
the original non-PIC Fortran numerical machine code is loaded as a fixed
address `.so`.

```text
captured PI-atm x2a fields
          ↓
ReplayBoundaryProvider → Python PICAMDriver
                              ├── cam_run2 schemes
                              ├── cam_run2 / cam_run3 dynamics
                              ├── cam_run4 history and restart
                              ├── Python completed-step clock
                              └── cam_run1 schemes
                                         ↓
                         fixed-address non-PIC CAM .so
                                         ↓
                              captured a2x BFB check
```

The source CAM lifecycle first exports the state created by `cam_init`, then
performs an initial-only, monolithic `atm_import -> cam_run1 -> atm_export`
priming pass, followed by one complete fine-grained startup look-ahead pass.
The replay therefore contains 53 boundary states: one initial state, one
priming state, one look-ahead state, and 50 public steps.  All startup
passes are preserved internally, while `cam.advance(50)` still means 50
completed public steps and ends at 0001-01-02 01:00:00.  Individual actions
remain visible:

```python
from pycam_sima.pi_cam import PICAMCase

case = PICAMCase.from_yaml("configs/pi_cam_icesm131.yaml")
with case.runtime(
    boundary="/path/to/replay",
    run_dir="/path/to/cam/run",
) as cam:
    cam.initialize()
    cam.physics.dadadj.run()       # experimental isolated scheme call
    cam.kernels.dadadj.run(        # only the original leaf routine;
        experimental=True,         # StatePool arrays go directly through ABI
    )
    cam.phases.cam_run3.run()      # experimental isolated phase call
    cam.step()                     # one complete source-ordered CAM step
    print(cam.pool["cam_out.a2x_rattr"])
    print(cam.pool["phys_state.t"])       # Python-owned rank-local CAM state
    print(cam.pool["cam_in.ts"])
```

The PI-CAM build now generates a state bridge directly from the original
`cam_in_t`, `cam_out_t`, `physics_state`, and `physics_tend` declarations.
For this source configuration it discovers 136 numeric fields; 134 are active
on each rank after `cam_init` and are registered as Fortran-contiguous NumPy
arrays in the rank-local `PICAMStatePool`.  The two inactive entries are
configuration-dependent pointer components and are deliberately not invented
by Python.

This is the first migration boundary, not the final zero-copy kernel layout.
`cam_init` still creates the legacy derived types and supplies their initial
scientific values.  The generated bridge then copies those values into the
Python arrays.  Before a native CAM call Python writes its authoritative state
to the temporary legacy mirror, and after the call it reads the results back.
As individual numerical kernels acquire flat generated ABIs, they consume the
same StatePool arrays directly and their corresponding mirror copies can be
removed.  Opaque `pbuf`, SE dycore, and process-private module state remain
Fortran-owned until they receive explicit serializers or field contracts.

All ordinary numerical entry points use one reusable Python adapter.  Its JSON
manifest declares the symbol, action id, field name, dtype, rank, intent, and
contiguity.  The adapter validates the NumPy arrays and constructs the generic
pointer/rank/shape table.  Generated device shims convert that table to normal
Fortran arguments; the current PI-CAM legacy image uses the same table through
one stable `bind(C)` action module while its state-transfer bodies are generated
from the original type declarations.  Compiler-specific module symbol mangling,
assumed-shape descriptors, and derived-type layouts therefore never leak into
the Python API.

Leaf routines whose original signatures already contain normal scalars and
arrays can now skip the legacy derived-type mirror entirely.  A compact YAML
descriptor supplies field names, dtypes, ranks, intents, chunk axes, and fixed
indices.  `kernel_codegen.py` generates a `bind(C)` pointer-table wrapper that
loops over this rank's CAM chunks and calls the unchanged original symbol.  It
does not copy or regenerate the numerical body.  The first admitted kernel is
`dadadj`:

```text
PICAMStatePool NumPy arrays
  phys_state.lchnk / ncol / pmid / pint / pdel / t / q
                         ↓ addresses + shapes, no copy
              generated pycam_pi_cam_dadadj_v1
                         ↓ one call per local chunk
                    original dadadj_
```

`cam.kernels.dadadj.run(experimental=True)` invokes only this leaf kernel.
It is intentionally different from `cam.physics.dadadj.run()`, which invokes
the complete CAM host stage around dry adjustment.  The direct leaf API does
not create tendencies, apply the host update, satisfy neighboring process
dependencies, or advance time, so it is restricted to explicit experiments
until the complete stage has a raw-array ABI.

For a Jupyter Notebook, `PICAMNotebookSession` submits one cpudev job and
starts the 512-rank CAM worker once. All later calls reuse those live MPI
ranks and their rank-local Python StatePools; the authenticated socket carries
only commands and selected field results:

```python
from pycam_sima.pi_cam import PICAMNotebookSession

with PICAMNotebookSession(
    "configs/pi_cam_icesm131.yaml",
    boundary="/path/to/replay",
    run_dir="/path/to/cam/run",
    env_script="/path/to/oracle-case/.env_mach_specific.sh",
    launch_mode="pbs",
) as cam:
    cam.step()
    print(cam.stats("cam_out.a2x_rattr", rank="global"))
    print(cam.stats("phys_state.t", rank="global"))
    cam.run_scheme("dadadj", phase="cam_run1")  # experimental host-stage call
    cam.run_kernel("dadadj")                     # raw-array leaf call
```

`cam.step()` follows the source order and advances the public CAM clock.
`run_scheme()` and `run_phase()` are deliberately experimental: they execute
only the requested action and do not supply missing scientific preconditions
or advance model time.

Build and validate with the reproducible cpudev jobs:

```bash
source /path/to/oracle-case/.env_mach_specific.sh
python tools/apply_pi_cam_source_patches.py --source-root /path/to/iCESM1.3.1
python tools/generate_pi_cam_state_bridge.py \
  --source-root /path/to/iCESM1.3.1 \
  --fortran /tmp/pycam_pi_cam_state_bridge.inc \
  --manifest /tmp/pycam_pi_cam_state_bridge.json
python tools/build_pi_cam_devices.py --case /path/to/oracle-case
qsub validation/jobs/pi_cam_boundary_capture_50step.pbs
qsub validation/jobs/pi_cam_python_state_bridge_50step.pbs
qsub validation/jobs/pi_cam_session_state_bridge_50step.pbs
qsub validation/jobs/pi_cam_direct_kernel_50step.pbs
```

`tools/build_pi_cam_devices.py` intentionally does not rebuild all of CAM with
`-fPIC`: that changes Intel floating-point code generation and fails BFB.  It
keeps the original non-PIC objects, replaces only the Python control surfaces,
links a fixed-address executable image, and changes its ELF type to ET_DYN for
`dlopen`.  Because Python rather than an Intel Fortran program is the process
main, the adapter also restores CAM's original `-ftz` MXCSR setting before the
first numerical call.  Validation is fail-closed and compares every numeric variable in
all CAM history and restart files; wall-clock strings are excluded.
The third job drives the same 50-step gate through one long-lived
`PICAMNotebookSession`, proving that later Notebook cells do not relaunch MPI.
`tools/apply_pi_cam_source_patches.py` is the reproducible source-preparation
entry point: it applies only the eight production patches, in dependency order,
to a clean `components/cam` checkout. Historical exploratory patches remain as
an audit trail but are not part of the build.
The current 512-rank gate is recorded in
[`validation/pi_cam_50step_gate.json`](validation/pi_cam_50step_gate.json):
both the direct Python driver and the persistent Notebook session are BFB for
all four CAM history/restart files after 50 completed steps.
The generated state bridge direct-driver gate is additionally recorded in
[`validation/pi_cam_python_state_bridge_vs_oracle_50step_bfb.json`](validation/pi_cam_python_state_bridge_vs_oracle_50step_bfb.json):
job `7030195.desched1` ran on `cpudev`, exposed 134 active legacy CAM fields to
Python on every rank, and remained BFB for all four files.
Persistent-session job `7030208.desched1` repeated the same 50-step BFB gate
with one MPI launch and retrieved active-column `phys_state.t` values directly
in Python before and after the run.
Direct-kernel job `7030781.desched1` then ran the final generated image on 512
cpudev ranks.  Its 50-step default CAM outputs remained BFB for all four files;
the disposable `dadadj` probe constructed 13,826 unstable columns and changed
`t` and `q` on all 512 ranks while preserving every NumPy address and every
declared read-only input byte.  The machine-readable records are
[`validation/pi_cam_direct_kernel_50step.json`](validation/pi_cam_direct_kernel_50step.json)
and
[`validation/pi_cam_direct_kernel_vs_oracle_50step_bfb.json`](validation/pi_cam_direct_kernel_vs_oracle_50step_bfb.json).

Runnable examples are split by execution mode:

- [`examples/try_notebook_session.ipynb`](examples/try_notebook_session.ipynb)
  keeps a 24-rank MPI worker alive for interactive phase, scheme, step, and
  field control through the authenticated socket bridge.
- [`examples/try_dask_fanout.ipynb`](examples/try_dask_fanout.ipynb) submits a
  restartable Dask task graph for independent checkpoint branches.
- [`examples/try_persistent_dask.ipynb`](examples/try_persistent_dask.ipynb)
  keeps one Dask-managed MPI model alive and demonstrates reusable rank-local
  in-memory snapshots for sequential branches.
- [`examples/try_pi_cam.ipynb`](examples/try_pi_cam.ipynb) is the independent
  PI-CAM-only walkthrough; it does not import or run CLM, CICE, DOCN, RTM, or
  the CESM coupler.

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

MPI rank count is read from `ModelConfig.mpi_size`; Dask and PBS launchers use
the same value and reject a launcher/config mismatch. Python reproduces
HOMME's 2/3/5 recursive space-filling curve, its non-factorable-grid fallback,
and its contiguous remainder-first rank partition. The current executable
`ne3` profile therefore supports 1 through 54 nonempty SE ranks instead of
being locked to 24:

```bash
RANKS=18 \
CONFIG=/path/to/model-with-mpi_size-18.yaml \
RUN_DIR=/path/containing/atm_in \
HISTORY_DIR=/new/history/directory \
qsub -V -l select=1:ncpus=18:mpiprocs=18:ompthreads=1:mem=20GB \
  jobs/fkessler_model_variable_mpi.pbs
```

The complete runtime accepts any positive `ne`, `pver`, and constituent count,
`np >= 2`, and a positive FVM cell count. A non-reference
`ne2np5.pg4`/L12/five-constituent/360-day profile is provided in
[`configs/configurable_ne2np5_pg4_l12.yaml`](configs/configurable_ne2np5_pg4_l12.yaml).
Its vertical-coordinate NetCDF file and matching `atm_in` remain case inputs.
Checkpoints are rank-local and must be restored with the same MPI size with
which they were written. The reference scientific/BFB gate remains
`ne3np4.pg3`; non-reference configurations require their own scientific
validation rather than inheriting that claim.

Startup cases choose a Python initial-state provider with
`analytic_ic_type`; the built-in providers are
`moist_baroclinic_wave_dcmip2016` and `resting_isothermal`. A durable restart
uses the same model-defining values and changes only the lifecycle fields:

```yaml
run_type: continue       # or branch
restart_path: /path/to/checkpoint
stop_n: 50
```

The checkpoint restores the calendar-aware clock, suite plan, registered
fields, plugin inventory, and every rank-local array without executing the
startup provider again. Suites with fewer than three moist constituents write
only the history diagnostics present in their StatePool; the FKESSLER
reference continues to write its exact 26-variable inventory.

The history gate compares filenames, timestamps, dtype, shape, and float64 bit
patterns for all 51 output times and 26 diagnostic variables. The upstream
CAM-SIMA executable is used only to produce an external test oracle; it is not
a selectable pycam-sima backend. The source-device build and full-run evidence
is recorded in
[`validation/source_preserving_devices.json`](validation/source_preserving_devices.json).

## Source-preserving Fortran devices

The pinned CAM-SIMA tree contains 7 suite XML files, 155 distinct active
schemes, and 340 scheme occurrences. PyCAM-SIMA audits all of them and
generates one deterministic connector descriptor per scheme:

```bash
pycam-sima audit-devices \
  --output validation/ccpp_device_catalog.json
pycam-sima generate-devices --clean
pycam-sima build-catalog-devices --strict
pycam-sima scheme-status \
  --output validation/all_scheme_support.json
```

`devices/generated/<scheme>/device.yaml` is generated from the pinned suite,
metadata, and unmodified Fortran source; it is not a second numerical
implementation. The status report distinguishes a built numerical device,
a Python-owned history service, and schemes that still require MPI, input
data, external source, or an allocatable-object policy. Thus “connector
generated” and “scientifically executable” are separate, machine-readable
claims.

All 155 descriptors live under this single generated tree. Host policy that
cannot be inferred from CCPP metadata is maintained separately in
`devices/overrides.yaml` and merged deterministically during generation.
For example, the Kessler entries preserve their validated
`reinitialize_each_run` policy and vertical-index bindings without keeping a
second hand-written descriptor directory. The clean rebuild, lifecycle, and
50-step BFB evidence for this consolidation is recorded in
[`validation/descriptor_unification.json`](validation/descriptor_unification.json).

For the pinned revision, all 155 active schemes and all 340 suite occurrences
are runtime executable: 121 schemes use native Fortran devices, 30 use
Python-owned host services, and four are explicit external services. No
connector remains unresolved. A clean standalone catalog build compiles all
121 ABI-compatible device targets with zero failures (120 in the shared core
bundle and MUSICA in its dependency-linked device library). The machine-readable
coverage is recorded in
[`validation/all_scheme_support.json`](validation/all_scheme_support.json)
and the clean-build result in
[`validation/portable_catalog_device_build.json`](validation/portable_catalog_device_build.json).
The independent 24-rank, 50-step, seven-suite BFB comparison covers the core
prognostic history fields `T`, `Q`, `U`, `V`, and `PS` at all 51 output
timestamps. Its machine-readable evidence is recorded in
[`validation/seven_suite_24x50_bfb.json`](validation/seven_suite_24x50_bfb.json).

The main `CAMDriver` uses the same XML-derived plan and standard-name bus as
the standalone host. It contains no Kessler scheme-order table:

```python
from pycam_sima import (
    CCPPDeviceHost,
    CCPPSuitePlan,
    DeviceCatalog,
    DeviceRegistry,
    HostServiceRegistry,
    ModelConfig,
)

config = ModelConfig.from_yaml("configs/fkessler_model.yaml")
catalog = DeviceCatalog.discover("/glade/work/ruitong/pycam-sima")
plan = CCPPSuitePlan.from_xml(config.resolve_suite_xml())
devices = DeviceRegistry(("build/devices", "build/catalog_devices"))
services = HostServiceRegistry.from_catalog(
    catalog, suite=config.physics_suite
)

# pool is a Python-owned StatePool satisfying this suite's standard names.
host = CCPPDeviceHost(pool, devices, plan, host_services=services)
host.run_lifecycle("initialize")
host.run_group("physics_before_coupler")
```

`CAMDriver` also compiles `CCPPStateSchema` for the selected suite during
construction. Suite-independent CAM component fields are combined with only
the process-field templates named by that suite, and missing primitive fields
are generated from CCPP metadata. For example, the Kessler profile includes
the previous-timestep temperature and precipitation fields, while the
Held-Suarez profile does not. `state_schema.report()` exposes dimensions,
conversion points, opaque objects, and the resulting StatePool field count.
`ModelConfig.suite_xml` may point to a custom suite name not present in the
pinned seven-suite catalog: its known scheme names still contribute metadata,
while unknown plugin processes are reported as unresolved until installed.

A device is defined by a small YAML description, the original CCPP `.meta`
file, and the original Fortran sources. `build-kernels` runs CAM-SIMA's own
CCPP parser to verify that metadata and source signatures agree, scans
Fortran module dependencies, generates the C ABI adapter and JSON manifest,
and compiles one isolated `.so`:

```text
devices/generated/kessler/device.yaml
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
uv run pycam-sima build-device devices/generated/kessler/device.yaml
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

### Runtime plugins and variables

Version 0.13 adds a collective runtime extension API. A plugin may be an
original-source `device.yaml` or a prebuilt `device.json` beside its `.so`.
Source plugins are built once in a hash-addressed shared cache; prebuilt
plugins pass the same ABI, source-hash, exported-symbol, ELF-dependency, and
RPATH checks. Explicit paths, `PYCAM_SIMA_PLUGIN_PATH`, and Python entry points
in the `pycam_sima.physics` group are discoverable.

```python
model.fields.create(
    "droplet_number",
    standard_name="cloud_droplet_number_concentration",
    dims=("column", "level"),
    units="kg-1",
    initial=0.0,
)

installed = model.physics.install(
    "/shared/my_microphysics/device.yaml",
    after="kessler",
    inputs={"my_required_input": 1.0},
    effective="now",
)

model.physics["my_microphysics"].run()

# Safe collective deletion after the field is no longer used by a device.
model.fields.delete("droplet_number")
```

CCPP argument metadata supplies missing primitive-variable contracts. Existing
standard names are reused zero-copy only when dtype, shape, and units match;
new `intent(in)`/`intent(inout)` fields require an initial value. Runtime
additions may use existing named dimensions or literal sizes but cannot resize
existing storage. `fields.delete()` only removes Python-owned dynamic fields
at an MPI action boundary. It rejects model-schema fields, history fields, and
fields still referenced by an installed plugin or loaded device; `remove()` is
an alias.
or replace an existing array.

Installation, activation, and deactivation are MPI-collective transactions at
phase/scheme boundaries. Every rank verifies the same cursor, plugin bytes,
and StatePool schema. `effective="next_step"` loads immediately but enables
the placement at the next complete step. Friendly dimensions such as
`column`, `level`, and `interface_level` map to the runtime dimensions
`nphys_local`, `pver`, and `pverp`.

`VariableSpec`, `PhysicsPluginSpec`, and `SchemePlacement` remain the
serializable low-level protocol for Dask action plans and advanced tooling.
The `model.fields` and `model.physics` façades compile to those same checked
objects; they do not bypass ABI, MPI, pointer-stability, or schema validation.

Dynamic fields default to checkpointed and not written to history. Checkpoint
schema v3 records complete contracts, plugin hashes, placements, activation
state, and Notebook Python-process inventories; restore fails closed if an
exact required artifact or callback hash is unavailable.

A runnable source plugin is included at
`examples/plugins/runtime_temperature_offset/device.yaml`. The 24-rank gate
builds it from the original Fortran source, executes it, checkpoints all
dynamic fields, reloads the generated prebuilt artifact, and executes it
again:

```bash
qsub jobs/dynamic_runtime_24.pbs
```

The all-rank hashes, artifact identities, and PBS result are recorded in
`validation/dynamic_plugins_and_variables.json`.

### Notebook Python processes

A trusted Notebook function can be inserted directly into the live
`SuitePlan`. `cloudpickle` freezes the function bytes once; the launcher sends
the payload and SHA-256 over the authenticated control socket, MPI world rank
0 broadcasts it, and every rank in the persistent model installs the same
callback.
The callback executes against only that rank's StatePool arrays:

```python
def custom_heating(fields, context, *, increment):
    temperature = fields["air_temperature"]
    temperature[...] += increment

process = model.physics.install_python(
    custom_heating,
    name="custom_heating",
    group="physics_before_coupler",
    after="kessler",
    reads=(),
    writes=("air_temperature",),
    parameters={"increment": 1.0},
)

process.run()                # uses the persistent increment=1.0
process.run(increment=2.0)   # one-call override; the default stays 1.0
process.parameters["increment"] = 0.5  # persistent update
process.disable()  # complete steps now skip it
process.enable()
model.advance(steps=1)
process.move(before="potential_temp_to_temp")
process.remove()
```

Unqualified names first resolve as CCPP standard names at the physics
boundary; use `field:<canonical-or-alias>` or `ccpp:<standard-name>` to make a
collision explicit. `reads` are non-owning read-only NumPy views. `writes` are
non-owning writable views, so values can change in place but storage, shape,
and dtype cannot be replaced. Access to an undeclared name fails immediately.
Custom parameters are keyword-only function arguments. Their defaults are
declared with `parameters={...}`, must be small JSON-compatible values, and
are validated collectively on all ranks. Persistent updates made through
`process.parameters` are included in retain/fork and checkpoint/restart;
per-call values passed to `process.run(...)` are not persisted. Large arrays
belong in StatePool fields rather than parameters.
The context is immutable and contains rank, model-communicator size, model
step, timestep, date, calendar, process name, and group; it intentionally does
not expose an MPI communicator. Each invocation reconstructs the callable
from the frozen payload, so mutable closure state is not persistent; durable
process state belongs in explicitly declared StatePool fields.
Callbacks must not create or call MPI collectives themselves: framework-owned
collectives define the safe process boundary, and an unmatched user
collective can deadlock the complete model.

By default, every rank copies only the declared `writes` before the call. If
one rank fails or returns a non-`None` value, all ranks restore those fields
and report rank-indexed tracebacks. `transactional=False` requires an explicit
`unsafe=True`; a failure then marks the model failed. Payloads default to an
8-MiB limit so large arrays must live in StatePool instead of a closure.
`cloudpickle` is trusted-code serialization, not a sandbox, and imported
packages must exist in every MPI rank's Python environment.

The same operation has a Future-returning form:

```python
installed = model.submit.install_python_process(
    custom_heating,
    name="custom_heating",
    after="kessler",
    writes=("air_temperature",),
    parameters={"increment": 1.0},
)
updated = model.submit.set_python_process_parameters(
    "custom_heating",
    {"increment": 0.5},
    depends_on=installed,
)
observed = model.submit.fields.physics_air_temperature.stats(
    rank=0,
    depends_on=updated,
)
```

`PlanBuilder.physics.install_python()` compiles to the serializable
`InstallPythonProcess` action, and `remove_python()` compiles to
`RemovePythonProcess`; use `experimental=True` because these actions alter
scientific ordering. Memory fork copies the callback inventory inside the
existing parent-rank to child-rank MPI payload. Disk checkpoints store the
base64 payload, hash, permissions, placement, enabled state, and rollback
policy and rebuild the registry without running the callback during restore.
Inside a persistent pool, a durable checkpoint can be restored into any idle
pre-launched slot without another `qsub` or `mpiexec`:

```python
saved = model.save()
with pool.restore("restarted", saved.path) as restarted:
    assert restarted.status.snapshot_transport == "checkpoint"
    assert restarted.status.details["python_processes"]
    restarted.advance(steps=1)
```

The real persistent-pool gate uses one PBS allocation and one
`mpiexec -n 96` for four 24-rank slots. It checks exact `+1 K` execution on
all ranks, collective rollback, in-memory fork inheritance, disk restart,
no-op 50-step scientific-state equality, destination-slot communicator
rebinding, and both no-op and unmodified FKESSLER history against the pinned
51-file oracle:

```bash
qsub jobs/python_runtime_process_4x24.pbs
```

The completed machine-readable evidence is kept in
`validation/python_runtime_process.json`.

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

The generic ABI additionally handles default Fortran logical values,
fixed-width character scalars/vectors, Python-owned shaped primitive
allocatable fields, and non-allocatable derived types.
Derived process objects are allocated by generated Fortran factory symbols;
Python stores their opaque handles in `StatePool`, passes the same object to
later schemes by CCPP standard name, and invokes the matching generated
destructor at finalize. Checkpointing fails closed while opaque state is live
because no byte-copy of a Fortran object is restart-safe.

CAM `cam_history`/`outfld` is a host service rather than a numerical kernel.
`HostServiceRegistry` replaces pure history-only schemes with Python
observations. Physical constants used by legacy helper modules are injected
from Python-owned StatePool fields before each native call; they are not
silently owned by a Fortran global.

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

The selected suite exposes every scheme occurrence individually.
`model.scheme_plan.describe()` shows its exact XML order, including subcycles,
and `run_scheme()` pauses after one scheme. In the validated Kessler profile
this is 19 `physics_before_coupler` schemes and 5
`physics_after_coupler` schemes:

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
`check_energy_scaling` occurs in both groups. Only an unmodified source-XML
plan can report `sequence_safe=True`; the FKESSLER source order is the
complete-step BFB gate currently validated.

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

### Single-model persistent runtime

The primary interactive path keeps one model alive in one MPI world. Resource
planning is internal: `model()` inherits the configured MPI size, estimates
memory, budgets one retained snapshot by default, starts MPI once, and closes
the hidden runtime with the model context. Users do not need to call
`plan_pool()` or specify `retained_snapshots=1` for the normal path.

```python
with experiments.model("base") as base:
    base.advance(steps=5)
    temperature = base.fields.air_temperature.stats(rank=0)
```

Internally, `model_slots == 1`, `world_size == ranks_per_model`, and
`retained_snapshots == 1`. Pass `retained_snapshots=0` only when the runtime
will never retain an in-memory continuation, or a larger value when several
rank-local snapshots must coexist.

To run several experiments sequentially from one base, retain an immutable
snapshot on the MPI ranks themselves. This advanced lifecycle exposes the
single-model runtime coordinator, but still performs no explicit planning:

```python
with experiments.runtime("cam-runtime") as runtime:
    base = runtime.model("base")
    base.advance(steps=25)
    state = base.retain("after-step-25")
    base.close()

    with runtime.restore_retained("control", state) as control:
        control.advance(steps=25)

    with runtime.restore_retained("warm", state) as warm:
        warm.fields.air_temperature += 1.0
        warm.advance(steps=25)

    state.close()
```

Each rank stores only its own `np.savez` bytes in its MPI worker process heap.
The launcher and Dask hold a small `RetainedModelState` containing the snapshot
ID, source slot, step, rank count, byte count, and config hash. Restore reads
rank-local bytes on the same rank; model data never passes through the socket,
world rank 0, Dask, or disk. A retained snapshot is reusable until
`state.close()` or `runtime.close()`.

The hidden planner includes the estimated snapshot bytes, active StatePool,
dynamic-field budget, and 15 percent runtime reserve in its PBS memory request.

Multi-model concurrency remains available explicitly:

```python
plan = experiments.plan_pool(max_concurrent_models=4)
with experiments.pool("parallel-models", resource_plan=plan) as pool:
    ...
```

This starts `4 * ranks_per_model` MPI ranks and preserves the existing
rank-to-rank `fork()` path. The default `worker_policy="exclusive"` requires
one launcher worker plus one ModelActor worker per live slot; use
`worker_policy="shared"` only for small local tests.

The maintained 24-rank validation runs a direct 50-step path and two
sequential restores from the same retained step-25 state: one continues to
step 50 and one verifies an exact NumPy `+1 K` field edit.

```bash
qsub jobs/single_slot_retained_25x25_bfb.pbs
```

`plan_pool()` discovers an active PBS allocation through `PBS_NODEFILE` and
`qstat`, or accepts explicit node/CPU/memory values to produce a PBS request
before allocation. It reserves 15 percent of memory by default and includes
Python StatePool plus a dynamic-field budget in its capacity calculation.

### Legacy single-model Persistent Dask Actor

Use a persistent Actor when many Notebook commands should operate on the same
live StatePool. Actor construction is the only operation that starts
`mpiexec`; every later method call is scheduled by Dask onto the same pinned
worker and forwarded to the already-running 24 MPI ranks:

```python
experiments = DaskExperimentClient(
    client,
    config=repo / "configs/fkessler_model.yaml",
    initial_run_dir=reference_run,
    run_root=scratch / "pycam-sima/persistent-experiments",
    python_executable=repo / ".venv/bin/python",
    execution_mode="allocation",
)

with experiments.open_persistent("interactive") as model:
    started = model.status
    model.advance(steps=2)
    temperature = model.fields.air_temperature.get(rank=0)
    checkpoint = model.save()
    finished = model.status

assert started.mpi_launch_count == finished.mpi_launch_count == 1
assert finished.step == started.step + 2
```

`open_persistent()` is the blocking compatibility API for the legacy Actor.
`model.status` returns a typed `ModelStatus`; `advance()` runs complete steps;
attribute-style `fields`, `phases`, and `physics` handles expose the live
StatePool and control graph; and `save()` returns typed checkpoint metadata.
`start_persistent()` preserves the Dask-native API where every method returns
an `ActorFuture`, and `model.submit` explicitly exposes that asynchronous
controller. The Actor holds the allocation-wide MPI lock until the `with`
block exits.

This mode still uses the authenticated `NotebookSession` socket internally:
the Dask Actor is the long-lived controller and the socket carries commands to
MPI rank 0, which broadcasts them to the other ranks. Dask replaces direct
Notebook-to-session ownership; it does not replace MPI or the IPC required to
control processes that remain alive.

This compatibility path remains available for existing code. Prefer the
single-model runtime for new interactive work. Use
`fork_models()` when a live base should become several independent,
long-lived MPI models without checkpoint files:

```python
control = experiments.plan("control")
warm = experiments.plan("warm")
warm.fields.edit("air_temperature", "add", 1.0)

with experiments.open_persistent("base") as base:
    base.advance(steps=10)
    children = experiments.fork_models(
        base, (control, warm), close_parent=True
    )
    with children:
        children.advance(steps=1)
        statuses = children.statuses
```

The base MPI ranks serialize their rank-local StatePools into one immutable
`CheckpointBundle`. A Dask Future retains that bundle in distributed memory
and supplies the same bytes to every child Actor. Each child starts its own
24-rank MPI job, restores new private NumPy arrays through the socket/MPI
bridge, applies its plan, and remains alive. No
`rank-*.npz`, `manifest.json`, or checkpoint directory is created. This is
bit-preserving memory transport, not zero-copy shared memory: separate PBS
jobs still deserialize and copy the arrays.

Persistent memory fork currently requires `execution_mode="pbs"` and one
distinct Dask worker per child so blocking Actor calls can proceed
concurrently. Every child consumes its own 24-rank PBS job. Set
`close_parent=True` to release the base PBS job after the distributed snapshot
is ready. The single-node allocation mode intentionally rejects this API
because one node cannot host several independent 24-rank MPI worlds.

Choose the checkpoint segment API (`submit_base`, `submit_plan`, `fork`) when
a boundary must survive worker/job failure or be resumed later. Choose
`fork_models()` for fast in-memory fan-out while the Dask cluster remains
alive. `model.save()` remains the explicit durable checkpoint boundary.

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

`validation/dask_persistent.json` records a real one-allocation Actor run in
which seven Actor calls advanced the same model from step 0 to step 2, read a
field, wrote a checkpoint, closed cleanly, and reported exactly one MPI launch.
Reproduce that gate with:

```bash
qsub jobs/dask_persistent_24.pbs
```

`validation/dask_persistent_fork.json` records the diskless persistent-fork
gate. A 24-rank base ran 10 steps, then three separate 24-rank PBS models
restored the same 37,660,986-byte Dask snapshot. Control and no-Kessler were
bitwise equal to the base across all 5,328 rank-local arrays, warm changed only
`air_temperature` and was exactly `numpy.add(base, 1.0)`, all children
advanced independently to step 11 with one MPI launch each, and the validation
root contained no checkpoint manifest, NPZ file, or checkpoint directory.
Reproduce it with:

```bash
python tools/dask_persistent_fork_smoke.py \
  --run-root /path/to/new/run-root \
  --initial-run-dir reference/cases/FKESSLER_ne3pg3_gnu_24x50/CaseDocs \
  --output /path/to/result.json
```

All persistent fields have `owner="python"`. Prognostic, tendency, and process
arrays are writable at phase boundaries. Static grid/topology arrays require
`unsafe=True`; kernel calls must preserve every NumPy address.

`FVMKernelConfig.from_pool(model.pool)` derives `nc`, `nlev`, tracer count,
halo widths, reconstruction order, quadrature count, jet-level range, active
level range, and the large-Courant switch in Python. The resulting C-compatible
configuration is supplied to both FVM kernel calls; the Fortran wrappers do not
define case dimensions or timestep controls.

To preserve CAM's BFB floating-point instruction order for each concrete
shape, `build-kernels` generates a compile-time specialization module from the
selected model YAML. ABI v2 exposes that specialization and Python checks all
four values before initialization; it never silently reuses a library with
the wrong layout:

```bash
uv run pycam-sima build-kernels \
  --config configs/configurable_ne2np5_pg4_l12.yaml
```

The default reference library remains `build/libpycam_sima_kernels.so`.
Non-reference builds are cached under
`build/kernels/<specialization-id>/libpycam_sima_kernels.so`.

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
