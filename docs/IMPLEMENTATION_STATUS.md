# Implementation status

## Implemented target

The repository implements a general, manifest-driven Python/Fortran device
boundary and one complete model configuration:

- CAM-SIMA `f8daa568eae2696b7c4ebff7768f02f5d097d9df`
- FKESSLER, `ne3np4.pg3`, L30, 24 ranks, one thread per rank
- 1800-second timestep and 50 requested steps
- DCMIP2016 moist baroclinic-wave initial condition
- startup, ATM-only, NO_LEAP execution

Python parses `atm_in`, initializes the grid and analytic state, owns all 222
canonical persistent fields, executes the model phase graph, performs MPI
communication through mpi4py, advances the clock, and writes history.
`libpycam_sima_kernels.so` exposes versioned dycore/mapping kernels and retains
no array pointer or mutable model state between calls. CCPP schemes are
separate source-preserving devices under `build/devices/`.

The Python scheme plan mirrors every active entry in the pinned Kessler CCPP
suite. Its default 19 before-coupler and 5 after-coupler schemes are
independently callable, observable, enabled/disabled, reorderable, and movable
between groups. Required-scheme changes demand `unsafe=True`; the untouched
XML order is the BFB contract.
The Kessler and Kessler-update boundaries are supplied by generated devices.
Their adapters call the pinned, unmodified `kessler.F90` and
`kessler_update.F90`; no second copy of either numerical algorithm is
maintained. The FP-sensitive geopotential helper remains a main numerical
kernel. The remaining conversion, thermodynamic, conservation, tendency, and
diagnostic boundaries execute against the same Python-owned StatePool.

## General device interface

Each device descriptor identifies:

- original Fortran source and CCPP metadata;
- source modules and portable dependency providers;
- lifecycle entrypoints and state policy;
- CCPP-dimension to StatePool-dimension bindings;
- named processes exposed to the Python scheduler.

The generator reuses CAM-SIMA's CCPP metadata and Fortran parsers to verify the
source signature. It emits an explicit-shape `bind(C)` adapter, a version
script, and `device.json`; then it builds and scans an isolated device `.so`.
The runtime `DeviceRegistry` discovers manifests without scheme-specific
Python code. It binds arguments by CCPP `standard_name`, validates dtype,
shape, units, intent, writable state, and Fortran contiguity, and preserves
all NumPy addresses.

The implemented state policies are:

- `stateless`: call the requested entrypoint directly;
- `reinitialize_each_run`: replay a Python-sourced initialize entrypoint before
  every process call;
- `initialize_once`: initialize once per Python worker and explicitly report
  persistent native state in the manifest.

Kessler and Kessler update use `reinitialize_each_run`. Their Fortran module
configuration can never become the checkpoint authority because each process
invocation reloads it from StatePool.

Dependency resolution is fail-closed. Intrinsic modules, declared source
modules, and portable providers are accepted. MPI, ESMF, PIO, CAM history, any
undeclared module, forbidden ELF dependency, RPATH, or RUNPATH fails the
build. The current device libraries depend only on `libgfortran`, `libm`, and
`libc`.

CSLAM/FVM geometry is generated independently by Python on every rank. The
generator ports CAM-SIMA's cubed-sphere vertices, exact spherical moments,
cross-face halo mapping, interpolation weights, rotation matrices, and
reconstruction metrics; the packaged NetCDF geometry asset has been removed.

Kernel ABI v2 receives spectral/FVM dimensions and runtime controls from
Python. The FVM wrapper obtains its array bounds, reconstruction order, halo
widths, quadrature size, active level range, jet range, and large-Courant
setting from a C-compatible configuration on each call. The SE kernels pass
their spectral-node dimensions directly; the flattened limiter layout is
checked by `pycam_sima_validate_se_dimensions_v2` immediately before the
FP-sensitive limiter call. Keeping `ngp` out of the limiter's numerical
signature preserves CAM's original floating-point instruction order.

GCC emits a different floating-point loop body when these small dimensions are
fully dynamic. The build therefore generates a specialization module from the
Python YAML configuration. The ABI values are still mandatory and are checked
against that module; unsupported dimensions fail instead of selecting an
implicit Fortran default. This keeps the case values out of handwritten
Fortran while retaining the fixed instruction shape required for BFB.

The library build invokes GCC 12.2 in an empty environment. Its build gate
rejects MPI/PMI/PALS, ESMF, PIO, NetCDF, HDF5, LibSci, RPATH, and RUNPATH ELF
entries in addition to forbidden control symbols.

The previous `cam_init`/`cam_run*` wrapper backend has been removed. The
upstream CAM-SIMA executable remains external to this package and is used only
to create immutable BFB oracle output.

The initial 222-field schema is now extensible. `VariableSpec` collectively
appends Python-owned Fortran-contiguous arrays without moving existing
addresses, while device manifests synthesize missing primitive contracts by
CCPP standard name. Live fields use existing named dimensions; redefinition,
resizing, and removal are rejected.

`PhysicsPluginManager` accepts source descriptors and prebuilt manifests,
validates every MPI rank against the same artifact and schema hashes, and
inserts new processes into the editable before/after-coupler plan. Loading may
occur after initialization or between completed phase/scheme actions.
Checkpoint schema v2 carries dynamic contracts and exact plugin identities
through disk restart and Dask in-memory fork.

`examples/plugins/runtime_temperature_offset` is the executable authoring
example. `jobs/dynamic_runtime_24.pbs` validates source compilation, prebuilt
reload, native execution, pointer stability, rank-local dynamic state, and
checkpoint recovery on all 24 ranks; the machine-readable record is
`validation/dynamic_plugins_and_variables.json`.

Dask execution supports both legacy edit-then-step `BranchSpec` segments and
versioned `SegmentPlan` action sequences. One plan may run phase, scheme,
scheme-group, observation, field-edit, and complete-step actions against one
live StatePool in one MPI launch. `submit_action()` makes one action a durable
checkpoint/Future boundary, while `field()` extracts one rank-local array
without gathering the complete checkpoint bundle. Granular model actions are
explicitly unsafe experiments; only the unchanged complete-step order carries
the model BFB contract.

Persistent Dask execution is a separate Actor path. One worker-pinned Actor
owns a live `NotebookSession`, holds the full-node MPI allocation lock, and
launches 24 ranks once. Subsequent Actor futures call phase, scheme, complete
step, field, plan, and checkpoint operations against the same in-memory
StatePool. In PBS mode, `fork_persistent()` captures the base's 24 rank-local
snapshots into one immutable in-memory `CheckpointBundle`, keeps it as a Dask
Future, and supplies it to multiple child Actors. Every child launches an
independent 24-rank MPI model, restores private NumPy arrays through the
socket/MPI bridge, applies a `BranchSpec` or `SegmentPlan`, and stays alive.
This path creates no checkpoint files. It requires a distinct Dask worker and
PBS job per concurrent child; single-node allocation mode rejects persistent
fan-out. Disk checkpoint segments remain the durable, restartable alternative.

## Verification contract

- Initialization must execute zero native calls and must not probe the kernel
  ABI.
- Initialization must not read a pre-generated FVM grid file.
- The native ELF dependency scan must pass after a clean-environment build.
- Every device's CCPP metadata must match its original Fortran signature.
- Generated adapters must call original scheme entrypoints and contain no
  copied numerical algorithm.
- Every device argument must resolve through a complete StatePool contract.
- Device errors must carry the original Fortran error message into Python.
- Every persistent field must report `owner="python"`.
- Every public phase checks NumPy pointer stability.
- Every scheme call checks NumPy pointer stability, and the default 24-scheme
  plan exactly matches `suite_kessler.xml`.
- The fixed 24-rank run must produce nstep=0 plus 50 steps.
- All 51 filenames, timestamps, shapes, dtypes, and bit patterns for the 26
  numeric history variables must match the oracle.
- A batched granular plan and the equivalent chain of single-action checkpoint
  segments must produce bitwise-identical StatePool arrays on all 24 ranks.
- A persistent Dask Actor must retain its model clock and StatePool across
  multiple Actor calls while reporting one MPI launch until explicit close.
- Persistent memory fork must restore every child at the exact parent clock,
  keep branch arrays independent, apply branch edits bit-for-bit, use one MPI
  launch per child, and create no checkpoint manifest, NPZ file, or checkpoint
  directory.

Machine-readable evidence is stored in
`validation/fkessler_model_bfb.json` and
`validation/source_preserving_devices.json`; Dask action evidence is stored in
`validation/dask_granular_actions.json`, and persistent Actor evidence is
stored in `validation/dask_persistent.json` and
`validation/dask_persistent_fork.json`. The granular record contains both real
PBS segments and the single-allocation batch-versus-checkpoint-chain
comparison. The persistent-fork record contains one 10-step base and three
independent 24-rank children restored from the same 37,660,986-byte Dask
snapshot, with exact comparison of all 5,328 rank-local arrays per branch and
zero checkpoint artifacts.

## Scope boundary

The all-suite connector layer now inventories all 7 pinned suite XML files,
155 distinct active schemes, and 340 occurrences. It emits 155 source-derived
descriptors, preserves XML groups/subcycles, supports all primitive metadata
arguments, fixed-width character data, default logical bridges, and
Python-owned shaped allocatable fields plus non-allocatable derived process
state. Pure CAM-history schemes are routed to an explicit Python history
service. The current build has 100 native devices and 29 Python host services;
26 connectors require an external service or dependency. Exact per-scheme
reasons are generated in `validation/all_scheme_support.json`; they must be
read instead of treating connector generation as proof that every scientific
configuration has run. The aggregate build, load, ELF, and 50-step regression
evidence is in `validation/all_scheme_connectors.json`.

The descriptor source of truth is a single
`devices/generated/<scheme>/device.yaml` tree. The two Kessler descriptors are
not maintained as special parallel directories; their non-metadata lifecycle
and vertical-index policy is declared in `devices/overrides.yaml` and merged
by the same deterministic catalog generator. The corresponding clean-build
and 24-rank/50-step BFB record is
`validation/descriptor_unification.json`.

The portable `ref_pres` provider implements the low-top behavior documented
inside the original Holtslag–Boville interstitial (`ntop_eddy = 1`). It is
valid for the seven pinned suites and is not a WACCM-X reference-pressure
implementation.

The remaining non-ready entries are fail-closed boundaries, principally input
data readers, MPI/global reductions, external MUSICA/physics sources,
allocatable derived objects, and large CAM infrastructure closures. They are
not replaced by no-op numerical kernels. Only the FKESSLER 24-rank/50-step
configuration currently carries a full-model BFB claim. Other complete
suites, grids, vertical levels, MPI sizes, timesteps, mediator or surface
components, MPAS, GPU execution, and FADIAB are not model-validated.
