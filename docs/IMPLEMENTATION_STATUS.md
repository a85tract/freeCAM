# Implementation status

## Implemented target

The repository implements one complete configuration:

- CAM-SIMA `f8daa568eae2696b7c4ebff7768f02f5d097d9df`
- FKESSLER, `ne3np4.pg3`, L30, 24 ranks, one thread per rank
- 1800-second timestep and 50 requested steps
- DCMIP2016 moist baroclinic-wave initial condition
- startup, ATM-only, NO_LEAP execution

Python parses `atm_in`, initializes the grid and analytic state, owns all 214
canonical persistent fields, executes the model phase graph, performs MPI
communication through mpi4py, advances the clock, and writes history.
`libpycam_sima_kernels.so` exposes versioned stateless numerical kernels and
retains no array pointer or mutable model state between calls.

The Python scheme plan mirrors every active entry in the pinned Kessler CCPP
suite. Its default 19 before-coupler and 5 after-coupler schemes are
independently callable, observable, enabled/disabled, reorderable, and movable
between groups. Required-scheme changes demand `unsafe=True`; the untouched
XML order is the BFB contract.
The two Kessler boundaries and the FP-sensitive geopotential helper call
stateless Fortran kernels. The remaining conversion, thermodynamic,
conservation, tendency, and diagnostic boundaries execute against the same
Python-owned StatePool.

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

## Verification contract

- Initialization must execute zero native calls and must not probe the kernel
  ABI.
- Initialization must not read a pre-generated FVM grid file.
- The native ELF dependency scan must pass after a clean-environment build.
- Every persistent field must report `owner="python"`.
- Every public phase checks NumPy pointer stability.
- Every scheme call checks NumPy pointer stability, and the default 24-scheme
  plan exactly matches `suite_kessler.xml`.
- The fixed 24-rank run must produce nstep=0 plus 50 steps.
- All 51 filenames, timestamps, shapes, dtypes, and bit patterns for the 26
  numeric history variables must match the oracle.

Machine-readable evidence is stored in
`validation/fkessler_model_bfb.json`.

## Scope boundary

Other suites, grids, vertical levels, MPI sizes, timesteps, restart/branch
runs, mediator or surface components, MPAS, GPU execution, and FADIAB are not
supported.
