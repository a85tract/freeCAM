# pycam-sima implementation status

## Implemented in this milestone

- The repository owns the Python driver and pins CAM-SIMA at
  `f8daa568eae2696b7c4ebff7768f02f5d097d9df`.
- `StatePool` owns every numeric array. Native routines receive stable raw
  pointers and do not allocate replacement model-state arrays.
- `DataInitialize` and `ModelAdvance` reproduce the ATM-only ordering around
  `physics_before_coupler` and `physics_after_coupler`.
- All 19 before-coupler and 5 after-coupler entries from `suite_kessler.xml`
  have typed Python-to-Fortran calls.
- The shared library compiles the pinned CAM-SIMA scheme sources, including
  `kessler_run`, `check_energy_chng`, `qneg`, state/tendency diagnostics and
  timestep lifecycle routines. CAM history calls are adapted to Python
  observers; the non-portable `cam_thermo` dependency is limited to the
  FKESSLER hydrostatic-energy algorithm.
- Taskflow expresses the dependent control sequence. mpi4py supplies the
  communicator, rank and gather operations. Derecho's active Cray MPICH ABI
  path is selected before Python loads mpi4py.
- Observers can read or modify arrays in interactive mode. Validation mode
  enforces read-only callbacks. CLI watchers and per-rank NPZ snapshots expose
  values at step or function boundaries.

## Verified

- GNU Fortran shared-library build succeeds.
- Seven unit/integration tests pass.
- A serial 50-step native Kessler kernel run succeeds.
- PBS job `6778826.desched1` completed a 24-rank, 50-step run with the marker
  `PYCAM_SIMA_JOB_OK job=6778826.desched1 ranks=24 steps=50`.

The 24-rank evidence is an MPI/ABI/control-flow kernel smoke. It is not a BFB
comparison with CAM-SIMA.

## Deliberately not claimed

- `libpycam_sima_se.so` and the real SE dynamics state transitions are not yet
  implemented. The current dynamics boundary is explicitly named `identity`.
- The Python initializer is a deterministic column-kernel state, not the full
  DCMIP2016 moist baroclinic-wave analytic initial condition on the ne3pg3
  spectral-element grid.
- No CAM-SIMA 50-step reference capture has been compared field-by-field, so
  there is no BFB claim.

These three items form the next integration gate. A future full-model
validation command must fail closed until all three are available.
