# pycam-sima implementation status

## Implemented target

The complete fixed target is implemented:

- CAM-SIMA `f8daa568eae2696b7c4ebff7768f02f5d097d9df`
- `FKESSLER`, `ne3pg3_ne3pg3_mg37`, L30
- DCMIP2016 moist baroclinic-wave analytic initial condition
- real SE dynamics with `se_nsplit=2`, `se_qsplit=1`, `se_tstep_type=4`
- Kessler `physics_before_coupler` and `physics_after_coupler`
- `FADIAB` dynamics profile with Held--Suarez analytic initial state and no
  parameterized-physics forcing
- mpi4py communicator, 24 ranks, 1800-second timestep
- Python/Taskflow control of `DataInitialize` and every `ModelAdvance` phase
- explicit Notebook control of the seven top-level calls in one advance cycle
- declarative `FullCAMStepPlan` control used directly by complete-CAM `step()`
- explicit `FullCAMRuntimeOptions` passed through the ABI into `cam_init`
- typed `FullCAMParameters` field handles for live MPI state inspection/editing
- Python observers and zero-copy state views at phase and step boundaries
- ordinary Jupyter kernel control through a separate authenticated 24-rank MPI
  worker, with synchronous `step`, `get_field`, `get_field_stats`, and
  `set_field` operations

`libpycam_sima_full.so` links the real PIC-enabled CAM-SIMA ATM archive. Its
explicit C ABI separates initialization, `cam_run1`, `cam_run2`, `cam_run3`,
`cam_run4`, timestep finalization, time advancement, finalization, and state
queries. The Python driver therefore owns the control loop; it does not launch
`cesm.exe`.

## State ownership

The Python `StatePool` is the single user-facing registry in both modes.

- Kernel mode: NumPy owns the allocations and native Kessler functions borrow
  their pointers.
- Full mode: CAM must retain allocation ownership for SE derived types and
  module state. The pool holds writable zero-copy NumPy views of those stable
  allocations. Reading or editing a pool array reads or edits the exact memory
  used by the next CAM phase; no snapshot copy is involved.

The full pool exposes 21 major state groups, including all primary atmospheric
state fields, physics tendencies and the complete constituent array. Temporary
SE scratch arrays are intentionally not part of the public state contract.

## Verification

- 28 unit/integration tests pass.
- The small Kessler library passes serial and 24-rank/50-step smoke tests.
- Native CAM-SIMA reference job `6779760.desched1` completed successfully.
- Current Python-controlled Kessler job `6792903.desched1` completed
  successfully with marker
  `PYCAM_SIMA_FULL_JOB_OK job=6792903.desched1 ranks=24 steps=50`.
- `compare-history` compared 51 timestamps: no missing/extra files and no
  differing value in any of the 26 numeric history variables. The five
  prognostic fields `T`, `Q`, `U`, `V`, and `PS` were REAL64 and bitwise
  identical. Numeric-state BFB is true for this fixed target.
- Kessler phase-session job `6790950.desched1` and adiabatic phase-session job
  `6790933.desched1` each paused after all seven phases, inspected temperature,
  completed the initial-send cycle and requested step 1, and finalized cleanly.
- The 24-rank, 50-step `FADIAB` reference job `6790929.desched1` and
  current Python-controlled job `6792935.desched1` each produced 51 history
  files. All 26 numeric variables were bitwise identical at every timestamp.
- Notebook session job `6788371.desched1` started from one non-MPI controller
  with `PBS_NODEFILE` intentionally removed,
  returned 21 live fields, completed two interactive steps, gathered all-rank
  statistics, and finalized cleanly. Its three output timestamps matched all
  26 numeric reference variables bitwise.
- Login-node test on `derecho6` automatically submitted PBS worker job
  `6788530.desched1`, returned a live rank-zero temperature array after one
  interactive step, and finalized cleanly. Both output timestamps matched all
  26 numeric reference variables bitwise.
- Complete-CAM control API job `6792924.desched1` exercised explicit runtime
  options, the seven-phase `FullCAMStepPlan`, typed temperature access, and one
  collective step from a login-node controller. Both available timestamps
  matched all 26 numeric reference variables bitwise.
- Non-default runtime-option job `6792973.desched1` ran 24 ranks for 50 steps
  with `timestep_seconds=900`. All 51 history files report `mdt=900`, proving
  that the Python option reached CAM's native time manager through ABI v3.

Machine-readable evidence is in `validation/fkessler_full_bfb.json`,
`validation/adiabatic_full_bfb.json`, and
`validation/notebook_session_smoke.json`; login-node evidence is in
`validation/notebook_login_session.json`, and the complete control-facade
evidence is in `validation/full_control_notebook_smoke.json`.
The non-default timestep evidence is in
`validation/fkessler_dt900_smoke.json`.

## Scope boundary

This is not yet a general replacement for every CAM-SIMA configuration. Only
the Kessler and adiabatic suites described above are supported. Other physics
suites, grids, vertical levels, mediator-enabled surface components,
restart/branch runs, MPAS dynamics and GPU configurations remain outside the
implemented target. BFB applies only to the fixed configurations above.
