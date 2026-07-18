# pycam-sima implementation status

## Implemented target

The complete fixed target is implemented:

- CAM-SIMA `f8daa568eae2696b7c4ebff7768f02f5d097d9df`
- `FKESSLER`, `ne3pg3_ne3pg3_mg37`, L30
- DCMIP2016 moist baroclinic-wave analytic initial condition
- real SE dynamics with `se_nsplit=2`, `se_qsplit=1`, `se_tstep_type=4`
- Kessler `physics_before_coupler` and `physics_after_coupler`
- mpi4py communicator, 24 ranks, 1800-second timestep
- Python/Taskflow control of `DataInitialize` and every `ModelAdvance` phase
- Python observers and zero-copy state views at phase and step boundaries
- ordinary Jupyter kernel control through a separate authenticated 24-rank MPI
  worker, with synchronous `step`, `get_field`, `get_field_stats`, and
  `set_field` operations

`libpycam_sima_full.so` links the real PIC-enabled CAM-SIMA ATM archive. Its
explicit C ABI separates initialization, `cam_run1`, `cam_run2`, `cam_run3`,
timestep finalization, time advancement, finalization, and state queries. The
Python driver therefore owns the control loop; it does not launch `cesm.exe`.

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

- 8 unit/integration tests pass.
- The small Kessler library passes serial and 24-rank/50-step smoke tests.
- Native CAM-SIMA reference job `6779760.desched1` completed successfully.
- Python-controlled full-model job `6779818.desched1` completed successfully
  with marker `PYCAM_SIMA_FULL_JOB_OK job=6779818.desched1 ranks=24 steps=50`.
- `compare-history` compared 51 timestamps: no missing/extra files and no
  differing value in any of the 26 numeric history variables. The five
  prognostic fields `T`, `Q`, `U`, `V`, and `PS` were REAL64 and bitwise
  identical. Numeric-state BFB is true for this fixed target.
- Notebook session job `6787846.desched1` started from one non-MPI controller,
  returned 21 live fields, completed two interactive steps, gathered all-rank
  statistics, and finalized cleanly. Its three output timestamps matched all
  26 numeric reference variables bitwise.

Machine-readable evidence is in `validation/fkessler_full_bfb.json` and
`validation/notebook_session_smoke.json`.

## Scope boundary

This is not yet a general replacement for every CAM-SIMA configuration. Other
physics suites, grids, vertical levels, mediator-enabled surface components,
restart/branch runs, MPAS dynamics and GPU configurations remain outside the
implemented target. BFB applies only to the fixed configuration above.
