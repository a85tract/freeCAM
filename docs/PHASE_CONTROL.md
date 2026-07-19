# Phase-by-phase control

`NotebookSession.run_phase()` exposes the seven top-level CAM-SIMA calls that
form one no-mediator `ModelAdvance` cycle:

```text
cam_run2
cam_run3
cam_run4
cam_timestep_final
advance_timestep
cam_timestep_init
cam_run1
```

The MPI worker returns to its command wait loop after every call, so fields can
be read or changed between phases:

```python
model.phase_status

model.run_phase("cam_run2")
after_physics = model.get_field("air_temperature", rank=0)

model.run_phase("cam_run3")
after_dynamics = model.get_field("air_temperature", rank=0)
```

Calling `run_phase()` without a name executes `model.next_phase`. The default
state machine rejects an invalid order before native CAM is called. An explicit
ordering experiment may use `allow_unsafe_order=True`, but doing so permanently
marks that session unsafe and disables `model.step()`:

```python
# Experimental: this is not a scientifically valid CAM ordering.
model.run_phase("cam_run3", allow_unsafe_order=True)
```

`model.step()` still uses exactly the same phase state machine. It is accepted
only at a complete-cycle boundary. The first cycle is CAM-SIMA's `nstep=0`
initial-send cycle and does not increment the Python step counter; the next
cycle produces requested step 1.

## No-physics-forcing dynamics configuration

CAM-SIMA already provides the `FADIAB` compset and `adiabatic` CCPP suite for a
dynamics run with no parameterized-physics forcing. This profile uses the
Held--Suarez analytic initial state, so its startup does not depend on a
parallel read of a moist CAM initial-condition file. It still runs required
state conversion, energy bookkeeping and diagnostics. This is safer than
skipping `cam_run2`, because `cam_run2` also performs the
physics-to-dynamics coupling.

Use the corresponding configuration and native library:

```python
model = NotebookSession(
    repo / "configs/adiabatic_ne3pg3.yaml",
    run_dir=adiabatic_run_dir,
    env_script=adiabatic_case / ".env_mach_specific.sh",
)
model.start()
```

The Kessler and adiabatic suites are selected at CAM initialization. Switching
suites inside an already initialized timestep is intentionally unsupported;
start a new session to change the physics configuration.
