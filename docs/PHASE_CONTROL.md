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

`model.step()` uses the declarative `model.step_plan`. The default plan contains
the same seven phases and is accepted only at a complete-cycle boundary:

```python
model.step_plan.describe()
model.step()
```

The first cycle is CAM-SIMA's `nstep=0`
initial-send cycle and does not increment the Python step counter; the next
cycle produces requested step 1.

Every full-CAM phase is required by the validated lifecycle. An on/off or
ordering experiment therefore requires explicit acknowledgement:

```python
# Control experiment only; this is not a valid dynamics-only CAM setup.
model.step_plan.disable("cam_run3", unsafe=True)

# Control experiment only; CAM may reject the native lifecycle.
model.step_plan.move("cam_run3", before="cam_run2", unsafe=True)

model.step()
```

Once an unsafe plan executes, that native session cannot be declared safe
again. Start a fresh session to return to the validated plan.

## Explicit initialization settings and live fields

The complete CAM initialization parameters are ordinary Python objects:

```python
options = FullCAMRuntimeOptions(
    timestep_seconds=1800,
    physics_profile="kessler",
    mediator_present=False,
)
```

They are passed through the C ABI into `cam_init`; they cannot be changed after
initialization. The 1800-second setting is the BFB-validated configuration;
other positive integer timesteps are explicit experiments. Live CAM fields can
be read or written at any phase boundary:

```python
temperature = model.parameters.air_temperature
values = temperature.get(rank=0)
values[0, 0] += 0.01
temperature.set(values, rank=0)
```

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
