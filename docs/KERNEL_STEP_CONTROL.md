# Kessler kernel step control

`FKesslerDriver` keeps CAM's default order, but the order and the main runtime
switches are now ordinary Python objects rather than an inline list hidden in
`run()`.

## Explicit setup

```python
from pycam_sima import FKesslerDriver, RuntimeOptions, StepPlan
from pycam_sima.config import CaseConfig

config = CaseConfig.from_yaml("configs/fkessler_ne3pg3.yaml")

options = RuntimeOptions(
    timestep_seconds=1800,
    physics_before=True,
    physics_after=True,
    dynamics=True,
)
plan = StepPlan.default()
model = FKesslerDriver(config, options=options, step_plan=plan)

model.allocate_minimal_state(ncol=1)
model.parameters.surface_reference_pressure = 100_000.0
model.parameters.dycore_energy_adjustment = True
model.parameters.constituent_minimum_values = (1.0e-12, 0.0, 0.0)
model.initialize()
```

The effective order is inspectable in a Notebook:

```python
model.step_plan.describe(model.options)
```

The default result has these eight phases:

1. `kessler_after_coupler`
2. `physics_to_dynamics`
3. `se_dynamics`
4. `physics_timestep_final`
5. `advance_clock`
6. `dynamics_to_physics`
7. `physics_timestep_initial`
8. `kessler_before_coupler`

`model.run(n)` is now just `model.step()` repeated `n` times.

## Switches and parameters

Options can be changed between calls to `step()`:

```python
# Keep the mapping and lifecycle calls, but skip both Kessler physics sections.
model.options.physics_before = False
model.options.physics_after = False

# Run or skip the injected dynamics implementation.
model.options.dynamics = True

# A direct option edit is synchronized at the start of the next step.
model.options.timestep_seconds = 900
model.step()
```

The typed parameter view writes the same NumPy arrays passed to the native
wrappers:

```python
model.parameters.surface_reference_pressure = 98_500.0
model.parameters.dycore_energy_adjustment = False
model.parameters.constituent_minimum_values[:] = (2.0e-12, 0.0, 0.0)
model.parameters.timestep_seconds = 600  # synchronized immediately

print(model.parameters.describe())
print(model.pool["air_temperature"])
```

Every other field remains directly available through `model.pool`. These are
live arrays, so edits are seen by the next native call without a copy.

For an edit at an internal phase boundary, register an observer:

```python
def edit_before_dynamics(context):
    context.state["air_temperature"][:] += 0.01

model.observe(
    "phase_begin:se_dynamics",
    edit_before_dynamics,
    access="readwrite",
)
```

## Order experiments

Optional phases can be disabled without changing the validated relative order:

```python
model.step_plan.disable("se_dynamics")
model.step_plan.enable("se_dynamics")
```

The five mapping, clock, and lifecycle phases are required. Disabling one or
moving any phase requires an explicit unsafe acknowledgement:

```python
model.step_plan.move(
    "se_dynamics",
    before="physics_to_dynamics",
    unsafe=True,
)
assert not model.step_plan.sequence_safe
```

`unsafe=True` means only that the requested experiment will be attempted. It
does not make the altered order physically valid, and disabling a required
Kessler lifecycle phase can make a later call fail its state check. Use
`model.step_plan.reset()` to restore the validated default.

This switchable kernel path uses `IdentityDynamics` unless another dynamics
object is supplied, so “physics off, dynamics on” is only a control-flow test
with the default object. For the complete SE dycore experiment, use the full
CAM `FADIAB` configuration described in `PHASE_CONTROL.md`; do not simulate it
by skipping `cam_run2` in an FKESSLER full-model run.
