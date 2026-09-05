# Decoupling the physics kernels

The aim is one shape for every active physical process of the PI-atm step: a
Python process class that owns the process's place in the workflow, its
parameters and its kernel slots, with the numerical kernels behind a declared,
replaceable, validated interface. The same kernel can then be called on its
own, sampled into training data, or replaced in the full model by a Python
function or a trained network, under one contract and one replacement
configuration. What runs when nothing is replaced is the original Fortran,
whole; what runs when something is replaced is the original Fortran up to the
replaced call, then Python, then the original Fortran again.

This page says how the pieces fit, what the inventory records, and where the
work stands. The ledger itself is
[`validation/physics_kernel_decoupling.json`](../validation/physics_kernel_decoupling.json),
written by `tools/build_physics_kernel_coverage.py` from the repository's
own records and checked current by the unit suite.

## The shape of a process

```
Python process class
    ├── parameters, lifecycle, place in the workflow
    ├── kernel contracts and kernels[...] slots
    └── execution
          ├── nothing replaced   -> the original Fortran process, called once
          └── something replaced -> Fortran to the replaced call
                                    -> frame to Python -> the replacement
                                    -> write-back, checked -> resume
```

Three pieces make this concrete today.

**The kernel slots.** `stage.kernels[name]` is the only registry of what
computes a kernel. Every way of installing a replacement lands there: assigning
into the mapping, naming a model file with `surrogate=`, or binding over the
method (`stage.mmacro_pcond = MethodType(fn, stage)`). The last used to rebind
the method alone, so a notebook's single-column call saw the new function
while the model, deciding how to run from the slots, ran the original Fortran.
`NativeStage.__setattr__` now puts a bound method into the slot as a
`MethodKernel`, and the walk, the runner's frame and the single-column caller
reach the same function. `OriginalKernel()` in a slot is the validation
replacement: the pause path runs, the original kernel answers through Python,
and bit-for-bit output proves the frame and the write-back.

**The segment-runner manifest.** Where the image can pause is declared in
[`native/pi_cam/segment_runners.yaml`](../native/pi_cam/segment_runners.yaml):
for each stage, the Fortran module, its generator, the descriptor its frames
are decoded with, the kernels it pauses at, and the gate records that
validated each pause. The backend (`freecam.pi_cam.native`) asks the manifest
which stage has a runner instead of knowing one by name; the stages ask it
whether a replacement can run segmented; the Workflow Builder reads it to say
which kernels are bindable and which are validated. A runner is
`ImageSegmentRunner(library, spec)`, one class for every prefix.

**The read-only description.** `stage.describe_kernels()` returns one record
per kernel: the owning class, whether the runner pauses at it, whether that
pause has passed a gate, the reviewed contract's inputs and outputs when one
exists, what is in the slot now, and how many times a model answered for it in
this run. The run summary written by the command line carries the same rows
under `stage_execution`, and the Workflow Builder consumes them rather than
keeping a list of its own.

## The inventory

The ledger classifies each of the step's 58 actions once and follows every
exposed kernel through the delivery loop:

```
contract reviewed -> original call captured -> standalone image built
  -> replayed bit for bit (full chunk, single column, public interface)
  -> replaced in the full model with the original kernel answering, bit for bit
  -> performance recorded -> supported
```

An action is one of: `numeric_scheme`, `process_control`, `diagnostics`,
`boundary`, `clock`, `dynamics`, `io`, `host_service`. A disabled action is
recorded as the alternate form of the enabled work it stands for (a stage
whose leaves run, or a leaf whose stage runs whole), never as a hole. An
enabled scheme whose body is expected to do nothing under this configuration
(`rayleigh_friction` without `rayk0`, the CARMA leaves with no CARMA model)
is marked inert-by-configuration and listed as unresolved until a targeted
test confirms it; it is not counted as covered. Execution is evidenced from
the recorded runs at two lengths, 50 steps and a month, not assumed from the
plan.

The record names nothing outside the repository: contract paths, evidence
files and catalog sources are relative, and there is no timestamp, so the
committed file equals a fresh build or the test fails.

## Where it stands

| Kernel | Owner | Contract | Runner pause | In-model gate | Loop |
| --- | --- | --- | --- | --- | --- |
| `mmacro_pcond` | Macrophysics (in the cloud stage) | reviewed | yes, validated | segmented, bit-for-bit | complete |
| `micro_mg_tend` | Microphysics (in the cloud stage) | reviewed | no | walk with the core through its image, bit-for-bit | open |
| `rad_rrtmg_sw` | Radiation | draft | no | walk with the original kernel, bit-for-bit | open |
| `rad_rrtmg_lw` | Radiation | draft | no | walk with the original kernel, bit-for-bit | open |

Twelve enabled scheme actions do numerical work in this configuration. Two
have a Python class today, both partial by the loop above. The other ten are
gaps with their candidate procedures listed from the catalog: vertical
diffusion, gravity-wave drag, the energy fixer, dry adjustment, deep and
shallow convection, and the wet deposition, dry deposition, convective
transport and chemistry leaves. For the four leaves the catalog's active call
graph lists no procedure yet; the ledger says so rather than choosing kernels
without it.

The stages that exist are delivered in this order: the generic contract and
runner registration (done, above); the cloud stage's second kernel and the two
radiation kernels through a segment runner, each gated with the original
kernel answering; convection and turbulence; the remaining active schemes in
dependency order; then the combined and long runs. Every step of the loop is
a record under `validation/` before the ledger counts it.
