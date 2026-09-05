# freeCAM Python process performance plan (v2)

Branch: `native-stage-batching` (from `standalone-physics-function`,
2026-09-04). This file is the task's plan. The historical performance evidence
lives in `validation/performance_overhead.md` and is never overwritten. v2
replaces the v1 of the same name: segmented execution now generates
start/resume entries at the original Fortran call sites rather than
translating the Python class back into Fortran.

## Summary

The existing Python process classes and the kernel-replacement interface stay
as they are; what changes is how production runs execute them.

- The Python class is not regenerated or compiled into Fortran.
- With no kernel replaced, the original whole Fortran process is called once.
- With a Python or AI kernel in a slot, the original Fortran runs up to the
  replacement point and returns to Python; Python runs the replacement kernel
  and calls `resume()`, and the Fortran continues from where it stopped.
- Fortran never calls back into Python: no `c_funptr`, no `ctypes.CFUNCTYPE`.
- The statement-by-statement Python transliteration is kept as a readable
  reference, a debugging tool and the bit-for-bit oracle; it is no longer the
  default hot path.
- The UI and the uncommitted `workflow_builder/` are out of scope.

Status: native-whole is implemented and passed a month with all 18 history
and restart files bit-for-bit; 424.30 s against 402.51 s for plain freeCAM,
about 5.4% slower. The remaining core work was segmented replacement and the
last of the wrapper overhead.

Status update (2026-09-04): on the trusted native path the median of three
native-whole months is 404.61 s, level with plain freeCAM (+0.5%).
Segmented-original passed the 512-rank 50-step gate (job 7322256):
`mmacro_pcond` replaced by the original kernel answered through the Python
boundary, the runner pausing twice a step (once per chunk), 150 segment calls
and 100 Python model calls over 50 steps, all four history and restart files
bit-for-bit (`validation/pi_cam_stage7_segmented_original_50step.json` and
its `_vs_oracle_50step_bfb.json`).

Status update (2026-09-04, evening): item 8 of the delivery order has landed.
Under `auto`, a stage with replacements runs segmented when the image's runner
pauses at every replaced kernel, and as the Python walk otherwise. A silent
fallback was fixed at the same time: a surrogate named by path used to load on
first use, so its slot was empty when the path was chosen, and one surrogate
month ran the original Fortran to the end and reported itself bit-for-bit (job
7322838). The slot is now held from construction by an unloaded
`PendingSurrogate`, and `tend` refuses to run when a replacement it was told
about does not show in its slots.

## 1. Execution semantics

The interface is unchanged:

```python
stage.kernels["mmacro_pcond"] = None
stage.kernels["mmacro_pcond"] = model
stage.kernels["micro_mg_tend"] = model
stage.kernels["rad_rrtmg_sw"] = model
stage.kernels["rad_rrtmg_lw"] = model
```

Three internal modes:

- **native-whole**: every replacement is `None`; the Python class calls the
  original whole Fortran process once a step.
- **segmented**: at least one kernel is replaced; the Fortran pauses only at
  the actual replacement points and returns to Python; helpers, kernels, loops
  and arithmetic that are not replaced keep running in the original Fortran.
- **legacy-python**: the statement-by-statement Python transliteration, for
  debugging, call-order checks and performance comparison only.

The default is `auto`: no replacement selects native-whole; a replacement
selects segmented.

Python still controls the workflow order, which processes are enabled, and
which kernels are replaced. The high-frequency loops and helper dispatch
inside a process that are not replaced stay in Fortran, so tens of thousands
of Python/Fortran crossings are avoided.

## 2. Implementation

### 2.1 Finish the native-whole fast path

Remove the overhead a built-in `NativeStage` pays for going through the Python
process registry:

- Mark the built-in stage processes as trusted native and treat them apart
  from ordinary notebook Python processes.
- native-whole creates no `PythonFieldView` and takes no field snapshot.
- Do not scan every StatePool pointer each step; pointer-stability checks move
  to initialization, field-registration changes and debug mode.
- Merge the duplicated MPI error collection into one collective status check.
- `native.run_action()` calls the backend primitive directly rather than
  re-entering the workflow dispatcher, so there is no recursion.

### 2.2 Give the original Fortran start/resume entries

The segmented code is generated from the original Fortran call sites at their
pinned revision and reviewed by hand; it is not derived from the Python class.

One internal ABI:

```
stage_context_create(stage_id) -> context_id
stage_start(context_id, replacement_mask) -> event
stage_frame(context_id) -> kernel arguments
stage_resume(context_id, completed_kernel_id) -> event
stage_context_destroy(context_id)
```

The only events are `DONE`, `NEEDS_PYTHON_KERNEL` and `ERROR`.

The sequence is: Python `stage.run()` calls `stage_start()`; the Fortran runs
until it reaches a replaced kernel; it saves its position and live state and
returns `NEEDS_PYTHON_KERNEL`; Python runs the model, checks and writes back
the result, and calls `stage_resume()`; the Fortran continues from where it
stopped.

### 2.3 Saving the paused Fortran state

Ordinary Fortran locals do not survive a subroutine's return, so each MPI rank
keeps a rank-local context holding: the program counter; the current chunk,
`lchnk` and `ncol`; the substep and kernel call index; the replacement mask;
the scalars that must survive a pause; automatic arrays and temporary
tendencies; and stable handles to CAM module and derived-type storage.

The generator does a live-variable analysis of the original call site and
emits a scaffold; the live-state list of every boundary is reviewed by hand.
A change in the source hash or the call-site anchors fails the build; a stale
adapter is never used silently.

### 2.4 The four replacement boundaries

The first version fully supports `mmacro_pcond`, `micro_mg_tend`,
`rad_rrtmg_sw` and `rad_rrtmg_lw`.

Nested calls: the cloud stage's context saves the enclosing `tphysbc` and
substep state; the macrophysics and microphysics sub-contexts pause before
their cores; when a sub-context returns a replacement event the outer context
saves its position too and returns to Python; `resume()` restores the inner
procedure first, then the outer; radiation uses one context for the SW and LW
pause points in turn.

Each original kernel call produces at most one pause/resume. The first
version does not reorder calls across chunks or substeps, so neither the
floating-point order nor the history order changes.

### 2.5 The Python replacement frame

`stage_frame()` returns the kernel id, call index, `lchnk`, `ncol`, substep,
and the argument names, pointers, shapes, dtypes and intents.

Python keeps the current model contract: only the live `ncol` columns reach
the model, never a padding lane; inputs are batched as the existing contract
says; the output must carry every required field; shapes, dtypes and names
must match exactly; the answer is written into the native context with
`np.copyto(..., casting="no")`; `resume()` is called only after the write-back
is complete.

When a model raises or returns an invalid result: the context is destroyed;
the model is marked tainted; further stepping, checkpointing and finalization
are refused; no claim is made to roll back the non-transactional Fortran work
already done.

### 2.6 Lifecycle limits

Only while a context is idle may a replacement be swapped or removed, the
workflow edited, a checkpoint or restart taken, history flushed, or the model
finalized. Each context carries a generation and a call token, and refuses a
repeated resume, a resume for the wrong kernel, or a write-back from a stale
frame.

## 3. Tests and validation

### Unit tests

- With no replacement the original stage is called once a step and
  `tend_chunk()` is never called.
- Adding a replacement switches `auto` to segmented; removing it restores
  native-whole.
- The runner returns only before a replaced kernel; unreplaced kernels never
  return to Python.
- The pause order over several replacements, chunks and substeps is right.
- A nested context propagates events outward and resumes from its position.
- The frame's names, shapes, dtypes, intents and `ncol` agree with the
  current interface.
- Invalid output, exceptions, repeated resumes and stale tokens are cleaned
  up safely.
- A static check confirms there is no Fortran-to-Python callback.
- GitHub Actions exercises the state machine against a fake backend, without
  Derecho.

### Bit-for-bit gates, in order

1. Single-rank synthetic: native-whole, legacy-python and segmented-original
   leave every array bitwise identical.
2. 512 ranks, 50 steps: `mmacro_pcond` segmented but still calling the
   original kernel; then `micro_mg_tend`; `rad_rrtmg_sw`; `rad_rrtmg_lw`; then
   all four boundaries enabled together.
3. Every StatePool, history and restart value compared with no tolerance.
4. One month, 1488 steps, all 18 files bit-for-bit.
5. After the month, one year, all 180 files bit-for-bit.

"Segmented but still calling the original kernel" must be bit-for-bit. If it
is not, the paused state, the call order or the write-back location is wrong,
and the difference cannot be attributed to a model.

### Performance gates

Same compiler, 512 ranks, four nodes, rank placement and inputs; three runs
each, the median taken:

- native-whole: within 5% of plain freeCAM.
- native-whole stage: at most 48.6 ms per step and rank.
- segmented-original: remove at least half of the current 92.13 ms excess;
  target about 65 ms per step and rank.
- Report separately: Python/Fortran crossings; pause/resume counts; Python
  model calls; pointer resolutions; bytes copied; mean-rank and slowest-rank
  time.

A real model's inference time and the framework's pause/resume overhead are
reported separately.

## 4. Delivery order

On the `native-stage-batching` branch:

1. Slim the native-whole registry path and MPI checks to the 5% gate.
2. Implement the generic context, event and frame ABI and a fake runner.
3. Attach `mmacro_pcond` and pass the 50-step bit-for-bit gate.
4. Attach `micro_mg_tend` and pass the nested-context gate.
5. Attach the two RRTMG kernels.
6. Pass the combined four-boundary, month and year gates.
7. Update the performance document and the machine-readable validation JSON.
8. Only after every gate passes, make `auto` with a replacement select
   segmented by default.

Each stage is its own commit. Historical performance results are kept, never
overwritten; the untracked `workflow_builder/` UI work is neither committed
nor modified.

## Assumptions

- The original Fortran is the only native source of truth for the science
  and the floating-point order.
- The Python class is the public process object, the replacement dispatcher,
  and the executable reference and debug implementation.
- The Python class is not translated back into Fortran.
- Not every helper-level `if` and loop inside a process has to be executed by
  interpreted Python; otherwise the ~30% loss cannot be removed at the root.
- The first version covers the current PI-CAM configuration and the four
  declared swappable kernels only; other configurations report unsupported.
- The MPI communicator, rank placement, numerical formulas and reduction
  order do not change.
