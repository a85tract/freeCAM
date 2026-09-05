# freeCAM performance plan: keep the UI and bit-for-bit, beat the original Fortran by 5%

Branch: `native-stage-batching` (from 2026-09-04). This file is the task's
plan (v3; it supersedes `native_stage_batching_plan.md` as the overall goal,
while that plan's three modes, native-whole / segmented / legacy-python, and
its runner design remain valid as phase 3 here). The historical performance
evidence lives in `validation/performance_overhead.md` and is never
overwritten.

## 1. Goal and acceptance

Keep the current Python classes, workflow editing, kernel replacement, dynamic
variables, dynamic Python functions, reload, step-by-step running and
plotting. Every Python function is still driven by Python; Fortran hands
control back through return statuses.

Goal: with the same resources, the same online coupling, and the same numbers
and output, freeCAM advances the complete model at least 5% faster than the
original Fortran.

Acceptance covers a month (1488 coupling steps) and a year (17520 coupling
steps) of the current PI-atm configuration on 512 MPI ranks and four nodes,
with every active component computing normally on real online x2a/a2x, the
same history and restart variables, frequencies, precision and compression,
the original kernel path bit-for-bit, and every existing user feature kept.
The goal is not reached by doing less science or writing less output.

The performance threshold is `median(freeCAM advance time / original Fortran
advance time) <= 0.95`, with no regression of the full lifecycle time, and
peak memory reported.

The earlier "at most 5% over plain freeCAM" is a historical milestone, not
completion. The native-whole month is already verified; the segment runner is
in development and continues from that work.

## 2. A trustworthy baseline and a budget

### Three groups

| Group | Control layer | Native implementation | Purpose |
| --- | --- | --- | --- |
| A | original Fortran driver | the pinned original | the target baseline |
| B | original Fortran driver | the optimised native code | the gain from native optimisation |
| C | freeCAM Python driver and classes | the same native code as B | the complete freeCAM |

C/A is the final comparison; B/A and C/B are reported too, to separate native
gains from the Python framework's cost. A native optimisation that can stand
alone is used by both B and C.

### Fixed conditions

- Fixed source, compiler, optimisation flags, MPI library, inputs, namelist
  and native library hash.
- Fixed rank-to-node placement, CPU affinity, and OpenMP and maths-library
  thread counts.
- The PBS requests are checked against the actual CPU binding, so a change of
  layout is not mistaken for a code gain.
- Paths, accounts and queues come from the site configuration; no new script
  names a personal path or account.
- Performance runs disable the fine-grained profiler; diagnostics run
  separately.
- A pair runs in one allocation where possible, A and C in alternating order,
  each in its own output directory.

Timing uses the same complete coupling-loop boundary for both: CAM, the other
components, the coupler, normal communication and in-loop output.
Initialization, final output and cleanup are reported as lifecycle time.
freeCAM's online time is never compared with the original's ATM timer alone.

### Finding enough to recover

A 50-step correctness run and a 300-step diagnostic first, covering the
Python workflow, adapters, parameter bindings and trace; CAM physics and
dynamics; the online provider, coupler and other components; MPI status
communication and waiting; native allocation, array copies and temporaries;
history, restart and directory switches. Mean-rank and slowest-rank times are
both recorded. Optimisations are ranked by the time they can recover from the
complete advance; overlapping timers are not simply added.

From the historical year, going from about 5510 s to 95% of the original's
about 5000 s means saving about 760 s. That figure only sizes the work; the
budget proper comes from the re-measured A and C.

## 3. Implementation route

### A. Cache the workflow and ABI call preparation

Internal execution caches, with the public UI unchanged.

- Build an execution list when the workflow changes, classifying native,
  Python callback, boundary, clock and I/O actions in advance.
- Turn stable actions from the generic `adapter.call()` into pre-bound calls
  that reuse the entry, pointer table, shapes, dtypes and error buffer.
- StatePool, workflow and kernel registry each keep a change version.
- In-place changes to array values do not rebuild anything; adding or
  removing fields, address changes, workflow edits and reload invalidate the
  caches concerned.
- Caches keep a reference to the array owner and release bindings when a
  field is removed dynamically.
- Ordinary user callbacks keep their field permissions, return-value checks
  and transactional semantics.

Caches optimise preparation only: never a time-dependent result, and never a
skipped user override.

### B. Finish the original process and the segment runner

Three internal modes: no replacement selects native-whole; a replacement
selects segmented; debugging and comparison use legacy-python. Finish the four
boundaries `mmacro_pcond`, `micro_mg_tend`, `rad_rrtmg_sw` and `rad_rrtmg_lw`.

- Cut control at the pinned original Fortran call sites, reusing the existing
  handles and argument descriptions.
- Numerical kernels keep the original implementation; adapters copy no
  arithmetic.
- The runner saves the variables, chunk, substep and position that survive a
  pause.
- Python drives start/frame/resume; there is no Fortran-to-Python callback.
- Pause only before a replaced kernel; the other native operations run
  contiguously.
- Ranks may differ in their local call counts, so no MPI collective is added
  at a pause.
- After a replacement fails, advancing is refused; explicit close and
  release are allowed.
- While a context is active, rebinding, checkpointing and moving processes
  are refused; normal operation resumes afterwards.

The original kernel executed through the replacement boundary must be
bit-for-bit: that is the independent proof that pause and resume are right.

### C. Batch adjacent native actions

Once the stage segmentation is verified, apply the same principle to the
workflow:

- Python builds the list of contiguous native actions from the current
  workflow.
- The native runner executes the list and returns the actions completed,
  their status, and any timing needed.
- Python callbacks, runtime Python conditions, field observation, the clock
  and I/O boundaries that cannot be merged end the current batch.
- No kernel reordering, and no batching across side effects or communication
  dependencies.
- A process a user runs on its own still runs alone.
- After a workflow edit or reload, the next action boundary takes the new
  plan.
- Per-step sampling and pause semantics are unchanged; no batch crosses a
  step the user asked to observe.

This executes a call list Python has already decided; it is not the Python
class translated back into Fortran.

### D. MPI control communication

- The normal path checks the error flag with an `Allreduce` on a
  preallocated integer buffer.
- Strings and tracebacks are gathered only when an error is found.
- Before a duplicated check is removed, show that the operations concerned
  share a communicator and a consistent execution order.
- Every rank confirms the error status before entering the next computation
  that needs collective participation.
- No synchronisation point is dropped casually, and no numerical reduction's
  order or algorithm changes.
- Configuration, payload-hash and workflow consistency checks move to the
  install and edit boundaries; a step checks only what must be verified
  dynamically.

Message counts, serialisation cost and synchronisation waits are measured
separately, to confirm the gain appears in the complete step rather than
moving the wait to another timer.

### E. Fortran-internal memory and repeated work

This is the phase that goes beyond the original; the candidates already seen
in the source come first.

**Microphysics packed workspace.** `micro_mg_cam` allocates and frees many
packed arrays on every call. Replace them with a model- or rank-owned reusable
workspace: allocated once, grown when capacity is short; the original initial
values restored on every call; the actual `mgncol`, shapes, leading dimensions
and valid column ranges kept; nothing depending on what the previous call
left behind; released together when the model closes.

**Physics state and tendency temporaries.** For `physics_state_copy`,
`physics_ptend_init` and their release paths: separate storage preparation
from numerical initialization; reuse allocated storage while keeping the
original per-call initialization semantics; audit whether allocation status
is read elsewhere, so an object that should be invalid does not keep looking
valid; remove only the copies dependency analysis proves redundant; keep
private storage for a `state_loc` that must be updated in isolation.

**Data movement and static lookups.** Cache field indices, metadata and
mapping lookups that are stable over a lifetime; keep the MCT component and
coupler layout differences and skip an intermediate copy only where layout
and ownership are proven identical; a view without an extra copy has an
explicit lifetime, and no Python callback may hold scratch about to be
reused; the online provider's directory switches merge over contiguous call
regions while every component's I/O still lands in the right directory.

Each optimisation is its own source patch, with A, B and C built alongside.
The original source and the oracle output stay read-only.

If these gains are not enough, continue with the measured slowest paths in
the dynamics, radiation or coupler: allocation, copies, indexing and
provably invariant repeated work. Floating-point expressions, summation order
and maths-library calls do not change; a candidate that alters an expression
is validated on its own and withdrawn on failure.

### F. Observation overhead

- The trace uses compact records and batch conversion, keeping the existing
  order, count and retention.
- Plot data follows the variables, statistics and step sampling the user
  registered; UI objects are created at display time.
- Requested per-step sampling, history and restart are never skipped.
- The performance threshold uses the original's output configuration; the
  cost of extra notebook observation is quantified separately.

## 4. Verification and performance acceptance

### Unit and interface tests

Cover pre-binding invalidation, array lifetimes, workflow reordering,
enable/disable, dynamic field addition and removal, callback installation and
reload, running a process alone, per-step sampling, and cleanup after a
segmented failure. In particular: an edit never leaves an old function or an
old array address being called; the native batch and the per-action path
execute in the same order; callback read/write permissions and rollback keep
their semantics; a failure on one rank does not leave the others waiting in
the next collective; the runner releases its resources on success, failure
and close.

### Numerical gates, in order

1. Capture and replay of real kernel inputs, compared array by array, byte
   for byte.
2. Coverage of `ncol < pcols`, different chunk counts, varying `mgncol`,
   several substeps, workspace growth, and restart.
3. A 512-rank 50-step gate for every change that can affect numbers or call
   order.
4. The original kernel executed through each of the four replacement
   boundaries, alone and combined, proving the pause path is taken and stays
   bit-for-bit.
5. A complete online month, then a year.
6. Every CAM history and restart file compared, the other active components'
   output and the coupling boundary checked, and every defined StatePool
   value compared with uninitialised padding explicitly excluded.

Each gate records the actual execution counts, the source and library hashes,
the first difference and the coverage. Model-replacement experiments are
evaluated separately; an AI approximation never stands in for a bit-for-bit
gate.

### Performance tests

- At least five A/C pairs for the month and three for the year.
- Group B measured alongside for the native gain, with C/B reported as the
  framework overhead.
- No picking the fastest run: every result is reported, with the median of
  the paired ratios and a confidence interval.
- Month and year both need a median ratio of 0.95 or less and an upper 95%
  confidence bound below 1.
- If scatter leaves the conclusion unclear, two more pairs are added by the
  predeclared rule; no sample is dropped because it came out badly.
- Only a verifiable cause, such as a scheduler fault, a node fault or an
  inconsistent input configuration, excludes a run, and the record is kept.

The final performance gate uses the original kernels on the default
scientific path with the Python class interface enabled. The speed of a
user's own Python or AI model is that model's own; no replacement is promised
to be faster than the original kernel.

Memory is accepted at the same time: workspaces reach a stable size over
repeated runs and do not grow with the step count, and the peak stays within
105% of the corresponding freeCAM configuration before the optimisation.

## 5. Order of work and deliverables

Continue from the current `native-stage-batching` work, keeping the runner in
development and the uncommitted UI content. Record the workspace and the
existing validation results first, then deliver in phases:

1. Online paired baselines, timing boundaries and the budget.
2. Workflow and ABI caching, and MPI status communication.
3. The four kernel boundaries and the original-kernel bit-for-bit proof.
4. Batched adjacent native actions.
5. Packed workspace, state and tendency storage reuse, and data movement.
6. Complete month and year performance and bit-for-bit acceptance.
7. Default execution policy, documentation and maintenance scripts.

The framework chooses the internal optimisations by default; existing
notebooks need no new performance parameter. The diagnostic paths that
reproduce the pre-optimisation behaviour are kept.

`validation/pi_cam_faster_than_fortran.json` is added, holding the baselines,
build information, every PBS run, the A/B/C times, memory, bit-for-bit
verdicts and whether the goal was met. `validation/performance_overhead.md`
is updated with its history kept.

The task is complete when the month and the year both show at least a 5%
reduction and pass the scientific and UI validation. If not, work continues
on verifiable optimisations ordered by the remaining time; if the candidates
run out, an accurate gain breakdown and its limits are delivered with the
goal marked unmet, without lowering the acceptance line or presenting a
phase result as completion.

## 6. Outcome and status (2026-09-04)

**The goal was not met. freeCAM runs level with the original Fortran, and
the work stops there by the user's decision.**

### Paired measurements (month, online coupling, one allocation, 512 ranks)

| Job | Code | Order | A, original Fortran | C, freeCAM | C/A | Bit-for-bit |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 7322199 | before | AC | 457.59 s | 491.14 s | 1.0733 | yes |
| 7322349 | before | CA | 437.88 s | 476.80 s | 1.0889 | yes |
| 7322467 | after, `ed685d5` | AC | 437.47 s | 440.07 s | **1.0060** | yes |
| 7322553 | after, `f565b67` | CA | 438.30 s | 439.86 s | **1.0036** | yes |

One year (17520 steps, one allocation, A then C):

| Job | Code | Order | A, original Fortran | C, freeCAM | C/A | Bit-for-bit |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 7322841 | after, `958bd60` | AC | 5067.39 s | 5130.82 s | **1.0125** | yes (180 files) |

The pairs are recorded in `validation/pi_cam_faster_than_fortran.json`, each
with its lifecycle times, memory, bit-for-bit verdict, the provider's
collective count and the code commit. Group B was not run: the native
optimisation phase on the Fortran side was not entered. One year pair was
run, after the decision to stop at parity.

### Where the time went

- **CAM itself was not slower.** Before the change, C's CAM actions averaged
  376 s against A's `CPL:ATM_RUN` average of 382 s, and the Python control
  layer cost about 2.6 ms a step (about 1%, from the timing table of the
  native-whole month 7314664). Items 3.A and 3.F could recover 1% at most
  and were not done.
- **The whole deficit was in the boundary path.** Before the change, C spent
  71 s in import and 44 s in export (A: about 48 s and 27 s). The online
  provider ran one pickled allreduce after every coupler action, about 30 a
  step. The original driver makes those calls back to back with each rank
  skipping the components it is not part of, so land (ranks 0-255), ice
  (256-383) and ocean (384-415) overlap; the per-action synchronisation
  lined them up behind one another and added a wait for every rotating
  straggler.
- **Item 3.D was implemented** (commit `3384292`): the step-begin group, the
  ATM iteration with its completion vote, and the closing group each make
  one two-integer `Allreduce`; the driver's boundary collectives reduce an
  integer flag and gather tracebacks only on error; the import carries its
  schedule. Five reductions a step. A protocol status is reported by every
  rank together; a rank-local failure inside a group aborts as
  `shr_sys_abort` would, so no other rank waits in a later collective. The
  512-rank online 50-step gate 7322441 is bit-for-bit.
- **Effect**: the boundary path fell from 115 s to 61 s; the median C/A from
  1.081 to 1.005 (the two pairs after the change: 1.0060 and 1.0036).

### The remaining budget (in-step perf, job 7322501, initialization excluded)

| Share | rank 100 | rank 400 |
| --- | ---: | ---: |
| CAM (`libfreecam_pi_cam.so`) | 67.7% | 66.3% |
| MPI (libmpi + libfabric, including waits) | 9.4% | 12.3% |
| Intel maths library | 7.4% | 7.8% |
| libc (mostly the progress engine's `sched_yield`) | 6.8% | 8.1% |
| Python | 4.2% | 4.1% |
| `memcpy` / `memset` inside CAM | 4.2% / 3.7% | 4.1% / 3.2% |

Page faults run at a median of about 52 per rank and step, and `malloc` and
`free` do not appear in the listing: allocation itself is not a cost, so what
item 3.E could recover is the copies and zero-fills (about 7%) plus the
boundary path's remaining 16 s a month (3.5%). DWARF call chains do not
unwind through the fixed-address image, so the callers of the copies were not
located. Another 5.5% would need most of both, and the work was not taken
further after the decision to stop at parity.

### Deliverables

- The original-kernel bit-for-bit gate of the segment runner (7322256), the
  grouped-collectives gate (7322441), the pair records and the perf record.
- The measured cost of the segmented path: the stage at 65 ms a step and rank
  with the original kernel answered through Python (gate 7322256), against
  39 ms native-whole and 92 ms for the legacy-python walk. The real network
  (`mmacro_pcond_soft_gated.pt`) run through the runner failed the way it did
  through the legacy walk: PIO refused a value at the first history write
  (job 7324422, about 100 steps, 285 GB); the network itself does not stand.
- Diagnostics: `validation/jobs/pi_cam_perf_online_50step.pbs`,
  `tools/perf_rank_wrapper.sh`, `tools/report_pi_cam_perf.py`; the paired job
  accepts `PYCAM_PAIR_DURATION=1year`.
