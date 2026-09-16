# Plan: every process a slot, at both granularities

Decided 2026-09-15 after the pausable and Python-driver forms were measured:
freeCAM's product is the replacement interface -- every process reachable
as a slot, coarse (the whole process or its compute blocks) and fine (the
kernels inside it), with the original answering bit-for-bit when nothing is
replaced, the overhead of a replacement bounded and measured, and the tools
to capture, replay, verify and time what a slot is given.  Model quality is
not promised.  What that positioning still lacks, in the order it is done:

## 1. Zero-overhead hooks for the thirteen kernels that have only a pause

A hook redirects a kernel's call sites to a wrapper of the callee's own
signature (`native/pi_cam/hooks.yaml`, `tools/generate_pi_cam_hooks.py`;
`rename-references` when the callee lives in another object,
`weaken-definition` when it lives in the caller's).  Unarmed, the wrapper
counts and calls the original; bound, it hands a TorchScript model (FTorch)
or a compiled plugin (a C-ABI function pointer) the arguments in place, with
no Python in the step.  Eleven kernels have one (the seven of batch A await
the gates of the image that first carries them); the runner's pause covers
the rest at three crossings a call.

| batch | kernels | what each needs |
| --- | --- | --- |
| A, plain arrays (written) | `compute_tms`, `compute_uwshcu_inv`, `mmacro_pcond`, `zm_conv_evap`, `momtran`, `zm_convr`, `compute_eddy_diff` | a function contract (`tools/draft_pi_cam_hook_contract.py` turns the drafter's output into one: structural extents with the configuration's values, logical dummies as `int32` carriers, pointer dummies as workspace handed on, intent-less dummies both ways), a `binding: fortran` hook entry naming the caller objects, one image |
| A, remaining | `convtran` (an extent read from a module array, `wtrc_ntype(iwtice)`), `wetdepa_v2` (not in the inventory), `dadadj` (its caller is the patched physpkg; 13 µs a call) | the first two need the contract loader to accept an extent that is a module variable, and the inventory to list the wet deposition kernel |
| B, awkward dummies | `rad_rrtmg_sw`, `rad_rrtmg_lw` (a derived type, `rrtmg_state_t`; the shortwave also has five optional dummies and a `0:pver` lower bound), `gw_drag_prof` (a derived type, `GWBand`), `compute_vdiff` (procedure dummies), `gas_phase_chemdr` (`state`, `pbuf`), `modal_aero_depvel_part` (same object as its caller; not in the inventory) | the generator taught to declare derived-type, optional and procedure dummies through the callee's own modules; `weaken-definition` for the last |

Gates per batch: the image with every hook unarmed, nothing replaced, fifty
steps bit-for-bit; the everything form (every kernel answered through its
pause) still bit-for-bit; per hook, a null plugin bound in shadow (the
original answers, the plugin's cost is timed) bit-for-bit.  `virtem` is a
function, not a subroutine, and costs 13 µs a call; it keeps its pause.

Batch A on image p29 (2026-09-16): nothing armed, bit-for-bit, step loop
16.18 s a rank (`validation/pi_cam_pausable_p29-nothing-armed_50step.json`;
the hooks cost nothing); everything through the pauses, bit-for-bit, 43.79 s,
kernel costs as on p28 (`..._p29-everything-timers_50step.json`).  The shadow
gates taught two things first: 512 ranks compiling a 57-argument plugin at
once need about 30 GB a node more than the run, so the shadow jobs ask 200 GB
a node (two runs killed at the 256 GB cap, failure records kept); and a
shadow gate that is bit-for-bit can still test nothing -- the command line
attached models to a pausable stage after installing its process, whose
pickled copy kept empty slots, so the stage ran the original whole
(`..._compute_uwshcu_inv-shadow_50step_failure.json`; fixed, with a test).
Every shadow gate then passed bit-for-bit with the plugin running on every
call of every chunk from the first step the class is attached (the stage in
`native-model` mode, 51200 plugin calls over 512 ranks).  The plugin path's
own cost -- the hook's argument tables, the Numba adapter, the call, the
write-back, with a plugin that writes zeros -- summed over the ranks and
divided by the calls:

| hook | job | plugin calls | plugin path, µs a call | original, µs a call | overhead |
| --- | --- | --- | --- | --- | --- |
| `compute_tms` | 7484398 | 51200 | 3 | 17 | 19% |
| `compute_uwshcu_inv` | 7484397 | 51200 | 686 | 6578 | 10% |
| `mmacro_pcond` | 7484216 | 51200 | 88 | 1975 | 4% |
| `zm_conv_evap` | 7484399 | 51200 | 40 | 37 | 108% |
| `momtran` | 7484400 | 51200 | 26 | 30 | 86% |
| `zm_convr` | 7484401 | 51200 | 152 | 1545 | 10% |
| `compute_eddy_diff` | 7484402 | 51200 | 21 | 1383 | 1% |

The path costs by the size of the argument list, not the kernel: a few
microseconds for compute_tms's fourteen arguments, 0.7 ms for the shallow
scheme's fifty (twenty-nine of them two- or three-dimensional).  Against the
expensive kernels that is ten per cent or less; for the cheap ones the path
costs as much as the kernel, which is the argument for replacing the process
rather than a 30 µs kernel inside it.

## 2. History for the coarse slots

A block model or a replay of `macro_microphysics` or of the radiation
branch does not produce the drivers' `outfld` diagnostics (141 fields for
the cloud drivers), so `h0` differs while the state is exact.  One entry per
driver, generated from the driver's own history statements, writes them
from the buffer as it stands after the block; then a replayed block is
bit-for-bit in every file.

## 3. Verification and documentation as tools

`tools/report_pi_cam_drift.py` (written: rms, largest and mean difference
per field of one CAM file of each run, in scaled units, as a table or a JSON
record naming no directory), the generated `docs/contracts.md`
(`tools/export_contracts_doc.py`, checked by a unit test: every function
contract with the hook it serves, the two cloud blocks and the radiation
branch), and the kernel-slot notebook `examples/replace_kernel.ipynb` (the
three ways a kernel is answered, on a live run).

## 4. Coarse contracts for the other ten numerical processes

The Python-driver form per process, drawn by the census (the original run
once, every field it changes recorded), proved by capture and replay
bit-for-bit -- in the order of what the process costs: shallow convection,
deep convection, chemistry, vertical diffusion, wet and dry deposition,
gravity-wave drag, dry adjustment, the energy fixer, convective transport.

## What it costs, what it is worth

What each kernel's replacement can return over a month, against the cost of
its hook path, is tabulated in `docs/kernel_replacement_returns.md`.

The all-paused month (7479754) prices the kernels: the seventeen sum to 17
percent of the step; the walk that reaches them from Python costs 6 percent
of the loop, the pause 1 to 2, a hook nothing.  Batch A is two days and an
image; batch B a day more; the history entry a day; the tools two; the ten
contracts a day each.
