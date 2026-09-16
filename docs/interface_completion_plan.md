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
no Python in the step.  Four kernels have one; the runner's pause covers the
rest at three crossings a call.

| batch | kernels | what each needs |
| --- | --- | --- |
| A, plain arrays | `rad_rrtmg_sw`, `rad_rrtmg_lw`, `compute_uwshcu_inv`, `zm_convr`, `zm_conv_evap`, `momtran`, `convtran`, `compute_eddy_diff`, `compute_tms`, `wetdepa_v2`, `mmacro_pcond`, `dadadj` | a function contract (reviewed: uwshcu, mmacro_pcond, dadadj; drafted: the rest), a `binding: fortran` hook entry naming the caller objects, one image |
| B, awkward dummies | `gw_drag_prof` (a derived type, `GWBand`), `compute_vdiff` (procedure dummies), `gas_phase_chemdr` (`state`, `pbuf`), `modal_aero_depvel_part` (same object as its caller; not in the inventory) | the generator taught to declare derived-type and procedure dummies through the callee's own modules; `weaken-definition` for the last |

Gates per batch: the image with every hook unarmed, nothing replaced, fifty
steps bit-for-bit; the everything form (every kernel answered through its
pause) still bit-for-bit; per hook, a null plugin bound in shadow (the
original answers, the plugin's cost is timed) bit-for-bit.  `virtem` is a
function, not a subroutine, and costs 13 µs a call; it keeps its pause.

## 2. History for the coarse slots

A block model or a replay of `macro_microphysics` or of the radiation
branch does not produce the drivers' `outfld` diagnostics (141 fields for
the cloud drivers), so `h0` differs while the state is exact.  One entry per
driver, generated from the driver's own history statements, writes them
from the buffer as it stands after the block; then a replayed block is
bit-for-bit in every file.

## 3. Verification and documentation as tools

`tools/report_pi_cam_drift.py` (the drift comparison now in a scratch
script), a generated `docs/contracts.md` from the function contracts and
block contracts (names, shapes, units, planes), and the kernel-slot notebook
beside `examples/replace_process.ipynb`.

## 4. Coarse contracts for the other ten numerical processes

The Python-driver form per process, drawn by the census (the original run
once, every field it changes recorded), proved by capture and replay
bit-for-bit -- in the order of what the process costs: shallow convection,
deep convection, chemistry, vertical diffusion, wet and dry deposition,
gravity-wave drag, dry adjustment, the energy fixer, convective transport.

## What it costs, what it is worth

The all-paused month (7479754) prices the kernels: the seventeen sum to 17
percent of the step; the walk that reaches them from Python costs 6 percent
of the loop, the pause 1 to 2, a hook nothing.  Batch A is two days and an
image; batch B a day more; the history entry a day; the tools two; the ten
contracts a day each.
