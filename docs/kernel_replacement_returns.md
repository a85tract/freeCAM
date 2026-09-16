# What replacing each kernel can return

The question this page answers: if a kernel is replaced by a model at its
hook, how much of the month can the replacement give back, and what does the
replacement path itself take.  Every number is per rank, for the admitted
configuration (512 ranks, ne16, 1488 steps a month).  The month of the
original is 402 s of step loop a rank
(`validation/pi_cam_1month_stage_fortran_performance.json`).

## The short table

Month cost of the kernel, month cost of the hook path, and what is left: the expected return of a free model, as seconds and as a share of the 402 s month.

| kernel | month cost, s | month path cost, s | expected return, s | return, % of the month |
| --- | ---: | ---: | ---: | ---: |
| `compute_uwshcu_inv` | 19.36 | 2.04 | 17.32 | 4.30% |
| `rad_rrtmg_sw` | 9.70 | 1.14 | 8.56 | 2.13% |
| `gas_phase_chemdr` | 9.16 | 0.68 | 8.48 | 2.11% |
| `rad_rrtmg_lw` | 8.18 | 0.76 | 7.42 | 1.84% |
| `mmacro_pcond` | 5.91 | 0.26 | 5.65 | 1.40% |
| `zm_convr` | 4.64 | 0.45 | 4.19 | 1.04% |
| `compute_eddy_diff` | 4.13 | 0.06 | 4.07 | 1.01% |
| `micro_mg_tend` | 3.96 | 1.67 | 2.29 | 0.57% |
| `wetdepa_v2` | 2.95 | 3.04 | 0.00 | 0.00% |
| `modal_aero_depvel_part` | 1.50 | 0.60 | 0.90 | 0.22% |
| `compute_vdiff` | 1.13 | 0.38 | 0.75 | 0.19% |
| `gw_drag_prof` | 0.83 | 1.23 | 0.00 | 0.00% |
| `convtran` | 0.45 | 0.12 | 0.32 | 0.08% |
| `zm_conv_evap` | 0.11 | 0.12 | 0.00 | 0.00% |
| `momtran` | 0.09 | 0.08 | 0.01 | 0.00% |
| `compute_tms` | 0.05 | 0.01 | 0.04 | 0.01% |
| `dadadj` | 0.04 | 0.06 | 0.00 | 0.00% |
| `virtem` | 0.04 | 0.01 | 0.03 | 0.01% |
| `cldfrc_fice` | 0.01 | 0.02 | 0.00 | 0.00% |
| `instratus_condensate` | 1.37 | 0.11 | 1.26 | 0.31% |
| **all** | **73.6** | **12.8** | **61.3** | **15.2%** |

## The full table

| # | kernel | process | calls a month | original, µs a call | plugin path, µs a call | month cost, s | ceiling, s | ceiling, % of month | with a 0.4 ms model, s | path source |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `compute_uwshcu_inv` | shallow_convection | 2,976 | 6,507 | 686 | 19.36 | 17.3 | 4.3% | 16.1 | measured (7484397) |
| 2 | `rad_rrtmg_sw` | radiation | 1,488 | 6,520 | 767 | 9.70 | 8.6 | 2.1% | 8.0 | estimated, 117k elements |
| 3 | `gas_phase_chemdr` | chemistry_tendencies_leaf | 2,976 | 3,077 | 229 | 9.16 | 8.5 | 2.1% | 7.3 | estimated, ~35k elements (state and pbuf not counted) |
| 4 | `rad_rrtmg_lw` | radiation | 1,488 | 5,498 | 510 | 8.18 | 7.4 | 1.8% | 6.8 | estimated, 78k elements |
| 5 | `mmacro_pcond` | macro_microphysics | 2,976 | 1,985 | 88 | 5.91 | 5.6 | 1.4% | 4.5 | measured (7484216) |
| 6 | `zm_convr` | deep_convection | 2,976 | 1,560 | 152 | 4.64 | 4.2 | 1.0% | 3.0 | measured (7484401) |
| 7 | `compute_eddy_diff` | vertical_diffusion | 2,976 | 1,388 | 21 | 4.13 | 4.1 | 1.0% | 2.9 | measured (7484402) |
| 8 | `micro_mg_tend` | macro_microphysics | 2,976 | 1,330 | 562 | 3.96 | 2.3 | 0.6% | 1.1 | measured (micro-null-shadow-p19) |
| 9 | `wetdepa_v2` | aerosol_wet_deposition_leaf | 89,280 | 33 | 34 | 2.95 | 0.0 | 0.0% | none | estimated, ~5k elements |
| 10 | `modal_aero_depvel_part` | aerosol_dry_deposition_leaf | 23,808 | 63 | 25 | 1.50 | 0.9 | 0.2% | none | estimated, ~3k elements |
| 11 | `compute_vdiff` | vertical_diffusion | 5,952 | 190 | 64 | 1.13 | 0.7 | 0.2% | none | estimated, 9k elements |
| 12 | `gw_drag_prof` | gravity_wave_drag | 2,976 | 278 | 412 | 0.83 | 0.0 | 0.0% | none | estimated, 63k elements |
| 13 | `convtran` | convective_tracer_transport_leaf | 2,976 | 150 | 41 | 0.45 | 0.3 | 0.1% | none | estimated, 6k elements |
| 14 | `zm_conv_evap` | deep_convection | 2,976 | 36 | 40 | 0.11 | 0.0 | 0.0% | none | measured (7484399) |
| 15 | `momtran` | deep_convection | 2,976 | 30 | 26 | 0.09 | 0.0 | 0.0% | none | measured (7484400) |
| 16 | `compute_tms` | vertical_diffusion | 2,976 | 18 | 3 | 0.05 | 0.0 | 0.0% | none | measured (7484398); the hook's own timer puts the original at 2 us |
| 17 | `dadadj` | dry_adjustment | 2,976 | 13 | 19 | 0.04 | 0.0 | 0.0% | none | estimated; the original is measured through its pause |
| 18 | `virtem` | vertical_diffusion | 2,976 | 13 | 3 | 0.04 | 0.0 | 0.0% | none | estimated; a function of three scalars, measured through its pause |
| 19 | `cldfrc_fice` | deep_convection | 2,980 | 1.9 | 6.1 | 0.01 | 0.0 | 0.0% | none | measured (7402506); the original from the hook's timer |
| 20 | `instratus_condensate` | macro_microphysics (inside mmacro_pcond) | 268,200 | 5 | 0.4 | 1.37 | 1.3 | 0.3% | none | measured (7402507); the original from the hook's timer |
| | **all** | | | | | **73.6** | **61.3** | **15.2%** | **49.6** | |

**Columns.**  *Calls a month*: what one rank makes, two chunks a step for the
per-chunk kernels, one a radiative step for the two solvers, more for the
kernels called inside loops.  *Original*: the kernel's own cost a call, from
the month in which every kernel was answered through its pause
(`validation/pi_cam_pausable_p28-everything-timers-1month_1month.json`, job
7479754, rank 0); for kernels under 50 µs that number includes the pause's
own copies, so it is an upper bound -- the hook's timer puts `compute_tms` at
2 µs, not 18.  *Plugin path*: the cost of the hook path with a plugin that
writes zeros -- argument tables, adapter, call, write-back -- measured in
the shadow gates on image p29 (`validation/pi_cam_pausable_p29-*-shadow_50step.json`)
and on earlier images for the three older hooks; for kernels without a hook
yet it is estimated from the fit below.  *Ceiling*: the month cost less the
path, the gain if the model itself were free.  *With a 0.4 ms model*: the
same with a model that costs what a small MLP costs through FTorch inside
the image (0.41 ms a call, measured on `micro_mg_tend`).

## The path cost follows the argument volume

Ten hooks measured with a null plugin, against the number of array elements
in their model blocks:

| hook | elements | measured, µs | fit, µs |
| --- | ---: | ---: | ---: |
| `instratus_condensate` | 417 | 0.4 | 6 |
| `cldfrc_fice` | 1,440 | 6.1 | 12 |
| `compute_tms` | 2,961 | 3.2 | 22 |
| `zm_conv_evap` | 8,739 | 40 | 60 |
| `momtran` | 9,142 | 26 | 62 |
| `compute_eddy_diff` | 13,956 | 21 | 94 |
| `zm_convr` | 21,860 | 152 | 145 |
| `mmacro_pcond` | 25,953 | 88 | 172 |
| `micro_mg_tend` | 56,705 | 562 | 372 |
| `compute_uwshcu_inv` | 100,977 | 686 | 659 |

Fit: path ≈ 3 µs + 0.0065 µs per element, about one nanosecond an
element, the speed of copying the arguments in and the answers out.  It is
right within a factor of two, which is enough for a ranking; the estimated
rows carry that uncertainty.  The path depends on how many numbers the model
block moves, not on what the kernel computes, so a small kernel with a large
argument list (`gw_drag_prof`, 278 µs of work behind 63k elements) is a poor
slot even before the model's own cost.

## What it says

- **Four kernels carry the return**: `compute_uwshcu_inv` (4.3% of the
  month if the model is free, 3.9% with a 0.4 ms one), `rad_rrtmg_sw` and
  `rad_rrtmg_lw` together (about 4%), `gas_phase_chemdr` (2.1%).  The next
  four (`mmacro_pcond`, `zm_convr`, `compute_eddy_diff`, `micro_mg_tend`) are
  about 1% each; with a 0.4 ms model, half of that.
- **Everything below 300 µs a call cannot pay for a model at all.** For
  `wetdepa_v2` and `modal_aero_depvel_part` the calls are too many and too
  small; for `compute_vdiff`, `gw_drag_prof`, `convtran` the path is a large
  part of the work; for the seven under 50 µs the path is the work.  These
  kernels are replaced by replacing the process they sit in.
- **The whole table is 74 s of the 402 s month**, 18% of it (68 s for the seventeen kernels of the timing run, 6 s for the three counted at their hooks).  A
  process-level replacement reaches more: `macro_microphysics` alone is
  about 12% of the step, `radiation` about 10%.

## Sources and caveats

The month costs are one run on exclusive nodes; the plugin paths are
50-step runs on the develop queue's shared half nodes, where the same
original ran 10 to 40% slower than in the month (`compute_uwshcu_inv` 9.0
against 6.5 ms), so a path measured there is if anything an overestimate.
The estimated paths take the element counts from the kernels' drafted
contracts (`native/pi_cam/functions/drafts/`) and, for the three kernels
not in the inventory, from a count of their real dummies; `gas_phase_chemdr`
also takes `state` and `pbuf`, whose fields are not in its count.  A hook for
the kernels marked estimated is batch B of `interface_completion_plan.md`;
until then their path is the runner's pause, which costs about 0.7 ms a
pause plus a per-step cost per stage that is being removed.
