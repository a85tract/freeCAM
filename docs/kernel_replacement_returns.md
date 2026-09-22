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
2 µs, not 18.  *Plugin path*: the whole cost of a Numba plugin that writes zeros, bound
at the hook, measured in the shadow gates on image p29
(`validation/pi_cam_pausable_p29-*-shadow_50step.json`) and on earlier
images for the three older hooks; for kernels without a hook yet it is
estimated from the fit below.  The hook's own part of it -- pointer tables
for the inputs, copying the outputs' live columns back -- is about 10 µs
(compute_uwshcu_inv: 0.686 ms in all, 0.010 outside the plugin call); the
rest is the adapter building one array view per argument and the plugin
writing every output value, which any plugin must do.  A TorchScript model
pays about 0.3 ms outside its forward instead (FTorch tensor wrapping and
output copies, compute_uwshcu_inv), and its own input assembly -- the
concatenation and scaling of the raw arguments into a feature matrix --
runs inside the forward and counts as model time.  *Ceiling*: the month cost less the
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

## A first surrogate at the largest slot, measured

A per-column MLP for `compute_uwshcu_inv` was trained on one 50-step frame
capture (512 ranks, 100 calls each; 96 ranks for training, 16 for validation;
1625 features -> 1722 targets, hidden 512, 1.98 M parameters, fifteen epochs;
the 38 tracers the scheme transports included, the water-tracer outputs
zero) and bound at the hook through FTorch on image p29.

| run | job | step loop, s a rank | model, ms a call | state after 50 steps |
| --- | --- | ---: | ---: | --- |
| nothing armed | 7484081 | 16.18 | -- | the oracle's |
| model in shadow (original answers) | 7493415 | 17.79 | 15.6 | bit-for-bit |
| model live | 7493416 | 21.78 | 16.7 | T rms 0.37 K, Q 0.25 g/kg, CLDLIQ 11.5 mg/kg (ref rms 15.6); 214,801 QNEG3 resets |

The path was right and the model was wrong twice over.  A 2 M-parameter
network costs 15.6 ms a call on one core of a shared node -- 63 MFLOP per
16-column call -- against the 6.5 ms of the scheme it replaces, so the run
slowed by a third; and with validation R2 between 0.2 and 0.8 it drove the
cloud water off within two days.  The returns table's "0.4 ms model" is a
72 k-parameter network; at this slot a model must stay under about 1 M
multiply-adds a column (hidden 256 without the tracers, roughly 2 ms a
call) to give anything back, and must be trained on more than one 50-step
capture to be worth running live.  Records:
`validation/pi_cam_pausable_p29-uwshcu-mlp-{shadow,live}_50step.json`.

## Measured over a month: the FTorch floor and the first surrogate

The plugin-path numbers above were fifty-step shadow gates of a compiled
null function, scaled to the month.  The table below is the month itself
(1488 steps, 512 ranks, image p29, the develop queue; one rank's figures):
a TorchScript model that does no arithmetic, bound at each hook and run in
shadow, bit for bit in every run.  This is what the FTorch path charges
before a model computes anything, and it is an order of magnitude above the
compiled null: TorchScript allocates and returns the output tensors inside
its forward and FTorch wraps every argument.  The baseline month with
nothing replaced is 402.7 s a rank (job 7500359).

| kernel | calls a month | kernel, µs a call | kernel, s a month | FTorch floor, µs a call | floor, s a month | return with a free model | shadow month |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `compute_uwshcu_inv` | 2,976 | 6,507.0 | 19.37 | 1,752 | 5.21 | 3.51 % | 7500362 |
| `mmacro_pcond` | 2,976 | 1,985.0 | 5.91 | 618 | 1.84 | 1.01 % | 7500471 |
| `zm_convr` | 2,976 | 1,560.0 | 4.64 | 661 | 1.97 | 0.66 % | 7500262 |
| `compute_eddy_diff` | 2,976 | 1,388.0 | 4.13 | 284 | 0.85 | 0.82 % | 7500473 |
| `micro_mg_tend` | 2,976 | 1,330.0 | 3.96 | 1,217 | 3.62 | 0.08 % | 7500474 |
| `instratus_condensate` | 267,840 | 5.1 | 1.37 | 62 | 16.72 | none | 7500475 |
| `zm_conv_evap` | 2,976 | 36.0 | 0.11 | 318 | 0.95 | none | 7500476 |
| `momtran` | 2,976 | 30.0 | 0.09 | 236 | 0.70 | none | 7500477 |
| `compute_tms` | 2,976 | 18.0 | 0.05 | 132 | 0.39 | none | 7500478 |
| `cldfrc_fice` | 2,976 | 1.9 | 0.01 | 175 | 0.52 | none | 7500479 |
| **the ten hooks** | | | **39.6** | | **32.8** | **6.1 %** | |

The surrogate of `compute_uwshcu_inv` over the same month: in shadow the
model costs 45.8 s a rank (15.4 ms a call) and the step loop is
452.8 s (job 7500360, bit for bit); live, the step loop is
606.9 s (job 7500361): the model's own 49.7 s plus what the rest
of the physics spends on a wrong state (10.7 M QNEG3 resets; after the month
T drifts 5.4 K rms, U 15 m/s, PS 16 hPa, CLDLIQ 25.7 mg/kg against a
reference rms of 13.2).  Records: `validation/pi_cam_pausable_p29-*-1month*_1month.json`.

### The floor, packed

The floor is the output tensors: TorchScript allocates and returns them
inside its forward, FTorch copies them out.  With `packed: true` the
compute_uwshcu_inv model block returns its 26 outputs as one 16 x 582
tensor and the hook zeroes the four tracer outputs itself, so nothing of
the 55 k tracer values crosses the model.  Measured over the same month on
image p30 with a do-nothing packed model in shadow (job 7504452, bit
for bit): **0.37 ms a call**, 1.1 s a rank a month, against 1.75 ms and
5.2 s unpacked -- 4.7 times less.  The budget at this slot becomes
6.1 ms a call, 18.3 s a month.  The other hooks keep their unpacked
blocks until a model is wanted there.

### A surrogate cheaper than the kernel, over a month

The next network at the same slot: no tracers, 485 features to 582 targets,
hidden 256, 340 k parameters, trained on the same fifty-step capture,
returning its outputs packed on image p30.  In shadow over the month (job
7504584, bit for bit) it costs **3.08 ms a call, 9.2 s a rank a month**,
against the kernel's 6.5 ms and 19.4 s: the first replacement at a kernel
slot that costs less than what it replaces, a saving of 10 s a month, 2.5 %
of it, if its answers were the kernel's.  They are not yet: live (job
7504585) the step loop is 514.9 s against 402.7, 3.9 M QNEG3 resets, and
after the month T drifts 5.1 K rms, U 15 m/s, PS 18 hPa.  The same network
trained on twice the data (the fifty-step capture plus one call in 25 across
a month, 296 k columns; job 7504871) is no faster live (514.5 s) and only a
little closer (T 4.8 K rms, PS 12 hPa): more of the same data does not fix
it.  The cost side of the kernel slot is settled; what remains is the model's
skill, and that is architecture and targets, not the interface.  Records:
`validation/pi_cam_pausable_p30-uwshcu-h256-{shadow,live}-1month_1month.json`.

### The wrong answers' collateral cost, removed

The 112 s the live month lost were not spent at the kernel.  The
surrogate's tendencies took cloud water below zero (3.6 M QNEG3 resets,
2.98 M water-tracer consistency warnings, 71 M lines of log), and with the
four tracer outputs zeroed the isotope tracers stopped following the water
they mirror.  One change to the hook block and two to the exported module
remove that cost without touching the network (image p31, commit 1a12df3a;
nothing armed, fifty steps bit for bit, job 7508625):

- the model block hands `tr0_inv` to the model and takes `trten_inv`,
  `wtqc_inv`, `wtprec` and `wtsnow` back inside the packed tensor, width
  4116 instead of 582, so a model can answer them;
- the module floors its water tendencies at `-q0/dt` and keeps snow within
  precipitation;
- it derives the twelve isotope tracer tendencies (four species, vapour,
  liquid and ice) from its own water tendencies at the ratios the column
  carries: the existing ratio for liquid and ice, the vapour ratio for
  vapour, detrained condensate and precipitation.  Given the kernel's own
  water tendencies, that rule reproduces the kernel's liquid and ice tracer
  tendencies on the capture to about 1e-3 relative, and vapour to 2 %
  (H218O) and 18 % (HDO): the fractionation it ignores.

| what answers the kernel | job | ms a call | s a rank a month | step loop, s a rank | over the month |
| --- | --- | ---: | ---: | ---: | --- |
| the original kernel | 7500359 | 6.5 | 19.4 | 402.7 | the oracle's answers |
| do-nothing model, tracer arrays returned too, shadow (p31) | 7508626 | 1.22 | 3.6 | 413.3 | bit for bit |
| 340 k-parameter MLP, floors and derived tracers, shadow (p31) | 7508627 | 6.00 | 17.9 | 429.6 | bit for bit |
| the same, live (p31) | 7508628 | 6.72 | 20.0 | **403.6** | 0 QNEG3 resets, 223 consistency warnings, 38 isotope precipitation errors |
| do-nothing model, tracer arrays at the twelve isotope constituents only, shadow (p32) | 7510376 | 0.48 | 1.4 | 405.9 | bit for bit |
| the MLP with floors and derived tracers, compact, shadow (p32) | 7510378 | 4.00 | 11.9 | 414.9 | bit for bit |
| the same, live (p32) | 7510379 | 4.17 | 12.4 | **396.0** | the p31 answers to the bit; 1.7 % under the baseline |
| the same network live before the change (p30) | 7504871 | 3.36 | 10.0 | 514.5 | 3.6 M QNEG3 resets, 2.98 M warnings, 121 k errors |

Live, the month takes 403.6 s against the baseline's 402.7: the collateral
cost is gone, and the model now costs what the kernel costs, 6.72 against
6.5 ms a call.  The 3.6 ms a call it gained over the p30 model is the price
of the wide tracer arrays, 3 534 more values a column to build, pack and
write back on 128 ranks a node that share its memory bandwidth: the
do-nothing model's floor rose from 0.37 to 1.22 ms.  The answers did not
change: T 5.2 K rms after the month (4.8 before), PS 17 hPa (12), history
relative rms median 0.59 (0.58).

Returning only the twelve isotope constituents removes that traffic (image
p32, commit 90d5e8c9).  A packed model block may now name, per output, the
1-based indices of the last axis the model returns; they sit in that order
inside the packed tensor and the hook zeroes the rest of the array before
scattering them.  The tracer arrays cross at 608 values a column instead of
3 534; nothing armed, fifty steps bit for bit (job 7510375).  Over the month
the do-nothing floor is back to 0.48 ms a call, the model costs 4.00 ms in
shadow and 4.17 ms live, and the live month takes **396.0 s against 402.7**,
1.7 % under the baseline, with the p31 answers to the bit (the two restart
files are identical).  At this slot the interface is settled: a 4.2 ms model
at a 6.5 ms kernel, and the run shows the difference.  What is left is the
network's skill.  Records: `validation/pi_cam_pausable_p3{1,2}-*_1month*.json`.

### A Python function at the same hook

A function put in the slot of a hooked kernel (`stage.kernels[name] = fn`,
written over the kernel's arrays) is compiled with Numba on every rank and
Fortran calls the compiled code at the hook; the interpreter never runs inside
a Fortran call.  A packed block takes compiled code as it takes a TorchScript
model: the adapter fills the one packed output, subsets included; a compiled
do-nothing kernel over the 50 arrays of compute_uwshcu_inv answered 51 200
calls in shadow over fifty steps, bit for bit, at 0.37 ms a call (7512361, on
the develop queue's shared nodes).  The other way, a C callback from the hook into the
interpreter, was built and measured before being withdrawn as a design: with
the same TorchScript surrogate run from Python instead of through FTorch on
image p33, the restart file was identical to the FTorch month's (7511232,
396.7 s) at 8.94 ms a call against FTorch's 4.18 and a step loop of 424.1 s
(7511231; shadow gate 7511790, bit for bit, 51 200 calls), so the detour cost
4.8 ms a call and 3.5 % of the run, and the interpreter with torch loaded on
every rank needed 200 GB a node (7511230, killed at 64 GB).  The interface
keeps the rule instead: Fortran calls compiled code or a model, never Python.
Records: `validation/pi_cam_pausable_p33-*.json`.

### The surrogate on a GPU, offline

Would the model be faster on a GPU?  Measured offline on one A100 against one
CPU thread of the same node, with the same 340 k-parameter packed model, real
captured columns, and the arrays as the hook hands them, float64 in and out
(job 7563406, `validation/pi_cam_uwshcu_surrogate_gpu_bench.json`; it
supersedes the first record, whose batches above 1 600 columns were capped by
the captured calls it stacked):

| columns a call | CPU, one thread | A100, arrays copied in and the output back | A100, device resident |
| ---: | ---: | ---: | ---: |
| 16 (one chunk, the hook's call) | 0.61 ms | 1.01 ms | 0.60 ms |
| 256 | 4.36 ms | 1.64 ms | 0.71 ms |
| 1 024 | -- | 2.99 ms | 0.72 ms |
| 4 096 | 170.5 ms | 14.4 ms | 0.88 ms |
| 16 384 | -- | 47.4 ms | 3.3 ms |

At the hook's granularity the GPU is slower than a CPU thread: a chunk's
forward is a few dozen kernel launches, and launching them costs more than
computing 16 columns.  A batched call pays from a few hundred columns (6.4 µs
a column at 256, about 3 µs from 1 024 upwards, against 17 µs on the CPU
thread at 256 and 42 µs at 4 096, where the strided Fortran arrays stop
fitting its caches), and what it pays for is the traffic of the twenty input
arrays to the device, not the arithmetic: resident, 16 384 columns take
3.3 ms.  No hook has hundreds of columns a call; it takes the node's ranks
batching their chunks into one call and waiting for one another, which the
batching plugin below measures inside the model.  The in-image CPU figure
itself (4.17 ms a call) is seven times this CPU thread's 0.61 ms because 128
ranks a node share the memory bandwidth; a GPU node's 64 cores would change
that number before any GPU did.

### The surrogate on a GPU, inside the image

The GPU path was then built (a CUDA libtorch under FTorch, the model bound
on each rank's GPU, image p35) and measured on the same case at the
validated layout moved to GPU nodes: four A100 nodes, 128 ranks a node, 32
ranks a GPU, one CUDA context a rank.  Fifty steps, the compact-subset
`compute_uwshcu_inv` model, 51 200 modelled calls; the CPU-path numbers are
the same image family on the same nodes (records
`pi_cam_pausable_g33-*-4x128`, `g35-*-gpu-4x128-repo3`):

| 50 steps, four A100 nodes x 128 ranks | nothing armed | model on the host, shadow | model on the GPU, shadow | model on the host, live | model on the GPU, live |
| --- | ---: | ---: | ---: | ---: | ---: |
| step loop, rank 0 | 17.2 s | 17.8 s | 87.0 s | 17.6 s | 86.6 s |
| model, a call (sum over ranks / calls) | -- | 4.08 ms | 216 ms | 4.31 ms | 220 ms |
| of which the forward | -- | 3.93 ms | 126 ms | 3.83 ms | 127 ms |
| first call, a rank (context, model load, fuser compile) | -- | 22 ms | 12.0 s | 24 ms | 12.0 s |
| a call after the first | -- | 3.86 ms | 96.6 ms | 4.07 ms | 100 ms |
| bit-for-bit / health | yes | yes | yes | 60 big-error lines | 60 big-error lines |

The GPU answers the same model 25 times slower than the host thread that
would otherwise be idle, and the step loop takes five times longer: 32
ranks share one device, each with its own context, so their calls are
time-sliced and a rank waits for the 31 others' launches before its own
few dozen run; the first call costs each rank twelve seconds of context
creation, model load and fuser compilation on top.  No month was run on the
GPU: at this rate a month would take about 45 minutes of four GPU nodes to
confirm a result the 50-step gate already settles.  The measurement cost
about five GPU-node-hours (twenty GPU-hours), most of it in the runs that failed first
(`validation/pi_cam_pausable_g34-uwshcu-sub-gpu-4x128_50step_failure.json`
records them: the launcher spreading the head node's device list, MPS
limits and footprints, CUDA memory).  The conclusion of the offline bench
stands: at this layout a GPU only pays for a model if the node's ranks
batch their chunks into one call -- a different design, whose upper bound
is the difference between the in-image host call and a shared batched call.

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
