# freeCAM Performance Overhead vs Original Fortran iCESM1.3.1

Measured overhead of the Python control layer against the original Fortran
CESM lifecycle, for the only admitted scientific configuration.

| Setting | Value |
| --- | --- |
| Configuration | `ne16` PI-atm, CAM5 physics, SE dynamics |
| MPI ranks | 512 (4 Derecho `cpu`/`cpudev` nodes) |
| Coupling | online — live CLM-SP, CICE%PRES, DOCN%DOM, RTM + CESM coupler |
| Timestep | 1800 s (17,520 steps per model year) |
| Calendar | NO_LEAP |
| Executable | identical native `.so` (`native_library_sha256` `8fceed56…`) |
| Measured | 2026-08-18 → 2026-08-19 |

Every freeCAM run below is **bit-for-bit identical** to the Fortran reference;
overhead is the cost of control, not of a different answer. See
[Correctness](#correctness) for the evidence.

---

## Summary

| | Time overhead | Memory overhead |
| --- | ---: | ---: |
| **1 model year** | **+10.2%** | **+10.3%** |
| **5 model years** | **+8.7%** | **+8.2%** |

That is the cost of Python owning the workflow.  Owning one *stage* as well —
`tphysbc` stage 7 as a Python class — cost a further **+40.4% of time** and
**+4.4% of memory** over freeCAM when first measured over a model month, and
**+17.4% of time** after the two rounds of control-path work recorded below,
still bit-for-bit; see [the section below](#what-the-python-cloud-macromicrophysics-stage-costs).

Overhead **does not grow with integration length** — it decreases slightly,
because the fixed Python startup cost is amortised over more steps.

---

## What the Python cloud macro/microphysics stage costs

The numbers above are the cost of Python owning the *workflow*.  This
section is the cost of Python owning a *stage*: `tphysbc` stage 7, the
macro/microphysics action, as `CloudMacroMicrophysics` — the substep loop,
both drivers and the aerosol activation walked statement for statement,
with every floating-point number still the oracle's.

Three runs of the same PI-atm month (1,488 steps, 512 ranks, the recorded
one-month boundary), all on 2026-08-27:

| Run | PBS job | What Python owns |
| --- | --- | --- |
| original Fortran | `7256750` | nothing |
| freeCAM | `7256751` | the workflow, the clock, the coupling |
| freeCAM + Python stage | `7256752` | the above, and stage 7's control flow |

### Correctness first

All three are bit-for-bit with the oracle month over all 18 CAM history and
restart files — the Fortran rerun, freeCAM, and freeCAM with the stage in
Python.  The overhead below is the cost of control, not of a different
answer.

### Time

Compared on the integration loop: the Fortran model's `CPL:ATM_RUN` — its
atmosphere inside the coupled loop — against freeCAM's `advance_seconds`.

| Run | 1,488 steps | vs Fortran | vs freeCAM |
| --- | ---: | ---: | ---: |
| original Fortran | 379.43 s | — | — |
| freeCAM | 402.51 s | **+6.1%** (+23.1 s) | — |
| freeCAM + Python stage, 2026-08-27 (`7256752`) | 565.31 s | **+49.0%** (+185.9 s) | **+40.4%** (+162.8 s) |
| … after round 1, 2026-09-03 (`7303107`) | 521.43 s | +37.4% (+141.9 s) | +29.5% (+118.9 s) |
| … after round 2, 2026-09-03 (`7303342`) | 472.41 s | **+24.5%** (+93.0 s) | **+17.4%** (+69.9 s) |
| … after round 4, 2026-09-04 (`7304288`) | 475.53 s | +25.3% (+96.1 s) | +18.1% (+73.0 s) |

The stage itself cost **146 ms per step per rank** in the first run, against
38.9 ms for the Fortran stage it replaces (the `CAM:macro_microphysics`
region of the freeCAM run); throughput fell from 18.23 to 12.98 SYPD.  Two
rounds of work on the control path, each gated bit-for-bit at 512 ranks
before the month was rerun, brought the stage to **100 ms** and then
**89 ms per step** (14.07 and 15.53 SYPD):

- round 1 — the cloud cores called once per chunk instead of once per
  column, and every direct kernel's argument table built once and kept
  (`PointerTableAdapter.bind`) instead of on every call;
- round 2 — the generated YAML tables parsed once with libyaml, a kernel's
  scratch slots resolved once per kernel and field map, a view of Fortran
  storage reused while the image reports the same address and extents, and
  a two-slot profiler region in place of the generator-backed one.

Rounds 3 and 4 (2026-09-04) then let a kernel read its `intent(in)` inputs
where they live instead of copying them into scratch, reused physics-buffer
views, local views and surface columns, and -- after the first attempt showed
why not -- kept the process's own error collective.  The first attempt
(`638b193`, month `7304103`: +33.4%) taught two things worth recording.
Binding a kernel to the address of its input storage rebuilt the argument
tables whenever the storage moved, and the per-chunk state copy moves on
every call; the rebuilds came in bursts on one rank at a time, invisible in
any rank's monthly total and fully visible on the step's critical path,
because the other 511 ranks waited at the next collective.  `BoundCall`
now points one argument at moved storage instead (`bc09299`).  And
deferring the stage's error collective to the boundary export saved
nothing: measured with the same code either way (`7304174`, `7304175`) the
wait simply appeared at the export, larger, so the immediate collective is
back (`951ad3f`).  Round 4 lands at the same figure as round 2 -- the stage's
own work is lower, but the step is set by the slowest rank's stage each
step, and that jitter (85-98 ms per step per rank over the month) is what
the collective after the stage waits for.

Everything outside the stage is unchanged between the runs to within
run-to-run noise (230.4 ms per step in the freeCAM run, 229.4 ms in the
round-4 run).  What remains of the stage's excess — about 50 ms per step — is
the count of Python-level operations per chunk: some 3,000 per step per rank
(entry calls, view constructions, history writes, scratch copies), each a
few microseconds to a few tens of microseconds, none of them arithmetic.  A
production-form cProfile of four ranks (`pi_cam_stage_python_cprofile_50step.pbs`)
found no single hot spot, and a one-node probe found compute-node Python
only 1.1–1.26× slower than the login node, so the remaining cost is the
operation count itself.  Bringing it further down means fewer operations per
chunk — history writes and views handed over as tables rather than one call
each, kernels run on the views without the scratch copy — which is work in
the handles modules, not in the walks.

Where that time goes, from the Gate M-4 profile (per step per rank):

| | Total | Lifted kernels | Handle calls | Copies |
| --- | ---: | ---: | ---: | ---: |
| whole stage | 152.9 ms | | | |
| macrophysics walk | 34.0 ms | 20.2 | 5.2 | 8.6 |
| microphysics walk | 32.7 ms | 16.3 | 14.1 | 2.3 |
| aerosol walk | 8.7 ms | 3.6 | 4.6 | 0.6 |

The rest is the glue, the physics-buffer reads and the hundred-odd history
writes.  Note that the profile is measured with the timer on every call, so
it over-reads the total; the 146 ms above is the timed month run's own
figure, and the table predates both rounds of control-path work.
The cost is per *call*, not per FLOP: about 700 crossings of the Python /
Fortran boundary per chunk per step, each cheap and none of them numerical.

### A trained network in mmacro_pcond's place

The same month was run once more with the Python stage and the soft-gated
surrogate `mmacro_pcond_soft_gated.pt` in the macrophysics core's slot
(`pi_cam_macro_surrogate_1month.pbs`, job `7305340`, 2026-09-04).  It is
not a bit-for-bit run and was never going to be; the questions were what
it costs and whether it stands.  It did not stand: at step 100, writing
the first two-day history record, PIO refused a value as not representable
in the file's type and CAM aborted.  The log carries thousands of
water-tracer consistency warnings from the first steps on (`wtrc q1q2
uqdiff error`, `BIG ERROR, diff value` of order 1e-2, `isotopic stratiform
precipitation mass error`), so the surrogate's condensation tendencies are
inconsistent with the isotopic water budget this configuration carries,
and something derived from it grew past float range within two days.  No
valid history record exists, so drift cannot be measured.  Until the
abort it ran at about 0.59 s per step against 0.32 s for the Python stage
with the Fortran core and 0.27 s for freeCAM: the network's inference on
the compute nodes costs more than the Fortran core it replaces, as the
fifty-step runs had suggested.  PBS high-water memory 296 GB, 74 GB above
the Python-stage month, for a PyTorch runtime on every rank.

### Memory

| Run | PBS high-water | vs Fortran |
| --- | ---: | ---: |
| original Fortran | 182.85 GB | — |
| freeCAM | 215.46 GB | +17.8% |
| freeCAM + Python stage | 224.95 GB | +23.0% |

The stage adds **+4.4% (9.5 GB)** over freeCAM: about 18 MB per rank, which
is the stage's scratch — one array per kernel field — plus the standalone
handles' views.  The driver's own cross-rank sampling agrees: peak total PSS
197.02 GB → 206.27 GB, **+4.70%**, and peak per-rank RSS 797 MB → 862 MB.

These month figures are not comparable with the year figures above: this
comparison replays a recorded boundary (14 GB of files read per rank set)
while the year runs coupled live components, so the freeCAM baseline sits
higher here than the +10.3% measured for the online year.

Evidence:
[`pi_cam_stage_python_1month_overhead.json`](pi_cam_stage_python_1month_overhead.json),
[`pi_cam_1month_stage_fortran_performance.json`](pi_cam_1month_stage_fortran_performance.json),
[`pi_cam_1month_stage_python_performance.json`](pi_cam_1month_stage_python_performance.json),
and the two bfb records beside them.  Reproduce with

```bash
qsub -A <account> validation/jobs/pi_atm_fortran_1month_performance.pbs
qsub -A <account> -v PYCAM_STAGE_FORM=fortran validation/jobs/pi_cam_stage_python_1month_performance.pbs
qsub -A <account> -v PYCAM_STAGE_FORM=python  validation/jobs/pi_cam_stage_python_1month_performance.pbs
tools/report_pi_cam_stage_overhead.py --fortran-report ... --python-report ...
```

---

## Time overhead

Compared on the model integration loop, which is the fair like-for-like
quantity: same step count, same rank count, both aggregated as the maximum
across all 512 ranks.

- freeCAM: `timing.advance_seconds` (`MPI_Wtime`, with barriers)
- Fortran: GPTL `CPL:RUN_LOOP` from `timing/cesm_timing_stats`

| Run | Steps | Fortran | freeCAM | Overhead | Absolute |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 year | 17,520 | 4,999.71 s | 5,510.10 s | **+10.21%** | +510 s (8.5 min) |
| 5 years | 87,600 | 25,556.49 s | 27,781.62 s | **+8.71%** | +2,225 s (37.1 min) |
| 1 year, Python stage (`7305332`, 2026-09-04) | 17,520 | 4,999.71 s | 6,636.20 s | +32.73% vs Fortran, +20.44% vs freeCAM | +1,636 s (27.3 min) |

The Python-stage year is the same online configuration with
`CloudMacroMicrophysics` in stage 7's place (`--cloud-macro-micro-python`,
code `951ad3f`); it is bit-for-bit with the Fortran year over all 180 CAM
history and restart files
([`pi_cam_python_memory_1year_stage_python_bfb.json`](pi_cam_python_memory_1year_stage_python_bfb.json)),
runs at 13.02 SYPD, and its PBS high-water memory is 221.8 GB (+3.4% over
freeCAM, +14.1% over Fortran).  Per step that is 64 ms of stage control on
top of freeCAM's 25 ms of workflow control, consistent with the month.

Per step this is ≈ 25 ms of Python control on a ≈ 292 ms Fortran step.

### Throughput

| Run | Fortran | freeCAM | Loss |
| --- | ---: | ---: | ---: |
| 1 year | 17.28 SYPD | 15.68 SYPD | −9.3% |
| 5 years | 16.90 SYPD | 15.55 SYPD | −8.0% |

### End-to-end wall clock

PBS elapsed time, which additionally includes job launch, Python import, MPI
session startup, and finalisation:

| Run | Fortran | freeCAM | Overhead |
| --- | ---: | ---: | ---: |
| 1 year | 1.41 h (`7113832`) | 1.55 h (`7149358`) | +9.9% |
| 5 years | 7.12 h (`7126500`) | 7.74 h (`7149359`) | +8.7% |

Consistent with the loop-only measurement — startup and shutdown are not a
material contribution.

### Initialisation and finalisation

| Phase | Fortran | freeCAM |
| --- | ---: | ---: |
| Initialise | 20.3 – 27.4 s | 23.1 – 25.2 s |
| Finalise | 0.01 – 0.12 s | 0.40 – 0.45 s |

Initialisation is dominated by filesystem variance and is the same magnitude
in both; against a multi-hour integration neither phase matters.

---

## Memory overhead

Job-aggregate high-water mark reported by PBS (`qhist` *Used Mem*), summed
over the 4 nodes.

| Run | Fortran | freeCAM | Overhead |
| --- | ---: | ---: | ---: |
| 1 year | 194.42 GB (`7113832`) | 214.51 GB (`7149358`) | **+10.33%** (+20.1 GB) |
| 5 years | 204.21 GB (`7126500`) | 221.04 GB (`7149359`) | **+8.24%** (+16.8 GB) |

Per rank that is 33 – 39 MB of Python control state on top of ≈ 380 MB of
Fortran model state.

### Memory does not scale with run length

This is the decisive result. Going from 1 year to 5 years (5× the steps):

| | 1 year | 5 years | Growth |
| --- | ---: | ---: | ---: |
| Fortran | 194.42 GB | 204.21 GB | +9.79 GB |
| freeCAM | 214.51 GB | 221.04 GB | **+6.53 GB** |

freeCAM grows *less* with integration length than the original Fortran model
does. There is no length-proportional Python term left.

### In-run sampling

Cross-rank total PSS, sampled every 2,000 steps by the driver
(`memory.samples` in the evidence JSON):

| Sample | 1 year (`7149358`) | 5 years (`7149359`) |
| --- | ---: | ---: |
| `initialized` | 187.53 GB | 187.53 GB |
| `step_2000` | 193.45 GB | 193.45 GB |
| last periodic sample | 195.11 GB (step 16,000) | 204.08 GB (step 86,000) |
| `finalized` | 196.43 GB | 205.48 GB |
| steady-state drift | +1.66 GB/yr | +2.13 GB/yr |

Residual drift is ≈ 4 MB per rank per model year and correlates with output
volume, not with step count: the `nhtfrq=0` runs, which write 60 monthly files
instead of 877 periodic ones, drift only +1.0 – 1.3 GB/yr. It is I/O buffer
growth in the native layer, not accumulation in the Python control plane.

### Bounded action trace

The Python driver records one action per dispatched operation. It is now a
`deque` capped at 4,096 records per rank, while the lifetime counter stays
exact:

| Run | Lifetime actions per rank | Retained | Resident |
| --- | ---: | ---: | ---: |
| 1 year | 841,012 | 4,096 | ~512 KB |
| 5 years | **4,204,852** | 4,096 | ~512 KB |

The 5-year count is identical on all 512 ranks and matches
87,600 × 48 + 52 exactly, so nothing is lost by the bound. Unbounded, those
4.2 M records would be ≈ 525 MB per rank ≈ **269 GB** across the job.

### Before the fix

For reference, the same 5-year case before the memory work
(job `7126501`, unbounded trace plus a duplicate shadow CAM instance):

| | Memory | Time | vs Fortran |
| --- | ---: | ---: | --- |
| Before (`7126501`) | 438.48 GB | 9.40 h | +114.7% memory, +32.0% time |
| After (`7149359`) | 221.04 GB | 7.74 h | **+8.2% memory, +8.7% time** |

---

## Correctness

Overhead is only meaningful because the answer is unchanged. The `nhtfrq=0`
runs were compared against an independent 20-year CESM production run
(`f.e13.F1850C5.ne16_g16.icesm131_ihesp.PI-atm.001`) with no numerical
tolerance:

| Run | Files compared | Result |
| --- | ---: | --- |
| 1 year (`7149429`) | 12 monthly | **BFB** — 215 numeric variables identical per file, 0 differing |
| 5 years (`7149430`) | 60 monthly | **BFB** — 215 numeric variables identical per file, 0 differing |

Only `date_written` and `time_written` are excluded; they record the wall-clock
instant of the write and are expected to differ.

---

## Job index

| Job | Name | Config | Memory | Elapsed | Ended |
| --- | --- | --- | ---: | ---: | --- |
| `7113832` | `fortran-1year` | Fortran baseline, `nhtfrq=-50` | 194.42 GB | 1.41 h | 2026-08-14 |
| `7126500` | `fortran-5year` | Fortran baseline, `nhtfrq=-50` | 204.21 GB | 7.12 h | 2026-08-16 |
| `7149358` | `freecam-mem-1y` | freeCAM, `nhtfrq=-50` | 214.51 GB | 1.55 h | 2026-08-18 |
| `7149359` | `freecam-mem-5y` | freeCAM, `nhtfrq=-50` | 221.04 GB | 7.74 h | 2026-08-19 |
| `7149429` | `freecam-monthly-1y` | freeCAM, `nhtfrq=0` | 201.24 GB | 1.56 h | 2026-08-18 |
| `7149430` | `freecam-monthly-5y` | freeCAM, `nhtfrq=0` | 206.89 GB | 7.51 h | 2026-08-19 |
| `7126501` | `freecam-online-5y` | freeCAM **before** memory fix | 438.48 GB | 9.40 h | 2026-08-17 |
| `7256750` | `fortran-1month` | Fortran baseline, replay month | 182.85 GB | 0.14 h | 2026-08-27 |
| `7256751` | `freecam-stage-1month` | freeCAM, stage 7 in Fortran | 215.46 GB | 0.12 h | 2026-08-27 |
| `7256752` | `freecam-stage-1month` | freeCAM, stage 7 in Python | 224.95 GB | 0.18 h | 2026-08-27 |

The memory and elapsed columns are PBS accounting values. `qhist` searches
only the current day's log by default, so retrieving them again requires the
period flag with the end dates above:

```bash
qhist -j 7113832,7126500,7126501,7149358,7149359,7149429,7149430 \
      -p 20260813-20260819 -w
qhist -j 7256750,7256751,7256752 -p 20260827-20260827 -w
```

## Evidence

- [`pi_cam_python_memory_1year.json`](pi_cam_python_memory_1year.json)
- [`pi_cam_python_memory_5year.json`](pi_cam_python_memory_5year.json)
- [`pi_cam_monthly_1year.json`](pi_cam_monthly_1year.json) ·
  [`pi_cam_monthly_1year_bfb.json`](pi_cam_monthly_1year_bfb.json)
- [`pi_cam_monthly_5year.json`](pi_cam_monthly_5year.json) ·
  [`pi_cam_monthly_5year_bfb.json`](pi_cam_monthly_5year_bfb.json)
- the Python stage over a month:
  [`pi_cam_stage_python_1month_overhead.json`](pi_cam_stage_python_1month_overhead.json) ·
  [`pi_cam_1month_stage_fortran_performance.json`](pi_cam_1month_stage_fortran_performance.json) ·
  [`pi_cam_1month_stage_python_performance.json`](pi_cam_1month_stage_python_performance.json)

Fortran baseline timing is read from `timing/cesm_timing_stats` in
`/glade/derecho/scratch/$USER/freeCAM/PI-cam/original-fortran-{one,five}-year-reference/run`.

## Reproducing

```bash
qsub validation/jobs/pi_atm_fortran_1year.pbs        # Fortran baseline
qsub validation/jobs/pi_atm_fortran_5year.pbs
qsub validation/jobs/pi_cam_python_memory_1year.pbs  # freeCAM, clean overhead
qsub validation/jobs/pi_cam_python_memory_5year.pbs
qsub validation/jobs/pi_cam_monthly_1year.pbs        # freeCAM, BFB vs production run
qsub validation/jobs/pi_cam_monthly_5year.pbs
```

Caveat: each row is one run of one configuration on a shared machine. Node
placement and filesystem contention move these numbers by a few percent; the
1-year and 5-year measurements agreeing to within 1.5 points is the reason to
trust them, not either one alone.
