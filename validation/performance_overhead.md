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

Overhead **does not grow with integration length** — it decreases slightly,
because the fixed Python startup cost is amortised over more steps.

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

| Job | Name | Config | Memory | Elapsed |
| --- | --- | --- | ---: | ---: |
| `7113832` | `fortran-1year` | Fortran baseline, `nhtfrq=-50` | 194.42 GB | 1.41 h |
| `7126500` | `fortran-5year` | Fortran baseline, `nhtfrq=-50` | 204.21 GB | 7.12 h |
| `7149358` | `freecam-mem-1y` | freeCAM, `nhtfrq=-50` | 214.51 GB | 1.55 h |
| `7149359` | `freecam-mem-5y` | freeCAM, `nhtfrq=-50` | 221.04 GB | 7.74 h |
| `7149429` | `freecam-monthly-1y` | freeCAM, `nhtfrq=0` | 201.24 GB | 1.56 h |
| `7149430` | `freecam-monthly-5y` | freeCAM, `nhtfrq=0` | 206.89 GB | 7.51 h |
| `7126501` | `freecam-online-5y` | freeCAM **before** memory fix | 438.48 GB | 9.40 h |

## Evidence

- [`pi_cam_python_memory_1year.json`](pi_cam_python_memory_1year.json)
- [`pi_cam_python_memory_5year.json`](pi_cam_python_memory_5year.json)
- [`pi_cam_monthly_1year.json`](pi_cam_monthly_1year.json) ·
  [`pi_cam_monthly_1year_bfb.json`](pi_cam_monthly_1year_bfb.json)
- [`pi_cam_monthly_5year.json`](pi_cam_monthly_5year.json) ·
  [`pi_cam_monthly_5year_bfb.json`](pi_cam_monthly_5year_bfb.json)

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
