"""Low-overhead hierarchical timing for the Python-owned CAM runtime.

The hierarchical reports intentionally resemble the two timing products
written by CESM's GPTL integration, while keeping a distinct
``freecam_timing`` prefix.  Finalization additionally renders a CIME-format
``cesm_timing.<case>.<lid>`` performance profile (Model Cost, throughput, and
Init/Run/Final times) from the same gathered totals, so the run leaves the
human-readable summary CESM users expect.  Timing is rank-local during
execution; MPI communication occurs only once, when the runtime is finalized
and the global summary is written.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


@dataclass(slots=True)
class _TimerRecord:
    calls: int = 0
    walltotal: float = 0.0
    wallmax: float = 0.0
    wallmin: float = float("inf")

    def add(self, elapsed: float) -> None:
        self.calls += 1
        self.walltotal += elapsed
        self.wallmax = max(self.wallmax, elapsed)
        self.wallmin = min(self.wallmin, elapsed)

    def payload(self) -> dict[str, int | float]:
        return {
            "calls": self.calls,
            "walltotal": self.walltotal,
            "wallmax": self.wallmax,
            "wallmin": 0.0 if self.calls == 0 else self.wallmin,
        }


@dataclass(frozen=True, slots=True)
class _ActiveTimer:
    name: str
    path: tuple[str, ...]
    started: float


class _Region:
    """``with profiler.region(name)``: one timer, entered and left by hand.

    A plain object with two slots rather than a generator-backed context
    manager: the runtime opens a few hundred regions per step per rank, and
    the generator machinery cost more than the timing it wrapped.
    """

    __slots__ = ("_profiler", "_name")

    def __init__(self, profiler: "FreeCAMProfiler", name: str) -> None:
        self._profiler = profiler
        self._name = name

    def __enter__(self) -> None:
        self._profiler.start(self._name)

    def __exit__(self, *exc_info: object) -> None:
        self._profiler.stop(self._name)


class FreeCAMProfiler:
    """Collect hierarchical wall-clock timings without action-time barriers."""

    def __init__(
        self,
        *,
        rank: int,
        size: int,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.rank = int(rank)
        self.size = int(size)
        self._clock = clock or _default_clock()
        self._records: dict[tuple[str, ...], _TimerRecord] = {}
        self._stack: list[_ActiveTimer] = []
        self._total_active = False
        self._written = False

    @property
    def active_path(self) -> tuple[str, ...]:
        return tuple(timer.name for timer in self._stack)

    @property
    def records(self) -> Mapping[tuple[str, ...], Mapping[str, int | float]]:
        return {
            path: record.payload() for path, record in self._records.items()
        }

    @property
    def region_count(self) -> int:
        return len(self._records)

    def start_total(self) -> None:
        if self._total_active:
            return
        if self._stack:
            raise RuntimeError("cannot start FREECAM:TOTAL inside another timer")
        self.start("FREECAM:TOTAL")
        self._total_active = True

    def stop_total(self) -> None:
        if not self._total_active:
            return
        if not self._stack or self._stack[-1].name != "FREECAM:TOTAL":
            raise RuntimeError(
                "cannot stop FREECAM:TOTAL while a child timer is active"
            )
        self.stop("FREECAM:TOTAL")
        self._total_active = False

    def start(self, name: str) -> None:
        normalized = name if type(name) is str and name.strip() == name else str(name).strip()
        if not normalized:
            raise ValueError("timer name cannot be empty")
        stack = self._stack
        path = (*stack[-1].path, normalized) if stack else (normalized,)
        # Create the record on entry, rather than on exit, so dictionary order
        # is a true pre-order traversal: parent first, followed by its children.
        if path not in self._records:
            self._records[path] = _TimerRecord()
        stack.append(_ActiveTimer(normalized, path, self._clock()))

    def stop(self, name: str) -> float:
        normalized = name if type(name) is str and name.strip() == name else str(name).strip()
        if not self._stack:
            raise RuntimeError(f"timer {normalized!r} is not active")
        active = self._stack[-1]
        if active.name != normalized:
            raise RuntimeError(
                f"timer stack mismatch: stopping {normalized!r}, "
                f"but {active.name!r} is active"
            )
        elapsed = max(0.0, self._clock() - active.started)
        self._stack.pop()
        self._records[active.path].add(elapsed)
        return elapsed

    def region(self, name: str) -> _Region:
        return _Region(self, name)

    def snapshot(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "size": self.size,
            "timers": {
                "/".join(path): record.payload()
                for path, record in self._records.items()
            },
        }

    def write(
        self,
        run_dir: str | Path,
        communicator: Any,
        *,
        profile: "CESMTimingContext | None" = None,
    ) -> tuple[Path, Path] | None:
        """Write rank-0 detail and all-rank summary reports once.

        When ``profile`` is supplied, rank 0 additionally writes a CIME-format
        ``cesm_timing.<case>.<lid>`` performance profile derived from the same
        gathered per-rank totals, so the run leaves the human-readable timing
        summary CESM users expect alongside the hierarchical reports.
        """

        if self._written:
            return None
        if self._stack:
            raise RuntimeError(
                "cannot write timing report while timers are active: "
                + " -> ".join(self.active_path)
            )
        local = self.snapshot()
        gather = getattr(communicator, "gather", None)
        if callable(gather):
            gathered = gather(local, root=0)
        else:
            gathered = communicator.allgather(local)
        self._written = True
        if self.rank != 0:
            return None
        records = tuple(gathered or ())
        timing_dir = Path(run_dir).resolve() / "timing"
        timing_dir.mkdir(parents=True, exist_ok=True)
        detail = timing_dir / "freecam_timing.0000"
        summary = timing_dir / "freecam_timing_stats"
        detail.write_text(_format_rank_report(local), encoding="utf-8")
        summary.write_text(_format_global_report(records), encoding="utf-8")
        if profile is not None:
            profile_path = (
                timing_dir / f"cesm_timing.{profile.case_name}.{profile.lid}"
            )
            profile_path.write_text(
                format_cesm_timing_profile(profile, _phase_totals(records)),
                encoding="utf-8",
            )
        return detail, summary


def _default_clock() -> Callable[[], float]:
    try:
        from mpi4py import MPI

        return MPI.Wtime
    except (ImportError, RuntimeError):
        return time.perf_counter


def _format_rank_report(snapshot: Mapping[str, object]) -> str:
    rank = int(snapshot["rank"])
    size = int(snapshot["size"])
    timers = snapshot.get("timers", {})
    assert isinstance(timers, Mapping)
    lines = [
        "FreeCAM hierarchical timing report",
        "Clock: MPI_Wtime (time.perf_counter fallback)",
        f"MPI tasks: {size}",
        f"Detail rank: {rank}",
        "",
        f"{'name':<68} {'called':>10} {'walltotal':>14} {'wallmax':>14} {'wallmin':>14}",
    ]
    for path, raw in timers.items():
        assert isinstance(raw, Mapping)
        parts = str(path).split("/")
        display_name = "  " * (len(parts) - 1) + parts[-1]
        lines.append(
            f"{display_name:<68.68} "
            f"{int(raw['calls']):>10d} "
            f"{float(raw['walltotal']):>14.6f} "
            f"{float(raw['wallmax']):>14.6f} "
            f"{float(raw['wallmin']):>14.6f}"
        )
    return "\n".join(lines) + "\n"


def _aggregate(
    snapshots: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    by_path: dict[str, list[tuple[int, Mapping[str, object]]]] = {}
    for snapshot in snapshots:
        rank = int(snapshot["rank"])
        timers = snapshot.get("timers", {})
        assert isinstance(timers, Mapping)
        for path, raw in timers.items():
            assert isinstance(raw, Mapping)
            by_path.setdefault(str(path), []).append((rank, raw))

    result: list[dict[str, object]] = []
    for path, per_rank in by_path.items():
        wall_by_rank = [
            (rank, float(raw["walltotal"])) for rank, raw in per_rank
        ]
        max_rank, wallmax = max(wall_by_rank, key=lambda item: item[1])
        min_rank, wallmin = min(wall_by_rank, key=lambda item: item[1])
        walltotal = sum(value for _, value in wall_by_rank)
        result.append(
            {
                "name": path,
                "processes": len(per_rank),
                "count": sum(int(raw["calls"]) for _, raw in per_rank),
                "walltotal": walltotal,
                "wallmax": wallmax,
                "max_rank": max_rank,
                "wallmin": wallmin,
                "min_rank": min_rank,
                "wallavg": walltotal / len(per_rank),
            }
        )
    return tuple(result)


def _format_global_report(snapshots: Sequence[Mapping[str, object]]) -> str:
    size = max((int(snapshot["size"]) for snapshot in snapshots), default=0)
    lines = [
        "FreeCAM global timing statistics",
        "Clock: MPI_Wtime (time.perf_counter fallback)",
        f"MPI tasks: {size}",
        "",
        (
            f"{'name':<72} {'processes':>9} {'count':>10} "
            f"{'walltotal':>14} {'wallmax':>14} {'max rank':>8} "
            f"{'wallmin':>14} {'min rank':>8} {'wallavg':>14}"
        ),
    ]
    for record in _aggregate(snapshots):
        lines.append(
            f"{str(record['name']):<72.72} "
            f"{int(record['processes']):>9d} "
            f"{int(record['count']):>10d} "
            f"{float(record['walltotal']):>14.6f} "
            f"{float(record['wallmax']):>14.6f} "
            f"{int(record['max_rank']):>8d} "
            f"{float(record['wallmin']):>14.6f} "
            f"{int(record['min_rank']):>8d} "
            f"{float(record['wallavg']):>14.6f}"
        )
    return "\n".join(lines) + "\n"


@dataclass(frozen=True, slots=True)
class CESMTimingContext:
    """Run metadata needed to render a CIME-format performance profile.

    Every field is supplied by the caller so the formatter stays pure: the
    driver reads the wall clock and environment once at finalization, while
    tests pass fixed values and assert on the exact rendered numbers.
    """

    case_name: str
    lid: str
    machine: str
    caseroot: str
    user: str
    curr_date: str
    driver: str
    grid: str
    compset: str
    run_type: str
    timestep_seconds: int
    mpi_ranks: int
    tasks_per_node: int
    components: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class _PhaseTotals:
    """Coarse init/run/final wall times, reduced over all ranks (slowest)."""

    steps: int
    init_seconds: float
    run_seconds: float
    final_seconds: float


def _rank_max_walltotal(
    snapshots: Sequence[Mapping[str, object]], key: str
) -> float:
    values = []
    for snapshot in snapshots:
        timers = snapshot.get("timers", {})
        assert isinstance(timers, Mapping)
        raw = timers.get(key)
        if raw is not None:
            assert isinstance(raw, Mapping)
            values.append(float(raw["walltotal"]))
    return max(values) if values else 0.0


def _phase_totals(snapshots: Sequence[Mapping[str, object]]) -> _PhaseTotals:
    """Extract the top-level phase totals the profile reports from snapshots.

    ``FREECAM:TOTAL`` brackets the whole run, with ``FREECAM:INITIALIZE``,
    ``FREECAM:STEP`` (one call per model step), and ``FREECAM:FINALIZE`` as its
    children.  Each phase is reduced to the slowest rank, matching CIME's
    max-across-tasks driver timers.
    """

    step_key = "FREECAM:TOTAL/FREECAM:STEP"
    steps = 0
    for snapshot in snapshots:
        timers = snapshot.get("timers", {})
        assert isinstance(timers, Mapping)
        raw = timers.get(step_key)
        if raw is not None:
            assert isinstance(raw, Mapping)
            steps = max(steps, int(raw["calls"]))
    return _PhaseTotals(
        steps=steps,
        init_seconds=_rank_max_walltotal(
            snapshots, "FREECAM:TOTAL/FREECAM:INITIALIZE"
        ),
        run_seconds=_rank_max_walltotal(snapshots, step_key),
        final_seconds=_rank_max_walltotal(
            snapshots, "FREECAM:TOTAL/FREECAM:FINALIZE"
        ),
    )


def format_cesm_timing_profile(
    context: CESMTimingContext, totals: _PhaseTotals
) -> str:
    """Render a CIME ``cesm_timing`` performance profile for a CAM-only run.

    freeCAM advances the CAM atmosphere as a single timed unit, so the
    component breakdown mirrors a standalone-atmosphere CESM case: ATM carries
    the whole run cost and every surface/coupler component reads zero, exactly
    as CIME reports for an ``atm``-only compset.  Model Cost bills whole nodes
    (``pe count for cost estimate``), matching CIME's node-granular accounting.
    """

    ranks = max(1, context.mpi_ranks)
    tasks_per_node = max(1, context.tasks_per_node)
    nodes = (ranks + tasks_per_node - 1) // tasks_per_node
    cost_pes = nodes * tasks_per_node

    run = totals.run_seconds
    simulated_days = totals.steps * context.timestep_seconds / 86400.0
    simulated_years = simulated_days / 365.0
    if run > 0.0 and simulated_days > 0.0:
        seconds_per_mday = run / simulated_days
        sypd = simulated_years / (run / 86400.0)
        model_cost = cost_pes * (run / 3600.0) / simulated_years
    else:
        seconds_per_mday = 0.0
        sypd = 0.0
        model_cost = 0.0

    lines = [
        "---------------- TIMING PROFILE ---------------------",
        f"  Case        : {context.case_name}",
        f"  LID         : {context.lid}",
        f"  Machine     : {context.machine}",
        f"  Caseroot    : {context.caseroot}",
        f"  Timeroot    : {context.caseroot}/Tools",
        f"  User        : {context.user}",
        f"  Curr Date   : {context.curr_date}",
        f"  Driver      : {context.driver}",
        f"  grid        : {context.grid}",
        f"  compset     : {context.compset}",
        f"  run type    : {context.run_type}",
        f"  stop option : nsteps, stop_n = {totals.steps}",
        f"  run length  : {simulated_days} days",
        "",
        "  component       comp_pes    root_pe   tasks  x threads "
        "instances (stride) ",
        "  ---------        ------     -------   ------   ------  "
        "---------  ------  ",
    ]
    for label, model in context.components:
        pes = 1 if label == "esp" else ranks
        lines.append(
            f"  {label} = {model:<9}{pes:>6}{0:>10}{pes:>10}     "
            "x 1       1      (1     ) "
        )
    lines.extend(
        [
            "",
            f"  total pes active           : {ranks} ",
            f"  mpi tasks per node         : {tasks_per_node} ",
            f"  pe count for cost estimate : {cost_pes} ",
            "",
            "  Overall Metrics: ",
            f"    Model Cost:{model_cost:>18.2f}   pe-hrs/simulated_year ",
            f"    Model Throughput:{sypd:>12.2f}   simulated_years/day ",
            "",
            f"    Init Time   :{totals.init_seconds:>12.3f} seconds ",
            f"    Run Time    :{run:>12.3f} seconds "
            f"{seconds_per_mday:>12.3f} seconds/day ",
            f"    Final Time  :{totals.final_seconds:>12.3f} seconds ",
            "",
            "",
            "Runs Time in total seconds, seconds/model-day, and "
            "model-years/wall-day ",
            "CPL Run Time represents time in CPL pes alone, not including "
            "time associated with data exchange with other components ",
            "",
        ]
    )
    for label, _model in (("tot", ""), *context.components):
        upper = label.upper()
        if upper == "ATM":
            seconds, spm, myr = run, seconds_per_mday, sypd
        elif upper == "TOT":
            seconds, spm, myr = run, seconds_per_mday, sypd
        else:
            seconds, spm, myr = 0.0, 0.0, 0.0
        lines.append(
            f"    {upper:<3} Run Time:{seconds:>12.3f} seconds "
            f"{spm:>12.3f} seconds/mday {myr:>12.2f} myears/wday "
        )
    lines.append(
        f"    CPL COMM Time:{0.0:>11.3f} seconds "
        f"{0.0:>12.3f} seconds/mday {0.0:>12.2f} myears/wday "
    )
    lines.append("   NOTE: min:max driver timers (seconds/day):   ")
    top = ranks - 1
    for index, (label, _model) in enumerate(context.components):
        upper = label.upper()
        indent = "                            " if index == 0 else (
            "                                                "
        )
        span = "0 to 0" if upper == "ESP" else f"0 to {top}"
        lines.append(f"{indent}{upper} (pes {span}) ")
    lines.extend(["", "", "More info on coupler timing: "])
    return "\n".join(lines) + "\n"


__all__ = ["CESMTimingContext", "FreeCAMProfiler", "format_cesm_timing_profile"]
