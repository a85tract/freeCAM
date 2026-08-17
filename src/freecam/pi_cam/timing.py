"""Low-overhead hierarchical timing for the Python-owned CAM runtime.

The report intentionally resembles the two timing products written by CESM's
GPTL integration, while keeping a distinct ``freecam_timing`` prefix.  Timing
is rank-local during execution; MPI communication occurs only once, when the
runtime is finalized and the global summary is written.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence


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
        normalized = str(name).strip()
        if not normalized:
            raise ValueError("timer name cannot be empty")
        path = (*self.active_path, normalized)
        # Create the record on entry, rather than on exit, so dictionary order
        # is a true pre-order traversal: parent first, followed by its children.
        self._records.setdefault(path, _TimerRecord())
        self._stack.append(_ActiveTimer(normalized, path, self._clock()))

    def stop(self, name: str) -> float:
        normalized = str(name).strip()
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

    @contextmanager
    def region(self, name: str) -> Iterator[None]:
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)

    def snapshot(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "size": self.size,
            "timers": {
                "/".join(path): record.payload()
                for path, record in self._records.items()
            },
        }

    def write(self, run_dir: str | Path, communicator: Any) -> tuple[Path, Path] | None:
        """Write rank-0 detail and all-rank summary reports once."""

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


__all__ = ["FreeCAMProfiler"]
