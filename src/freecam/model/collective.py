"""Compact formatting for errors reported by many MPI ranks."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence


def _rank_ranges(ranks: Sequence[int]) -> str:
    """Format sorted rank ids without printing hundreds of repeated numbers."""

    if not ranks:
        return ""
    ranges: list[str] = []
    first = previous = int(ranks[0])
    for raw_rank in ranks[1:]:
        rank = int(raw_rank)
        if rank == previous + 1:
            previous = rank
            continue
        ranges.append(str(first) if first == previous else f"{first}-{previous}")
        first = previous = rank
    ranges.append(str(first) if first == previous else f"{first}-{previous}")
    return ", ".join(ranges)


def collective_error_message(
    operation: str,
    errors: Sequence[str | None],
    *,
    max_groups: int = 8,
) -> str | None:
    """Group identical rank-local failures into one readable message.

    MPI ranks usually fail at the same statement.  Repeating an identical
    traceback hundreds of times hides the useful part of the error, so this
    formatter prints each distinct traceback once together with its rank set.
    """

    grouped: defaultdict[str, list[int]] = defaultdict(list)
    for rank, error in enumerate(errors):
        if error is not None:
            grouped[str(error).strip()].append(rank)
    if not grouped:
        return None

    failed = sum(len(ranks) for ranks in grouped.values())
    lines = [
        f"{operation} failed collectively on {failed}/{len(errors)} MPI ranks "
        f"({len(grouped)} distinct error{'s' if len(grouped) != 1 else ''}):"
    ]
    ordered = sorted(grouped.items(), key=lambda item: item[1][0])
    for error, ranks in ordered[:max_groups]:
        label = "rank" if len(ranks) == 1 else "ranks"
        lines.append(f"{label} {_rank_ranges(ranks)}:\n{error}")
    omitted = len(ordered) - max_groups
    if omitted > 0:
        lines.append(f"... {omitted} additional distinct errors omitted")
    return "\n".join(lines)


__all__ = ["collective_error_message"]
