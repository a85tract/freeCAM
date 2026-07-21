"""Serializable edits applied to one isolated model branch."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping

import numpy as np


_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_FIELD_OPERATIONS = frozenset(("set", "add", "multiply"))


@dataclass(frozen=True, slots=True)
class FieldEdit:
    name: str
    operation: str
    value: float
    unsafe: bool = False

    def __post_init__(self) -> None:
        if self.operation not in _FIELD_OPERATIONS:
            raise ValueError(
                f"field operation must be one of {sorted(_FIELD_OPERATIONS)}"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "operation": self.operation,
            "value": self.value,
            "unsafe": self.unsafe,
        }

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "FieldEdit":
        return cls(
            name=str(values["name"]),
            operation=str(values["operation"]),
            value=float(values["value"]),
            unsafe=bool(values.get("unsafe", False)),
        )


@dataclass(frozen=True, slots=True)
class SchemeMove:
    name: str
    before: str | None = None
    after: str | None = None
    to_group: str | None = None

    def __post_init__(self) -> None:
        if self.before is not None and self.after is not None:
            raise ValueError("scheme move accepts at most one of before or after")
        if self.before is None and self.after is None and self.to_group is None:
            raise ValueError("scheme move requires before, after, or to_group")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "before": self.before,
            "after": self.after,
            "to_group": self.to_group,
        }

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "SchemeMove":
        return cls(
            name=str(values["name"]),
            before=(None if values.get("before") is None else str(values["before"])),
            after=(None if values.get("after") is None else str(values["after"])),
            to_group=(
                None
                if values.get("to_group") is None
                else str(values["to_group"])
            ),
        )


@dataclass(frozen=True, slots=True)
class BranchSpec:
    """One branch from a common model snapshot."""

    name: str
    steps: int = 1
    disable_schemes: tuple[str, ...] = ()
    enable_schemes: tuple[str, ...] = ()
    scheme_moves: tuple[SchemeMove, ...] = ()
    field_edits: tuple[FieldEdit, ...] = ()

    def __post_init__(self) -> None:
        if not _SAFE_NAME.fullmatch(self.name):
            raise ValueError(
                "branch name may contain only letters, digits, dot, dash, and underscore"
            )
        if self.steps < 0:
            raise ValueError("branch steps must be non-negative")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "steps": self.steps,
            "disable_schemes": list(self.disable_schemes),
            "enable_schemes": list(self.enable_schemes),
            "scheme_moves": [move.as_dict() for move in self.scheme_moves],
            "field_edits": [edit.as_dict() for edit in self.field_edits],
        }

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "BranchSpec":
        return cls(
            name=str(values["name"]),
            steps=int(values.get("steps", 1)),
            disable_schemes=tuple(
                str(value) for value in values.get("disable_schemes", ())
            ),
            enable_schemes=tuple(
                str(value) for value in values.get("enable_schemes", ())
            ),
            scheme_moves=tuple(
                SchemeMove.from_mapping(value)
                for value in values.get("scheme_moves", ())
            ),
            field_edits=tuple(
                FieldEdit.from_mapping(value)
                for value in values.get("field_edits", ())
            ),
        )

    def apply(self, driver: Any) -> None:
        """Apply branch-local edits after restoring private arrays."""

        for name in self.disable_schemes:
            driver.scheme_plan.disable(name, unsafe=True)
        for name in self.enable_schemes:
            driver.scheme_plan.enable(name)
        for move in self.scheme_moves:
            driver.scheme_plan.move(
                move.name,
                before=move.before,
                after=move.after,
                to_group=move.to_group,
                unsafe=True,
            )
        for edit in self.field_edits:
            current = driver.pool.get(edit.name)
            if edit.operation == "set":
                updated = np.full_like(current, edit.value)
            elif edit.operation == "add":
                updated = np.add(current, edit.value)
            else:
                updated = np.multiply(current, edit.value)
            driver.pool.set(edit.name, updated, unsafe=edit.unsafe)
