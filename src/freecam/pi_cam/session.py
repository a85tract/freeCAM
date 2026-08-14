"""Interactive Jupyter controller for one persistent PI-CAM MPI model."""

from __future__ import annotations

import base64
import json
import os
import queue
import secrets
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from html import escape
from multiprocessing.connection import Connection, Listener
from pathlib import Path
from typing import Any

import cloudpickle
import numpy as np

from freecam.core.runtime_env import mpi_loader_environment
from freecam.model.python_processes import PythonProcessSpec

from .boundary import CAMBoundaryProvider
from .config import PICAMConfig
from .expressions import DistributedOperand
from .physics_catalog import (
    PICAMPhysicsCatalog,
    merge_runtime_process_records,
)
from .state import PICAMVariableSpec
from .ui import PICAMStateView, PICAMWorkflowView


class PICAMNotebookError(RuntimeError):
    """The persistent PI-CAM MPI worker could not complete a request."""


def _normalize_field_selection(index: Any) -> tuple[Any, ...]:
    """Validate a NumPy-style index before broadcasting it to MPI ranks."""

    items = index if isinstance(index, tuple) else (index,)
    normalized: list[Any] = []
    ellipses = 0
    for item in items:
        if item is Ellipsis:
            ellipses += 1
            if ellipses > 1:
                raise IndexError("a field selection may contain only one ellipsis")
            normalized.append(Ellipsis)
        elif isinstance(item, (int, np.integer)) and not isinstance(
            item, (bool, np.bool_)
        ):
            normalized.append(int(item))
        elif isinstance(item, slice):
            values = []
            for value in (item.start, item.stop, item.step):
                if value is None:
                    values.append(None)
                elif isinstance(value, (int, np.integer)) and not isinstance(
                    value, (bool, np.bool_)
                ):
                    values.append(int(value))
                else:
                    raise TypeError("field slice bounds must be integers or None")
            if values[2] == 0:
                raise ValueError("field slice step cannot be zero")
            normalized.append(slice(*values))
        else:
            raise TypeError(
                "distributed fields support integer, slice, and ellipsis indexing"
            )
    return tuple(normalized)


def _authkey_argument(authkey: bytes) -> str:
    """Encode a secret without letting a leading dash confuse argparse."""

    return "--authkey=" + base64.urlsafe_b64encode(authkey).decode("ascii")


class _SessionFieldReference(DistributedOperand):
    """One rank-local StatePool field exposed through the live MPI session."""

    def __init__(
        self,
        session: "PICAMNotebookSession",
        name: str,
        *,
        selection: tuple[Any, ...] | None = None,
    ) -> None:
        self.session = session
        self.name = name
        self.selection = selection

    @property
    def _expression_session(self) -> Any:
        return self.session

    @property
    def _expression_payload(self) -> Mapping[str, Any]:
        return {
            "type": "field",
            "name": self.name,
            "selection": self.selection,
        }

    @property
    def metadata(self) -> Mapping[str, Any]:
        fields = self.session.status.get("fields", {})
        if self.name not in fields:
            raise KeyError(self.name)
        return dict(fields[self.name])

    def get(self, *, rank: int = 0) -> Any:
        return self.session.field(self.name, rank=rank, selection=self.selection)

    def values(self, *, rank: int = 0) -> Any:
        """Return a copy of this field from one MPI rank."""

        return self.get(rank=rank)

    def stats(self, *, rank: int | str = 0) -> Mapping[str, Any]:
        return self.session.stats(
            self.name, rank=rank, selection=self.selection
        )

    def mean(self, *, rank: int | str = "global") -> float:
        return float(self.stats(rank=rank)["mean"])

    def min(self, *, rank: int | str = "global") -> float:
        return float(self.stats(rank=rank)["min"])

    def max(self, *, rank: int | str = "global") -> float:
        return float(self.stats(rank=rank)["max"])

    def fill(self, value: float | int) -> "_SessionFieldReference":
        self.session.edit_field(
            self.name,
            operation="fill",
            value=value,
            selection=self.selection,
        )
        return self

    def _inplace(
        self,
        operation: str,
        value: Any,
    ) -> "_SessionFieldReference":
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
            value, (bool, np.bool_)
        ):
            self.session.edit_field(
                self.name,
                operation=operation,
                value=value,
                selection=self.selection,
            )
            return self
        ufunc = {
            "add": np.add,
            "subtract": np.subtract,
            "multiply": np.multiply,
            "divide": np.divide,
        }[operation]
        expression = ufunc(self, value)
        self.session.assign_expression(
            self.name,
            expression.payload,
            selection=self.selection,
        )
        return self

    def __iadd__(self, value: Any) -> "_SessionFieldReference":
        return self._inplace("add", value)

    def __isub__(self, value: Any) -> "_SessionFieldReference":
        return self._inplace("subtract", value)

    def __imul__(self, value: Any) -> "_SessionFieldReference":
        return self._inplace("multiply", value)

    def __itruediv__(self, value: Any) -> "_SessionFieldReference":
        return self._inplace("divide", value)

    def __getitem__(self, index: Any) -> "_SessionFieldReference":
        if self.selection is not None:
            raise TypeError("combine distributed field indices in one [] expression")
        return _SessionFieldReference(
            self.session,
            self.name,
            selection=_normalize_field_selection(index),
        )

    def __setitem__(self, index: Any, value: Any) -> None:
        selection = _normalize_field_selection(index)
        if (
            isinstance(value, _SessionFieldReference)
            and value.session is self.session
            and value.name == self.name
            and value.selection == selection
        ):
            # ``field[index] += scalar`` writes the selection proxy back after
            # its in-place operation has already run on every MPI rank.
            return
        target = _SessionFieldReference(
            self.session, self.name, selection=selection
        )
        if isinstance(value, DistributedOperand):
            if value._expression_session is not self.session:
                raise ValueError("a distributed expression cannot mix different models")
            self.session.assign_expression(
                self.name,
                value._expression_payload,
                selection=selection,
            )
        else:
            target.fill(value)

    def __repr__(self) -> str:
        metadata = self.metadata
        selected = "" if self.selection is None else f", selection={self.selection!r}"
        return (
            f"Field(name={self.name!r}, shape={tuple(metadata.get('shape', ()))}, "
            f"dtype={metadata.get('dtype')!r}, units={metadata.get('units')!r}"
            f"{selected})"
        )

    def delete(self) -> Mapping[str, Any]:
        return self.session.delete_field(self.name)


class _SessionFieldCollection:
    def __init__(self, session: "PICAMNotebookSession") -> None:
        self.session = session
        self._ui_aliases: dict[str, str] = {}

    @property
    def names(self) -> tuple[str, ...]:
        return self.session.field_names

    @property
    def short_names(self) -> Mapping[str, str]:
        """Unambiguous leaf names derived from canonical StatePool names."""

        candidates: dict[str, list[str]] = {}
        for canonical in self.session.status.get("fields", {}):
            leaf = str(canonical).rsplit(".", 1)[-1]
            candidates.setdefault(leaf, []).append(str(canonical))
        return {
            leaf: names[0]
            for leaf, names in candidates.items()
            if len(names) == 1
        }

    @property
    def aliases(self) -> Mapping[str, str]:
        """All client-side explicit aliases plus safe automatic short names."""

        return {**self.short_names, **self._ui_aliases}

    def alias(
        self,
        alias: str,
        field: str,
        *,
        replace: bool = False,
    ) -> _SessionFieldReference:
        """Register one explicit Notebook alias without changing MPI state."""

        short = str(alias)
        if not short.isidentifier():
            raise ValueError("field alias must be a valid Python identifier")
        fields = self.session.status.get("fields", {})
        if short in fields:
            raise ValueError(f"field alias {short!r} is already a canonical field")
        if short in self._ui_aliases and not replace:
            raise ValueError(f"field alias {short!r} already exists")
        canonical = self._resolve(str(field))
        self._ui_aliases[short] = canonical
        return _SessionFieldReference(self.session, canonical)

    def __dir__(self) -> list[str]:
        fields = self.session.status.get("fields", {})
        candidates = set(fields)
        candidates.update(self.aliases)
        for metadata in fields.values():
            candidates.update(metadata.get("aliases", ()))
            if metadata.get("standard_name"):
                candidates.add(str(metadata["standard_name"]))
        return sorted(
            set(super().__dir__())
            | {item for item in candidates if item.isidentifier()}
        )

    def __getattr__(self, name: str) -> _SessionFieldReference:
        try:
            canonical = self._resolve(name)
        except KeyError as exc:
            raise AttributeError(str(exc)) from exc
        return _SessionFieldReference(self.session, canonical)

    def __getitem__(self, name: str) -> _SessionFieldReference:
        return _SessionFieldReference(self.session, self._resolve(name))

    def _resolve(self, name: str) -> str:
        fields = self.session.status.get("fields", {})
        if name in fields:
            return name
        if name in self._ui_aliases:
            return self._ui_aliases[name]
        matches = [
            canonical
            for canonical, metadata in fields.items()
            if name in metadata.get("aliases", ())
            or name == metadata.get("standard_name")
        ]
        if len(matches) == 1:
            return str(matches[0])
        if len(matches) > 1:
            raise KeyError(
                f"field alias {name!r} is ambiguous: " + ", ".join(matches)
            )
        leaf_matches = [
            str(canonical)
            for canonical in fields
            if str(canonical).rsplit(".", 1)[-1] == name
        ]
        if len(leaf_matches) == 1:
            return leaf_matches[0]
        if len(leaf_matches) > 1:
            raise KeyError(
                f"field short name {name!r} is ambiguous: "
                + ", ".join(leaf_matches)
            )
        raise KeyError(f"unknown PI-CAM field {name!r}")

    def create(
        self,
        name: str,
        *,
        dims: Sequence[str],
        dtype: str = "float64",
        units: str = "1",
        initial: float | int = 0.0,
        writable: bool = True,
        restart: bool = True,
        aliases: Sequence[str] = (),
        standard_name: str | None = None,
    ) -> _SessionFieldReference:
        self.session.create_field(
            name,
            dimensions=dims,
            dtype=dtype,
            units=units,
            initial=initial,
            writable=writable,
            restart=restart,
            aliases=aliases,
            standard_name=standard_name,
        )
        return _SessionFieldReference(self.session, name)

    def create_array(
        self, name: str, values: np.ndarray
    ) -> _SessionFieldReference:
        self.session.create_array(name, values)
        return _SessionFieldReference(self.session, name)

    def delete(self, name: str) -> Mapping[str, Any]:
        return self.session.delete_field(name)


class _SessionProcessCallResult(Mapping[str, _SessionFieldReference]):
    """Named StatePool outputs returned by one original CAM process call."""

    def __init__(
        self,
        session: "PICAMNotebookSession",
        record: Mapping[str, Any],
        trace: Mapping[str, Any],
    ) -> None:
        self.session = session
        self.name = str(record["name"])
        self.trace = dict(trace)
        self.bindings = tuple(record.get("bindings", ()))
        self._outputs = {
            str(binding["argument"]): str(binding["field"])
            for binding in self.bindings
            if str(binding.get("intent", "inout")).lower() in {"out", "inout"}
        }

    @property
    def fields(self) -> Mapping[str, _SessionFieldReference]:
        return {
            name: _SessionFieldReference(self.session, field)
            for name, field in self._outputs.items()
        }

    @property
    def process(self) -> Any:
        return self.session.physics.process(self.name)

    def remove(self) -> Mapping[str, Any]:
        return self.session.remove_promoted_process(self.name)

    def __getitem__(self, name: str) -> _SessionFieldReference:
        return _SessionFieldReference(self.session, self._outputs[str(name)])

    def __iter__(self):
        return iter(self._outputs)

    def __len__(self) -> int:
        return len(self._outputs)

    def __getattr__(self, name: str) -> _SessionFieldReference:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __repr__(self) -> str:
        return f"ProcessResult(name={self.name!r}, outputs={tuple(self)!r})"


class _SessionActionReference:
    """A physics action that can be run or edited without string plumbing."""

    def __init__(
        self,
        session: "PICAMNotebookSession",
        name: str,
        phase: str,
        *,
        kind: str = "scheme",
        record: Mapping[str, Any] | None = None,
    ) -> None:
        self.session = session
        self.name = name
        self.phase = phase
        self.kind = kind
        self._snapshot = None if record is None else dict(record)

    def run(self) -> Mapping[str, Any]:
        # Prefer the generated rank-local StatePool adapter when this process
        # has a complete explicit field contract.  Other admitted workflow
        # boundaries use the native action adapter directly.
        if self.operation in tuple(self.session.status.get("kernels", ())):
            return self.session.run_kernel(self.operation)
        return self.session.run_action(self.name, phase=self.phase)

    @property
    def qualified_name(self) -> str:
        return f"{self.phase}.{self.name}"

    @property
    def operation(self) -> str:
        return str(self._record()["operation"])

    @property
    def native_id(self) -> int | None:
        value = self._record().get("native_id")
        return None if value is None else int(value)

    @property
    def granularity(self) -> str:
        record = self._record()
        return str(
            record.get(
                "granularity",
                "leaf" if str(record["operation"]).startswith("leaf_") else "stage",
            )
        )

    @property
    def parent_stage(self) -> str | None:
        """Return the composite source stage that this leaf expands."""

        value = self._record().get("parent_stage")
        return None if value is None else str(value)

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self._record()

    @property
    def runnable(self) -> bool:
        if self._snapshot is None:
            return True
        return bool(self._snapshot.get("native_available", True))

    @property
    def capability(self) -> str:
        return "runtime"

    @property
    def enabled(self) -> bool:
        return bool(self._record()["enabled"])

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self.session.set_action_enabled(self.name, bool(value), phase=self.phase)
        if self._snapshot is not None:
            self._snapshot["enabled"] = bool(value)

    def enable(self) -> Mapping[str, Any]:
        result = self.session.set_action_enabled(self.name, True, phase=self.phase)
        if self._snapshot is not None:
            self._snapshot["enabled"] = True
        return result

    def disable(self) -> Mapping[str, Any]:
        result = self.session.set_action_enabled(self.name, False, phase=self.phase)
        if self._snapshot is not None:
            self._snapshot["enabled"] = False
        return result

    def move(
        self,
        *,
        before: str | None = None,
        after: str | None = None,
    ) -> Mapping[str, Any]:
        return self.session.move_action(
            self.name,
            phase=self.phase,
            before=before,
            after=after,
        )

    def reload(
        self,
        function: Any,
        *,
        reads: Sequence[str] | None = None,
        writes: Sequence[str] | None = None,
        parameters: Mapping[str, Any] | None = None,
        transactional: bool | None = None,
        unsafe: bool = False,
    ) -> "_SessionActionReference":
        """Replace a live Python callback on every MPI rank in place."""

        if self.kind != "python_process":
            raise TypeError("reload() is available only for Notebook Python processes")
        # Import lazily: facade imports this module to construct the session.
        from .facade import _python_callable_access, _runtime_state_callback

        inferred_reads, inferred_writes = _python_callable_access(function)
        callback = _runtime_state_callback(function, owner=self.name)
        self.session.reload_python(
            callback,
            name=self.name,
            phase=self.phase,
            reads=(inferred_reads if reads is None else tuple(reads)),
            writes=(inferred_writes if writes is None else tuple(writes)),
            parameters=parameters,
            transactional=transactional,
            unsafe=unsafe,
        )
        return self

    def remove(self) -> Mapping[str, Any]:
        if self.kind == "python_process":
            return self.session.remove_python(self.name)
        if self.kind == "runtime_fortran_process":
            return self.session.remove_fortran(self.name)
        if self.kind == "runtime_catalog_process":
            return self.session.remove_promoted_process(self.name)
        raise TypeError(f"source physics action {self.phase}.{self.name} cannot be removed")

    def _record(self) -> Mapping[str, Any]:
        if self._snapshot is not None:
            return dict(self._snapshot)
        matches = [
            row
            for row in self.session.status.get("step_plan", ())
            if row["phase"] == self.phase and row["name"] == self.name
        ]
        if len(matches) != 1:
            raise KeyError(f"{self.phase}.{self.name}")
        return dict(matches[0])

    def __repr__(self) -> str:
        return (
            f"PhysicsProcess(name={self.name!r}, operation={self.operation!r}, "
            f"enabled={self.enabled}, granularity={self.granularity!r})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _SessionActionReference):
            return NotImplemented
        return (
            self.session is other.session
            and self.qualified_name == other.qualified_name
        )

    __hash__ = None


class _SessionCatalogPhysicsReference:
    """A flat source physics entry that is not yet an independent boundary."""

    def __init__(
        self,
        session: "PICAMNotebookSession",
        record: Mapping[str, Any],
    ) -> None:
        self.session = session
        self._snapshot = dict(record)
        self.name = str(record["api_name"])
        self.phase = str(record["phase"])
        self.kind = "catalog_process"

    @property
    def operation(self) -> str:
        return str(self._snapshot["operation"])

    @property
    def qualified_name(self) -> str:
        return str(self._snapshot["qualified_name"])

    @property
    def source(self) -> str:
        return str(self._snapshot["source"])

    @property
    def level(self) -> str:
        return str(self._snapshot["level"])

    @property
    def granularity(self) -> str:
        return self.level

    @property
    def enabled(self) -> None:
        return None

    @property
    def runnable(self) -> bool:
        return bool(self._snapshot.get("runnable", False))

    @property
    def capability(self) -> str:
        return str(self._snapshot["capability"])

    @property
    def blockers(self) -> tuple[str, ...]:
        return tuple(str(item) for item in self._snapshot.get("blockers", ()))

    @property
    def parent_processes(self) -> tuple[str, ...]:
        return tuple(
            str(item) for item in self._snapshot.get("parent_processes", ())
        )

    @property
    def metadata(self) -> Mapping[str, Any]:
        return dict(self._snapshot)

    def _unavailable(self, operation: str) -> None:
        reason = ", ".join(self.blockers) or self.capability
        parents = ", ".join(self.parent_processes) or "its enclosing CAM process"
        raise PICAMNotebookError(
            f"{self.name!r} is a cataloged {self.level}, not an independently "
            f"runnable boundary; cannot {operation}. Run {parents}, or first "
            f"admit explicit StatePool/context bindings. Current blockers: {reason}"
        )

    def run(self) -> Mapping[str, Any]:
        if not self.runnable:
            self._unavailable("run it")
        return self().trace

    def __call__(self, **arguments: Any) -> _SessionProcessCallResult:
        bindings, initials = self.session._process_call_arguments(arguments)
        promoted_names = {
            str(record["name"])
            for record in self.session.status.get("promoted_processes", ())
        }
        if self.name not in promoted_names:
            record = self.session.promote_process(
                self.name,
                bindings=bindings,
                initials=initials,
            )
        elif arguments:
            raise PICAMNotebookError(
                f"{self.name!r} is already bound; remove its previous result "
                "before changing call arguments"
            )
        else:
            record = next(
                item
                for item in self.session.status.get("promoted_processes", ())
                if str(item["name"]) == self.name
            )
        trace = self.session.run_promoted_process(self.name)
        return _SessionProcessCallResult(self.session, record, trace)

    def bind(self, **arguments: Any) -> "_SessionPromotedPhysicsReference":
        """Bind field handles/literals without executing the process."""

        bindings, initials = self.session._process_call_arguments(arguments)
        return self.promote(bindings=bindings, initials=initials)

    def insert(
        self,
        *,
        before: str | None = None,
        after: str | None = None,
        enabled: bool = True,
    ) -> "_SessionPromotedPhysicsReference":
        """Auto-bind this process and insert it into the live workflow."""

        if not self.runnable:
            self._unavailable("insert it")
        return self.bind().insert(
            before=before,
            after=after,
            enabled=enabled,
        )

    def _install(
        self,
        session: "PICAMNotebookSession",
        *,
        before: str | None = None,
        after: str | None = None,
    ) -> "_SessionPromotedPhysicsReference":
        if session is not self.session:
            raise PICAMNotebookError(
                "a physics process can only be inserted into its originating model"
            )
        return self.insert(before=before, after=after)

    def promote(
        self,
        *,
        bindings: Mapping[str, str] | None = None,
        initials: Mapping[str, Any] | None = None,
        dimensions: Mapping[str, int] | None = None,
    ) -> "_SessionPromotedPhysicsReference":
        self.session.promote_process(
            self.name,
            bindings=bindings,
            initials=initials,
            dimensions=dimensions,
        )
        reference = self.session.physics.process(self.name)
        assert isinstance(reference, _SessionPromotedPhysicsReference)
        return reference

    def enable(self) -> Mapping[str, Any]:
        self._unavailable("enable it")
        raise AssertionError("unreachable")

    def disable(self) -> Mapping[str, Any]:
        self._unavailable("disable it")
        raise AssertionError("unreachable")

    def move(self, *, before: str | None = None, after: str | None = None) -> None:
        del before, after
        self._unavailable("move it")

    def __repr__(self) -> str:
        return (
            f"PhysicsProcess(name={self.name!r}, source={self.qualified_name!r}, "
            f"level={self.level!r}, runnable=False, capability={self.capability!r})"
        )


class _SessionPromotedPhysicsReference(_SessionCatalogPhysicsReference):
    """Notebook handle for one StatePool-bound original CAM routine."""

    @property
    def runnable(self) -> bool:
        return bool(self._snapshot.get("native_available"))

    @property
    def capability(self) -> str:
        if self._runtime_row() is not None:
            return "runtime"
        return (
            "statepool_bound"
            if self.runnable
            else "statepool_bound_no_native"
        )

    @property
    def bindings(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self._snapshot.get("bindings", ()))

    @property
    def fields(self) -> Mapping[str, _SessionFieldReference]:
        return {
            str(binding["argument"]): _SessionFieldReference(
                self.session, str(binding["field"])
            )
            for binding in self.bindings
            if str(binding.get("intent", "inout")).lower() in {"out", "inout"}
        }

    def __getitem__(self, name: str) -> _SessionFieldReference:
        return self.fields[str(name)]

    def __getattr__(self, name: str) -> _SessionFieldReference:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    @property
    def enabled(self) -> bool | None:
        row = self._runtime_row()
        return None if row is None else bool(row["enabled"])

    def run(self) -> Mapping[str, Any]:
        return self.session.run_promoted_process(self.name)

    def __call__(self, **arguments: Any) -> _SessionProcessCallResult:
        if arguments:
            raise PICAMNotebookError(
                f"{self.name!r} is already bound; call it without arguments or "
                "remove it before rebinding"
            )
        trace = self.run()
        return _SessionProcessCallResult(self.session, self._snapshot, trace)

    def remove(self) -> Mapping[str, Any]:
        return self.session.remove_promoted_process(self.name)

    def enable(self) -> Mapping[str, Any]:
        row = self._require_runtime_row()
        return self.session.set_action_enabled(
            self.name, True, phase=str(row["phase"])
        )

    def disable(self) -> Mapping[str, Any]:
        row = self._require_runtime_row()
        return self.session.set_action_enabled(
            self.name, False, phase=str(row["phase"])
        )

    def move(
        self, *, before: str | None = None, after: str | None = None
    ) -> Mapping[str, Any]:
        row = self._require_runtime_row()
        return self.session.move_action(
            self.name,
            phase=str(row["phase"]),
            before=before,
            after=after,
        )

    def insert(
        self,
        *,
        before: str | None = None,
        after: str | None = None,
        enabled: bool = True,
    ) -> "_SessionPromotedPhysicsReference":
        self.session.install_promoted_process(
            self.name,
            before=before,
            after=after,
            enabled=enabled,
        )
        return self.session.physics.process(self.name)

    def _install(
        self,
        session: "PICAMNotebookSession",
        *,
        before: str | None = None,
        after: str | None = None,
    ) -> "_SessionPromotedPhysicsReference":
        if session is not self.session:
            raise PICAMNotebookError(
                "a bound process can only be inserted into its originating model"
            )
        return self.insert(before=before, after=after)

    def _runtime_row(self) -> Mapping[str, Any] | None:
        matches = tuple(
            row
            for row in self.session.status.get("step_plan", ())
            if str(row.get("name")) == self.name
            and str(row.get("kind")) == "runtime_catalog_process"
        )
        return matches[0] if len(matches) == 1 else None

    def _require_runtime_row(self) -> Mapping[str, Any]:
        row = self._runtime_row()
        if row is None:
            raise PICAMNotebookError(
                f"{self.name!r} is bound but not in the complete workflow; "
                "insert it with driver.cam.workflow.insert(process, before=... "
                "or after=...)"
            )
        return row


class _SessionPhysicsCollection:
    _RUNTIME_KINDS = {"scheme", "python_process", "runtime_fortran_process"}

    def __init__(self, session: "PICAMNotebookSession") -> None:
        self.session = session
        self.catalog = PICAMPhysicsCatalog.load_default()

    @property
    def names(self) -> tuple[str, ...]:
        """Flat Python names for all case-reachable physics interfaces."""

        return tuple(str(row["api_name"]) for row in self.records)

    @property
    def operation_names(self) -> tuple[str, ...]:
        return tuple(str(row["operation"]) for row in self.records)

    @property
    def action_names(self) -> tuple[str, ...]:
        """Human-readable aliases used by the Python workflow."""

        return tuple(
            str(row["name"])
            for row in self.runtime_records
        )

    @property
    def runtime_records(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(
            dict(row)
            for row in self.session.status.get("step_plan", ())
            if row["kind"] in self._RUNTIME_KINDS
        )

    @property
    def records(self) -> tuple[Mapping[str, Any], ...]:
        records = [
            dict(row)
            for row in merge_runtime_process_records(
                self.runtime_records,
                self.catalog,
            )
        ]
        promoted = {
            str(record["name"]): dict(record)
            for record in self.session.status.get("promoted_processes", ())
        }
        process_adapters = {
            str(name) for name in self.session.status.get("process_adapters", ())
        }
        for row in records:
            if str(row["kind"]) == "catalog_process":
                identity = (
                    f"{row.get('qualified_name')}@{row.get('source')}"
                )
                available = identity in process_adapters
                row.update(
                    {
                        "runnable": available,
                        "capability": (
                            "compiled_process_device"
                            if available
                            else row["capability"]
                        ),
                    }
                )
            record = promoted.get(str(row["api_name"]))
            if record is None:
                continue
            runtime_action = next(
                (
                    action
                    for action in self.session.status.get("step_plan", ())
                    if str(action.get("name")) == str(record["name"])
                    and str(action.get("kind")) == "runtime_catalog_process"
                ),
                None,
            )
            row.update(
                {
                    "phase": (
                        "promoted_process"
                        if runtime_action is None
                        else str(runtime_action["phase"])
                    ),
                    "kind": (
                        "promoted_process"
                        if runtime_action is None
                        else "runtime_catalog_process"
                    ),
                    "runnable": bool(record.get("native_available")),
                    "enabled": (
                        None
                        if runtime_action is None
                        else bool(runtime_action["enabled"])
                    ),
                    "capability": (
                        "runtime"
                        if runtime_action is not None
                        else "statepool_bound"
                        if bool(record.get("native_available"))
                        else "statepool_bound_no_native"
                    ),
                    "bindings": tuple(record.get("bindings", ())),
                    "created_fields": tuple(record.get("created_fields", ())),
                    "native_available": bool(record.get("native_available")),
                    "requires_binding": False,
                }
            )
        return tuple(records)

    @property
    def interfaces(self) -> tuple[Any, ...]:
        references = []
        for row in self.records:
            if str(row["kind"]) in {
                "promoted_process",
                "runtime_catalog_process",
            }:
                references.append(_SessionPromotedPhysicsReference(self.session, row))
            elif str(row["kind"]) == "catalog_process":
                references.append(_SessionCatalogPhysicsReference(self.session, row))
            elif bool(row["runnable"]):
                references.append(_SessionActionReference(
                    self.session,
                    str(row["name"]),
                    str(row["phase"]),
                    kind=str(row["kind"]),
                    record=row,
                ))
            else:
                references.append(_SessionCatalogPhysicsReference(self.session, row))
        return tuple(references)

    @property
    def runnable(self) -> tuple[Any, ...]:
        return tuple(reference for reference in self.interfaces if reference.runnable)

    @property
    def catalog_only(self) -> tuple[Any, ...]:
        return tuple(
            reference for reference in self.interfaces if not reference.runnable
        )

    @property
    def coverage(self) -> Mapping[str, int]:
        base = self._coverage(self.records)
        source_only = tuple(
            row
            for row in merge_runtime_process_records(
                self.runtime_records,
                self.catalog,
            )
            if row["kind"] == "catalog_process"
        )
        process_adapters = {
            str(name) for name in self.session.status.get("process_adapters", ())
        }
        result = {
            **base,
            "source_reachable": self.catalog.reachable_procedures,
            "source_catalog": len(self.catalog.processes),
            "physical_processes": len(self.catalog.physics_processes),
            "compiled_process_adapters": sum(
                process.generated_adapter
                for process in self.catalog.physics_processes
            ),
            "formerly_catalog_only_interfaces": len(source_only),
            "catalog_adapters_compiled": sum(
                bool(row.get("generated_adapter")) for row in source_only
            ),
            "catalog_current_case_loadable": sum(
                f"{row['qualified_name']}@{row['source']}" in process_adapters
                for row in source_only
            ),
            "runtime_templates": sum(
                bool(row.get("generated_adapter")) for row in source_only
            ),
            "runtime_templates_loadable": sum(
                f"{row['qualified_name']}@{row['source']}" in process_adapters
                for row in source_only
            ),
            "runtime_bound": sum(
                str(row["kind"])
                in {"promoted_process", "runtime_catalog_process"}
                for row in self.records
            ),
            "runtime_inserted": sum(
                str(row["kind"]) == "runtime_catalog_process"
                for row in self.records
            ),
            "current_case_loadable": sum(
                f"{process.qualified_name}@{process.source}" in process_adapters
                for process in self.catalog.physics_processes
            ),
            "helper_routines": len(self.catalog.helpers),
            "runtime_overlap": len(self.catalog.physics_processes) - len(source_only),
            "excluded_lifecycle": self.catalog.excluded_lifecycle,
        }
        result["configuration_specific"] = (
            result["physical_processes"] - result["current_case_loadable"]
        )
        return result

    @staticmethod
    def _coverage(records: Sequence[Mapping[str, Any]]) -> Mapping[str, int]:
        runnable = tuple(row for row in records if bool(row["runnable"]))
        catalog_only = tuple(row for row in records if not bool(row["runnable"]))
        leaf = sum(
            str(row.get("granularity", "")) == "leaf"
            or str(row["operation"]).startswith("leaf_")
            for row in runnable
        )
        planned = tuple(row for row in runnable if row.get("enabled") is not None)
        enabled = sum(bool(row["enabled"]) for row in planned)
        result = {
            "interfaces": len(records),
            "runnable": len(runnable),
            "catalog_only": len(catalog_only),
            "enabled": enabled,
            "disabled": len(planned) - enabled,
            "leaf": leaf,
            "stage": len(runnable) - leaf,
        }
        return result

    def describe(self) -> tuple[Mapping[str, Any], ...]:
        """Return the complete process table in source execution order."""

        return self.records

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.interfaces)

    def __dir__(self) -> list[str]:
        return sorted(
            set(super().__dir__())
            | {
                name
                for name in (*self.names, *self.operation_names, *self.action_names)
                if name.isidentifier()
            }
        )

    def __getattr__(self, name: str) -> _SessionActionReference:
        try:
            return self.scheme(name)
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __getitem__(self, name: str) -> _SessionActionReference:
        return self.scheme(name)

    def scheme(self, name: str) -> _SessionActionReference:
        matches = [
            row
            for row in self.records
            if (
                row["name"] == name
                or row["api_name"] == name
                or row["operation"] == name
                or name in row.get("aliases", ())
                or row.get("qualified_name") == name
            )
        ]
        if len(matches) != 1:
            raise KeyError(f"physics action {name!r} is unknown or ambiguous")
        row = matches[0]
        if str(row["kind"]) in {"promoted_process", "runtime_catalog_process"}:
            return _SessionPromotedPhysicsReference(self.session, row)
        if str(row["kind"]) == "catalog_process":
            return _SessionCatalogPhysicsReference(self.session, row)
        if bool(row["runnable"]):
            return _SessionActionReference(
                self.session,
                str(row["name"]),
                str(row["phase"]),
                kind=str(row["kind"]),
                record=row,
            )
        return _SessionCatalogPhysicsReference(self.session, row)

    def process(self, name: str) -> Any:
        """Resolve one flat physics process; ``scheme`` remains an alias."""

        return self.scheme(name)

    def __repr__(self) -> str:
        coverage = self.coverage
        return (
            "PhysicsProcesses("
            f"interfaces={coverage['interfaces']}, runnable={coverage['runnable']}, "
            f"catalog_only={coverage['catalog_only']})"
        )

    def _repr_html_(self) -> str:
        records = self.records
        coverage = self.coverage

        def table(selected: Sequence[Mapping[str, Any]]) -> str:
            rows = []
            for record in selected:
                operation = str(record["operation"])
                level = str(
                    record.get(
                        "granularity",
                        "leaf" if operation.startswith("leaf_") else "stage",
                    )
                )
                parent_stage = (
                    str(record.get("parent_stage") or "unknown")
                    if level == "leaf"
                    else "—"
                )
                if not bool(record["runnable"]):
                    state = str(record["capability"])
                elif record.get("enabled") is None:
                    state = "bindable runtime template"
                else:
                    state = "enabled" if bool(record["enabled"]) else "disabled"
                rows.append(
                    "<tr>"
                    f"<td><code>{escape(str(record['api_name']))}</code></td>"
                    f"<td><code>{escape(operation)}</code></td>"
                    f"<td>{escape(level)}</td>"
                    f"<td><code>{escape(parent_stage)}</code></td>"
                    f"<td>{escape(state)}</td>"
                    "</tr>"
                )
            return (
                "<table><thead><tr><th>Python API</th>"
                "<th>Original routine</th><th>Level</th>"
                "<th>Parent stage</th><th>Capability</th>"
                f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
            )

        runnable = tuple(record for record in records if bool(record["runnable"]))
        catalog_only = tuple(
            record for record in records if not bool(record["runnable"])
        )
        return (
            "<div class='freecam-physics'>"
            "<style>"
            ".freecam-physics table{border-collapse:collapse;font-size:13px}"
            ".freecam-physics th,.freecam-physics td{padding:4px 9px;"
            "border-bottom:1px solid #ddd;text-align:left}"
            ".freecam-physics code{font-size:12px}"
            ".freecam-physics details{margin-top:10px}"
            ".freecam-physics summary{cursor:pointer;font-weight:600}"
            "</style>"
            f"<p><strong>{coverage['interfaces']} flat physics interfaces</strong>. "
            f"All {coverage['formerly_catalog_only_interfaces']} former "
            "catalog-only interfaces have compiled StatePool pointer adapters; "
            f"{coverage['catalog_current_case_loadable']} of those devices load "
            "in this PI-CAM executable.</p>"
            "<h4>Runnable workflow boundaries</h4>"
            f"{table(runnable)}"
            "<details><summary>Show all source-catalog entries</summary>"
            f"{table(catalog_only)}</details></div>"
        )

    def install_python(
        self,
        function: Any,
        *,
        name: str,
        before: str | None = None,
        after: str | None = None,
        reads: Sequence[str] = (),
        writes: Sequence[str] = (),
        parameters: Mapping[str, Any] | None = None,
        enabled: bool = True,
        transactional: bool = True,
        unsafe: bool = False,
    ) -> _SessionActionReference:
        phase, before, after = self._resolve_placement(
            phase=None, before=before, after=after
        )
        result = self.session.install_python(
            function,
            name=name,
            phase=phase,
            before=before,
            after=after,
            reads=reads,
            writes=writes,
            parameters=parameters,
            enabled=enabled,
            transactional=transactional,
            unsafe=unsafe,
        )
        return _SessionActionReference(
            self.session,
            str(result["name"]),
            str(result["phase"]),
            kind="python_process",
        )

    def install_fortran(
        self,
        source: str | Path,
        *,
        process: str,
        before: str | None = None,
        after: str | None = None,
        project_root: str | Path | None = None,
        enabled: bool = True,
        unsafe: bool = False,
    ) -> _SessionActionReference:
        phase, before, after = self._resolve_placement(
            phase=None, before=before, after=after
        )
        result = self.session.install_fortran(
            source,
            process=process,
            phase=phase,
            before=before,
            after=after,
            project_root=project_root,
            enabled=enabled,
            unsafe=unsafe,
        )
        return _SessionActionReference(
            self.session,
            str(result["name"]),
            str(result["phase"]),
            kind="runtime_fortran_process",
        )

    def _resolve_placement(
        self,
        *,
        phase: str | None,
        before: str | None,
        after: str | None,
    ) -> tuple[str, str | None, str | None]:
        """Infer the internal source region from a neighboring process."""

        if before is None and after is None:
            before = "state_export"
        if before is not None and after is not None:
            raise PICAMNotebookError("provide only one of before= or after=")
        # Explicit internal placement remains accepted for compatibility and
        # is validated collectively by the live worker.  The normal public
        # path below infers it from the neighboring process name.
        if phase is not None:
            return str(phase), before, after
        anchor = before or after or ""
        records = list(self.runtime_records)
        known = {
            (str(row["phase"]), str(row["name"])) for row in records
        }
        records.extend(
            row
            for row in self.session.status.get("step_plan", ())
            if (str(row["phase"]), str(row["name"])) not in known
        )
        matches = [
            row
            for row in records
            if row["name"] == anchor
            or row["operation"] == anchor
            or row.get("api_name") == anchor
            or f"{row['phase']}.{row['name']}" == anchor
        ]
        if len(matches) != 1:
            raise PICAMNotebookError(
                f"placement process {anchor!r} is unknown or ambiguous"
            )
        inferred = str(matches[0]["phase"])
        return inferred, before, after


class _SessionPhaseReference:
    def __init__(self, session: "PICAMNotebookSession", name: str) -> None:
        self.session = session
        self.name = name

    @property
    def actions(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(
            dict(row)
            for row in self.session.status.get("step_plan", ())
            if row["phase"] == self.name
        )

    def run(self) -> tuple[Mapping[str, Any], ...]:
        return self.session.run_phase(self.name)

    def expand(self) -> tuple[Mapping[str, Any], ...]:
        expanders = {
            "cam_run1": self.session.expand_cam_run1_leaves,
            "cam_run2": self.session.expand_cam_run2_leaves,
            "cam_run4": self.session.expand_cam_run4_leaves,
        }
        if self.name not in expanders:
            raise TypeError(f"phase {self.name!r} has no finer validated expansion")
        return expanders[self.name]()


class _SessionPhaseCollection:
    def __init__(self, session: "PICAMNotebookSession") -> None:
        self.session = session

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                str(row["phase"])
                for row in self.session.status.get("step_plan", ())
            )
        )

    def __getattr__(self, name: str) -> _SessionPhaseReference:
        if name not in self.names:
            raise AttributeError(name)
        return _SessionPhaseReference(self.session, name)

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()) | set(self.names))

    def __getitem__(self, name: str) -> _SessionPhaseReference:
        if name not in self.names:
            raise KeyError(name)
        return _SessionPhaseReference(self.session, name)


class _SessionKernelReference:
    def __init__(self, session: "PICAMNotebookSession", name: str) -> None:
        self.session = session
        self.name = name

    def run(self) -> Mapping[str, Any]:
        return self.session.run_kernel(self.name)


class _SessionKernelCollection:
    def __init__(self, session: "PICAMNotebookSession") -> None:
        self.session = session

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.session.status.get("kernels", ()))

    def __getattr__(self, name: str) -> _SessionKernelReference:
        if name not in self.names:
            raise AttributeError(name)
        return _SessionKernelReference(self.session, name)

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()) | set(self.names))

    def __getitem__(self, name: str) -> _SessionKernelReference:
        if name not in self.names:
            raise KeyError(name)
        return _SessionKernelReference(self.session, name)


class PICAMNotebookSession:
    """Keep 512 CAM ranks alive and execute one Python-controlled action at a time."""

    def __init__(
        self,
        config: str | Path | PICAMConfig,
        *,
        boundary: str | Path | CAMBoundaryProvider,
        run_dir: str | Path,
        env_script: str | Path,
        python_executable: str | Path | None = None,
        launcher: str | Sequence[str] = "mpiexec",
        launch_mode: str = "auto",
        pbs_account: str | None = None,
        pbs_queue: str = "develop",
        pbs_walltime: str = "02:00:00",
        pbs_memory_per_node: str = "110GB",
        verify_boundary_exports: bool = True,
        startup_timeout: float = 1200.0,
        request_timeout: float = 300.0,
        log_path: str | Path | None = None,
    ) -> None:
        if isinstance(config, PICAMConfig):
            raise TypeError("PICAMNotebookSession currently requires a YAML config path")
        self.config_path = Path(config).resolve()
        self.config = PICAMConfig.from_yaml(self.config_path)
        self.boundary = (
            Path(boundary).resolve()
            if isinstance(boundary, (str, Path))
            else boundary
        )
        self.run_dir = Path(run_dir).resolve()
        self.env_script = Path(env_script).resolve()
        self.python_executable = str(python_executable or sys.executable)
        self.launcher = tuple(
            shlex.split(launcher) if isinstance(launcher, str) else launcher
        )
        if launch_mode not in {"auto", "local", "pbs"}:
            raise ValueError("launch_mode must be auto, local, or pbs")
        self.launch_mode = launch_mode
        self.pbs_account = pbs_account
        self.pbs_queue = pbs_queue
        self.pbs_walltime = pbs_walltime
        normalized_memory = str(pbs_memory_per_node).strip().upper()
        if not normalized_memory[:-2].isdigit() or not normalized_memory.endswith(
            "GB"
        ):
            raise ValueError("pbs_memory_per_node must have form '<integer>GB'")
        self.pbs_memory_per_node = normalized_memory
        self.verify_boundary_exports = bool(verify_boundary_exports)
        self.startup_timeout = float(startup_timeout)
        self.request_timeout = float(request_timeout)
        self.log_path = Path(log_path or self.run_dir / "pi_cam_notebook_worker.log").resolve()
        if not self.env_script.is_file():
            raise FileNotFoundError(self.env_script)
        if not (self.run_dir / "atm_in").is_file():
            raise FileNotFoundError(f"PI-CAM run directory lacks atm_in: {self.run_dir}")
        if isinstance(self.boundary, Path) and not (
            self.boundary / "manifest.json"
        ).is_file():
            raise FileNotFoundError(
                f"PI-CAM boundary replay is incomplete: {self.boundary}"
            )
        if not isinstance(self.boundary, (Path, CAMBoundaryProvider)):
            raise TypeError(
                "boundary must be a replay path or CAMBoundaryProvider"
            )
        self._connection: Connection | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._job_id: str | None = None
        self._log_handle: Any = None
        self._pbs_script: Path | None = None
        self._status: dict[str, Any] = {}
        self._request_lock = threading.RLock()
        self._step_plots: list[Any] = []
        self.fields = _SessionFieldCollection(self)
        self.physics = _SessionPhysicsCollection(self)
        self.phases = _SessionPhaseCollection(self)
        self.kernels = _SessionKernelCollection(self)
        self.state = PICAMStateView(self)
        self.workflow = PICAMWorkflowView(self)

    @property
    def running(self) -> bool:
        return self._connection is not None

    @property
    def job_id(self) -> str | None:
        return self._job_id

    @property
    def status(self) -> Mapping[str, Any]:
        if self.running:
            self._status = dict(self._request({"op": "status"}))
        return dict(self._status)

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(self.status.get("fields", ()))

    @property
    def has_step_plots(self) -> bool:
        return bool(self._step_plots)

    def _register_step_plot(self, plot: Any) -> None:
        if plot not in self._step_plots:
            self._step_plots.append(plot)

    def capture_step_plots(self) -> None:
        for plot in tuple(self._step_plots):
            plot.capture()

    def start(self) -> "PICAMNotebookSession":
        if self.running:
            raise RuntimeError("PI-CAM Notebook session is already running")
        environment = self._environment()
        mode = self._launch_mode(environment)
        authkey = secrets.token_bytes(32)
        listener = Listener(("0.0.0.0" if mode == "pbs" else "127.0.0.1", 0), authkey=authkey)
        host = socket.getfqdn() if mode == "pbs" else "127.0.0.1"
        _, port = listener.address
        command = [
            *self.launcher,
            "-n",
            str(self.config.mpi_size),
            self.python_executable,
            "-m",
            "freecam.pi_cam.session_worker",
            "--host",
            host,
            "--port",
            str(port),
            # URL-safe base64 may start with "-".  Use argparse's
            # --option=value form so such a secret is never mistaken for a
            # new command-line option on all 512 ranks.
            _authkey_argument(authkey),
            "--config",
            str(self.config_path),
            "--run-dir",
            str(self.run_dir),
            "--expected-ranks",
            str(self.config.mpi_size),
        ]
        command.extend(self._boundary_arguments())
        command.append(
            "--verify-exports"
            if self.verify_boundary_exports
            else "--no-verify-exports"
        )
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if mode == "pbs":
                self._submit_pbs(command, environment)
            else:
                self._log_handle = self.log_path.open("ab", buffering=0)
                self._process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=self._log_handle,
                    stderr=subprocess.STDOUT,
                    env=environment,
                    start_new_session=True,
                )
            self._connection = self._accept(listener)
            self._status = dict(self._unwrap(self._receive(self.startup_timeout)))
        except BaseException:
            self._abort()
            raise
        finally:
            listener.close()
        return self

    def _boundary_arguments(self) -> list[str]:
        if isinstance(self.boundary, Path):
            return ["--boundary", str(self.boundary)]
        payload = cloudpickle.dumps(self.boundary, protocol=5)
        maximum_bytes = 8 * 1024 * 1024
        if len(payload) > maximum_bytes:
            raise ValueError(
                "online boundary provider payload exceeds 8 MiB; do not capture "
                "large arrays in the update callback"
            )
        path = self.run_dir / ".online-boundary-provider.pkl"
        path.write_bytes(payload)
        path.chmod(0o600)
        return ["--boundary-provider", str(path)]

    def step(self, count: int = 1) -> Mapping[str, Any]:
        if count < 1:
            raise ValueError("count must be positive")
        # ``request_timeout`` is the bound for one interactive action.  A bulk
        # run is still one socket request but legitimately performs many
        # complete CAM/CESM coupling steps before replying.
        timeout = max(self.request_timeout, 15.0 * int(count))
        command = {"op": "step", "count": int(count)}
        response = (
            self._request(command)
            if timeout == self.request_timeout
            else self._request(command, timeout=timeout)
        )
        self._status = dict(response)
        return dict(self._status)

    def advance(self, steps: int = 1) -> Mapping[str, Any]:
        """Advance complete CAM steps while keeping ``step`` compatible."""

        return self.step(steps)

    def configure_output(
        self,
        *,
        history_every: int | None = 1,
        restart_every: int | None | str = "end",
    ) -> Mapping[str, Any]:
        """Set history/restart cadence without editing ``atm_in`` by hand."""

        if restart_every == "end":
            restart_interval = None
            restart_at_end = True
        else:
            restart_interval = restart_every
            restart_at_end = False
        self._status = dict(
            self._request(
                {
                    "op": "configure_output",
                    "history_every": history_every,
                    "restart_every": restart_interval,
                    "restart_at_end": restart_at_end,
                }
            )
        )
        return dict(self._status)

    def workflow_action(
        self,
        name: str,
        *,
        phase: str,
        kind: str,
    ) -> Any:
        """Return a live handle for any workflow row, including control rows."""

        if kind == "runtime_catalog_process":
            return self.physics.process(name)
        return _SessionActionReference(self, name, phase, kind=kind)

    def run_action(self, name: str, *, phase: str | None = None) -> Mapping[str, Any]:
        """Run one scheme or installed runtime process without advancing time."""

        return dict(
            self._request({"op": "run_action", "name": name, "phase": phase})
        )

    def run_scheme(self, name: str, *, phase: str | None = None) -> Mapping[str, Any]:
        """Compatibility alias for :meth:`run_action`."""

        return self.run_action(name, phase=phase)

    def run_phase(self, name: str) -> tuple[Mapping[str, Any], ...]:
        return tuple(self._request({"op": "run_phase", "phase": name}))

    def run_kernel(self, name: str) -> Mapping[str, Any]:
        """Run one experimental raw-array kernel on all live MPI ranks."""

        return dict(self._request({"op": "run_kernel", "name": name}))

    def _process_call_arguments(
        self, arguments: Mapping[str, Any]
    ) -> tuple[dict[str, str], dict[str, Any]]:
        """Convert field objects to bindings and literals to initial values."""

        bindings: dict[str, str] = {}
        initials: dict[str, Any] = {}
        for raw_name, value in arguments.items():
            name = str(raw_name)
            if isinstance(value, _SessionFieldReference):
                bindings[name] = value.name
                continue
            if isinstance(value, str):
                try:
                    bindings[name] = self.fields._resolve(value)
                    continue
                except KeyError:
                    pass
            initials[name] = value
        return bindings, initials

    def promote_process(
        self,
        name: str,
        *,
        bindings: Mapping[str, str] | None = None,
        initials: Mapping[str, Any] | None = None,
        dimensions: Mapping[str, int] | None = None,
    ) -> Mapping[str, Any]:
        """Move one original routine's explicit caller arguments into StatePool."""

        result = dict(
            self._request(
                {
                    "op": "promote_process",
                    "name": str(name),
                    "bindings": dict(bindings or {}),
                    "initials": dict(initials or {}),
                    "dimensions": {
                        str(key): int(value)
                        for key, value in (dimensions or {}).items()
                    },
                }
            )
        )
        self._status = dict(self._request({"op": "status"}))
        return result

    def run_promoted_process(self, name: str) -> Mapping[str, Any]:
        """Run one StatePool-bound routine without advancing model time."""

        return dict(
            self._request({"op": "run_promoted_process", "name": str(name)})
        )

    def install_promoted_process(
        self,
        name: str,
        *,
        before: str | None = None,
        after: str | None = None,
        enabled: bool = True,
    ) -> Mapping[str, Any]:
        """Insert one bound source routine into the complete live workflow."""

        result = dict(
            self._request(
                {
                    "op": "install_promoted_process",
                    "name": str(name),
                    "before": before,
                    "after": after,
                    "enabled": bool(enabled),
                }
            )
        )
        self._update_step_plan(result)
        self._status = dict(self._request({"op": "status"}))
        return result

    def remove_promoted_process(self, name: str) -> Mapping[str, Any]:
        result = dict(
            self._request({"op": "remove_promoted_process", "name": str(name)})
        )
        self._status = dict(self._request({"op": "status"}))
        return result

    def trace(self, *, since: int = 0) -> tuple[Mapping[str, Any], ...]:
        """Return actual worker-side action records from ``since`` onward."""

        if int(since) < 0:
            raise ValueError("since cannot be negative")
        return tuple(self._request({"op": "trace", "since": int(since)}))

    def expand_cam_run1_leaves(self) -> tuple[Mapping[str, Any], ...]:
        """Replace three composite stages with ordered native leaf actions."""

        return tuple(self._request({"op": "expand_cam_run1_leaves"}))

    def expand_cam_run2_leaves(self) -> tuple[Mapping[str, Any], ...]:
        """Replace admitted ``cam_run2`` composites with native leaves."""

        return tuple(self._request({"op": "expand_cam_run2_leaves"}))

    def expand_cam_run4_leaves(self) -> tuple[Mapping[str, Any], ...]:
        """Replace the ``cam_run4`` finish composite with native leaves."""

        return tuple(self._request({"op": "expand_cam_run4_leaves"}))

    def expand_cam_run2_run4_leaves(self) -> tuple[Mapping[str, Any], ...]:
        """Expand admitted leaf actions from ``cam_run2`` through run4."""

        return tuple(self._request({"op": "expand_cam_run2_run4_leaves"}))

    def create_field(
        self,
        name: str,
        *,
        dimensions: Sequence[str],
        dtype: str = "float64",
        units: str = "1",
        initial: float | int = 0.0,
        writable: bool = True,
        restart: bool = True,
        aliases: Sequence[str] = (),
        standard_name: str | None = None,
    ) -> Mapping[str, Any]:
        spec = PICAMVariableSpec(
            name=name,
            dimensions=tuple(dimensions),
            dtype=dtype,
            units=units,
            initial=initial,
            writable=writable,
            restart=restart,
            aliases=tuple(aliases),
            standard_name=standard_name,
        )
        return dict(self._request({"op": "create_field", "spec": spec.to_payload()}))

    def create_array(self, name: str, values: np.ndarray) -> Mapping[str, Any]:
        """Copy one rank-independent NumPy array into every MPI rank."""

        if not isinstance(values, np.ndarray):
            raise TypeError("values must be a NumPy array")
        return dict(
            self._request(
                {
                    "op": "create_array",
                    "name": str(name),
                    "values": np.array(values, copy=True, order="F", subok=False),
                }
            )
        )

    def edit_field(
        self,
        name: str,
        *,
        operation: str,
        value: float | int,
        selection: tuple[Any, ...] | None = None,
    ) -> Mapping[str, Any]:
        """Apply one scalar edit independently to every rank-local array."""

        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise TypeError("distributed field edits require a numeric scalar")
        if not np.isfinite(value):
            raise ValueError("distributed field edit value must be finite")
        command = {
            "op": "edit_field",
            "name": str(name),
            "operation": str(operation),
            "value": value.item() if isinstance(value, np.generic) else value,
        }
        if selection is not None:
            command["selection"] = tuple(selection)
        return dict(self._request(command))

    def assign_expression(
        self,
        name: str,
        expression: Mapping[str, Any],
        *,
        selection: tuple[Any, ...] | None = None,
    ) -> Mapping[str, Any]:
        """Evaluate an expression independently on every MPI rank and write it."""

        command: dict[str, Any] = {
            "op": "assign_expression",
            "name": str(name),
            "expression": dict(expression),
        }
        if selection is not None:
            command["selection"] = tuple(selection)
        return dict(self._request(command))

    def evaluate_expression(
        self,
        expression: Mapping[str, Any],
        *,
        rank: int = 0,
    ) -> Any:
        """Evaluate an expression on one selected rank and return a copy."""

        return self._request(
            {
                "op": "evaluate_expression",
                "rank": int(rank),
                "expression": dict(expression),
            }
        )

    def delete_field(self, name: str) -> Mapping[str, Any]:
        return dict(self._request({"op": "delete_field", "name": name}))

    def install_python(
        self,
        function: Any,
        *,
        name: str,
        phase: str,
        before: str | None = None,
        after: str | None = None,
        reads: Sequence[str] = (),
        writes: Sequence[str] = (),
        parameters: Mapping[str, Any] | None = None,
        enabled: bool = True,
        transactional: bool = True,
        unsafe: bool = False,
    ) -> Mapping[str, Any]:
        spec = PythonProcessSpec.from_callable(
            function,
            name=name,
            group=phase,
            before=before,
            after=after,
            reads=reads,
            writes=writes,
            parameters=parameters,
            enabled=enabled,
            transactional=transactional,
        )
        return dict(
            self._request(
                {
                    "op": "install_python",
                    "spec": spec.as_dict(),
                    "unsafe": bool(unsafe),
                }
            )
        )

    def remove_python(self, name: str) -> Mapping[str, Any]:
        return dict(self._request({"op": "remove_python", "name": name}))

    def reload_python(
        self,
        function: Any,
        *,
        name: str,
        phase: str,
        reads: Sequence[str] = (),
        writes: Sequence[str] = (),
        parameters: Mapping[str, Any] | None = None,
        transactional: bool | None = None,
        unsafe: bool = False,
    ) -> Mapping[str, Any]:
        """Serialize and collectively replace an installed Python callback."""

        spec = PythonProcessSpec.from_callable(
            function,
            name=name,
            group=phase,
            reads=reads,
            writes=writes,
            parameters=parameters,
            transactional=True if transactional is None else transactional,
        )
        return dict(
            self._request(
                {
                    "op": "reload_python",
                    "name": name,
                    "spec": spec.as_dict(),
                    "preserve_parameters": parameters is None,
                    "preserve_transactional": transactional is None,
                    "unsafe": bool(unsafe),
                }
            )
        )

    def install_fortran(
        self,
        source: str | Path,
        *,
        process: str,
        phase: str,
        before: str | None = None,
        after: str | None = None,
        project_root: str | Path | None = None,
        enabled: bool = True,
        unsafe: bool = False,
    ) -> Mapping[str, Any]:
        return dict(
            self._request(
                {
                    "op": "install_fortran",
                    "spec": {
                        "schema_version": 1,
                        "source": str(Path(source).expanduser().resolve()),
                        "process": process,
                        "phase": phase,
                        "before": before,
                        "after": after,
                        "project_root": (
                            None
                            if project_root is None
                            else str(Path(project_root).expanduser().resolve())
                        ),
                        "enabled": bool(enabled),
                    },
                    "unsafe": bool(unsafe),
                }
            )
        )

    def remove_fortran(self, name: str) -> Mapping[str, Any]:
        return dict(self._request({"op": "remove_fortran", "name": name}))

    def set_action_enabled(
        self,
        name: str,
        enabled: bool,
        *,
        phase: str | None = None,
    ) -> Mapping[str, Any]:
        result = dict(
            self._request(
                {
                    "op": "set_action_enabled",
                    "name": name,
                    "phase": phase,
                    "enabled": bool(enabled),
                }
            )
        )
        self._update_step_plan(result)
        return result

    def move_action(
        self,
        name: str,
        *,
        phase: str | None = None,
        before: str | None = None,
        after: str | None = None,
    ) -> Mapping[str, Any]:
        result = dict(
            self._request(
                {
                    "op": "move_action",
                    "name": name,
                    "phase": phase,
                    "before": before,
                    "after": after,
                }
            )
        )
        self._update_step_plan(result)
        return result

    def replace_workflow(self, order: Sequence[str]) -> Mapping[str, Any]:
        """Atomically replace the enabled workflow order on all MPI ranks."""

        result = dict(
            self._request(
                {
                    "op": "replace_workflow",
                    "order": tuple(str(name) for name in order),
                }
            )
        )
        self._update_step_plan(result)
        return result

    def _update_step_plan(self, result: Mapping[str, Any]) -> None:
        plan = result.get("plan")
        if plan is not None:
            self._status["step_plan"] = tuple(dict(row) for row in plan)

    def field(
        self,
        name: str,
        *,
        rank: int = 0,
        selection: tuple[Any, ...] | None = None,
    ) -> Any:
        command: dict[str, Any] = {
            "op": "field",
            "name": name,
            "rank": int(rank),
        }
        if selection is not None:
            command["selection"] = tuple(selection)
        return self._request(command)

    def stats(
        self,
        name: str,
        *,
        rank: int | str = 0,
        selection: tuple[Any, ...] | None = None,
    ) -> Mapping[str, Any]:
        command: dict[str, Any] = {"op": "stats", "name": name, "rank": rank}
        if selection is not None:
            command["selection"] = tuple(selection)
        return dict(self._request(command))

    def close(self) -> None:
        if self._connection is not None:
            try:
                self._request({"op": "close"})
            finally:
                self._cleanup()

    def __enter__(self) -> "PICAMNotebookSession":
        return self.start()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        try:
            self.close()
        except BaseException:
            if exc_type is None:
                raise

    def _request(
        self,
        command: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> Any:
        with self._request_lock:
            if self._connection is None:
                raise RuntimeError("PI-CAM Notebook session is not running")
            try:
                self._connection.send(command)
                return self._unwrap(
                    self._receive(
                        self.request_timeout if timeout is None else float(timeout)
                    )
                )
            except (BrokenPipeError, EOFError, OSError, TimeoutError):
                # A timed-out or disconnected collective cannot be safely reused.
                # Tear down the MPI/PBS worker now instead of leaving hundreds of
                # ranks running until the user notices and calls qdel manually.
                self._abort()
                raise

    def _receive(self, timeout: float) -> Any:
        assert self._connection is not None
        if not self._connection.poll(timeout):
            raise TimeoutError(f"PI-CAM worker timed out after {timeout}s\n{self._log_tail()}")
        return self._connection.recv()

    @staticmethod
    def _unwrap(response: Any) -> Any:
        if not isinstance(response, dict) or response.get("status") not in {"ok", "error"}:
            raise PICAMNotebookError(f"invalid PI-CAM worker response: {response!r}")
        if response["status"] == "error":
            raise PICAMNotebookError(str(response.get("error", "unknown worker error")))
        return response.get("result")

    def _environment(self) -> dict[str, str]:
        command = f"source {shlex.quote(str(self.env_script))} >/dev/null 2>&1 && env -0"
        raw = subprocess.check_output(["bash", "-c", command], env=os.environ)
        environment: dict[str, str] = {}
        for entry in raw.split(b"\0"):
            if entry and b"=" in entry:
                key, value = entry.split(b"=", 1)
                environment[os.fsdecode(key)] = os.fsdecode(value)
        environment = mpi_loader_environment(environment)
        manifest = self.config.native_manifest
        if manifest is not None:
            payload = json.loads(manifest.read_text())
            math_library = payload.get("intel_math_library")
            if math_library:
                math_path = Path(str(math_library)).resolve()
                if not math_path.is_file():
                    raise FileNotFoundError(
                        f"PI-CAM Intel math runtime does not exist: {math_path}"
                    )
                existing = environment.get("LD_PRELOAD", "")
                entries = [item for item in existing.split(":") if item]
                if str(math_path) not in entries:
                    entries.insert(0, str(math_path))
                environment["LD_PRELOAD"] = ":".join(entries)
        return environment

    def _launch_mode(self, environment: Mapping[str, str]) -> str:
        if self.launch_mode != "auto":
            return self.launch_mode
        if environment.get("PBS_NODEFILE"):
            return "local"
        short_host = socket.gethostname().split(".", 1)[0]
        return "pbs" if short_host.startswith("derecho") else "local"

    def _submit_pbs(self, command: Sequence[str], environment: Mapping[str, str]) -> None:
        if not self.pbs_account:
            raise PICAMNotebookError(
                "no PBS account is configured; pass account=... to freecam.Driver, "
                "set PBS_ACCOUNT_DERECHO, or provide a CESM reference case whose "
                "env_batch.xml defines CHARGE_ACCOUNT or PROJECT"
            )
        script = self.run_dir / f".pi-cam-notebook-{secrets.token_hex(6)}.pbs"
        nodes = (self.config.mpi_size + 127) // 128
        rendered = " ".join(shlex.quote(item) for item in command)
        preload = environment.get("LD_PRELOAD")
        preload_export = (
            f"export LD_PRELOAD={shlex.quote(preload)}\n" if preload else ""
        )
        script.write_text(
            "#!/bin/bash\n"
            "#PBS -N pi-cam-nb\n"
            f"#PBS -A {self.pbs_account}\n"
            f"#PBS -q {self.pbs_queue}\n"
            f"#PBS -l select={nodes}:ncpus=64:mpiprocs=128:ompthreads=1:"
            f"mem={self.pbs_memory_per_node}\n"
            f"#PBS -l walltime={self.pbs_walltime}\n"
            "#PBS -j oe\n"
            f"#PBS -o {self.log_path}\n"
            "set -euo pipefail\n"
            f"source {shlex.quote(str(self.env_script))} >/dev/null 2>&1\n"
            f"export LD_LIBRARY_PATH={shlex.quote(environment['LD_LIBRARY_PATH'])}\n"
            f"{preload_export}"
            f"export PYTHONPATH={shlex.quote(str(Path(__file__).resolve().parents[2]))}\n"
            f"exec {rendered}\n"
        )
        self._pbs_script = script
        try:
            result = subprocess.run(
                ["qsub", str(script)],
                check=True,
                capture_output=True,
                text=True,
                env=dict(environment),
            )
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or str(exc)).strip()
            raise PICAMNotebookError(
                "cannot submit PI-CAM Notebook worker "
                f"with PBS account {self.pbs_account!r}: {detail}\n"
                f"Generated script: {script}"
            ) from exc
        self._job_id = result.stdout.strip().splitlines()[-1]
        print(
            f"PI-CAM Notebook worker submitted as {self._job_id}; "
            f"waiting for {self.config.mpi_size} ranks ...",
            flush=True,
        )

    def _accept(self, listener: Listener) -> Connection:
        results: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

        def accept() -> None:
            try:
                results.put((True, listener.accept()))
            except BaseException as error:
                results.put((False, error))

        threading.Thread(target=accept, daemon=True).start()
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            try:
                ok, result = results.get(timeout=0.1)
            except queue.Empty:
                if self._process is not None and self._process.poll() is not None:
                    raise PICAMNotebookError(
                        f"PI-CAM worker exited with {self._process.returncode}\n{self._log_tail()}"
                    )
                continue
            if not ok:
                raise PICAMNotebookError(f"cannot accept PI-CAM worker: {result}")
            return result
        raise TimeoutError(f"PI-CAM worker did not connect\n{self._log_tail()}")

    def _log_tail(self) -> str:
        if not self.log_path.is_file():
            return f"worker log not created: {self.log_path}"
        return "\n".join(self.log_path.read_text(errors="replace").splitlines()[-80:])

    def _abort(self) -> None:
        if self._process is not None and self._process.poll() is None:
            try:
                os.killpg(self._process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        if self._job_id is not None:
            subprocess.run(["qdel", self._job_id], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self._cleanup()

    def _cleanup(self) -> None:
        if self._connection is not None:
            self._connection.close()
        if self._log_handle is not None:
            self._log_handle.close()
        self._connection = None
        self._process = None
        self._job_id = None
        self._log_handle = None
