"""Jupyter-friendly views over one live PI-CAM session.

The views deliberately keep all scientific state in the MPI workers.  They
request only the selected rank-local arrays needed for a profile or the small
step-plan table needed for workflow inspection.
"""

from __future__ import annotations

from io import BytesIO
from dataclasses import dataclass
from html import escape
from typing import TYPE_CHECKING, Any, Iterator, Mapping, Sequence

import numpy as np

from .expressions import DistributedOperand

if TYPE_CHECKING:
    from .session import PICAMNotebookSession


@dataclass(frozen=True, slots=True)
class PICAMWorkflowAction:
    """One row in the live, Python-owned PI-CAM workflow."""

    index: int
    phase: str
    name: str
    operation: str
    kind: str
    enabled: bool
    native_id: int | None = None
    control_owner: str = "python"
    implementation: str = "python"

    @property
    def qualified_name(self) -> str:
        return f"{self.phase}.{self.name}"

    @property
    def display_name(self) -> str:
        """Readable public name for a source leaf implementation."""

        return self.name.removesuffix("_leaf")

    def __str__(self) -> str:
        marker = "" if self.enabled else " [disabled]"
        return f"{self.display_name}{marker}"


class PICAMWorkflowView:
    """Live view of the model's scientific process order.

    CAM control, clock, I/O, and StatePool housekeeping actions still execute,
    but they are intentionally absent from the default user-facing sequence.
    Use :attr:`debug` when those implementation details are needed.
    """

    _SCIENTIFIC_KINDS = frozenset(
        {
            "scheme",
            "coupling",
            "dynamics",
            "python_process",
            "runtime_fortran_process",
            "runtime_catalog_process",
        }
    )

    def __init__(
        self,
        session: "PICAMNotebookSession",
        *,
        include_internal: bool = False,
    ) -> None:
        self._session = session
        self._include_internal = bool(include_internal)

    @property
    def debug(self) -> "PICAMWorkflowView":
        """The complete source-faithful workflow, including control and I/O."""

        return PICAMWorkflowView(self._session, include_internal=True)

    @property
    def all(self) -> "PICAMWorkflowView":
        """Alias for :attr:`debug`."""

        return self.debug

    @classmethod
    def is_scientific(cls, row: Mapping[str, Any] | PICAMWorkflowAction) -> bool:
        kind = row.kind if isinstance(row, PICAMWorkflowAction) else row.get("kind")
        return str(kind) in cls._SCIENTIFIC_KINDS

    def actions(
        self, *, include_disabled: bool = False
    ) -> tuple[PICAMWorkflowAction, ...]:
        rows = self._session.status.get("step_plan", ())
        actions = tuple(
            PICAMWorkflowAction(
                index=int(row.get("index", index)),
                phase=str(row["phase"]),
                name=str(row["name"]),
                operation=str(row["operation"]),
                kind=str(row["kind"]),
                enabled=bool(row["enabled"]),
                native_id=(
                    None if row.get("native_id") is None else int(row["native_id"])
                ),
                control_owner=str(row.get("control_owner", "python")),
                implementation=str(row.get("implementation", "python")),
            )
            for index, row in enumerate(rows)
        )
        if not self._include_internal:
            actions = tuple(action for action in actions if self.is_scientific(action))
        return actions if include_disabled else tuple(
            action for action in actions if action.enabled
        )

    def describe(self, *, include_disabled: bool = False) -> tuple[dict[str, object], ...]:
        return tuple(
            {
                "index": action.index,
                "name": (
                    action.name if self._include_internal else action.display_name
                ),
                "operation": action.operation,
                "kind": action.kind,
                "native_id": action.native_id,
                "enabled": action.enabled,
                "control_owner": action.control_owner,
                "implementation": action.implementation,
            }
            for action in self.actions(include_disabled=include_disabled)
        )

    def expand(self) -> tuple[Mapping[str, Any], ...]:
        """Expose every validated leaf process without naming source phases."""

        results = (
            *self._session.expand_cam_run1_leaves(),
            *self._session.expand_cam_run2_run4_leaves(),
        )
        return tuple(results)

    def __iter__(self) -> Iterator[Any]:
        return iter(self[:])

    def __len__(self) -> int:
        return len(self.actions())

    def __getitem__(self, name_or_index: str | int | slice) -> Any:
        """Return one live process handle by workflow index or name."""

        if isinstance(name_or_index, slice):
            return [self._process(action) for action in self.actions()[name_or_index]]
        if isinstance(name_or_index, int):
            return self._process(self.actions()[name_or_index])
        name = str(name_or_index)
        matches = tuple(
            action
            for action in self.actions(include_disabled=True)
            if name in {action.name, action.operation, action.qualified_name}
        )
        if len(matches) != 1:
            raise KeyError(f"workflow action {name!r} is unknown or ambiguous")
        return self._process(matches[0])

    def __setitem__(self, index: int | slice, value: Any) -> None:
        """Apply normal list assignment as one atomic remote reorder.

        The resulting list must still contain every enabled action exactly
        once.  This makes ``workflow[:] = new_order`` safe: omission does not
        silently disable a CAM process.
        """

        order = self[:]
        order[index] = value
        self.replace(order)

    def index(self, process: str | Any) -> int:
        """Return the current index of one enabled process."""

        target = self._resolve(process)
        for index, item in enumerate(self[:]):
            if item.qualified_name == target.qualified_name:
                return index
        raise ValueError(f"{target.qualified_name!r} is not enabled in workflow")

    def count(self, process: str | Any) -> int:
        """Return zero or one; source processes cannot be duplicated."""

        try:
            self.index(process)
        except (KeyError, ValueError):
            return 0
        return 1

    def replace(self, processes: Any) -> Mapping[str, Any]:
        """Replace the visible process order on every MPI rank.

        In the normal scientific view, hidden control and I/O actions retain
        their slots while the requested scientific processes are reordered
        around them.  The debug view replaces the complete enabled order.
        """

        resolved = tuple(self._resolve(process) for process in processes)
        if not self._include_internal:
            full_rows = tuple(
                PICAMWorkflowAction(
                    index=int(row.get("index", index)),
                    phase=str(row["phase"]),
                    name=str(row["name"]),
                    operation=str(row["operation"]),
                    kind=str(row["kind"]),
                    enabled=bool(row["enabled"]),
                    native_id=(
                        None
                        if row.get("native_id") is None
                        else int(row["native_id"])
                    ),
                    control_owner=str(row.get("control_owner", "python")),
                    implementation=str(row.get("implementation", "python")),
                )
                for index, row in enumerate(
                    self._session.status.get("step_plan", ())
                )
                if bool(row.get("enabled", True))
            )
            visible = tuple(row for row in full_rows if self.is_scientific(row))
            if len(resolved) != len(visible):
                raise ValueError(
                    "workflow must contain every visible scientific process "
                    "exactly once"
                )
            replacements = iter(resolved)
            resolved = tuple(
                next(replacements) if self.is_scientific(row) else self._process(row)
                for row in full_rows
            )
        return self._session.replace_workflow(
            tuple(process.qualified_name for process in resolved)
        )

    def move(
        self,
        process: str | Any,
        *,
        before: str | Any | None = None,
        after: str | Any | None = None,
    ) -> Mapping[str, Any]:
        """Move a process anywhere in the complete Python workflow."""

        if (before is None) == (after is None):
            raise ValueError("provide exactly one of before= or after=")
        source = self._resolve(process)
        target = self._resolve(before if before is not None else after)
        return self._session.move_action(
            source.name,
            phase=source.phase,
            before=target.qualified_name if before is not None else None,
            after=target.qualified_name if after is not None else None,
        )

    def enable(self, process: str | Any) -> Mapping[str, Any]:
        """Enable one process in subsequent complete steps."""

        return self._resolve(process).enable()

    def disable(self, process: str | Any) -> Mapping[str, Any]:
        """Disable one process in subsequent complete steps."""

        return self._resolve(process).disable()

    def _resolve(self, process: str | Any | None) -> Any:
        if process is None:
            raise ValueError("workflow process cannot be None")
        if hasattr(process, "name") and hasattr(process, "phase"):
            return process
        try:
            return self[str(process)]
        except KeyError:
            return self._session.physics.process(str(process))

    def _process(self, action: PICAMWorkflowAction) -> Any:
        factory = getattr(self._session, "workflow_action", None)
        if callable(factory):
            return factory(
                action.name,
                phase=action.phase,
                kind=action.kind,
            )
        physics = getattr(self._session, "physics", None)
        if physics is None:
            return action
        return physics.process(action.name)

    def insert(
        self,
        process_or_index: Any,
        process: Any | None = None,
        *,
        before: str | None = None,
        after: str | None = None,
    ) -> None:
        """Insert a process and return ``None``, like :meth:`list.insert`.

        ``workflow.insert(process)`` uses placement declared on the process.
        The FreeCESM-style ``workflow.insert(index, process)`` form is also
        supported: the new process is inserted before the action currently at
        that index.  Retrieve its live handle afterwards with
        ``workflow[process_name]``; use :meth:`install` when an immediate
        return handle is more convenient.
        """

        self.install(
            process_or_index,
            process,
            before=before,
            after=after,
        )

    def install(
        self,
        process_or_index: Any,
        process: Any | None = None,
        *,
        before: str | None = None,
        after: str | None = None,
    ) -> Any:
        """Install a process and return its live control handle."""

        if process is None:
            candidate = process_or_index
        else:
            index = int(process_or_index)
            rows = self.actions()
            if not 0 <= index <= len(rows):
                raise IndexError(index)
            candidate = process
            if before is not None or after is not None:
                raise ValueError(
                    "index placement cannot be combined with before= or after="
                )
            if index == len(rows):
                if not rows:
                    raise ValueError("cannot place a process in an empty workflow")
                target = rows[-1]
                after = target.name
            else:
                target = rows[index]
                before = target.name
        installer = getattr(candidate, "_install", None)
        if callable(installer):
            return installer(
                self._session,
                before=before,
                after=after,
            )
        existing = self._resolve(candidate)
        if existing.enabled:
            raise ValueError(
                f"workflow already contains {existing.qualified_name!r}"
            )
        existing.enable()
        try:
            self.move(existing, before=before, after=after)
        except BaseException:
            existing.disable()
            raise
        return existing

    def append(self, process: Any) -> None:
        """Append after the final visible process and return ``None``."""

        self.install(len(self), process)

    def extend(self, processes: Sequence[Any]) -> None:
        """Append several processes and return ``None``, like :meth:`list.extend`."""

        for process in processes:
            self.append(process)

    def pop(self, index: int = -1) -> Any:
        """Remove a runtime process or disable an original CAM process."""

        process = self[index]
        if process.operation in {
            "boundary_import",
            "advance_timestep",
            "boundary_export",
        }:
            raise ValueError(
                f"required CAM control action {process.operation!r} cannot be popped"
            )
        if process.kind in {
            "python_process",
            "runtime_fortran_process",
            "runtime_catalog_process",
        }:
            process.remove()
        else:
            process.disable()
        return process

    def remove(self, process: str | Any) -> None:
        """List-style removal with the same safety rules as :meth:`pop`."""

        self.pop(self.index(process))

    def _repr_html_(self) -> str:
        rows = self.actions(include_disabled=True)
        if self._include_internal:
            body = "".join(
                "<tr class='{}'><td>{}</td><td>{}</td>"
                "<td><code>{}</code></td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                    "freecam-disabled" if not row.enabled else "",
                    row.index,
                    escape(row.display_name),
                    escape(row.operation),
                    escape(row.kind),
                    escape(row.control_owner),
                    escape(row.implementation),
                )
                for row in rows
            )
            heading = (
                "<th>#</th><th>process</th><th>operation</th>"
                "<th>kind</th><th>control</th><th>implementation</th>"
            )
        else:
            body = "".join(
                "<tr class='{}'><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                    "freecam-disabled" if not row.enabled else "",
                    index,
                    escape(row.display_name),
                    escape(row.implementation),
                )
                for index, row in enumerate(rows)
            )
            heading = "<th>#</th><th>process</th><th>implementation</th>"
        return (
            "<style>"
            ".freecam-workflow{border-collapse:collapse;width:100%;font-size:.9em}"
            ".freecam-workflow th,.freecam-workflow td{padding:.35rem .55rem;"
            "border-bottom:1px solid var(--jp-border-color2,#ddd);text-align:left}"
            ".freecam-workflow th{position:sticky;top:0;background:"
            "var(--jp-layout-color1,#fff)}"
            ".freecam-workflow .freecam-disabled{opacity:.45;text-decoration:line-through}"
            "</style>"
            "<table class='freecam-workflow'><thead><tr>"
            f"{heading}"
            "</tr></thead><tbody>"
            f"{body}</tbody></table>"
        )


@dataclass(frozen=True, slots=True)
class _ProfileSpec:
    field: str
    label: str
    scale: float = 1.0
    constituent: int | None = None


_PROFILE_SPECS: dict[str, _ProfileSpec] = {
    "t": _ProfileSpec("phys_state.t", "temperature (K)"),
    "temperature": _ProfileSpec("phys_state.t", "temperature (K)"),
    "u": _ProfileSpec("phys_state.u", "zonal wind u (m/s)"),
    "v": _ProfileSpec("phys_state.v", "meridional wind v (m/s)"),
    "q": _ProfileSpec(
        "phys_state.q",
        "water-vapor mixing ratio (g/kg)",
        scale=1.0e3,
        constituent=0,
    ),
    "omega": _ProfileSpec(
        "phys_state.omega",
        "vertical pressure velocity omega (Pa/s)",
    ),
    "pmid": _ProfileSpec(
        "phys_state.pmid",
        "midpoint pressure (Pa)",
    ),
}


class PICAMProfilePlot:
    """Reusable latest-value or history plot backed by the live StatePool."""

    def __init__(
        self,
        state: "PICAMStateView",
        *,
        mode: str,
        rank: int,
        variables: Sequence[str],
        columns: int | None,
        figsize: tuple[float, float],
        axes: Any,
        plot_kwargs: Mapping[str, Any],
    ) -> None:
        selected_mode = str(mode).strip().lower()
        if selected_mode not in {"latest", "history"}:
            raise ValueError("plot mode must be 'latest' or 'history'")
        self.state = state
        self.mode = selected_mode
        self.rank = int(rank)
        self.variables = tuple(str(variable) for variable in variables)
        self.plot_kwargs = dict(plot_kwargs)
        self.figure, self.axes = state._plot_grid(
            variables=self.variables,
            columns=columns,
            figsize=figsize,
            axes=axes,
        )
        # The reusable object owns rendering through _repr_png_. Removing the
        # raw Figure from pyplot prevents the inline backend from showing a
        # second, stale copy at the end of the cell.
        import matplotlib.pyplot as plt

        plt.close(self.figure)
        self.snapshots: list[dict[str, Any]] = []
        self._last_profiles: tuple[np.ndarray, ...] | None = None
        initial_label = self.plot_kwargs.pop("label", None)
        if self.mode == "latest":
            self.refresh(label=initial_label)
        else:
            self.capture(label=initial_label, force=True)

    def _profiles(self) -> tuple[np.ndarray, ...]:
        return tuple(
            self.state.profile(variable, rank=self.rank)
            for variable in self.variables
        )

    def _changed(self, profiles: tuple[np.ndarray, ...]) -> bool:
        return self._last_profiles is None or any(
            not np.array_equal(current, previous)
            for current, previous in zip(profiles, self._last_profiles)
        )

    def _draw(
        self,
        profiles: tuple[np.ndarray, ...],
        *,
        clear: bool,
        label: str | None,
        plot_kwargs: Mapping[str, Any],
    ) -> None:
        if clear:
            for axis in self.axes.flat:
                axis.clear()
                axis.set_visible(True)
        options = {**self.plot_kwargs, **dict(plot_kwargs)}
        if label is not None:
            options["label"] = label
        for axis, variable, values in zip(
            self.axes.flat, self.variables, profiles
        ):
            self.state._draw_profile_values(
                variable,
                values,
                ax=axis,
                plot_kwargs=options,
            )
        for axis in self.axes.flat[len(self.variables) :]:
            axis.set_visible(False)
        status = self.state._session.status
        self.figure.suptitle(
            f"PI-CAM rank {self.rank} mean profiles ({self.mode}) · "
            f"step {status.get('step', '?')} · date {status.get('date', '?')}"
        )
        self.figure.tight_layout()

    def refresh(
        self, *, label: str | None = None, **plot_kwargs: Any
    ) -> "PICAMProfilePlot":
        """Replace every curve with current StatePool values."""

        if self.mode != "latest":
            raise TypeError("refresh() is available only in latest mode")
        profiles = self._profiles()
        status = self.state._session.status
        self._draw(
            profiles,
            clear=True,
            label=label,
            plot_kwargs=plot_kwargs,
        )
        self._last_profiles = tuple(values.copy() for values in profiles)
        self.snapshots[:] = [
            {
                "step": status.get("step"),
                "date": status.get("date"),
                "label": label,
            }
        ]
        return self

    def capture(
        self,
        *,
        label: str | None = None,
        force: bool = False,
        **plot_kwargs: Any,
    ) -> "PICAMProfilePlot":
        """Append current values when they differ from the last snapshot."""

        if self.mode != "history":
            raise TypeError("capture() is available only in history mode")
        profiles = self._profiles()
        if not force and not self._changed(profiles):
            return self
        status = self.state._session.status
        if label is None:
            label = f"step {status.get('step', '?')}"
            if self.snapshots and self.snapshots[-1]["step"] == status.get("step"):
                label += f" · update {len(self.snapshots)}"
        self._draw(
            profiles,
            clear=False,
            label=label,
            plot_kwargs=plot_kwargs,
        )
        self._last_profiles = tuple(values.copy() for values in profiles)
        self.snapshots.append(
            {
                "step": status.get("step"),
                "date": status.get("date"),
                "label": label,
            }
        )
        return self

    def _repr_png_(self) -> bytes:
        # IPython calls this for every display(plot), so latest values are
        # fetched at display time rather than when the object was created.
        if self.mode == "latest":
            self.refresh()
        else:
            self.capture()
        output = BytesIO()
        self.figure.savefig(output, format="png", bbox_inches="tight")
        return output.getvalue()

    def __iter__(self) -> Iterator[Any]:
        yield self.figure
        yield self.axes

    def __repr__(self) -> str:
        return (
            f"PICAMProfilePlot(mode={self.mode!r}, rank={self.rank}, "
            f"variables={self.variables!r}, snapshots={len(self.snapshots)})"
        )


class PICAMStateView:
    """Rank-local profile and summary UI for the live MPI StatePool."""

    default_variables = ("T", "u", "v", "q")

    def __init__(self, session: "PICAMNotebookSession") -> None:
        object.__setattr__(self, "_session", session)

    def __getattr__(self, name: str) -> Any:
        aliases = {
            "T": "phys_state.t",
            "u": "phys_state.u",
            "v": "phys_state.v",
            "q": "phys_state.q",
        }
        try:
            return self._session.fields[aliases.get(name, name)]
        except KeyError as exc:
            raise AttributeError(str(exc)) from exc

    def _field_name(self, name: str) -> str:
        aliases = {
            "T": "phys_state.t",
            "u": "phys_state.u",
            "v": "phys_state.v",
            "q": "phys_state.q",
        }
        return aliases.get(str(name), str(name))

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        # Delayed import avoids a facade -> session -> ui -> facade cycle.
        from .facade import Variable

        # Python writes the result of ``state.T += value`` back through this
        # method after ``T.__iadd__`` has already changed the remote field.
        # Accept that exact field handle without issuing the edit twice.
        if getattr(value, "session", None) is self._session:
            try:
                canonical = self._session.fields._resolve(self._field_name(name))
            except KeyError:
                canonical = None
            if (
                canonical is not None
                and getattr(value, "name", None) == canonical
                and getattr(value, "selection", None) is None
            ):
                return

        if isinstance(value, DistributedOperand):
            if value._expression_session is not self._session:
                raise ValueError("a distributed expression cannot mix different models")
            try:
                canonical = self._session.fields._resolve(self._field_name(name))
            except KeyError:
                raise TypeError(
                    "assign a freecam.Variable before writing a distributed "
                    f"expression into new field {name!r}"
                ) from None
            if (
                getattr(value, "name", None) == canonical
                and getattr(value, "selection", None) is None
            ):
                # Augmented assignment writes the field reference back after
                # ``__iadd__``/etc. already edited all MPI ranks.
                return
            self._session.assign_expression(
                canonical,
                value._expression_payload,
            )
            return

        if isinstance(value, np.ndarray):
            self._session.fields.create_array(name, value)
            return
        if not isinstance(value, Variable):
            raise TypeError(
                "assign a NumPy array for rank-independent state or "
                "freecam.Variable(dims=(...), initial=...) for a distributed field"
            )
        self._session.fields.create(
            name,
            dims=value.dims,
            dtype=value.dtype,
            units=value.units,
            initial=value.initial,
            writable=value.writable,
            restart=value.restart,
            aliases=value.aliases,
            standard_name=value.standard_name,
        )

    def __getitem__(self, name: str) -> Any:
        return self._session.fields[self._field_name(name)]

    def __setitem__(self, name: str, value: Any) -> None:
        self.__setattr__(str(name), value)

    def __delitem__(self, name: str) -> None:
        self._session.fields[self._field_name(name)].delete()

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self.keys())

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        try:
            self._session.fields._resolve(self._field_name(name))
        except KeyError:
            return False
        return True

    def keys(self) -> tuple[str, ...]:
        return tuple(self._session.status.get("fields", ()))

    def values(self) -> tuple[Any, ...]:
        return tuple(self._session.fields[name] for name in self.keys())

    def items(self) -> tuple[tuple[str, Any], ...]:
        return tuple((name, self._session.fields[name]) for name in self.keys())

    def describe(self) -> tuple[Mapping[str, Any], ...]:
        """Return one compact metadata row per StatePool field."""

        fields = self._session.status.get("fields", {})
        return tuple(
            {
                "name": name,
                "shape": tuple(metadata.get("shape", ())),
                "dtype": metadata.get("dtype"),
                "units": metadata.get("units", "1"),
                "dimensions": tuple(metadata.get("dimensions", ())),
                "writable": bool(metadata.get("writable", True)),
                "dynamic": bool(metadata.get("dynamic", False)),
            }
            for name, metadata in fields.items()
        )

    def create(
        self,
        name: str,
        *,
        like: str | Any | None = None,
        dims: Sequence[str] | None = None,
        units: str | None = None,
        initial: float | int = 0.0,
        dtype: str | None = None,
        writable: bool = True,
        restart: bool = True,
        aliases: Sequence[str] = (),
        standard_name: str | None = None,
    ) -> Any:
        """Create a distributed field, optionally copying another field's layout.

        ``state.create("tracer", like="T")`` resolves the rank-local CAM
        dimensions on every MPI rank, avoiding repetition of implementation
        dimensions such as ``pcols`` and ``chunks`` in Notebook code.
        """

        metadata: Mapping[str, Any] = {}
        if like is not None:
            reference = (
                getattr(self, like)
                if isinstance(like, str)
                else like
            )
            metadata = dict(getattr(reference, "metadata", {}))
            if not metadata:
                raise TypeError("like= must name or reference a StatePool field")
        resolved_dims = (
            tuple(str(item) for item in dims)
            if dims is not None
            else tuple(str(item) for item in metadata.get("dimensions", ()))
        )
        if not resolved_dims and not (
            dims is not None and len(tuple(dims)) == 0
        ):
            raise ValueError("provide dims= or like= for a distributed field")
        self._session.fields.create(
            str(name),
            dims=resolved_dims,
            dtype=str(dtype or metadata.get("dtype", "float64")),
            units=str(units if units is not None else metadata.get("units", "1")),
            initial=initial,
            writable=writable,
            restart=restart,
            aliases=tuple(str(item) for item in aliases),
            standard_name=standard_name,
        )
        return self._session.fields[str(name)]

    @property
    def aliases(self) -> Mapping[str, str]:
        """Safe short-name mapping currently available in this Notebook."""

        return {
            "T": "phys_state.t",
            "u": "phys_state.u",
            "v": "phys_state.v",
            "q": "phys_state.q",
            **dict(self._session.fields.aliases),
        }

    def alias(
        self,
        alias: str,
        field: str | Any,
        *,
        replace: bool = False,
    ) -> Any:
        """Give a canonical field an explicit, client-side attribute name."""

        if hasattr(type(self), str(alias)):
            raise ValueError(
                f"field alias {alias!r} conflicts with the State API; "
                "use state['canonical.name'] instead"
            )
        target = getattr(field, "name", field)
        return self._session.fields.alias(
            str(alias), str(target), replace=replace
        )

    def zeros_like(self, name: str, like: str | Any, **metadata: Any) -> Any:
        """Create a zero-filled distributed field with another field's layout."""

        return self.create(name, like=like, initial=0.0, **metadata)

    def ones_like(self, name: str, like: str | Any, **metadata: Any) -> Any:
        """Create a one-filled distributed field with another field's layout."""

        return self.create(name, like=like, initial=1.0, **metadata)

    def __delattr__(self, name: str) -> None:
        if name.startswith("_"):
            raise AttributeError(name)
        self._session.fields[name].delete()

    def __dir__(self) -> list[str]:
        fields = self._session.status.get("fields", {})
        names = {
            alias
            for alias in ("T", "u", "v", "q", *fields, *self.aliases)
            if str(alias).isidentifier()
        }
        return sorted(set(super().__dir__()) | names)

    def profile(
        self,
        variable: str = "T",
        *,
        rank: int = 0,
        constituent: int | None = None,
    ) -> np.ndarray:
        """Return one horizontal-mean vertical profile from a selected rank.

        Native ``(pcols, pver, chunks)`` fields use ``phys_state.ncol`` to
        discard padded columns.  Four-dimensional constituent fields require
        a constituent index; ``q`` defaults to water vapor (index zero).
        """

        spec = self._spec(variable, constituent=constituent)
        values = np.asarray(self._session.field(spec.field, rank=rank))
        selected_constituent = (
            spec.constituent if constituent is None else int(constituent)
        )
        if values.ndim == 4:
            if selected_constituent is None:
                raise ValueError(
                    f"field {spec.field!r} has a constituent axis; provide constituent="
                )
            if not 0 <= selected_constituent < values.shape[2]:
                raise IndexError(
                    f"constituent {selected_constituent} is outside 0..{values.shape[2] - 1}"
                )
            values = values[:, :, selected_constituent, :]
        if values.ndim == 3:
            ncol = np.asarray(
                self._session.field("phys_state.ncol", rank=rank), dtype=np.int64
            ).reshape(-1)
            if ncol.size != values.shape[-1]:
                raise ValueError(
                    "phys_state.ncol does not match the field's chunk dimension"
                )
            columns = [
                values[: int(count), :, chunk]
                for chunk, count in enumerate(ncol)
                if int(count) > 0
            ]
            if not columns:
                raise ValueError(f"rank {rank} contains no active physics columns")
            values = np.concatenate(columns, axis=0)
        if values.ndim == 2:
            result = np.mean(values, axis=0, dtype=np.float64)
        elif values.ndim == 1:
            result = np.asarray(values, dtype=np.float64)
        else:
            raise ValueError(
                f"field {spec.field!r} with shape {values.shape} is not a vertical profile"
            )
        return np.asarray(result * spec.scale, dtype=np.float64)

    def summary(
        self,
        *,
        rank: int = 0,
        variables: tuple[str, ...] = default_variables,
    ) -> str:
        """Return a compact FreeCESM-style summary for one MPI rank."""

        status = self._session.status
        parts = [
            f"rank={rank}",
            f"step={status.get('step', '?')}",
            f"date={status.get('date', '?')}",
        ]
        for variable in variables:
            profile = self.profile(variable, rank=rank)
            key = variable.lower()
            if key in {"t", "temperature"}:
                parts.append(f"T_mean={profile.mean():.2f} K")
            elif key == "q":
                parts.append(f"q_mean={profile.mean():.3f} g/kg")
            elif key in {"u", "v"}:
                parts.append(f"{key}_mean={profile.mean():.2f} m/s")
            else:
                parts.append(f"{variable}_mean={profile.mean():.6g}")
        return "   ".join(parts)

    def plot_profile(
        self,
        variable: str = "T",
        *,
        rank: int = 0,
        constituent: int | None = None,
        ax: Any = None,
        **plot_kwargs: Any,
    ) -> Any:
        """Plot one rank-local horizontal-mean profile on a Matplotlib axis."""

        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "PI-CAM plotting requires the notebook extra: "
                "uv sync --extra notebook"
            ) from exc
        values = self.profile(
            variable,
            rank=rank,
            constituent=constituent,
        )
        if ax is None:
            _, ax = plt.subplots(figsize=(3.2, 4.2))
        return self._draw_profile_values(
            variable,
            values,
            ax=ax,
            constituent=constituent,
            plot_kwargs=plot_kwargs,
        )

    def _draw_profile_values(
        self,
        variable: str,
        values: np.ndarray,
        *,
        ax: Any,
        constituent: int | None = None,
        plot_kwargs: Mapping[str, Any] | None = None,
    ) -> Any:
        """Draw already-fetched profile values without another MPI request."""

        spec = self._spec(variable, constituent=constituent)
        plot_kwargs = dict(plot_kwargs or {})
        plot_kwargs.setdefault("marker", "o")
        plot_kwargs.setdefault("markersize", 3.5)
        ax.plot(values, np.arange(values.size), **plot_kwargs)
        if not ax.yaxis_inverted():
            ax.invert_yaxis()
        ax.set_xlabel(spec.label)
        ax.set_ylabel("model level (0 = top)")
        ax.spines[["top", "right"]].set_visible(False)
        if ax.get_legend_handles_labels()[1]:
            ax.legend(frameon=False)
        return ax

    def _plot_grid(
        self,
        *,
        variables: Sequence[str],
        columns: int | None,
        figsize: tuple[float, float],
        axes: Any,
    ) -> tuple[Any, Any]:
        """Create or validate axes shared by snapshot and reusable plots."""

        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "PI-CAM plotting requires the notebook extra: "
                "uv sync --extra notebook"
            ) from exc
        selected = tuple(str(variable) for variable in variables)
        if not selected:
            raise ValueError("variables must contain at least one field")
        if columns is None:
            columns = min(3, int(np.ceil(np.sqrt(len(selected)))))
        if int(columns) < 1:
            raise ValueError("columns must be positive")
        ncols = min(int(columns), len(selected))
        nrows = (len(selected) + ncols - 1) // ncols
        expected_shape = (nrows, ncols)
        if axes is None:
            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=figsize,
                sharey=True,
                squeeze=False,
            )
        else:
            axes = np.asarray(axes, dtype=object)
            if axes.ndim == 0:
                axes = axes.reshape(1, 1)
            elif axes.ndim == 1 and nrows == 1:
                axes = axes.reshape(1, -1)
            elif axes.ndim == 1 and ncols == 1:
                axes = axes.reshape(-1, 1)
            if axes.shape != expected_shape:
                raise ValueError(f"axes must have shape {expected_shape}")
            fig = axes.flat[0].figure
        return fig, axes

    def plot(
        self,
        *,
        rank: int = 0,
        variables: Sequence[str] = default_variables,
        columns: int | None = None,
        figsize: tuple[float, float] = (9.0, 6.5),
        axes: Any = None,
        mode: str = "snapshot",
        **plot_kwargs: Any,
    ) -> tuple[Any, Any] | PICAMProfilePlot:
        """Plot a snapshot or create a reusable latest/history display."""

        selected = tuple(str(variable) for variable in variables)
        selected_mode = str(mode).strip().lower()
        if selected_mode != "snapshot":
            return PICAMProfilePlot(
                self,
                mode=selected_mode,
                rank=rank,
                variables=selected,
                columns=columns,
                figsize=figsize,
                axes=axes,
                plot_kwargs=plot_kwargs,
            )
        fig, axes = self._plot_grid(
            variables=selected,
            columns=columns,
            figsize=figsize,
            axes=axes,
        )
        for axis, variable in zip(axes.flat, selected):
            self.plot_profile(variable, rank=rank, ax=axis, **plot_kwargs)
        for axis in axes.flat[len(selected) :]:
            axis.set_visible(False)
        status = self._session.status
        fig.suptitle(
            f"PI-CAM rank {rank} mean profiles · step {status.get('step', '?')} · "
            f"date {status.get('date', '?')}"
        )
        fig.tight_layout()
        return fig, axes

    def _spec(self, variable: str, *, constituent: int | None) -> _ProfileSpec:
        key = str(variable).lower()
        if key in _PROFILE_SPECS:
            return _PROFILE_SPECS[key]
        fields = self._session.status.get("fields", {})
        resolver = getattr(getattr(self._session, "fields", None), "_resolve", None)
        try:
            canonical = (
                str(resolver(str(variable)))
                if callable(resolver)
                else str(variable)
            )
        except KeyError:
            canonical = str(variable)
        if canonical not in fields:
            raise KeyError(
                f"unknown profile variable {variable!r}; use T, u, v, q, "
                "a unique StatePool short name, or a canonical field name"
            )
        units = str(fields[canonical].get("units", "1"))
        suffix = "" if units == "1" else f" ({units})"
        return _ProfileSpec(
            canonical,
            f"{variable}{suffix}",
            constituent=constituent,
        )


__all__ = [
    "PICAMStateView",
    "PICAMWorkflowAction",
    "PICAMWorkflowView",
]
