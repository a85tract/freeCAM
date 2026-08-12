"""Jupyter-friendly views over one live PI-CAM session.

The views deliberately keep all scientific state in the MPI workers.  They
request only the selected rank-local arrays needed for a profile or the small
step-plan table needed for workflow inspection.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Any, Iterator, Mapping, Sequence, TYPE_CHECKING

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

    def __str__(self) -> str:
        marker = "" if self.enabled else " [disabled]"
        return f"{self.name}{marker}"


class PICAMWorkflowView:
    """Live, iterable view of the session's current action order."""

    def __init__(self, session: "PICAMNotebookSession") -> None:
        self._session = session

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
        return actions if include_disabled else tuple(
            action for action in actions if action.enabled
        )

    def describe(self, *, include_disabled: bool = False) -> tuple[dict[str, object], ...]:
        return tuple(
            {
                "index": action.index,
                "name": action.name,
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
        """Replace the complete enabled process order on every MPI rank."""

        resolved = tuple(self._resolve(process) for process in processes)
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
    ) -> Any:
        """Install a Python ``Physics`` object or bound CAM process.

        ``workflow.insert(process)`` uses placement declared on the process.
        The FreeCESM-style ``workflow.insert(index, process)`` form is also
        supported: the new process is inserted before the action currently at
        that index.
        """

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
                # boundary_export must remain the final action, so list-style
                # append means "at the end of the executable model body".
                target = rows[-1]
                before = target.name
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

    def append(self, process: Any) -> Any:
        """Append before the required final ``boundary_export`` action."""

        return self.insert(len(self), process)

    def extend(self, processes: Sequence[Any]) -> tuple[Any, ...]:
        """Append several runtime processes and return their live handles."""

        return tuple(self.append(process) for process in processes)

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
        body = "".join(
            "<tr class='{}'><td>{}</td><td>{}</td>"
            "<td><code>{}</code></td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                "freecam-disabled" if not row.enabled else "",
                row.index,
                escape(row.name),
                escape(row.operation),
                escape(row.kind),
                escape(row.control_owner),
                escape(row.implementation),
            )
            for row in rows
        )
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
            "<th>#</th><th>process</th><th>operation</th>"
            "<th>kind</th><th>control</th><th>implementation</th>"
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
}


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
            raise AttributeError(name) from exc

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

    def __delattr__(self, name: str) -> None:
        if name.startswith("_"):
            raise AttributeError(name)
        self._session.fields[name].delete()

    def __dir__(self) -> list[str]:
        fields = self._session.status.get("fields", {})
        names = {
            alias
            for alias in ("T", "u", "v", "q", *fields)
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
        spec = self._spec(variable, constituent=constituent)
        values = self.profile(
            variable,
            rank=rank,
            constituent=constituent,
        )
        if ax is None:
            _, ax = plt.subplots(figsize=(3.2, 4.2))
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

    def plot(
        self,
        *,
        rank: int = 0,
        variables: Sequence[str] = default_variables,
        columns: int | None = None,
        figsize: tuple[float, float] = (9.0, 6.5),
        axes: Any = None,
        **plot_kwargs: Any,
    ) -> tuple[Any, Any]:
        """Plot selected rank-local profiles in an automatically sized grid."""

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
        if variable not in fields:
            raise KeyError(
                f"unknown profile variable {variable!r}; use T, u, v, q, "
                "or a canonical StatePool field name"
            )
        units = str(fields[variable].get("units", "1"))
        suffix = "" if units == "1" else f" ({units})"
        return _ProfileSpec(
            str(variable),
            f"{variable}{suffix}",
            constituent=constituent,
        )


__all__ = [
    "PICAMStateView",
    "PICAMWorkflowAction",
    "PICAMWorkflowView",
]
