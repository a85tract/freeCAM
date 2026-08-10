"""Jupyter-friendly views over one live PI-CAM session.

The views deliberately keep all scientific state in the MPI workers.  They
request only the selected rank-local arrays needed for a profile or the small
step-plan table needed for workflow inspection.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Any, Iterator, Mapping, TYPE_CHECKING

import numpy as np

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

    @property
    def qualified_name(self) -> str:
        return f"{self.phase}.{self.name}"

    def __str__(self) -> str:
        marker = "" if self.enabled else " [disabled]"
        return f"{self.qualified_name}{marker}"


class PICAMWorkflowView:
    """Live, iterable view of the session's current action order."""

    def __init__(self, session: "PICAMNotebookSession") -> None:
        self._session = session

    def actions(self, *, include_disabled: bool = False) -> tuple[PICAMWorkflowAction, ...]:
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
                "phase": action.phase,
                "name": action.name,
                "operation": action.operation,
                "kind": action.kind,
                "native_id": action.native_id,
                "enabled": action.enabled,
            }
            for action in self.actions(include_disabled=include_disabled)
        )

    def __iter__(self) -> Iterator[PICAMWorkflowAction]:
        return iter(self.actions())

    def __len__(self) -> int:
        return len(self.actions())

    def insert(
        self,
        process_or_index: Any,
        process: Any | None = None,
        *,
        phase: str | None = None,
        before: str | None = None,
        after: str | None = None,
    ) -> Any:
        """Install a ``Physics`` object into the live workflow.

        ``workflow.insert(process)`` uses placement declared on the process.
        The FreeCESM-style ``workflow.insert(index, process)`` form is also
        supported: the new process is inserted before the action currently at
        that index, and its phase is inferred from that action.
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
                    raise ValueError("cannot infer a phase from an empty workflow")
                target = rows[-1]
                phase = phase or target.phase
                after = target.name
            else:
                target = rows[index]
                phase = phase or target.phase
                before = target.name
        installer = getattr(candidate, "_install", None)
        if not callable(installer):
            raise TypeError(
                "workflow.insert expects a freecam.Physics instance"
            )
        return installer(
            self._session,
            phase=phase,
            before=before,
            after=after,
        )

    def _repr_html_(self) -> str:
        rows = self.actions(include_disabled=True)
        body = "".join(
            "<tr class='{}'><td>{}</td><td><code>{}</code></td>"
            "<td>{}</td><td><code>{}</code></td><td>{}</td></tr>".format(
                "freecam-disabled" if not row.enabled else "",
                row.index,
                escape(row.phase),
                escape(row.name),
                escape(row.operation),
                escape(row.kind),
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
            "<th>#</th><th>phase</th><th>process</th><th>native operation</th>"
            "<th>kind</th></tr></thead><tbody>"
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

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        # Delayed import avoids a facade -> session -> ui -> facade cycle.
        from .facade import Variable

        if not isinstance(value, Variable):
            raise TypeError(
                "real MPI fields require a distributed definition; assign "
                "freecam.Variable(dims=(...), initial=...)"
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
        figsize: tuple[float, float] = (9.0, 6.5),
        axes: Any = None,
        **plot_kwargs: Any,
    ) -> tuple[Any, Any]:
        """Plot temperature, winds, and water vapor in a 2x2 panel."""

        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "PI-CAM plotting requires the notebook extra: "
                "uv sync --extra notebook"
            ) from exc
        if axes is None:
            fig, axes = plt.subplots(2, 2, figsize=figsize, sharey=True)
        else:
            axes = np.asarray(axes)
            if axes.shape != (2, 2):
                raise ValueError("axes must have shape (2, 2)")
            fig = axes.flat[0].figure
        for axis, variable in zip(axes.flat, self.default_variables):
            self.plot_profile(variable, rank=rank, ax=axis, **plot_kwargs)
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
