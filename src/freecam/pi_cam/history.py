"""Lazy, Notebook-friendly access to CAM history and restart NetCDF files."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re
from typing import Any, Iterator, Mapping

import numpy as np


_STREAM = re.compile(r"\.cam\.(h\d+|r|rh\d+|rs)\.")


class PICAMOutputView:
    """Discover CAM NetCDF streams without loading arrays until requested."""

    def __init__(self, driver: Any, kind: str) -> None:
        if kind not in {"history", "restart"}:
            raise ValueError("output kind must be history or restart")
        self._driver = driver
        self.kind = kind

    @property
    def run_dir(self) -> Path | None:
        return self._driver.run_dir

    @property
    def files(self) -> tuple[Path, ...]:
        run_dir = self.run_dir
        if run_dir is None or not run_dir.is_dir():
            return ()
        selected: list[Path] = []
        for path in run_dir.glob("*.cam.*.nc"):
            stream = self._stream(path)
            if stream is None:
                continue
            if self.kind == "history" and stream.startswith("h"):
                selected.append(path)
            elif self.kind == "restart" and stream.startswith("r"):
                selected.append(path)
        return tuple(sorted(selected))

    @property
    def streams(self) -> Mapping[str, tuple[Path, ...]]:
        grouped: defaultdict[str, list[Path]] = defaultdict(list)
        for path in self.files:
            stream = self._stream(path)
            if stream is not None:
                grouped[stream].append(path)
        return {
            name: tuple(sorted(paths)) for name, paths in sorted(grouped.items())
        }

    def latest(self, stream: str | None = None) -> Path:
        files = self._select(stream)
        if not files:
            raise FileNotFoundError(
                f"no CAM {self.kind} files are available in {self.run_dir}"
            )
        return files[-1]

    def open(self, stream: str | None = None, **kwargs: Any) -> Any:
        """Open one stream as an Xarray Dataset, lazily backed by NetCDF."""

        try:
            import xarray as xr
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "CAM output browsing requires the notebook extra: "
                "uv sync --extra notebook"
            ) from exc
        files = self._select(stream)
        if not files:
            raise FileNotFoundError(
                f"no CAM {self.kind} files are available in {self.run_dir}"
            )
        datasets = [xr.open_dataset(path, **kwargs) for path in files]
        if len(datasets) == 1:
            return datasets[0]
        try:
            combined = xr.concat(
                datasets,
                dim="time",
                data_vars="minimal",
                coords="minimal",
                compat="override",
            )
        except BaseException:
            for dataset in datasets:
                dataset.close()
            raise
        combined.set_close(lambda: [dataset.close() for dataset in datasets])
        return combined

    def __getitem__(self, variable: str) -> Any:
        return self.open()[str(variable)]

    def plot(
        self,
        variable: str,
        *,
        stream: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Open one field and delegate plotting to Xarray."""

        return self.open(stream)[str(variable)].plot(**kwargs)

    def step_series(
        self,
        variable: str,
        *,
        statistic: str = "global_mean",
        stream: str | None = None,
        level: int | None = None,
        step_variable: str = "nsteph",
        **open_kwargs: Any,
    ) -> Any:
        """Return one in-memory diagnostic value for every history step.

        ``global_mean`` uses CAM's ``area`` weights along ``ncol`` when both
        are available, then averages any remaining spatial dimensions.  Use
        ``level=`` to select one ``lev`` or ``ilev`` before reduction.
        """

        if self.kind != "history":
            raise TypeError("step_series() is available only for history output")
        selected_statistic = str(statistic).strip().lower()
        if selected_statistic not in {"global_mean", "mean", "min", "max"}:
            raise ValueError(
                "statistic must be 'global_mean', 'mean', 'min', or 'max'"
            )
        with self.open(stream, **open_kwargs) as dataset:
            field_name = str(variable)
            if field_name not in dataset:
                raise KeyError(
                    f"history stream does not contain variable {field_name!r}"
                )
            if step_variable not in dataset:
                raise KeyError(
                    f"history stream does not contain step variable "
                    f"{step_variable!r}"
                )
            values = dataset[field_name]
            if "time" not in values.dims:
                raise ValueError(
                    f"history variable {field_name!r} has no time dimension"
                )
            metadata = dict(values.attrs)
            selected_level_dimension: str | None = None
            if level is not None:
                selected_level_dimension = next(
                    (name for name in ("lev", "ilev") if name in values.dims),
                    None,
                )
                if selected_level_dimension is None:
                    raise ValueError(
                        f"history variable {field_name!r} has no lev or ilev "
                        "dimension"
                    )
                values = values.isel({selected_level_dimension: int(level)})

            if selected_statistic == "global_mean":
                if "ncol" in values.dims and "area" in dataset:
                    area = dataset["area"]
                    if area.dims == ("ncol",):
                        values = values.weighted(area).mean(dim="ncol")
                remaining = tuple(name for name in values.dims if name != "time")
                if remaining:
                    values = values.mean(dim=remaining)
            else:
                dimensions = tuple(name for name in values.dims if name != "time")
                if dimensions:
                    values = getattr(values, selected_statistic)(dim=dimensions)

            steps = dataset[str(step_variable)]
            if steps.dims != ("time",) or steps.sizes["time"] != values.sizes["time"]:
                raise ValueError(
                    f"step variable {step_variable!r} must have only the time "
                    "dimension and match the selected field"
                )
            series = values.assign_coords(
                model_step=("time", np.asarray(steps.values))
            ).load()
            series.name = f"{field_name}_{selected_statistic}"
            series.attrs.update(metadata)
            series.attrs.update(
                {
                    "source_variable": field_name,
                    "statistic": selected_statistic,
                    "step_variable": str(step_variable),
                }
            )
            if selected_level_dimension is not None:
                series.attrs["selected_level_dimension"] = selected_level_dimension
                series.attrs["selected_level_index"] = int(level)
            return series

    def plot_steps(
        self,
        variable: str,
        *,
        statistic: str = "global_mean",
        stream: str | None = None,
        level: int | None = None,
        step_variable: str = "nsteph",
        ax: Any = None,
        **plot_kwargs: Any,
    ) -> Any:
        """Plot one reduced CAM history variable against model step."""

        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "CAM history plotting requires the notebook extra: "
                "uv sync --extra notebook"
            ) from exc
        series = self.step_series(
            variable,
            statistic=statistic,
            stream=stream,
            level=level,
            step_variable=step_variable,
        )
        if ax is None:
            _, ax = plt.subplots(figsize=(5.0, 3.2))
        options = dict(plot_kwargs)
        options.setdefault("marker", "o")
        ax.plot(series.coords["model_step"].values, series.values, **options)
        ax.set_xlabel("model step")
        units = str(series.attrs.get("units", "1"))
        unit_suffix = "" if units in {"", "1"} else f" ({units})"
        label = str(series.attrs.get("long_name", variable))
        ax.set_ylabel(f"{statistic.replace('_', ' ')} {label}{unit_suffix}")
        level_suffix = "" if level is None else f" at level {level}"
        ax.set_title(f"{variable}{level_suffix} through the CAM run")
        ax.spines[["top", "right"]].set_visible(False)
        if ax.get_legend_handles_labels()[1]:
            ax.legend(frameon=False)
        ax.figure.tight_layout()
        return ax

    def __iter__(self) -> Iterator[Path]:
        return iter(self.files)

    def __len__(self) -> int:
        return len(self.files)

    def __repr__(self) -> str:
        return (
            f"CAMOutput(kind={self.kind!r}, files={len(self.files)}, "
            f"streams={tuple(self.streams)!r})"
        )

    def _select(self, stream: str | None) -> tuple[Path, ...]:
        grouped = self.streams
        if stream is None:
            preferred = "h0" if self.kind == "history" else "r"
            if preferred in grouped:
                return grouped[preferred]
            if len(grouped) == 1:
                return next(iter(grouped.values()))
            if not grouped:
                return ()
            raise ValueError(
                f"CAM {self.kind} has multiple streams {tuple(grouped)}; "
                "select one with open(stream=...)"
            )
        try:
            return grouped[str(stream)]
        except KeyError as exc:
            raise KeyError(
                f"unknown CAM {self.kind} stream {stream!r}; available: "
                + ", ".join(grouped)
            ) from exc

    @staticmethod
    def _stream(path: Path) -> str | None:
        match = _STREAM.search(path.name)
        return None if match is None else match.group(1)


__all__ = ["PICAMOutputView"]
