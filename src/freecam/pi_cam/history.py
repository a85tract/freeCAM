"""Lazy, Notebook-friendly access to CAM history and restart NetCDF files."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re
from typing import Any, Iterator, Mapping


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
