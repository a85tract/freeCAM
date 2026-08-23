"""Training pairs from a physics function, in memory and on disk.

One row per sample: the inputs and parameters that were drawn, the outputs
and updated in/out values the routine returned, and a status that says
whether it returned at all.  Invalid samples keep their inputs and carry
NaN outputs with a message -- they are never written as if they were data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .result import STATUSES
from .spec import FunctionSpec


@dataclass
class Dataset:
    function: str
    inputs: dict[str, np.ndarray]
    parameters: dict[str, np.ndarray]
    outputs: dict[str, np.ndarray]
    updated: dict[str, np.ndarray]
    status: np.ndarray
    message: list[str | None]
    sample_id: np.ndarray
    attributes: dict[str, Any] = field(default_factory=dict)
    axes: dict[str, str] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.status.shape[0])

    def __repr__(self) -> str:
        counts = ", ".join(f"{name}={count}" for name, count in self.status_counts.items())
        return (f"Dataset({self.function}: {len(self)} samples [{counts}]; "
                f"{len(self.inputs)} inputs, {len(self.parameters)} parameters, "
                f"{len(self.outputs)} outputs, {len(self.updated)} updated)")

    @property
    def first_valid_index(self) -> int:
        indices = np.flatnonzero(self.valid)
        if indices.size == 0:
            raise ValueError("the dataset has no valid sample")
        return int(indices[0])

    def to_xarray(self):
        """The same content as an ``xarray.Dataset`` (``input__*`` etc. variables)."""

        import xarray as xr

        variables = {}
        for prefix, table in (("input", self.inputs), ("parameter", self.parameters), ("output", self.outputs), ("updated", self.updated)):
            for name, values in table.items():
                dims = ["sample"] + [self.axes.get(f"{name}:{axis}", f"dim_{extent}") for axis, extent in enumerate(values.shape[1:])]
                variables[f"{prefix}__{name}"] = (dims, values)
        variables["status"] = (["sample"], np.asarray(self.status, dtype=object))
        variables["message"] = (["sample"], np.asarray([item or "" for item in self.message], dtype=object))
        variables["sample_id"] = (["sample"], self.sample_id)
        return xr.Dataset(variables, attrs={k: (v if isinstance(v, (int, float)) else str(v)) for k, v in self.attributes.items()})

    def verify_sample(self, function: Any, index: int | str = "first_valid") -> "SampleVerification":
        """Re-execute one stored sample and compare every returned value bit for bit."""

        from .image import _first_difference

        position = self.first_valid_index if index == "first_valid" else int(index)
        sample = self.sample(position)
        result = function.try_run(inputs=sample["inputs"], parameters=sample["parameters"])
        differences: dict[str, dict[str, Any]] = {}
        stored_status = sample["status"]
        if result.status == stored_status == "ok":
            for table, stored in (("outputs", sample["outputs"]), ("updated_inputs", sample["updated"])):
                for name, value in stored.items():
                    difference = _first_difference(np.asarray(value), np.asarray(getattr(result, table)[name]))
                    if difference is not None:
                        differences[name] = difference
        return SampleVerification(
            index=position, stored_status=stored_status, replayed_status=result.status,
            compared=len(sample["outputs"]) + len(sample["updated"]), differences=differences,
        )

    @property
    def valid(self) -> np.ndarray:
        return self.status == "ok"

    @property
    def status_counts(self) -> dict[str, int]:
        return {name: int(np.sum(self.status == name)) for name in STATUSES if np.any(self.status == name)}

    def sample(self, index: int) -> dict[str, Any]:
        return {
            "inputs": {name: values[index] for name, values in self.inputs.items()},
            "parameters": {name: values[index].item() for name, values in self.parameters.items()},
            "outputs": {name: values[index] for name, values in self.outputs.items()},
            "updated": {name: values[index] for name, values in self.updated.items()},
            "status": str(self.status[index]),
            "message": self.message[index],
            "sample_id": int(self.sample_id[index]),
        }

    def to_netcdf(self, path: str | Path) -> Path:
        return self.save(path)

    def save(self, path: str | Path) -> Path:
        from netCDF4 import Dataset as NetCDF

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with NetCDF(str(path), "w") as handle:
            handle.createDimension("sample", len(self))
            declared: set[str] = set()

            def write(prefix: str, table: Mapping[str, np.ndarray]) -> None:
                for name, values in table.items():
                    dims = ["sample"]
                    for axis_index, extent in enumerate(values.shape[1:]):
                        axis = self.axes.get(f"{name}:{axis_index}", f"dim_{extent}")
                        if axis not in declared:
                            handle.createDimension(axis, int(extent))
                            declared.add(axis)
                        dims.append(axis)
                    variable = handle.createVariable(f"{prefix}__{name}", values.dtype.str if values.dtype.kind != "f" else "f8", tuple(dims))
                    variable[...] = values

            write("input", self.inputs)
            write("parameter", self.parameters)
            write("output", self.outputs)
            write("updated", self.updated)
            status = handle.createVariable("status", str, ("sample",))
            message = handle.createVariable("message", str, ("sample",))
            for index in range(len(self)):
                status[index] = str(self.status[index])
                message[index] = self.message[index] or ""
            sample_id = handle.createVariable("sample_id", "i8", ("sample",))
            sample_id[...] = self.sample_id
            for key, value in self.attributes.items():
                setattr(handle, key, value if isinstance(value, (int, float)) else str(value))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Dataset":
        from netCDF4 import Dataset as NetCDF

        tables: dict[str, dict[str, np.ndarray]] = {"input": {}, "parameter": {}, "output": {}, "updated": {}}
        axes: dict[str, str] = {}
        with NetCDF(str(path)) as handle:
            for name, variable in handle.variables.items():
                prefix, _, item = name.partition("__")
                if prefix in tables and item:
                    tables[prefix][item] = np.asarray(variable[...])
                    for axis, dimension in enumerate(variable.dimensions[1:]):
                        axes[f"{item}:{axis}"] = str(dimension)
            status = np.asarray([str(item) for item in handle.variables["status"][...]])
            message = [str(item) or None for item in handle.variables["message"][...]]
            sample_id = np.asarray(handle.variables["sample_id"][...])
            attributes = {key: handle.getncattr(key) for key in handle.ncattrs()}
        return cls(
            function=str(attributes.get("function", "")),
            inputs=tables["input"], parameters=tables["parameter"], outputs=tables["output"], updated=tables["updated"],
            status=status, message=message, sample_id=sample_id, attributes=attributes, axes=axes,
        )


@dataclass(frozen=True)
class SampleVerification:
    """One stored sample re-executed: same status, and every value identical?"""

    index: int
    stored_status: str
    replayed_status: str
    compared: int
    differences: dict[str, dict[str, Any]]

    @property
    def equal(self) -> bool:
        return self.stored_status == self.replayed_status and not self.differences

    def assert_equal(self) -> "SampleVerification":
        if not self.equal:
            raise AssertionError(
                f"sample {self.index} did not re-execute identically: status {self.stored_status!r} -> "
                f"{self.replayed_status!r}, differences in {sorted(self.differences)}"
            )
        return self

    def __repr__(self) -> str:
        verdict = "identical" if self.equal else f"DIFFERENT in {sorted(self.differences)}"
        return (f"SampleVerification(index={self.index}, status={self.stored_status!r} -> {self.replayed_status!r}, "
                f"{self.compared} values compared: {verdict})")


def open_dataset(path: str | Path) -> Dataset:
    """Read a dataset written by :meth:`Dataset.to_netcdf`."""

    return Dataset.load(path)


def assemble(spec: FunctionSpec, rows: Sequence[Mapping[str, Any]], attributes: Mapping[str, Any]) -> Dataset:
    """Stack per-sample records into one Dataset; failed samples get NaN outputs."""

    n = len(rows)
    dims = spec.dimensions

    def stack(kind: str, items) -> dict[str, np.ndarray]:
        table: dict[str, np.ndarray] = {}
        for item in items:
            shape = (n, *item.public_extent(dims))
            values = np.full(shape, np.nan, dtype=np.float64) if item.dtype == "float64" else np.zeros(shape, dtype=np.dtype(item.dtype))
            for index, row in enumerate(rows):
                value = row.get(kind, {}).get(item.name)
                if value is not None:
                    values[index] = value
            table[item.name] = values
        return table

    parameters: dict[str, np.ndarray] = {}
    for name, parameter in spec.parameters.items():
        column = np.full(n, np.nan if parameter.dtype == "float64" else 0, dtype=np.float64 if parameter.dtype == "float64" else np.int32)
        for index, row in enumerate(rows):
            column[index] = row.get("parameters", {}).get(name, parameter.default)
        parameters[name] = column
    axes: dict[str, str] = {}
    for item in spec.arguments:
        if item.public_shape:
            axes[f"{item.name}:0"] = spec.public_axis(item.public_shape[0])
    return Dataset(
        function=spec.function,
        inputs=stack("inputs", spec.user_arguments),
        parameters=parameters,
        outputs=stack("outputs", spec.outputs),
        updated=stack("updated", spec.inouts),
        status=np.asarray([row["status"] for row in rows]),
        message=[row.get("message") for row in rows],
        sample_id=np.arange(n, dtype=np.int64),
        attributes=dict(attributes),
        axes=axes,
    )


__all__ = ["Dataset", "SampleVerification", "assemble", "open_dataset"]
