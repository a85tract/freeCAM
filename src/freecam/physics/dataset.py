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
        with NetCDF(str(path)) as handle:
            for name, variable in handle.variables.items():
                prefix, _, item = name.partition("__")
                if prefix in tables and item:
                    tables[prefix][item] = np.asarray(variable[...])
            status = np.asarray([str(item) for item in handle.variables["status"][...]])
            message = [str(item) or None for item in handle.variables["message"][...]]
            sample_id = np.asarray(handle.variables["sample_id"][...])
            attributes = {key: handle.getncattr(key) for key in handle.ncattrs()}
        return cls(
            function=str(attributes.get("function", "")),
            inputs=tables["input"], parameters=tables["parameter"], outputs=tables["output"], updated=tables["updated"],
            status=status, message=message, sample_id=sample_id, attributes=attributes,
        )


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


__all__ = ["Dataset", "assemble"]
