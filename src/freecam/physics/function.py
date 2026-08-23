"""A CAM physics routine as an ordinary single-column function."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .column import InvalidInput, coerce_inputs, pack_column, unpack_column
from .errors import PhysicsError
from .host import InProcessHost, SubprocessHost
from .result import FunctionResult
from .spec import FunctionSpec, load_function_spec

REPO = Path(__file__).resolve().parents[3]


class PhysicsFunction:
    """``y = f(x, p)`` on one vertical column, executed by the original Fortran.

    ``inputs``, ``inouts``, ``outputs`` and ``parameters`` describe the
    boundary; :meth:`run` takes one column's inputs (``(lev,)`` profiles and
    scalars) and optional parameter values, and returns the outputs, the
    updated in/out values, and a status.  Parameters are written for the
    call and restored afterwards, so calls never leak state into each other.
    """

    def __init__(self, spec: FunctionSpec, host: Any, *, metadata: Mapping[str, Any] | None = None) -> None:
        self.spec = spec
        self.host = host
        self.metadata = dict(metadata or {})

    # -- description ---------------------------------------------------------

    @property
    def name(self) -> str:
        return self.spec.function

    @property
    def inputs(self):
        return self.spec.inputs

    @property
    def inouts(self):
        return self.spec.inouts

    @property
    def outputs(self):
        return self.spec.outputs

    @property
    def parameters(self):
        return self.spec.parameters

    def describe(self) -> str:
        return self.spec.describe()

    # -- calling --------------------------------------------------------------

    def run(self, inputs: Mapping[str, Any], parameters: Mapping[str, Any] | None = None) -> FunctionResult:
        metadata = {**self.metadata, "parameters": dict(parameters or {})}
        try:
            resolved = coerce_inputs(self.spec, inputs)
        except InvalidInput as error:
            return FunctionResult({}, {}, "invalid_input", str(error), metadata)
        pool = pack_column(self.spec, resolved)
        returned = tuple(f"{self.spec.function}.{item.name}" for item in self.spec.arguments if item.returned)
        if parameters:
            self.host.set_parameters(parameters)
        try:
            outcome = self.host.call(pool, returned)
        finally:
            if parameters:
                self.host.restore_parameters()
        if outcome.status != "ok":
            return FunctionResult({}, {}, outcome.status, outcome.message, metadata)
        outputs, updated = unpack_column(self.spec, outcome.pool)
        return FunctionResult(outputs, updated, "ok", None, metadata)

    __call__ = run

    def batch(self, samples: Iterable[Mapping[str, Any]], parameters: Mapping[str, Any] | None = None) -> list[FunctionResult]:
        """Run many columns; each sample may carry its own ``parameters``."""

        results = []
        for sample in samples:
            own = dict(sample.get("parameters", {}) or {}) if isinstance(sample, Mapping) and "parameters" in sample else {}
            inputs = sample.get("inputs", sample) if isinstance(sample, Mapping) and "inputs" in sample else sample
            merged = {**(parameters or {}), **own}
            results.append(self.run(inputs, merged or None))
        return results

    def close(self) -> None:
        self.host.close()

    def __enter__(self) -> "PhysicsFunction":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"PhysicsFunction({self.spec.qualified_name!r}, host={type(self.host).__name__})"


def load_function(
    name: str,
    *,
    manifest: str | Path | None = None,
    module_state: str | Path | None = None,
    host: str = "subprocess",
    max_restarts: int = 100,
) -> PhysicsFunction:
    """Load a routine's standalone image and its model snapshot as a function."""

    spec = load_function_spec(name)
    manifest_path = Path(manifest) if manifest else REPO / "build" / "pi_cam_standalone" / name / "manifest.json"
    snapshot_path = Path(module_state) if module_state else REPO / "validation" / f"pi_cam_{name}_module_state.json"
    if not manifest_path.is_file():
        raise PhysicsError(f"no standalone image manifest for {name!r}: {manifest_path}")
    if not snapshot_path.is_file():
        raise PhysicsError(f"no module-state snapshot for {name!r}: {snapshot_path}")
    snapshot = json.loads(snapshot_path.read_text())
    if host == "subprocess":
        backend: Any = SubprocessHost(manifest_path, snapshot, max_restarts=max_restarts)
    elif host == "inprocess":
        backend = InProcessHost(manifest_path, snapshot)
    else:
        raise PhysicsError(f"unknown host {host!r}")
    metadata = {
        "function": spec.qualified_name,
        "image": str(manifest_path),
        "image_sha256": json.loads(manifest_path.read_text())["library_sha256"],
        "module_state": str(snapshot_path),
        "module_state_digest": snapshot.get("digest"),
    }
    return PhysicsFunction(spec, backend, metadata=metadata)


__all__ = ["PhysicsFunction", "load_function"]
