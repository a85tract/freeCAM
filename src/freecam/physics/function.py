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

    def example_input(self, name: str = "captured-anchor"):
        """A real column shipped with the package, ready to pass as ``inputs``."""

        from .examples import load_example_column

        return load_example_column(self.spec.function, name)

    def sampling_space(
        self,
        *,
        base: Mapping[str, Any],
        inputs: Mapping[str, Any] | None = None,
        parameters: Mapping[str, Any] | None = None,
        fixed_parameters: Mapping[str, Any] | None = None,
    ):
        """Distributions over inputs and parameters, everything else from ``base``."""

        from .distributions import SamplingSpace

        return SamplingSpace(self.spec, inputs=inputs, parameters=parameters, base=base, fixed_parameters=fixed_parameters)

    def run(self, inputs: Mapping[str, Any], parameters: Mapping[str, Any] | None = None, *, errors: str = "raise") -> FunctionResult:
        """Call the routine on one column.

        With ``errors="raise"`` (the default) an input the routine refuses
        raises ``FortranAbortError`` and the like; ``errors="return"`` --
        also :meth:`try_run` -- returns the result with its ``status`` set,
        which is what batch generation uses so one bad sample never stops
        the rest.
        """

        if errors not in ("raise", "return"):
            raise ValueError("errors must be 'raise' or 'return'")
        result = self._run(inputs, parameters)
        return result.raise_for_status() if errors == "raise" else result

    def try_run(self, inputs: Mapping[str, Any], parameters: Mapping[str, Any] | None = None) -> FunctionResult:
        return self._run(inputs, parameters)

    def _run(self, inputs: Mapping[str, Any], parameters: Mapping[str, Any] | None) -> FunctionResult:
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
            results.append(self.try_run(inputs, merged or None))
        return results

    def generate_dataset(
        self,
        n_samples: int,
        space: Any,
        *,
        seed: int = 0,
        base: Mapping[str, Any] | None = None,
        parameters: Mapping[str, Any] | None = None,
        progress: Any = None,
    ):
        """Draw ``n_samples`` joint samples from ``space``, run each, return a Dataset.

        ``space`` is a :class:`SamplingSpace` (preferred; see
        :meth:`sampling_space`) or a plain ``{name: distribution}`` mapping,
        in which case ``base`` supplies the undrawn inputs and
        ``parameters`` any fixed parameter values.  Every sample is one
        independent column call; a sample the routine refuses keeps its
        inputs and a status, never fabricated outputs.
        """

        import subprocess

        from .dataset import assemble
        from .distributions import SamplingSpace

        if isinstance(space, SamplingSpace):
            if base is not None or parameters is not None:
                raise ValueError("pass base and fixed parameters when building the SamplingSpace")
        else:
            drawn_inputs = {k: v for k, v in dict(space).items() if k not in self.spec.parameters}
            drawn_parameters = {k: v for k, v in dict(space).items() if k in self.spec.parameters}
            space = SamplingSpace(self.spec, inputs=drawn_inputs, parameters=drawn_parameters, base=base, fixed_parameters=parameters)
        rng = np.random.default_rng(seed)
        rows = []
        for index in range(int(n_samples)):
            sample_inputs, sample_parameters = space.draw(rng)
            result = self.try_run(sample_inputs, sample_parameters or None)
            rows.append({
                "inputs": coerce_inputs(self.spec, sample_inputs) if result.status != "invalid_input" else {
                    k: np.asarray(v) for k, v in sample_inputs.items()
                },
                "parameters": sample_parameters,
                "outputs": dict(result.outputs),
                "updated": dict(result.updated_inputs),
                "status": result.status,
                "message": result.message,
            })
            if progress is not None:
                progress(index + 1, int(n_samples), result.status)
        try:
            commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=False).stdout.strip()
        except OSError:
            commit = ""
        attributes = {
            **self.metadata,
            "seed": int(seed),
            "samples": int(n_samples),
            "sampling_space": space.describe(),
            "fixed_parameters": json.dumps(dict(space.fixed_parameters)),
            "freecam_commit": commit,
        }
        return assemble(self.spec, rows, attributes)

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
    # Provenance travels as hashes, not as this machine's paths.
    metadata = {
        "function": spec.qualified_name,
        "image_sha256": json.loads(manifest_path.read_text())["library_sha256"],
        "module_state_digest": snapshot.get("digest"),
    }
    return PhysicsFunction(spec, backend, metadata=metadata)


__all__ = ["PhysicsFunction", "load_function"]
