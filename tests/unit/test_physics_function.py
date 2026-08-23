"""Column packing, results, and the function object over a fake host."""

from __future__ import annotations

import numpy as np
import pytest

from freecam.physics.column import InvalidInput, coerce_inputs, pack_column, unpack_column
from freecam.physics.function import PhysicsFunction
from freecam.physics.host import CallOutcome
from freecam.physics.result import FunctionResult
from freecam.physics.spec import load_function_spec


def _column():
    spec = load_function_spec("dadadj")
    pint = np.linspace(225.0, 1.0e5, 31)
    pmid = 0.5 * (pint[:-1] + pint[1:])
    return spec, {"pmid": pmid, "pint": pint, "pdel": np.diff(pint), "t": np.full(30, 250.0), "q": np.full(30, 1e-3)}


def test_pack_puts_the_column_in_lane_zero_with_ncol_one() -> None:
    spec, inputs = _column()
    pool = pack_column(spec, coerce_inputs(spec, inputs))
    assert pool["dadadj.ncol"].tolist() == [1] and pool["dadadj.lchnk"].tolist() == [1]
    assert pool["dadadj.t"].shape == (16, 30, 1) and pool["dadadj.t"].flags.f_contiguous
    assert np.array_equal(pool["dadadj.t"][0, :, 0], inputs["t"])
    assert np.all(pool["dadadj.t"][1:, :, 0] == 0.0)
    pool["dadadj.t"][0, 3, 0] = 1.0
    outputs, updated = unpack_column(spec, pool)
    assert outputs == {} and updated["t"][3] == 1.0 and updated["t"].shape == (30,)


def test_inputs_are_validated_and_case_insensitive() -> None:
    spec, inputs = _column()
    resolved = coerce_inputs(spec, {**inputs, "T": inputs.pop("t")})
    assert "t" in resolved
    with pytest.raises(InvalidInput, match="missing inputs"):
        coerce_inputs(spec, {"t": np.zeros(30)})
    with pytest.raises(InvalidInput, match="expected \\(31,\\)"):
        coerce_inputs(spec, {**inputs, "pint": np.zeros(30)})


class _Host:
    def __init__(self, status="ok"):
        self.status = status
        self.parameters: list = []

    def set_parameters(self, values):
        self.parameters.append(("set", dict(values)))

    def restore_parameters(self):
        self.parameters.append(("restore", {}))

    def call(self, pool, returned=None):
        if self.status != "ok":
            return CallOutcome(self.status, None, "boom")
        pool["dadadj.t"][0, :, 0] += 1.0
        return CallOutcome("ok", pool)

    def close(self):
        pass


def test_function_run_returns_outputs_and_restores_parameters() -> None:
    spec, inputs = _column()
    host = _Host()
    function = PhysicsFunction(spec, host)
    result = function.run(inputs=inputs, parameters={"nlvdry": 5})
    assert result.ok and np.array_equal(result.updated_inputs["t"], inputs["t"] + 1.0)
    assert result["t"] is result.updated_inputs["t"]
    assert result.metadata["parameters"] == {"nlvdry": 5}
    assert host.parameters == [("set", {"nlvdry": 5}), ("restore", {})]
    assert function(inputs).ok
    assert "status='ok'" in repr(result)


def test_interactive_run_raises_and_try_run_returns_a_status() -> None:
    from freecam.physics import FortranAbortError

    spec, inputs = _column()
    with pytest.raises(InvalidInput, match="missing inputs"):
        PhysicsFunction(spec, _Host()).run({"t": np.zeros(30)})
    assert PhysicsFunction(spec, _Host()).try_run({"t": np.zeros(30)}).status == "invalid_input"
    with pytest.raises(FortranAbortError, match="boom"):
        PhysicsFunction(spec, _Host("fortran_abort")).run(inputs)
    aborted = PhysicsFunction(spec, _Host("fortran_abort")).run(inputs, errors="return")
    assert aborted.status == "fortran_abort" and aborted.message == "boom" and aborted.outputs == {}
    assert "fortran_abort" in repr(aborted)
    with pytest.raises(FortranAbortError):
        aborted.raise_for_status()
    with pytest.raises(ValueError):
        PhysicsFunction(spec, _Host()).run(inputs, errors="maybe")


def test_result_rejects_unknown_status() -> None:
    with pytest.raises(ValueError):
        FunctionResult({}, {}, "weird")
