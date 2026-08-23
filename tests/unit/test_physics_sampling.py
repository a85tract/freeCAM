"""Distributions, joint draws, the dataset, and generate_dataset over a fake host."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from freecam.physics.dataset import Dataset, assemble
from freecam.physics.distributions import (
    Anchored, Constant, Derived, HybridPressure, LogUniform, Normal, SamplingSpace, Uniform,
)
from freecam.physics.errors import PhysicsError
from freecam.physics.function import PhysicsFunction
from freecam.physics.host import CallOutcome
from freecam.physics.spec import load_function_spec

HYAI = np.array([0.00225, 0.005, 0.02, 0.0])
HYBI = np.array([0.0, 0.05, 0.3, 1.0])


def test_plain_distributions_respect_their_bounds_and_seeds() -> None:
    rng = np.random.default_rng(3)
    u = Uniform(1.0, 2.0).sample(rng, (100,), {})
    assert u.shape == (100,) and u.min() >= 1.0 and u.max() <= 2.0
    lu = LogUniform(1e-6, 1e-2).sample(rng, (1000,), {})
    assert lu.min() >= 1e-6 and lu.max() <= 1e-2 and np.median(lu) < 1e-3
    n = Normal(0.0, 1.0, clip=(-1.0, 1.0)).sample(rng, (500,), {})
    assert n.min() >= -1.0 and n.max() <= 1.0
    assert np.array_equal(Constant(5.0).sample(rng, (3,), {}), [5.0, 5.0, 5.0])
    a = Anchored(np.array([1.0, 100.0]), 0.1, relative=True, clip=(0.0, None)).sample(rng, (2,), {})
    assert a.shape == (2,) and abs(a[1] - 100.0) < 60.0
    with pytest.raises(PhysicsError):
        LogUniform(0.0, 1.0)
    first = Uniform(0.0, 1.0).sample(np.random.default_rng(11), (4,), {})
    second = Uniform(0.0, 1.0).sample(np.random.default_rng(11), (4,), {})
    assert np.array_equal(first, second)


def test_hybrid_pressure_is_structurally_consistent() -> None:
    draw = HybridPressure(HYAI, HYBI, 1.0e5, Uniform(60000.0, 100000.0)).sample(np.random.default_rng(0), (), {})
    pint = draw["pint"]
    assert np.all(np.diff(pint) > 0)
    assert np.allclose(draw["dp"], np.diff(pint)) and np.allclose(draw["p"], 0.5 * (pint[:-1] + pint[1:]))
    assert 60000.0 <= float(draw["ps"]) <= 100000.0 and float(pint[-1]) == float(draw["ps"])


def test_sampling_space_orders_dependencies_and_separates_parameters() -> None:
    spec = load_function_spec("dadadj")
    pressure = HybridPressure(
        np.array([0.002] * 3 + [0.0] * 28), np.concatenate([np.zeros(3), np.linspace(0.05, 1.0, 28)]), 1.0e5,
        Uniform(80000.0, 100000.0), produces=("pmid", "pdel", "pint"),
    )

    def humidity(rng, t, pmid):
        return 1e-3 * (pmid / pmid[-1]) * (t / 300.0)

    space = SamplingSpace(spec, {
        "q": Derived(humidity, depends=("t", "pmid")),
        "t": Uniform(200.0, 300.0),
        "pmid": pressure,
        "nlvdry": Uniform(2, 6),
    })
    assert space.order.index("pmid") < space.order.index("q") and space.order.index("t") < space.order.index("q")
    inputs, parameters = space.draw(np.random.default_rng(1))
    assert set(inputs) == {"pmid", "pdel", "pint", "t", "q"}
    assert inputs["pint"].shape == (31,) and inputs["q"].shape == (30,)
    assert isinstance(parameters["nlvdry"], int) and 2 <= parameters["nlvdry"] <= 6
    with pytest.raises(PhysicsError, match="cannot be sampled"):
        SamplingSpace(spec, {"lchnk": Constant(1)})
    with pytest.raises(PhysicsError, match="not drawn"):
        SamplingSpace(spec, {"q": Derived(humidity, depends=("t", "pmid"))})


class _Host:
    """Doubles t and aborts whenever level 0 of t is below 100 K."""

    restarts = 0

    def set_parameters(self, values): pass
    def restore_parameters(self): pass
    def close(self): pass

    def call(self, pool, returned=None):
        if pool["dadadj.t"][0, 0, 0] < 100.0:
            return CallOutcome("fortran_abort", None, "Impossible case")
        pool["dadadj.t"][0, :, 0] *= 2.0
        return CallOutcome("ok", pool)


def _pressure():
    pint = np.linspace(225.0, 1.0e5, 31)
    return {"pint": Constant(pint), "pmid": Constant(0.5 * (pint[:-1] + pint[1:])), "pdel": Constant(np.diff(pint))}


def test_generate_dataset_is_reproducible_and_keeps_failed_samples(tmp_path: Path) -> None:
    spec = load_function_spec("dadadj")
    function = PhysicsFunction(spec, _Host(), metadata={"function": spec.qualified_name})
    distributions = {**_pressure(), "t": Uniform(50.0, 300.0), "q": Constant(np.full(30, 1e-3)), "nlvdry": Uniform(2, 5)}

    first = function.generate_dataset(40, distributions, seed=7)
    second = function.generate_dataset(40, distributions, seed=7)
    assert np.array_equal(first.inputs["t"], second.inputs["t"])
    assert np.array_equal(first.parameters["nlvdry"], second.parameters["nlvdry"])
    assert len(first) == 40 and set(first.status_counts) <= {"ok", "fortran_abort"}
    aborted = ~first.valid
    assert aborted.any() and first.valid.any()
    # Failed samples keep their inputs and carry NaN outputs with a message.
    assert np.isnan(first.updated["t"][aborted]).all() and not np.isnan(first.inputs["t"][aborted]).any()
    assert all(message == "Impossible case" for message, bad in zip(first.message, aborted) if bad)
    ok = np.nonzero(first.valid)[0][0]
    assert np.array_equal(first.updated["t"][ok], 2.0 * first.inputs["t"][ok])

    # Any stored sample re-executes to the same output.
    sample = first.sample(ok)
    again = function.run(sample["inputs"], sample["parameters"])
    assert np.array_equal(again["t"], first.updated["t"][ok])

    path = first.save(tmp_path / "dadadj.nc")
    loaded = Dataset.load(path)
    assert np.array_equal(loaded.inputs["t"], first.inputs["t"])
    assert np.array_equal(loaded.status, first.status) and loaded.message[int(np.nonzero(aborted)[0][0])] == "Impossible case"
    assert loaded.attributes["seed"] == 7 and loaded.attributes["function"] == spec.qualified_name
    from netCDF4 import Dataset as NetCDF
    with NetCDF(str(path)) as handle:
        assert handle.variables["input__t"].dimensions == ("sample", "lev")
        assert handle.variables["input__pint"].dimensions == ("sample", "ilev")
        assert "parameter__nlvdry" in handle.variables and "updated__t" in handle.variables
