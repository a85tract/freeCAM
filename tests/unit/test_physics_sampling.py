"""Distributions, joint draws, the dataset, and generate_dataset over a fake host."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from freecam.physics import (
    Anchored, Constant, Dataset, Derived, HybridCoordinate, HybridPressure, LogUniform, Normal,
    SamplingSpace, Uniform, load_example_column, open_dataset,
)
from freecam.physics.errors import PhysicsError
from freecam.physics.function import PhysicsFunction
from freecam.physics.host import CallOutcome
from freecam.physics.spec import load_function_spec

COORDINATE = HybridCoordinate(
    np.concatenate([np.array([0.00225, 0.005, 0.02]), np.zeros(28)]),
    np.concatenate([np.zeros(3), np.linspace(0.05, 1.0, 28)]),
    1.0e5,
)


def test_plain_distributions_respect_their_bounds_and_seeds() -> None:
    rng = np.random.default_rng(3)
    u = Uniform(1.0, 2.0).sample(rng, (100,), {})
    assert u.shape == (100,) and u.min() >= 1.0 and u.max() <= 2.0
    lu = LogUniform(1e-6, 1e-2).sample(rng, (1000,), {})
    assert lu.min() >= 1e-6 and lu.max() <= 1e-2 and np.median(lu) < 1e-3
    n = Normal(0.0, 1.0, clip=(-1.0, 1.0)).sample(rng, (500,), {})
    assert n.min() >= -1.0 and n.max() <= 1.0
    assert np.array_equal(Constant(5.0).sample(rng, (3,), {}), [5.0, 5.0, 5.0])
    with pytest.raises(PhysicsError):
        LogUniform(0.0, 1.0)
    first = Uniform(0.0, 1.0).sample(np.random.default_rng(11), (4,), {})
    second = Uniform(0.0, 1.0).sample(np.random.default_rng(11), (4,), {})
    assert np.array_equal(first, second)


def test_anchored_relative_noise_cannot_leave_a_zero_level_frozen() -> None:
    anchor = np.array([0.0, 1e-3, 0.0])
    rng = np.random.default_rng(0)
    relative_only = Anchored(anchor, relative_scale=0.1, clip=(0.0, None)).sample(rng, (3,), {})
    assert relative_only[0] == 0.0 and relative_only[2] == 0.0
    with_floor = np.stack([Anchored(anchor, relative_scale=0.1, absolute_scale=1e-8, clip=(0.0, None)).sample(rng, (3,), {}) for _ in range(50)])
    assert (with_floor[:, 0] > 0).any() and (with_floor >= 0).all()
    with pytest.raises(PhysicsError):
        Anchored(anchor)


def test_hybrid_pressure_is_structurally_consistent_and_named_by_the_spec() -> None:
    spec = load_function_spec("dadadj")
    space = SamplingSpace(spec, inputs={"pmid": HybridPressure(COORDINATE, Uniform(60000.0, 100000.0))})
    inputs, _ = space.draw(np.random.default_rng(0))
    pint = inputs["pint"]
    assert np.all(np.diff(pint) > 0)
    assert np.allclose(inputs["pdel"], np.diff(pint)) and np.allclose(inputs["pmid"], 0.5 * (pint[:-1] + pint[1:]))
    assert 60000.0 <= float(pint[-1]) <= 100000.0
    # A HybridPressure keyed under a name it does not produce is an error.
    with pytest.raises(PhysicsError, match="does not produce"):
        SamplingSpace(spec, inputs={"t": HybridPressure(COORDINATE, Constant(9e4))})


def test_sampling_space_takes_undrawn_inputs_from_the_base_and_orders_dependencies() -> None:
    spec = load_function_spec("dadadj")
    column = load_example_column("dadadj")

    def humidity(rng, t, pmid):
        return 1e-3 * (pmid / pmid[-1]) * (t / 300.0)

    space = spec and SamplingSpace(
        spec,
        inputs={"q": Derived(humidity, depends=("t", "pmid")), "t": Uniform(200.0, 300.0),
                "pmid": HybridPressure.from_column(column, Uniform(0.9 * column.surface_pressure, column.surface_pressure))},
        parameters={"nlvdry": Uniform(2, 6)},
        base=column,
    )
    assert space.order.index("pmid") < space.order.index("q") and space.order.index("t") < space.order.index("q")
    # Nothing the pressure distribution produces is also taken from the base.
    assert not ({"pmid", "pdel", "pint"} & set(space.base))
    inputs, parameters = space.draw(np.random.default_rng(1))
    assert set(inputs) == {"pmid", "pdel", "pint", "t", "q"}
    assert isinstance(parameters["nlvdry"], int) and 2 <= parameters["nlvdry"] <= 6
    assert "drawn inputs" in repr(space)
    with pytest.raises(PhysicsError, match="cannot be sampled"):
        SamplingSpace(spec, inputs={"lchnk": Constant(1)})
    with pytest.raises(PhysicsError, match="neither drawn nor in the base"):
        SamplingSpace(spec, inputs={"q": Derived(humidity, depends=("t", "pmid"))})


def test_example_column_is_a_complete_input_with_its_provenance() -> None:
    column = load_example_column("mmacro_pcond")
    spec = load_function_spec("mmacro_pcond")
    assert set(column) == {item.name for item in spec.user_arguments} - {"dt", "do_cldice"} | set(column) & {"dt", "do_cldice"}
    assert column.hybrid is not None and column.surface_pressure > 50000.0
    assert "captured" in repr(column) and column.source["ncol"] >= 1
    merged = {**column, "t0": column["t0"] + 1.0}
    assert merged["t0"][0] == column["t0"][0] + 1.0


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


def test_generate_dataset_is_reproducible_keeps_failed_samples_and_verifies(tmp_path: Path) -> None:
    spec = load_function_spec("dadadj")
    function = PhysicsFunction(spec, _Host(), metadata={"function": spec.qualified_name})
    column = load_example_column("dadadj")
    space = function.sampling_space(
        base=column,
        inputs={"t": Uniform(50.0, 300.0)},
        parameters={"nlvdry": Uniform(2, 5)},
    )

    first = function.generate_dataset(40, space, seed=7)
    second = function.generate_dataset(40, space, seed=7)
    assert np.array_equal(first.inputs["t"], second.inputs["t"])
    assert np.array_equal(first.parameters["nlvdry"], second.parameters["nlvdry"])
    assert np.array_equal(first.inputs["q"][0], column["q"])  # undrawn input from the base
    assert len(first) == 40 and set(first.status_counts) <= {"ok", "fortran_abort"}
    aborted = ~first.valid
    assert aborted.any() and first.valid.any()
    assert np.isnan(first.updated["t"][aborted]).all() and not np.isnan(first.inputs["t"][aborted]).any()
    assert all(message == "Impossible case" for message, bad in zip(first.message, aborted) if bad)
    assert "Dataset(dadadj" in repr(first)

    report = first.verify_sample(function)
    assert report.equal and report.index == first.first_valid_index and report.compared == 2
    report.assert_equal()

    path = first.to_netcdf(tmp_path / "dadadj.nc")
    loaded = open_dataset(path)
    assert np.array_equal(loaded.inputs["t"], first.inputs["t"])
    assert np.array_equal(loaded.status, first.status)
    assert loaded.attributes["seed"] == 7 and loaded.attributes["function"] == spec.qualified_name
    xr = loaded.to_xarray()
    assert xr["input__t"].dims == ("sample", "lev") and xr["input__pint"].dims == ("sample", "ilev")
    assert "parameter__nlvdry" in xr and "updated__t" in xr

    # A mapping of distributions still works, with base and fixed parameters alongside.
    legacy = function.generate_dataset(5, {"t": Constant(column["t"]), "nlvdry": Uniform(2, 5)}, seed=1, base=column, parameters={})
    assert len(legacy) == 5 and legacy.valid.all()
    with pytest.raises(ValueError):
        function.generate_dataset(5, space, seed=1, base=column)
