"""The mmacro_pcond kernel boundary seen from Python."""

from __future__ import annotations

import numpy as np
import pytest

from freecam.physics.errors import PhysicsError
from freecam.physics.surrogate import (
    INPUTS, RETURNED, IdentityKernel, MacroKernel, ShadowSurrogate, _Lanes,
)

PCOLS, PVER, NCHUNKS = 16, 30, 2
NCOL = np.array([13, 14], dtype=np.int32)


def _pool(seed: int = 0) -> dict[str, np.ndarray]:
    """A StatePool-shaped record: every field at the shape the spec declares."""

    from freecam.physics.spec import load_function_spec

    spec = load_function_spec("mmacro_pcond")
    rng = np.random.default_rng(seed)
    pool: dict[str, np.ndarray] = {}
    for prefix, names in (("in", INPUTS), ("out", RETURNED), ("ref", RETURNED)):
        for name in names:
            extent = spec.argument(name).native_extent(spec.dimensions)
            pool[f"macro_split.{prefix}_{name}"] = rng.random((*extent, NCHUNKS))
    # ncol is what says which lanes are live, so it is set, not drawn.
    pool["macro_split.in_ncol"] = NCOL.copy()
    return pool


def test_lanes_gather_and_scatter_leave_the_padding_alone() -> None:
    lanes = _Lanes(NCOL)
    assert lanes.columns == 27
    field = np.arange(PCOLS * PVER * NCHUNKS, dtype=np.float64).reshape(PCOLS, PVER, NCHUNKS)
    before = field.copy()
    gathered = lanes.gather(field)
    assert gathered.shape == (27, PVER)
    lanes.scatter(field, np.zeros_like(gathered))
    for chunk, count in enumerate(NCOL):
        assert np.all(field[:count, :, chunk] == 0.0)
        assert np.array_equal(field[count:, :, chunk], before[count:, :, chunk])


def test_identity_kernel_changes_nothing_at_all() -> None:
    """The plumbing gate: every value makes the round trip and none may move."""

    pool = _pool(3)
    before = {name: value.copy() for name, value in pool.items()}
    IdentityKernel().run(pool, None)
    for name, value in pool.items():
        assert np.array_equal(value, before[name]), name


def test_a_kernel_that_answers_only_part_of_the_boundary_is_refused() -> None:
    class Partial(MacroKernel):
        def predict(self, batch, columns):
            return {"cld": np.zeros((columns, PVER))}

    with pytest.raises(PhysicsError, match="missing"):
        Partial().run(_pool(4), None)


def test_a_kernel_writes_only_the_live_lanes() -> None:
    class Constant(MacroKernel):
        def predict(self, batch, columns):
            assert batch["t0"].shape == (columns, PVER)
            assert batch["landfrac"].shape == (columns,)
            return {name: np.full((columns, PVER), 0.5) for name in RETURNED}

    pool = _pool(5)
    before = {name: value.copy() for name, value in pool.items()}
    Constant().run(pool, None)
    for name in RETURNED:
        field = pool[f"macro_split.out_{name}"]
        original = before[f"macro_split.out_{name}"]
        for chunk, count in enumerate(NCOL):
            assert np.all(field[:count, :, chunk] == 0.5)
            assert np.array_equal(field[count:, :, chunk], original[count:, :, chunk])
    # Inputs and the shadow slot are untouched.
    for name in INPUTS:
        assert np.array_equal(pool[f"macro_split.in_{name}"], before[f"macro_split.in_{name}"])


def test_shadow_scores_the_model_without_touching_the_run() -> None:
    class Offset(ShadowSurrogate):
        def predict(self, batch, columns):
            return {name: np.zeros((columns, PVER)) for name in RETURNED}

    pool = _pool(6)
    for name in RETURNED:
        pool[f"macro_split.ref_{name}"][...] = 2.0
    before = {name: value.copy() for name, value in pool.items()}
    shadow = Offset(model=None)
    shadow.run(pool, None)

    for name, value in pool.items():
        assert np.array_equal(value, before[name]), name
    report = shadow.drift_report()
    assert report["columns"] == 27
    assert report["arguments"]["cld"]["rmse"] == pytest.approx(2.0)
    assert report["arguments"]["cld"]["relative_rmse"] == pytest.approx(1.0)
    assert report["arguments"]["cld"]["max_absolute"] == pytest.approx(2.0)


def test_the_declared_fields_are_the_ones_the_record_publishes() -> None:
    from pathlib import Path
    import re

    module = Path(__file__).resolve().parents[2] / "native/pi_cam/support/pycam_macro_split.F90"
    body = module.read_text().split("type pycam_macro_record_t", 1)[1].split("end type", 1)[0]
    published = {f"macro_split.{m}" for m in re.findall(r"::\s*((?:in|out|ref)_\w+)", body)}
    for declared in (*MacroKernel.reads, *MacroKernel.writes,
                     *IdentityKernel.reads, *ShadowSurrogate.reads):
        assert declared in published, declared


def test_the_stage_and_its_halves_are_never_both_enabled() -> None:
    """Running the stage beside its halves would do the macrophysics twice."""

    from freecam.physics.surrogate import FIRST_HALF, SECOND_HALF, STAGE, split_macro_stage

    class Action:
        def __init__(self, enabled: bool) -> None:
            self.enabled = enabled

        def enable(self, **_: object) -> None:
            self.enabled = True

        def disable(self, **_: object) -> None:
            self.enabled = False

    class Workflow:
        def __init__(self) -> None:
            self.items = {STAGE: Action(True), FIRST_HALF: Action(False), SECOND_HALF: Action(False)}

        def process(self, name: str) -> Action:
            return self.items[name]

    workflow = Workflow()
    assert split_macro_stage(workflow) == (FIRST_HALF, SECOND_HALF)
    assert [workflow.items[k].enabled for k in (STAGE, FIRST_HALF, SECOND_HALF)] == [False, True, True]
    assert split_macro_stage(workflow, split=False) == (STAGE,)
    assert [workflow.items[k].enabled for k in (STAGE, FIRST_HALF, SECOND_HALF)] == [True, False, False]
