"""The answers cldwat2m_macro computes from its other answers.

If the identity table is wrong, a surrogate built on it is wrong everywhere
and quietly: it would answer states that look plausible and do not follow
from the tendencies it reported.  So the table is checked against the
routine's own recorded output rather than against itself.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from freecam.physics.identities import (CPAIR, NON_NEGATIVE, derived_targets,
                                        identities_for)

REPO = Path(__file__).resolve().parents[2]
DATASET = REPO / "examples" / "mmacro_pcond_training.nc"


@pytest.fixture(scope="module")
def recorded():
    """The routine's own answers, read once.

    Opened once and closed on the way out: two live netCDF4 handles on one
    file in one process segfault the reader rather than raising.
    """

    netCDF4 = pytest.importorskip("netCDF4")
    if not DATASET.is_file():
        pytest.skip(f"{DATASET.name} is not present")
    data = netCDF4.Dataset(DATASET)
    try:
        yield {name: np.asarray(variable[:]) for name, variable in data.variables.items()}
    finally:
        data.close()


def test_the_table_says_which_answers_are_not_free() -> None:
    derived = derived_targets("mmacro_pcond")
    assert derived == ("t0", "qv0", "ql0", "qi0", "nl0", "ni0", "cld")
    assert derived_targets("dadadj") == ()          # no table, no claim
    for item in identities_for("mmacro_pcond"):
        assert item.provenance.startswith("cldwat2m_macro.F90:")
    assert set(NON_NEGATIVE["mmacro_pcond"]) < set(derived)


def test_every_identity_reproduces_the_routines_own_answer(recorded) -> None:
    """Against a dataset the routine itself produced, column by column.

    The tolerance is not a fitting tolerance.  These are the same numbers
    written two ways, so the only difference admitted is the order the
    additions happen in.
    """

    column = {name[len("input__"):]: value
              for name, value in recorded.items() if name.startswith("input__")}
    answer = {name[len("output__"):]: value
              for name, value in recorded.items() if name.startswith("output__")}
    truth = {name[len("updated__"):]: value
             for name, value in recorded.items() if name.startswith("updated__")}
    truth["cld"] = answer["cld"]
    dt = column["dt"][:, None]

    for item in identities_for("mmacro_pcond"):
        got, expected = item(column, answer, dt), truth[item.target]
        scale = float(np.percentile(np.abs(expected), 99)) or 1.0
        assert np.abs(got - expected).max() / scale < 1e-12, item.target
        # And the identity really reads what it says it reads.
        for name in item.reads:
            assert name in column
        for name in item.derives_from:
            assert name in answer


def test_the_temperature_identity_needs_the_right_specific_heat(recorded) -> None:
    """cpair is not a fitted constant; the wrong one breaks the identity."""

    t0 = recorded["input__t0"]
    dt = recorded["input__dt"][:, None]
    change = recorded["updated__t0"] - t0
    forcing = recorded["input__a_t"] + recorded["input__c_t"]
    energy = recorded["output__s_tendout"]

    right = np.abs(change - (energy / CPAIR + forcing) * dt).max()
    wrong = np.abs(change - (energy / (CPAIR * 1.01) + forcing) * dt).max()
    assert right < 1e-10
    assert wrong > 1e3 * max(right, 1e-30)
