"""Answers a routine computes from its other answers, exactly.

A surrogate that fits every output independently is fitting more than the
routine contains.  cldwat2m_macro's six updated states are not separate
answers at all: the routine forms a tendency, then sets the state to the
value that tendency came from, so state and tendency are two views of one
number.  ``mmacro_pcond`` at cldwat2m_macro.F90:1174-1183 and 1208-1213::

    qv_tendout = (qv_star - qv0)/dt - (A_qv + C_qv)
    ...
    qv0        = qv_star

Substituting one into the other leaves an identity with no freedom in it,
and it holds to machine precision in every captured column: the residual of
``qv0_new - qv0 - (qv_tendout + A_qv + C_qv)*dt`` is 2e-19 at its worst
against a state of 1e-2.

Fitting both sides separately is worse than wasteful.  It spends a third of
the output space on numbers already determined, and it lets the two drift
apart -- which is what a model run sees as water that does not add up.  The
first gated surrogate violated the state identities in 100% of columns and
stopped a 512-rank run at step 3 with an isotopic mass error of 2.5e+36.

So the surrogate predicts the free answers and *derives* the rest.  The
identity is then exact by construction rather than approximately satisfied,
which is the difference that matters for a kernel asked to run for thousands
of steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

#: Specific heat of dry air, physconst.  s_tendout is a dry static energy
#: tendency, so the temperature identity divides by it.
CPAIR = 1004.64


@dataclass(frozen=True)
class Identity:
    """One answer computed from others, with where it comes from."""

    target: str
    #: Argument names read from the column.
    reads: tuple[str, ...]
    #: Other target names read from the model's own answer.
    derives_from: tuple[str, ...]
    compute: Callable[[Mapping[str, Any], Mapping[str, Any], float], np.ndarray]
    provenance: str

    def __call__(self, column, answer, dt) -> np.ndarray:
        return self.compute(column, answer, dt)


def _state(name: str, tendency: str, scale: float = 1.0) -> Identity:
    """A state the routine sets to the value its tendency was formed from."""

    def compute(column, answer, dt):
        return (np.asarray(column[name], dtype=np.float64)
                + (np.asarray(answer[tendency], dtype=np.float64) / scale
                   + np.asarray(column[f"a_{name[:-1]}"], dtype=np.float64)
                   + np.asarray(column[f"c_{name[:-1]}"], dtype=np.float64)) * dt)

    return Identity(
        target=name,
        reads=(name, f"a_{name[:-1]}", f"c_{name[:-1]}"),
        derives_from=(tendency,),
        compute=compute,
        provenance="cldwat2m_macro.F90:1174-1183 with 1208-1213",
    )


def _cloud_fraction() -> Identity:
    """Net cloud fraction, cldwat2m_macro.F90:1201.

    ``cld = a_st_star + a_cu0``, and the total stratus fraction is the larger
    of the liquid and ice fractions rather than their sum -- which the capture
    confirms exactly, to the last bit, in every column.
    """

    def compute(column, answer, dt):
        return (np.maximum(np.asarray(answer["al_st_star"], dtype=np.float64),
                           np.asarray(answer["ai_st_star"], dtype=np.float64))
                + np.asarray(column["a_cu0"], dtype=np.float64))

    return Identity(target="cld", reads=("a_cu0",),
                    derives_from=("al_st_star", "ai_st_star"),
                    compute=compute, provenance="cldwat2m_macro.F90:1201")


#: Per function, the answers that are not free.  ``t0`` divides by cpair
#: because s_tendout is an energy tendency; the water and number states do
#: not, because theirs are tendencies of the state itself.
IDENTITIES: dict[str, tuple[Identity, ...]] = {
    "mmacro_pcond": (
        _state("t0", "s_tendout", CPAIR),
        _state("qv0", "qv_tendout"),
        _state("ql0", "ql_tendout"),
        _state("qi0", "qi_tendout"),
        _state("nl0", "nl_tendout"),
        _state("ni0", "ni_tendout"),
        _cloud_fraction(),
    ),
}

#: States the routine can never leave negative.  The identities keep the
#: answer self-consistent; they do not keep it physical, and a tendency the
#: network overshoots would otherwise take condensate below zero -- which is
#: what CAM's own QNEG3 check exists to catch.
NON_NEGATIVE: dict[str, tuple[str, ...]] = {
    "mmacro_pcond": ("qv0", "ql0", "qi0", "nl0", "ni0"),
}


def identities_for(function: str) -> tuple[Identity, ...]:
    return IDENTITIES.get(function, ())


def derived_targets(function: str) -> tuple[str, ...]:
    return tuple(item.target for item in identities_for(function))


__all__ = ["CPAIR", "Identity", "IDENTITIES", "NON_NEGATIVE",
           "identities_for", "derived_targets"]
