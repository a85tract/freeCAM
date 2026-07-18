import numpy as np
import pytest

from pycam_sima.mpi_runtime import SerialComm
from pycam_sima.observer import ObserverContext, ObserverRegistry
from pycam_sima.state_pool import FieldSpec, StatePool


def context(pool):
    return ObserverContext(0, 0, 1, "after", "kessler", pool, 0, SerialComm())


def test_interactive_callback_can_modify_authoritative_state():
    pool = StatePool()
    field = pool.allocate(FieldSpec("air_temperature", np.float64, ("ncol",)), (2,))
    registry = ObserverRegistry(mode="interactive")
    registry.observe("after:*", lambda ctx: ctx.state["air_temperature"].fill(7.0))
    registry.emit("after:kessler", context(pool))
    assert np.array_equal(field, [7.0, 7.0])


def test_validation_callback_is_readonly():
    pool = StatePool()
    pool.allocate(FieldSpec("air_temperature", np.float64, ("ncol",)), (2,))
    registry = ObserverRegistry(mode="validation")

    def mutate(ctx):
        ctx.state["air_temperature"][0] = 1.0

    registry.observe("after:*", mutate, access="readonly")
    with pytest.raises(ValueError):
        registry.emit("after:kessler", context(pool))
