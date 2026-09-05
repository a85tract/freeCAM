"""A surrogate named by path is a replacement from the moment the stage is built.

The stage decides how to run -- the original Fortran whole, or paused at the
kernels something else computes -- by looking at which kernel slots are
filled.  A surrogate that only loaded on first use left its slot empty at
that decision, and a month that named a model ran the original Fortran to
the end and reported it bit-for-bit.
"""

import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from freecam.physics.errors import PhysicsError
from freecam.physics.macrophysics import RETURNED, Macrophysics
from freecam.physics import surrogate as surrogate_module
from freecam.physics.surrogate import PendingSurrogate


class _Loaded:
    takes_parameters = True
    x_arguments = ["t0"]

    def __init__(self) -> None:
        self.calls: list = []

    def __call__(self, column, parameters=None):
        self.calls.append(("one", parameters))
        return {name: np.zeros(30) for name in RETURNED}

    def batched(self, columns, parameters=None):
        self.calls.append(("batched", parameters))
        ncol = np.asarray(columns["t0"]).shape[0]
        return {name: np.zeros((ncol, 30), dtype=np.float32) for name in RETURNED}


def test_a_surrogate_named_by_path_is_a_replacement_before_it_loads(monkeypatch) -> None:
    def never(path):
        raise AssertionError(f"nothing should load {path} while the stage only decides how to run")

    monkeypatch.setattr(surrogate_module, "load_surrogate", never)
    stage = Macrophysics(surrogate="somewhere/model.pt")
    assert stage.replacements() == ("mmacro_pcond",)
    assert stage.configured_replacements() == ("mmacro_pcond",)
    assert isinstance(stage.kernels["mmacro_pcond"], PendingSurrogate)
    assert not stage.kernels["mmacro_pcond"].loaded

    from freecam.physics.cloud_macro_microphysics import CloudMacroMicrophysics

    whole = CloudMacroMicrophysics(macro_surrogate="somewhere/model.pt")
    assert "mmacro_pcond" in whole.replacements()
    runner = SimpleNamespace(kernels=("mmacro_pcond",))
    image = SimpleNamespace(segment_runner=lambda stage: runner)
    assert whole.select_mode(image) == "segmented"     # auto: the runner covers the replacement
    assert whole.select_mode(SimpleNamespace(segment_runner=lambda stage: None)) == "legacy-python"
    assert whole.select_mode() == "legacy-python"      # no image to ask: the proven walk


def test_a_kernel_and_a_surrogate_cannot_both_stand_in_the_slot() -> None:
    with pytest.raises(PhysicsError, match="both a kernel and a surrogate"):
        Macrophysics(kernel=lambda column: {}, surrogate="model.pt")


def test_the_pending_surrogate_loads_once_on_use_and_travels_unloaded(monkeypatch) -> None:
    loaded = _Loaded()
    loads: list[str] = []

    def load(path):
        loads.append(str(path))
        return loaded

    monkeypatch.setattr(surrogate_module, "load_surrogate", load)
    pending = PendingSurrogate("model.pt")
    assert pending.takes_parameters
    pending.batched({"t0": np.zeros((2, 30))}, {"cldfrc_rhminl": 0.9})
    pending({"t0": np.zeros(30)})
    assert pending.x_arguments == ["t0"]                 # attributes reach the loaded kernel
    assert loads == ["model.pt"]                         # once
    assert loaded.calls == [("batched", {"cldfrc_rhminl": 0.9}), ("one", None)]

    copy = pickle.loads(pickle.dumps(pending))
    assert copy.path == "model.pt" and not copy.loaded  # the weights never travel


def test_the_frame_adapter_answers_the_returned_values_and_passes_the_workspace_through() -> None:
    stage = Macrophysics()
    model = _Loaded()
    answer = stage.frame_kernel("mmacro_pcond", model, native=None)
    tke = np.ones((3, 30))
    batch = {"ncol": np.int32(3), "dt": np.float64(1800.0), "t0": np.zeros((3, 30)), "tke": tke,
             "landfrac": np.zeros(3)}

    out = answer(batch)

    assert model.calls == [("batched", None)]
    for name in RETURNED:
        assert out[name].dtype == np.float64 and out[name].shape == (3, 30)
    assert out["tke"] is tke                              # the workspace goes back as it came
    assert out["landfrac"] is batch["landfrac"]

    plain = lambda batch: {}                              # noqa: E731
    assert stage.frame_kernel("something_else", plain, native=None) is plain


def test_a_model_short_of_an_answer_is_refused_by_the_adapter() -> None:
    stage = Macrophysics()

    class Short:
        def batched(self, columns, parameters=None):
            return {"cld": np.zeros((2, 30))}

    answer = stage.frame_kernel("mmacro_pcond", Short(), native=None)
    with pytest.raises(PhysicsError, match="missing"):
        answer({"ncol": np.int32(2), "t0": np.zeros((2, 30))})


def test_a_stage_told_to_replace_a_kernel_refuses_to_run_the_original_in_its_place() -> None:
    stage = Macrophysics(surrogate="model.pt")
    stage.kernels["mmacro_pcond"] = None                 # someone emptied the slot
    with pytest.raises(PhysicsError, match="was told to replace"):
        stage.tend({}, SimpleNamespace(native=object()))
