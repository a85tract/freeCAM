"""The command line fills a pausable stage's slots before the stage is installed."""
from __future__ import annotations

import cloudpickle

from freecam.physics.segments import FrameCapture, OriginalKernel
from freecam.pi_cam.cli import _pausable_stage


class _Model:
    pass


def test_every_slot_is_filled_on_the_stage_whose_tend_is_installed() -> None:
    model = _Model()
    stage = _pausable_stage("deep_convection", "auto", original_kernels=["zm_convr"],
                            capture_kernels=["momtran"], kernel_models={"cldfrc_fice": model, "compute_tms": model})
    assert isinstance(stage.kernels["zm_convr"], OriginalKernel)
    assert isinstance(stage.kernels["momtran"], FrameCapture)
    assert stage.kernels["cldfrc_fice"] is model                 # a model for this stage's kernel
    assert stage.kernels["zm_conv_evap"] is None                 # untouched slot
    assert "compute_tms" not in stage.kernels                    # another stage's kernel is not taken
    assert stage.execution_policy == "auto"
    # what install pickles is the bound method of this very stage: the slots travel with it
    restored = cloudpickle.loads(cloudpickle.dumps(stage.tend)).__self__
    assert restored.replacements() == ("zm_convr", "momtran", "cldfrc_fice")


def test_an_unknown_stage_is_refused() -> None:
    import pytest

    with pytest.raises(SystemExit, match="not one of"):
        _pausable_stage("no_such_stage", "auto", original_kernels=[], capture_kernels=[], kernel_models={})
