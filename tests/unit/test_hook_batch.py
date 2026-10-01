"""A stage whose kernel's hook answers every chunk at once: when it batches, and in what order it runs."""

from __future__ import annotations

import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from freecam.physics.errors import PhysicsError
from freecam.physics.native_model import NativeModel
from freecam.physics.pausable import DryAdjustment, ShallowConvection


def _torchscript(tmp_path: Path) -> Path:
    """A file NativeModel takes for a TorchScript archive (its code and constants entries)."""

    path = tmp_path / "model.pt"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("model/constants.pkl", b"")
        archive.writestr("model/code/__torch__.py", b"")
    return path


class _Entry:
    """A ctypes function pointer: callable, and takes restype and argtypes."""

    def __init__(self, function) -> None:
        self._function = function

    def __call__(self, *args):
        return self._function(*args)


class _Library:
    """The image's entries a hook batch goes through, logging the order they are called in."""

    def __init__(self, *, pending: int = 0) -> None:
        self.log: list[tuple] = []

        def stats(hook, forwards, chunks, rows, untaken, pending_out, seconds, diff):
            pending_out._obj.value = pending
            return 0

        self.pycam_stagehost_bind_v1 = _Entry(lambda: self.log.append(("bind_hosts",)) or 0)
        self.pycam_hooks_batch_v1 = _Entry(lambda hook, mode, check: self.log.append(("batch_mode", hook, mode)) or 0)
        self.pycam_shcu_batch_v1 = _Entry(lambda: self.log.append(("gather",)) or 0)
        self.pycam_hooks_batch_stats_v1 = _Entry(stats)


def _native(library: _Library) -> SimpleNamespace:
    return SimpleNamespace(library=library, run_action=lambda stage: library.log.append(("run", stage)))


def test_a_stage_batches_its_hook_only_when_told_and_only_for_a_native_model(tmp_path) -> None:
    stage = ShallowConvection()
    assert stage.hook_batch_mode() is None                                  # call by call unless told
    stage.batch_chunks = True
    assert stage.hook_batch_mode() is None                                  # nothing to batch
    stage.kernels["compute_uwshcu_inv"] = NativeModel(_torchscript(tmp_path))
    assert stage.hook_batch_mode() == "model"
    stage.kernels["compute_uwshcu_inv"] = NativeModel(_torchscript(tmp_path), shadow=True)
    with pytest.raises(PhysicsError, match="a shadow model"):
        stage.hook_batch_mode()
    stage.kernels["compute_uwshcu_inv"] = None
    stage.batch_original = True
    assert stage.hook_batch_mode() == "original"                            # the gate, nothing replaced
    stage.kernels["compute_uwshcu_inv"] = lambda batch: {}
    assert stage.hook_batch_mode() is None                                  # a Python model batches at the runner
    other = DryAdjustment()
    other.batch_chunks = other.batch_original = True
    assert other.hook_batch_mode() is None                                  # no batched hook in its driver


def test_a_batched_step_gathers_then_runs_the_action_and_every_chunk_is_taken(tmp_path) -> None:
    stage = ShallowConvection()
    stage.batch_chunks = stage.batch_original = True
    library = _Library()
    stage.run_whole(_native(library))
    stage.run_whole(_native(library))
    hook = [entry for entry in library.log if entry[0] == "batch_mode"][0][1]
    assert library.log == [("bind_hosts",), ("batch_mode", hook, 2), ("gather",), ("run", stage.STAGE),
                           ("gather",), ("run", stage.STAGE)]                  # set once, gathered every step
    assert stage.execution.batch == "original" and stage.execution.batched_steps == 2
    assert stage.execution.describe()["hook_batch"] == "original"


def test_a_chunk_no_call_took_fails_the_step() -> None:
    stage = ShallowConvection()
    stage.batch_chunks = stage.batch_original = True
    with pytest.raises(PhysicsError, match="did not take every gathered chunk"):
        stage.run_whole(_native(_Library(pending=1)))


def test_a_stage_without_a_batch_runs_its_action_alone() -> None:
    stage = ShallowConvection()
    library = _Library()
    stage.run_whole(_native(library))
    assert library.log == [("run", stage.STAGE)]
    assert "hook_batch" not in stage.execution.describe()
