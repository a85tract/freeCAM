"""The hook table and the counters an image with hooks reports."""
import ctypes
from pathlib import Path

import pytest
import yaml

from freecam.pi_cam.errors import PICAMConfigurationError
from freecam.pi_cam.hooks import load_hooks, read_hook_counts

REPO = Path(__file__).resolve().parents[2]


def test_the_committed_hook_table_names_its_kernels_and_redirections() -> None:
    table = load_hooks(REPO / "native/pi_cam/hooks.yaml")
    assert table.kernel_names == ("cldfrc_fice", "fluxbelowinv") and table.fiber_stack_bytes >= (64 << 20)
    fice = table.hook("cldfrc_fice")
    assert fice.callee_symbol == "cloud_fraction_mp_cldfrc_fice_" and fice.original_module == "cloud_fraction"
    assert [c.object for c in fice.callers] == ["zm_conv.o"]
    flux = table.hook("fluxbelowinv")
    assert flux.original_symbol == "uwshcu_mp_fluxbelowinv_original_" and flux.symbol == flux.callee_symbol
    for hook in table.hooks:
        assert (REPO / hook.contract).is_file()


def _table(tmp_path: Path, **hook):
    record = {"kernel": "toy", "contract": "native/pi_cam/functions/toy.yaml", "callee_symbol": "m_mp_toy_",
              "original": {"module": "m", "routine": "toy"}, "callers": [{"object": "c.o", "routine": "caller"}]}
    record.update(hook)
    path = tmp_path / "hooks.yaml"
    path.write_text(yaml.safe_dump({"schema_version": 1, "hooks": [record]}))
    return path


def test_the_loader_refuses_inconsistent_hooks(tmp_path: Path) -> None:
    assert load_hooks(_table(tmp_path)).hook("toy").symbol == "pycam_hook_toy_"
    with pytest.raises(PICAMConfigurationError):
        load_hooks(_table(tmp_path, redirect="sideways"))
    with pytest.raises(PICAMConfigurationError):
        load_hooks(_table(tmp_path, original={"symbol": "x_", "module": "m"}))
    with pytest.raises(PICAMConfigurationError):
        load_hooks(_table(tmp_path, redirect="weaken-definition"))      # needs the second name
    with pytest.raises(PICAMConfigurationError):
        load_hooks(_table(tmp_path, callers=[]))


class _Entry:
    """A ctypes function pointer: callable, and accepts restype/argtypes."""

    def __init__(self, function) -> None:
        self._function = function

    def __call__(self, *args):
        return self._function(*args)


class _Library:
    """What ctypes shows of an image with two hooks."""

    def __init__(self) -> None:
        names = {1: b"cldfrc_fice", 2: b"fluxbelowinv"}
        counts = {1: (200, 100), 2: (600, 0)}

        def name(hook, buffer, length):
            if hook not in names:
                return 1
            ctypes.memmove(buffer, names[hook] + b"\0", len(names[hook]) + 1)
            return 0

        def count(hook, calls, paused):
            if hook not in counts:
                return 1
            calls._obj.value, paused._obj.value = counts[hook]
            return 0

        self.pycam_hooks_count_v1 = _Entry(lambda: 2)
        self.pycam_hooks_name_v1 = _Entry(name)
        self.pycam_hooks_counts_v1 = _Entry(count)


def test_hook_counts_are_read_by_name() -> None:
    assert read_hook_counts(None) == {}
    assert read_hook_counts(object()) == {}                              # an image without hooks
    assert read_hook_counts(_Library()) == {"cldfrc_fice": {"calls": 200, "paused": 100},
                                            "fluxbelowinv": {"calls": 600, "paused": 0}}
