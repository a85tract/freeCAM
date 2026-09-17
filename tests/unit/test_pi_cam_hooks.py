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
    assert table.kernel_names[:5] == ("cldfrc_fice", "fluxbelowinv", "instratus_condensate", "micro_mg_tend", "compute_tms")
    assert set(table.kernel_names[5:]) == {"compute_uwshcu_inv", "mmacro_pcond", "zm_conv_evap", "momtran", "zm_convr", "compute_eddy_diff"}
    assert table.fiber_stack_bytes >= (64 << 20)
    core = table.hook("instratus_condensate")              # inside mmacro_pcond, same object: weakened
    assert core.redirect == "weaken-definition" and core.id == 3
    assert core.original_symbol == "cldwat2m_macro_mp_instratus_condensate_original_"
    fice = table.hook("cldfrc_fice")
    assert fice.callee_symbol == "cloud_fraction_mp_cldfrc_fice_" and fice.original_module == "cloud_fraction"
    assert [c.object for c in fice.callers] == ["zm_conv.o"]
    flux = table.hook("fluxbelowinv")
    assert flux.original_symbol == "uwshcu_mp_fluxbelowinv_original_" and flux.symbol == flux.callee_symbol
    micro = table.hook("micro_mg_tend")                    # logical, character and pointer dummies: Fortran-bound
    assert micro.binding == "fortran" and micro.symbol == "pycam_hooks_mp_hook_micro_mg_tend_" and not micro.pausable
    assert micro.redirect == "rename-references" and micro.original_module == "micro_mg1_0" and micro.id == 4
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
    with pytest.raises(PICAMConfigurationError):
        load_hooks(_table(tmp_path, binding="sideways"))
    with pytest.raises(PICAMConfigurationError):                  # a Fortran-bound hook is reached by renamed references
        load_hooks(_table(tmp_path, binding="fortran", redirect="weaken-definition", original={"symbol": "x_"}))
    assert not load_hooks(_table(tmp_path, binding="fortran")).hook("toy").pausable
    with pytest.raises(PICAMConfigurationError):                  # a subset names an output the model does not return
        load_hooks(_table(tmp_path, model={"inputs": ["a"], "outputs": ["b"], "packed": True, "subset": {"c": [1]}}))
    with pytest.raises(PICAMConfigurationError):                  # a subset lives inside a packed tensor only
        load_hooks(_table(tmp_path, model={"inputs": ["a"], "outputs": ["b"], "subset": {"b": [1, 2]}}))
    with pytest.raises(PICAMConfigurationError):                  # its indices are 1-based and distinct
        load_hooks(_table(tmp_path, model={"inputs": ["a"], "outputs": ["b"], "packed": True, "subset": {"b": [0, 1]}}))
    table = load_hooks(_table(tmp_path, model={"inputs": ["a"], "outputs": ["b", "c"], "packed": True, "subset": {"c": [3, 1]}}))
    assert table.hook("toy").model_subset("c") == (3, 1) and table.hook("toy").model_subset("b") is None


class _Entry:
    """A ctypes function pointer: callable, and accepts restype/argtypes."""

    def __init__(self, function) -> None:
        self._function = function

    def __call__(self, *args):
        return self._function(*args)


class _Library:
    """What ctypes shows of an image with two hooks."""

    def __init__(self) -> None:
        names = {1: b"cldfrc_fice", 2: b"fluxbelowinv", 3: b"instratus_condensate", 4: b"micro_mg_tend"}
        counts = {1: (200, 100), 2: (600, 0), 3: (18000, 18000), 4: (200, 0)}

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

        self.pycam_hooks_count_v1 = _Entry(lambda: 4)
        self.pycam_hooks_name_v1 = _Entry(name)
        self.pycam_hooks_counts_v1 = _Entry(count)


def test_hook_counts_are_read_by_name() -> None:
    assert read_hook_counts(None) == {}
    assert read_hook_counts(object()) == {}                              # an image without hooks
    assert read_hook_counts(_Library()) == {"cldfrc_fice": {"calls": 200, "paused": 100},
                                            "fluxbelowinv": {"calls": 600, "paused": 0},
                                            "instratus_condensate": {"calls": 18000, "paused": 18000},
                                            "micro_mg_tend": {"calls": 200, "paused": 0}}


def test_a_packed_model_block_is_read_and_rendered(tmp_path) -> None:
    import subprocess
    import sys

    from freecam.pi_cam.hooks import load_hooks

    table = load_hooks()
    hook = table.hook("compute_uwshcu_inv")
    assert hook.model_packed and hook.model_zero_outputs == ()
    assert "tr0_inv" in hook.model_inputs and hook.model_outputs[-4:] == ("trten_inv", "wtqc_inv", "wtprec", "wtsnow")
    assert hook.model_subset("trten_inv") == (10, 11, 12, 17, 18, 19, 24, 25, 26, 31, 32, 33)   # the isotope V/L/I constituents
    assert hook.model_subset("wtprec") == (10, 17, 24, 31) and hook.model_subset("umf_inv") is None
    text = (REPO / "native/pi_cam/support/pycam_hooks.F90").read_text()
    body = text[text.index("subroutine model_compute_uwshcu_inv"):text.index("end subroutine model_compute_uwshcu_inv")]
    assert "out_t(1)" in body and "o_packed(16, 1190)" in body           # one tensor: 582 bulk columns + 360 + 240 + 4 + 4
    assert "w_umf_inv(1:hk_n, hk_j) = o_packed(1:hk_n, 1 + hk_j)" in body  # cush takes column 1, umf the next 31
    # a subset output: the hook zeroes the live columns, then scatters the packed columns (level-major) to its indices
    assert "integer, parameter :: sub_trten_inv(12) = (/ 10, 11, 12, 17, 18, 19, 24, 25, 26, 31, 32, 33 /)" in body
    assert "w_trten_inv(1:hk_n, :, :) = 0.0_c_double" in body
    assert "w_trten_inv(1:hk_n, hk_j, sub_trten_inv(hk_s)) = o_packed(1:hk_n, 582 + (hk_j - 1) * 12 + hk_s)" in body
    assert "w_wtprec(1:hk_n, sub_wtprec(hk_s)) = o_packed(1:hk_n, 1182 + hk_s)" in body
    assert "w_wtsnow(1:hk_n, sub_wtsnow(hk_s)) = o_packed(1:hk_n, 1186 + hk_s)" in body
    # a plugin (compiled code or a Python callback) fills the one packed output through the same tables
    assert "plugin_status = plugin(20_c_int, in_p, in_s, 1_c_int, out_p, out_s)" in body
    assert "out_p(1) = c_loc(o_packed); out_s(:, 1) = (/ int(16, c_int64_t), int(1190, c_int64_t), 0_c_int64_t /)" in body
    assert subprocess.run([sys.executable, str(REPO / "tools/generate_pi_cam_hooks.py"), "--check"],
                          capture_output=True, text=True, cwd=REPO).returncode == 0
