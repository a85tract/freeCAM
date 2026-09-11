"""A TorchScript model bound at a hooked kernel: the image answers, Python stays out of the step."""
from __future__ import annotations

import ctypes
import re
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from freecam.physics.errors import PhysicsError
from freecam.physics.native_model import NativeModel
from freecam.pi_cam.errors import PICAMConfigurationError
from freecam.pi_cam.hooks import BIND_STATUS, bind_hook_model, load_hooks, read_hook_counts, unbind_hook_model

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import generate_pi_cam_hooks as generator  # noqa: E402


def torchscript_archive(path: Path) -> Path:
    """The shape of a torch.jit.save archive, without torch: code and constants in a zip."""

    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("model/code/__torch__.py", "def forward(self, x): return x\n")
        archive.writestr("model/constants.pkl", b"\x80\x02.")
        archive.writestr("model/data.pkl", b"\x80\x02.")
    return path


def test_a_torchscript_archive_is_recognised_and_other_files_are_not(tmp_path: Path) -> None:
    model = NativeModel(torchscript_archive(tmp_path / "m.pt"))
    assert model.describe()["binding"] == "torchscript" and len(model.sha256) == 64
    assert model.takes_frame is False
    with pytest.raises(PhysicsError):
        model({"x": 1})                                            # never called from Python
    plain = tmp_path / "weights.pkl"
    plain.write_bytes(b"\x80\x04not a zip")
    assert not NativeModel.is_torchscript(plain)
    with pytest.raises(PhysicsError):
        NativeModel(plain)
    with zipfile.ZipFile(tmp_path / "other.zip", "w") as archive:
        archive.writestr("readme.txt", "no model here")
    assert not NativeModel.is_torchscript(tmp_path / "other.zip")


def test_the_hook_table_says_which_hooks_take_a_model() -> None:
    table = load_hooks(REPO / "native/pi_cam/hooks.yaml")
    core = table.hook("instratus_condensate")
    assert core.takes_model
    assert core.model_inputs[:3] == ("k", "p_in", "t0_in") and len(core.model_inputs) == 19
    assert core.model_outputs == ("t_out", "qv_out", "ql_out", "qi_out", "al_st_out", "ai_st_out", "ql_st_out", "qi_st_out")
    assert not table.hook("cldfrc_fice").takes_model and not table.hook("fluxbelowinv").takes_model


def _table(tmp_path: Path, model) -> Path:
    record = {"kernel": "toy", "contract": "native/pi_cam/functions/toy.yaml", "callee_symbol": "m_mp_toy_",
              "original": {"module": "m", "routine": "toy"}, "callers": [{"object": "c.o", "routine": "caller"}],
              "model": model}
    path = tmp_path / "hooks.yaml"
    path.write_text(yaml.safe_dump({"schema_version": 1, "hooks": [record]}))
    return path


def test_the_loader_refuses_a_malformed_model_block(tmp_path: Path) -> None:
    with pytest.raises(PICAMConfigurationError):
        load_hooks(_table(tmp_path, {"inputs": ["a"]}))                       # no outputs
    with pytest.raises(PICAMConfigurationError):
        load_hooks(_table(tmp_path, {"inputs": ["a", "a"], "outputs": ["b"]}))  # listed twice
    # a name on both sides is the generator's to judge (an inout array may be both)
    assert load_hooks(_table(tmp_path, {"inputs": ["a"], "outputs": ["a"]})).hook("toy").takes_model
    assert load_hooks(_table(tmp_path, {"inputs": ["a"], "outputs": ["b"]})).hook("toy").takes_model


def test_the_generated_module_answers_a_bound_model_inside_the_image() -> None:
    text = generator.render(load_hooks())
    assert "use ftorch, only: torch_model, torch_tensor, torch_kCPU" in text
    assert "subroutine model_instratus_condensate(" in text
    assert "call torch_model_forward(models(3), in_t, out_t)" in text
    assert "type(torch_tensor) :: in_t(19), out_t(8)" in text
    # the scalar k travels as a one-element real tensor; arrays are wrapped where they live
    assert "s_k(1) = real(k, c_double)" in text
    assert "call c_f_pointer(c_loc(p_in(1)), v_p_in, (/ 16 /))" in text
    # the live columns are written back, the padding lanes left as CAM had them
    assert "n = min(int(ncol), 16)" in text and "t_out(1:n) = o_t_out(1:n)" in text
    # only the hooks with a model block have the branch; the others cannot be bound
    assert text.count("call model_") == 2
    assert "has_model(nhooks) = (/ .false., .false., .true., .true. /)" in text
    assert "can_pause(nhooks) = (/ .true., .true., .true., .false. /)" in text
    # a Fortran-bound hook: the callee's own kinds, no bind(C), no frame, a model over rank-2 and rank-3 arrays
    assert "subroutine hook_micro_mg_tend(microp_uniform, pcols, pver, ncol" in text and "bind(C, name='pycam_hooks_mp" not in text
    assert "    logical, intent(in) :: microp_uniform" in text and "    character(len=*), intent(out) :: errstring" in text
    assert "    real(c_double), pointer, intent(in) :: tnd_qsnow(:,:)" in text
    assert "type(torch_tensor) :: in_t(26), out_t(89)" in text
    # packed arrays: the callee's own pcols/pver dummies size every array, not the contract's constants
    assert "real(c_double), intent(in), target :: tn(pcols, pver)" in text
    assert "call c_f_pointer(c_loc(rndst), v_rndst, (/ pcols, pver, 4 /))" in text
    assert "real(c_double), target :: o_rflx(pcols, pver+1)" in text and "n = min(int(ncol), pcols)" in text
    assert "l_tnd_qsnow = tnd_qsnow(1:pcols, 1:pver)" in text
    assert "w_qc(1:n, :) = o_qc(1:n, :)" in text and "w_prect(1:n) = o_prect(1:n)" in text
    # the bind(C) hook keeps the contract's fixed extents (module arrays of pcols)
    assert "call c_f_pointer(c_loc(p_in(1)), v_p_in, (/ 16 /))" in text
    assert "if (associated(tnd_qsnow)) then" in text and "errstring = ' '" in text
    # arming refuses a hook without a frame
    assert "if (flag /= 0_c_int .and. .not. can_pause(hook)) then" in text
    # shadow: the model runs, the original answers, nothing is written back
    assert text.count("if (.not. shadow(") >= 4 and "shadow(hook) = shadow_flag /= 0_c_int" in text
    for entry in ("pycam_hooks_bind_model_v1", "pycam_hooks_unbind_model_v1", "pycam_hooks_modeled_v1"):
        assert f"bind(C, name='{entry}')" in text
    # a bound model and a Python replacement cannot share a hook
    assert "if (flag /= 0_c_int .and. modeled(hook)) then" in text


class _Entry:
    def __init__(self, function) -> None:
        self._function = function

    def __call__(self, *args):
        return self._function(*args)


class _Library:
    """An image with the model entries: it remembers what was bound where."""

    def __init__(self, status: int = 0) -> None:
        self.bound: dict[int, bytes] = {}
        self.unbound: list[int] = []

        def bind(hook, path, length, shadow):
            if status:
                return status
            self.bound[hook] = (path[:length], int(shadow))
            return 0

        def unbind(hook):
            self.unbound.append(hook)
            return 0

        def modeled(hook, out):
            out._obj.value = 4242
            return 0

        self.pycam_hooks_bind_model_v1 = _Entry(bind)
        self.pycam_hooks_unbind_model_v1 = _Entry(unbind)
        self.pycam_hooks_modeled_v1 = _Entry(modeled)
        names = {1: b"cldfrc_fice", 2: b"fluxbelowinv", 3: b"instratus_condensate", 4: b"micro_mg_tend"}

        def name(hook, buffer, length):
            if hook not in names:
                return 1
            ctypes.memmove(buffer, names[hook] + b"\0", len(names[hook]) + 1)
            return 0

        def count(hook, calls, paused):
            calls._obj.value, paused._obj.value = (9, 0)
            return 0

        self.pycam_hooks_count_v1 = _Entry(lambda: 4)
        self.pycam_hooks_name_v1 = _Entry(name)
        self.pycam_hooks_counts_v1 = _Entry(count)


def test_binding_reaches_the_image_and_refusals_are_named(tmp_path: Path) -> None:
    library = _Library()
    bind_hook_model(library, 3, tmp_path / "m.pt")
    assert library.bound == {3: (str(tmp_path / "m.pt").encode(), 0)}
    bind_hook_model(library, 4, tmp_path / "m.pt", shadow=True)
    assert library.bound[4] == (str(tmp_path / "m.pt").encode(), 1)
    unbind_hook_model(library, 3)
    assert library.unbound == [3]
    with pytest.raises(PICAMConfigurationError, match=re.escape(BIND_STATUS[2])):
        bind_hook_model(_Library(status=2), 1, tmp_path / "m.pt")
    with pytest.raises(PICAMConfigurationError, match="built without FTorch"):
        bind_hook_model(object(), 3, tmp_path / "m.pt")
    assert read_hook_counts(library)["instratus_condensate"] == {"calls": 9, "paused": 0, "modeled": 4242}


def test_a_stage_with_a_native_model_runs_whole_and_binds_once(tmp_path: Path, monkeypatch) -> None:
    from freecam.physics.cloud_macro_microphysics import CloudMacroMicrophysics

    model = NativeModel(torchscript_archive(tmp_path / "instratus.pt"))
    stage = CloudMacroMicrophysics()
    stage.kernels["instratus_condensate"] = model
    assert stage.select_mode(None) == "native-model"
    library = _Library()
    ran: list[str] = []
    native = SimpleNamespace(library=library, run_action=lambda name, phase=None: ran.append(name),
                             segment_runner=lambda stage_name: None)
    context = SimpleNamespace(native=native, step=1)
    stage.tend(None, context)
    stage.tend(None, context)
    assert ran == [stage.STAGE, stage.STAGE]                    # the whole original stage, twice
    assert list(library.bound) == [3]                          # bound at instratus's hook, once
    # a shadow model binds with the flag and rebinds when the mode changes
    shadowed = NativeModel(torchscript_archive(tmp_path / "instratus.pt"), shadow=True)
    stage.kernels["instratus_condensate"] = shadowed
    stage.tend(None, context)
    assert library.bound[3][1] == 1 and shadowed.describe()["shadow"] is True
    assert stage.execution.describe()["execution_mode"] == "native-model"
    assert stage.execution.describe()["python_fortran_crossings_per_step"] == 1
    # a Python replacement next to a native model has no path
    stage.kernels["cldfrc_fice"] = lambda batch: {}
    with pytest.raises(PhysicsError, match="one kind per stage"):
        stage.select_mode(None)
    # a kernel that is not a hook cannot take a native model
    other = CloudMacroMicrophysics()
    other.kernels["mmacro_pcond"] = model
    with pytest.raises(PhysicsError, match="not a hooked kernel"):
        other.tend(None, context)


def test_the_command_line_tells_a_torchscript_archive_from_a_pickle(tmp_path: Path) -> None:
    from freecam.pi_cam.cli import _kernel_models_summary, _load_kernel_model

    archive = torchscript_archive(tmp_path / "m.pt")
    assert isinstance(_load_kernel_model(archive), NativeModel)
    assert _load_kernel_model(archive, shadow=True).shadow is True
    summary = _kernel_models_summary({"instratus_condensate": archive}, {"micro_mg_tend": archive})
    assert summary["instratus_condensate"]["binding"] == "torchscript" and "shadow" not in summary["instratus_condensate"]
    assert summary["micro_mg_tend"]["shadow"] is True
    plain = tmp_path / "weights.pkl"; plain.write_bytes(b"\x80\x04not a zip")
    with pytest.raises(SystemExit):
        _load_kernel_model(plain, shadow=True)                   # only a native model can shadow


def test_the_run_record_sums_the_calls_a_model_answered_over_the_ranks() -> None:
    from freecam.pi_cam.cli import _hook_summary

    records = [{"hook_counts": {"instratus_condensate": {"calls": 9360, "paused": 0, "modeled": 9000}}},
               {"hook_counts": {"instratus_condensate": {"calls": 9360, "paused": 0, "modeled": 9000}}}]
    assert _hook_summary(records) == {"instratus_condensate": {"calls": 18720, "paused": 0, "ranks_called": 2, "modeled": 18000}}
