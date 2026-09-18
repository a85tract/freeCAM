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
    assert table.hook("cldfrc_fice").takes_model and not table.hook("fluxbelowinv").takes_model
    assert table.hook("cldfrc_fice").model_inputs == ("t",) and table.hook("cldfrc_fice").model_outputs == ("fice", "fsnow")


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
    assert "hk_n = min(int(ncol), 16)" in text and "t_out(1:hk_n) = o_t_out(1:hk_n)" in text
    # only the hooks with a model block have the branch; the others cannot be bound
    table = load_hooks()
    flags = lambda values: "(/ " + ", ".join(".true." if v else ".false." for v in values) + " /)"  # noqa: E731
    assert text.count("call model_") == sum(1 for hook in table.hooks if hook.takes_model)
    assert not table.hook("fluxbelowinv").takes_model and table.hook("cldfrc_fice").takes_model
    assert f"has_model(nhooks) = {flags(hook.takes_model for hook in table.hooks)}" in text
    assert f"can_pause(nhooks) = {flags(hook.pausable for hook in table.hooks)}" in text
    # a Fortran-bound hook: the callee's own kinds, no bind(C), no frame, a model over rank-2 and rank-3 arrays
    assert "subroutine hook_micro_mg_tend(microp_uniform, pcols, pver, ncol" in text and "bind(C, name='pycam_hooks_mp" not in text
    assert "    logical, intent(in) :: microp_uniform" in text and "    character(len=*), intent(out) :: errstring" in text
    assert "    real(c_double), pointer, intent(in) :: tnd_qsnow(:,:)" in text
    assert "type(torch_tensor) :: in_t(26), out_t(89)" in text
    # packed arrays: the callee's own pcols/pver dummies size every array, not the contract's constants
    assert "real(c_double), intent(in), target :: tn(pcols, pver)" in text
    assert "call c_f_pointer(c_loc(rndst), v_rndst, (/ pcols, pver, 4 /))" in text
    assert "real(c_double), target :: o_rflx(pcols, pver+1)" in text and "hk_n = min(int(ncol), pcols)" in text
    assert "l_tnd_qsnow = tnd_qsnow(1:pcols, 1:pver)" in text
    assert "w_qc(1:hk_n, :) = o_qc(1:hk_n, :)" in text and "w_prect(1:hk_n) = o_prect(1:hk_n)" in text
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
    # the warm-up at bind: one forward on zeros of the contract's extents, timed apart from the calls
    # the model is loaded where it will run: the host, or this rank's GPU (bind v2); v1 keeps the host
    assert "call torch_model_load(models(hook), filename(1:length), device_type, int(device_index))\n" in text
    assert "status = pycam_hooks_bind_model_v2(hook, path, length, shadow_flag, torch_kCPU, -1_c_int)" in text
    assert "bind(C, name='pycam_hooks_bind_model_v2')" in text and "integer(c_int), save :: model_device(nhooks) = torch_kCPU" in text
    assert text.count("\n  subroutine warm_") == len(table.hooks) and "      call warm_micro_mg_tend()" in text
    # a compiled plugin takes the same arrays as pointer and extent tables, in place of the forward
    assert "bind(C, name='pycam_hooks_bind_plugin_v1')" in text and "type(c_funptr), save :: plugins(nhooks)" in text
    assert "plugin_status = plugin(19_c_int, in_p, in_s, 8_c_int, out_p, out_s)" in text
    assert "plugin_status = plugin(1_c_int, in_p, in_s, 2_c_int, out_p, out_s)" in text
    assert "in_p(1) = c_loc(s_k); in_s(:, 1) = (/ 1_c_int64_t, 0_c_int64_t, 0_c_int64_t /)" in text
    assert "out_p(1) = c_loc(o_qc); out_s(:, 1) = (/ int(pcols, c_int64_t), int(pver, c_int64_t), 0_c_int64_t /)" in text
    # input tensors are made on the model's device; the output tensor stays on the host
    assert "if (.not. plugged(4)) call torch_tensor_from_array(in_t(1), sp_deltatin, model_device(4), model_device_index(4))" in text
    assert "if (.not. plugged(4)) call torch_tensor_from_array(out_t(1), op_qc, torch_kCPU)" in text
    assert "real(c_double), target :: z_tn(16, 30)" in text and "real(c_double), target :: y_rflx(16, 31)" in text
    assert "warm_ticks(hook) = w1 - w0" in text and "warm_seconds = real(warm_ticks(hook), c_double)" in text
    # in shadow the original is timed too, on the same calls: both prices from one run
    assert "original_ticks(4) = original_ticks(4) + (h1 - h0)" in text and text.count("original_ticks(") == 2 * len(table.hooks)
    # per-rank timers: the whole model call as the hook sees it, and the first call alone
    assert "hook_ticks(4) = hook_ticks(4) + (h1 - h0)" in text
    assert "if (answered(4) == 1_c_int64_t) first_ticks(4) = h1 - h0" in text


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
        self.plugged: list[tuple[int, int, int]] = []

        def bind(hook, path, length, shadow):
            if status:
                return status
            self.bound[hook] = (path[:length], int(shadow))
            return 0

        def bind_v2(hook, path, length, shadow, device, index):
            if status:
                return status
            self.bound[hook] = (path[:length], int(shadow), int(device), int(index))
            return 0

        def unbind(hook):
            self.unbound.append(hook)
            return 0

        def modeled(hook, out):
            out._obj.value = 4242
            return 0

        def bind_plugin(hook, address, shadow):
            if status:
                return status
            self.plugged.append((int(hook), int(getattr(address, "value", address)), int(shadow)))
            return 0

        self.pycam_hooks_bind_model_v1 = _Entry(bind)
        self.pycam_hooks_bind_model_v2 = _Entry(bind_v2)
        self.pycam_hooks_bind_plugin_v1 = _Entry(bind_plugin)
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
    import freecam.pi_cam.hooks as hooks_module

    table_loads: list[int] = []
    real_load_hooks = hooks_module.load_hooks
    monkeypatch.setattr(hooks_module, "load_hooks", lambda *a, **k: (table_loads.append(1), real_load_hooks(*a, **k))[1])
    stage.tend(None, context)
    stage.tend(None, context)
    assert ran == [stage.STAGE, stage.STAGE]                    # the whole original stage, twice
    assert list(library.bound) == [3]                          # bound at instratus's hook, once
    assert len(table_loads) == 1                               # and the hooks table read once, not every step
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
    other.kernels["macrop_advective_forcing"] = model
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
    # the timers are summed over the ranks, and the slowest rank's own value is kept beside the sum
    timed = [{"hook_counts": {"micro_mg_tend": {"calls": 100, "paused": 0, "modeled": 100, "model_seconds": 0.10,
                                                "forward_seconds": 0.08, "first_call_seconds": 0.04, "warm_seconds": 0.03}}},
             {"hook_counts": {"micro_mg_tend": {"calls": 100, "paused": 0, "modeled": 100, "model_seconds": 0.30,
                                                "forward_seconds": 0.20, "first_call_seconds": 0.20, "warm_seconds": 0.05}}}]
    summary = _hook_summary(timed)["micro_mg_tend"]
    assert summary["model_seconds"] == pytest.approx(0.40) and summary["model_seconds_max"] == pytest.approx(0.30)
    assert summary["first_call_seconds_max"] == pytest.approx(0.20) and summary["warm_seconds"] == pytest.approx(0.08)


def _fice_module():
    import importlib.util

    path = Path(__file__).resolve().parents[2] / "examples/plugins/numba_kernels/cldfrc_fice.py"
    spec = importlib.util.spec_from_file_location("fice_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_python_kernel_compiles_to_the_hooks_plugin_interface_and_answers_like_the_formula() -> None:
    pytest.importorskip("numba")
    import numpy as np

    from freecam.physics.native_model import NativePlugin
    from freecam.physics.numba_kernel import adapter_source, call_plugin_from_python, compile_kernel, _model_arguments

    hook, inputs, outputs = _model_arguments("cldfrc_fice")
    source = adapter_source("cldfrc_fice", inputs, outputs)
    assert "a0 = farray(in_ptrs[0], (in_shapes[0], in_shapes[1],), float64)" in source
    assert "o1 = farray(out_ptrs[1], (out_shapes[3], out_shapes[4],), float64)" in source and "kernel(a0, o0, o1)" in source
    module = _fice_module()
    plugin = compile_kernel("cldfrc_fice", module.cldfrc_fice)
    assert isinstance(plugin, NativePlugin) and plugin.address and plugin.describe()["binding"] == "numba"
    rng = np.random.default_rng(0)
    t = np.asfortranarray(rng.uniform(200.0, 300.0, size=(16, 30)))
    fice, fsnow = np.zeros((16, 30), order="F"), np.zeros((16, 30), order="F")
    assert call_plugin_from_python(plugin, [t], [fice, fsnow]) == 0
    ref_fice, ref_fsnow = module.cldfrc_fice_reference(t)
    assert np.array_equal(fice, ref_fice) and np.array_equal(fsnow, ref_fsnow)
    # a wrong table is refused, not read
    assert call_plugin_from_python(plugin, [t, t], [fice, fsnow]) == 1
    with pytest.raises(PhysicsError, match="not called from Python"):
        plugin(t)


def test_a_stage_binds_a_compiled_plugin_at_the_hook_and_runs_whole(tmp_path: Path) -> None:
    pytest.importorskip("numba")
    from freecam.physics.cloud_macro_microphysics import CloudMacroMicrophysics
    from freecam.physics.numba_kernel import compile_kernel

    plugin = compile_kernel("cldfrc_fice", _fice_module().cldfrc_fice)
    stage = CloudMacroMicrophysics()
    stage.kernels["cldfrc_fice"] = plugin
    assert stage.select_mode(None) == "native-model"
    library = _Library()
    ran: list[str] = []
    native = SimpleNamespace(library=library, run_action=lambda name, phase=None: ran.append(name),
                             segment_runner=lambda stage_name: None)
    context = SimpleNamespace(native=native, step=1)
    stage.tend(None, context)
    stage.tend(None, context)
    assert ran == [stage.STAGE, stage.STAGE]
    assert library.plugged == [(1, plugin.address, 0)]              # fice is hook 1; bound once, live
    stage.kernels["cldfrc_fice"] = compile_kernel("cldfrc_fice", _fice_module().cldfrc_fice, shadow=True)
    stage.tend(None, context)
    assert library.plugged[-1][2] == 1 and len(library.plugged) == 2
    # installing a stage cloudpickles it into the process registry: the plugin survives that in this
    # process, keeps its address, and its code stays callable after the original object is gone
    import gc

    import cloudpickle
    import numpy as np

    from freecam.physics.numba_kernel import call_plugin_from_python

    address = stage.kernels["cldfrc_fice"].address
    payload = cloudpickle.dumps(stage.kernels["cldfrc_fice"])
    copy = cloudpickle.loads(cloudpickle.dumps(stage))
    del plugin, stage
    gc.collect()
    revived = copy.kernels["cldfrc_fice"]
    assert revived.address == address and revived.describe()["binding"] == "numba"
    t = np.asfortranarray(np.full((16, 30), 250.0))
    fice, fsnow = np.zeros((16, 30), order="F"), np.zeros((16, 30), order="F")
    assert call_plugin_from_python(revived, [t], [fice, fsnow]) == 0 and fice[0, 0] == pytest.approx((263.15 - 250.0) / 30.0)
    # the payload must hash the same on every rank: a second compilation of the same function pickles
    # identically although its code sits at another address (7402200)
    again = compile_kernel("cldfrc_fice", _fice_module().cldfrc_fice, shadow=True)
    assert again.address != address and cloudpickle.dumps(again) == payload


def test_a_function_over_the_kernels_arrays_in_a_hooked_slot_is_compiled_and_fortran_calls_it() -> None:
    pytest.importorskip("numba")
    import numpy as np

    from freecam.physics.cloud_macro_microphysics import CloudMacroMicrophysics
    from freecam.physics.native_model import NativePlugin
    from freecam.physics.numba_kernel import call_plugin_from_python

    module = _fice_module()
    stage = CloudMacroMicrophysics()
    stage.kernels["cldfrc_fice"] = module.cldfrc_fice                        # def cldfrc_fice(t, fice, fsnow): the arrays
    assert stage.select_mode(None) == "native-model" and stage.binding_kind("cldfrc_fice") == "numba"
    plugin = stage._hook_bindings()["cldfrc_fice"]
    assert isinstance(plugin, NativePlugin) and plugin.describe()["binding"] == "numba" and not plugin.shadow
    assert stage._hook_bindings()["cldfrc_fice"] is plugin                     # compiled once, while the function stays
    rng = np.random.default_rng(0)
    t = np.asfortranarray(rng.uniform(200.0, 300.0, size=(16, 30)))
    fice, fsnow = np.zeros((16, 30), order="F"), np.zeros((16, 30), order="F")
    assert call_plugin_from_python(plugin, [t], [fice, fsnow]) == 0             # what Fortran calls: compiled code, no interpreter
    ref_fice, ref_fsnow = module.cldfrc_fice_reference(t)
    assert np.array_equal(fice, ref_fice) and np.array_equal(fsnow, ref_fsnow)
    library = _Library()
    ran: list[str] = []
    native = SimpleNamespace(library=library, run_action=lambda name, phase=None: ran.append(name),
                             segment_runner=lambda stage_name: None)
    context = SimpleNamespace(native=native, step=1)
    stage.tend(None, context)
    stage.tend(None, context)
    assert ran == [stage.STAGE, stage.STAGE] and library.plugged == [(1, plugin.address, 0)]   # fice is hook 1, bound once
    # the same function under the segmented policy answers at the pause; a function of one batch always does
    stage.execution_policy = "segmented"
    covering = SimpleNamespace(segment_runner=lambda stage_name: SimpleNamespace(kernels=("cldfrc_fice",)))
    assert stage.select_mode(covering) == "segmented"
    stage.execution_policy = "auto"
    stage.kernels["cldfrc_fice"] = lambda batch: {}
    assert stage.binding_kind("cldfrc_fice") == "callable" and stage.select_mode(covering) == "segmented"
    # a function of the arrays that Numba cannot compile is refused, not run from the interpreter
    def not_compilable(t, fice, fsnow):
        fice[0, 0] = float(id(object()))                # a Python object: nopython mode has none

    stage.kernels["cldfrc_fice"] = not_compilable
    with pytest.raises(PhysicsError, match="must compile with Numba"):
        stage.select_mode(None)


def test_a_compiled_function_fills_a_packed_output_in_the_hooks_layout() -> None:
    pytest.importorskip("numba")
    import numpy as np

    from freecam.physics.numba_kernel import _model_block, adapter_source, call_plugin_from_python, compile_kernel, packed_layout

    hook, spec, inputs, outputs = _model_block("compute_uwshcu_inv")
    layout = {entry["name"]: entry for entry in packed_layout(hook, spec, outputs)}
    assert layout["cush"]["offset"] == 0 and layout["umf_inv"]["offset"] == 1 and layout["umf_inv"]["width"] == 31
    assert layout["trten_inv"]["offset"] == 582 and layout["trten_inv"]["packed"] == [30, 12] and layout["trten_inv"]["subset"][0] == 10
    assert layout["wtprec"]["offset"] == 1182 and layout["wtsnow"]["offset"] == 1186 and sum(e["width"] for e in layout.values()) == 1190
    source = adapter_source("compute_uwshcu_inv", inputs, outputs, layout=packed_layout(hook, spec, outputs))
    assert "packed = farray(out_ptrs[0], (out_shapes[0], out_shapes[1]), float64)" in source
    assert "o26 = np.zeros((n, 30, 12,))" in source and "packed[i, 582 + j * 12 + k] = o26[i, j, k]" in source
    names = [f"in_{item.name}" for item in inputs] + [f"out_{item.name}" for item in outputs]   # cush is in and out
    body = "\n".join([
        f"def constants({', '.join(names)}):",
        "    n = in_t0_inv.shape[0]",
        "    for i in range(n):",
        "        out_cush[i] = 3.0",
        "        out_cnb_inv[i] = float(i)",
        "        for j in range(31):",
        "            out_umf_inv[i, j] = 4.0",
        "        for j in range(30):",
        "            out_trten_inv[i, j, 0] = 1.0",       # H2OV, constituent 10: the first slot of the subset
        "        out_trten_inv[i, 5, 11] = 2.0",          # H218OI, constituent 33: the last slot, one level
        "        for s in range(4):",
        "            out_wtprec[i, s] = 5.0",
    ])
    namespace: dict = {}
    exec(body, namespace)
    plugin = compile_kernel("compute_uwshcu_inv", namespace["constants"])
    shapes = {item.name: [16] + [int(spec.dimensions[a]) if str(a) in spec.dimensions else int(a) for a in item.native_shape[1:]]
              for item in inputs if item.rank}
    ins = [np.zeros(shapes[item.name], order="F") + 1.0 if item.rank else 1800.0 for item in inputs]
    packed = np.zeros((16, 1190), order="F")
    assert call_plugin_from_python(plugin, ins, [packed]) == 0
    assert np.all(packed[:, 0] == 3.0) and np.all(packed[:, 1:32] == 4.0) and np.all(packed[:, 32:581] == 0.0)
    assert np.array_equal(packed[:, 581], np.arange(16.0))                       # cnb_inv is the last bulk column
    tr = packed[:, 582:942].reshape(16, 30, 12)                                   # level-major, the subset's 12 slots
    assert np.all(tr[:, :, 0] == 1.0) and tr[0, 5, 11] == 2.0 and tr[0, 4, 11] == 0.0 and np.all(tr[:, :, 1:11] == 0.0)
    assert np.all(packed[:, 942:1182] == 0.0) and np.all(packed[:, 1182:1186] == 5.0) and np.all(packed[:, 1186:] == 0.0)


def test_a_native_model_on_a_gpu_is_bound_on_this_ranks_device(tmp_path: Path, monkeypatch) -> None:
    from freecam.physics.cloud_macro_microphysics import CloudMacroMicrophysics
    from freecam.physics.native_model import NativeModel, local_gpu_index
    from freecam.pi_cam.hooks import bind_hook_model

    # the node-local rank spread over the visible GPUs
    monkeypatch.setenv("PMI_LOCAL_RANK", "6"); monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert local_gpu_index() == 2                                              # 6 of 4 GPUs
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    assert local_gpu_index() == 0
    model = NativeModel(torchscript_archive(tmp_path / "m.pt"), device="cuda")
    assert model.device == "cuda" and model.resolved_device_index() == 0 and model.describe()["device_index"] == 0
    assert model.key.endswith(":cuda:0") and "device='cuda'" in repr(model)
    pinned = NativeModel(torchscript_archive(tmp_path / "m.pt"), device="cuda", device_index=3, shadow=True)
    assert pinned.resolved_device_index() == 3 and pinned.key.endswith(":cuda:3:shadow")
    with pytest.raises(PhysicsError):
        NativeModel(torchscript_archive(tmp_path / "m.pt"), device="tpu")
    # the stage binds it through the device-aware entry with the rank's index; the host model through v1
    stage = CloudMacroMicrophysics()
    stage.kernels["instratus_condensate"] = pinned
    library = _Library()
    native = SimpleNamespace(library=library, run_action=lambda name, phase=None: None, segment_runner=lambda s: None)
    stage.tend(None, SimpleNamespace(native=native, step=1))
    assert library.bound[3][1:] == (1, 1, 3)                                   # shadow, torch_kCUDA, device 3
    # an image without the entry refuses a GPU model instead of binding it on the host
    old = _Library(); del old.pycam_hooks_bind_model_v2
    with pytest.raises(Exception, match="device-aware bind entry"):
        bind_hook_model(old, 3, pinned.path, shadow=True, device="cuda", device_index=3)
