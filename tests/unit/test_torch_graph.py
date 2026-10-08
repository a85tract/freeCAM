"""A bound model's batched forward replayed as a CUDA graph: the request, the binding, the record.

The graph captures the model's own forward, or an AOTInductor package compiled from it
(``compiled=``): the package is checked here against its record and its model, the image's
entries for it against what the generated hook module does with them.

The graph itself is captured and replayed in native/pi_cam/support/pycam_torch_graph.cpp on a
GPU; what is tested here is everything around it that runs without one: the generated hook
module's path, the Python requests and refusals, and the run record.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_native_model import _Entry, _Library, torchscript_archive  # noqa: E402

from freecam.physics.native_model import NativeModel
from freecam.pi_cam.errors import PICAMConfigurationError
from freecam.pi_cam.hooks import load_hooks, read_hook_counts, read_hook_graph, set_hook_graph
from freecam.physics.errors import PhysicsError

REPO = Path(__file__).resolve().parents[2]


class _GraphLibrary(_Library):
    """An image with the graph entries: it remembers what was asked, and answers a state."""

    def __init__(self, status: int = 0, mode: int = 2) -> None:
        super().__init__()
        self.graphs: list[tuple[int, int]] = []

        def set_graph(hook, on):
            if status:
                return status
            self.graphs.append((int(hook), int(on)))
            return 0

        def state(hook, mode_out, status_out, replays, seconds):
            mode_out._obj.value, status_out._obj.value = mode, 5 if mode == 3 else 0
            replays._obj.value, seconds._obj.value = (48 if mode == 2 else 0), 1.25
            return 0

        def message(buffer, length):
            text = b"the replayed forward does not answer as the ordinary forward does"
            ctypes.memmove(buffer, text + b"\0", len(text) + 1)
            return len(text)

        self.pycam_hooks_set_graph_v1 = _Entry(set_graph)
        self.pycam_hooks_graph_state_v1 = _Entry(state)
        self.pycam_hooks_graph_message_v1 = _Entry(message)


class _CompiledLibrary(_GraphLibrary):
    """An image that also takes a compiled package for the graph, and says how far it answered from the model."""

    def __init__(self, gap: tuple[float, float] = (1.0e-5, 3.0e-5), **kwargs) -> None:
        super().__init__(**kwargs)
        self.packages: list[tuple[int, int, str]] = []
        self.package_set = False

        def set_graph_v2(hook, on, package):
            self.packages.append((int(hook), int(on), ctypes.string_at(package).decode()))
            self.package_set = True
            return 0

        def compiled(hook, flag, median, largest):
            flag._obj.value, median._obj.value, largest._obj.value = int(self.package_set), *gap
            return 0

        self.pycam_hooks_set_graph_v2 = _Entry(set_graph_v2)
        self.pycam_hooks_graph_compiled_v1 = _Entry(compiled)


def compiled_package(tmp_path: Path, model: Path, *, kernel: str = "compute_uwshcu_inv", device: str = "cuda:0",
                     model_sha256: str | None = None) -> Path:
    """An AOTInductor package's shape (a zip) and the record tools/compile_torch_model.py writes beside it."""

    package = tmp_path / "model.pt2"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("model/data/aotinductor/model/model.wrapper.so", b"\x7fELF")
    Path(f"{package}.json").write_text(json.dumps({
        "kernel": kernel, "model": model.name, "device": device,
        "model_sha256": model_sha256 or hashlib.sha256(model.read_bytes()).hexdigest(),
        "package_sha256": hashlib.sha256(package.read_bytes()).hexdigest()}))
    return package


def test_a_graph_replays_a_model_on_a_gpu_and_is_part_of_its_binding(tmp_path: Path) -> None:
    archive = torchscript_archive(tmp_path / "m.pt")
    with pytest.raises(PhysicsError, match="needs device='cuda'"):
        NativeModel(archive, graph=True)
    graphed = NativeModel(archive, device="cuda", device_index=3, graph=True)
    assert graphed.key.endswith(":cuda:3:graph") and graphed.describe()["graph"] is True
    assert "graph=True" in repr(graphed)
    plain = NativeModel(archive, device="cuda", device_index=3)
    assert plain.key != graphed.key and "graph" not in plain.describe()        # a change of mode rebinds


def test_the_stage_asks_for_the_graph_after_binding_the_model(tmp_path: Path, monkeypatch) -> None:
    from freecam.physics.cloud_macro_microphysics import CloudMacroMicrophysics
    import freecam.pi_cam.hooks as hooks_module

    monkeypatch.setattr(hooks_module, "_cuda_device_report", lambda: (4, "four devices"))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__="2.14.0+cu126", version=SimpleNamespace(cuda="12.6")))
    archive = torchscript_archive(tmp_path / "instratus.pt")
    stage = CloudMacroMicrophysics()
    stage.kernels["instratus_condensate"] = NativeModel(archive, device="cuda", device_index=1, graph=True)
    library = _GraphLibrary()
    native = SimpleNamespace(library=library, run_action=lambda name, phase=None: None, segment_runner=lambda s: None)
    stage.tend(None, SimpleNamespace(native=native, step=1))
    assert library.bound[3][2:] == (1, 1) and library.graphs == [(3, 1)]       # bound on cuda:1, then the graph
    stage.tend(None, SimpleNamespace(native=native, step=2))
    assert library.graphs == [(3, 1)]                                           # bound once, asked once
    # the same model without the graph: bound anew (which drops the image's graph), not asked again
    stage.kernels["instratus_condensate"] = NativeModel(archive, device="cuda", device_index=1)
    stage.tend(None, SimpleNamespace(native=native, step=3))
    assert library.graphs == [(3, 1)]


def test_a_graph_request_names_its_refusal(tmp_path: Path) -> None:
    old = _Library()                                                            # an image before CUDA graphs
    with pytest.raises(PICAMConfigurationError, match="predates CUDA graphs"):
        set_hook_graph(old, 3, True)
    set_hook_graph(old, 3, False)                                               # nothing to turn off
    with pytest.raises(PICAMConfigurationError, match="no TorchScript model is bound there on a CUDA device"):
        set_hook_graph(_GraphLibrary(status=9), 3, True)
    with pytest.raises(PICAMConfigurationError, match="does not answer in batches"):
        set_hook_graph(_GraphLibrary(status=10), 3, True)
    library = _GraphLibrary()
    set_hook_graph(library, 3, True)
    set_hook_graph(library, 3, False)
    assert library.graphs == [(3, 1), (3, 0)]


def test_the_record_says_whether_each_rank_replayed_or_why_it_was_refused() -> None:
    from freecam.pi_cam.cli import _hook_summary

    replaying = _GraphLibrary(mode=2)
    assert read_hook_graph(replaying, 3) == {"mode": "replaying", "replays": 48, "capture_seconds": 1.25}
    refused = read_hook_graph(_GraphLibrary(mode=3), 3)
    assert refused["mode"] == "refused" and refused["status"] == "the replay does not answer as the ordinary forward does"
    assert "does not answer as the ordinary forward" in refused["message"]
    assert read_hook_graph(_GraphLibrary(mode=0), 3) is None                   # off: nothing to say
    assert read_hook_graph(_Library(), 3) is None                               # an image without graphs
    assert read_hook_counts(replaying)["instratus_condensate"]["graph"]["mode"] == "replaying"
    # over the ranks: how many replayed and how many were refused, and the first refusal's reason
    records = [{"hook_counts": {"compute_uwshcu_inv": {"calls": 2, "paused": 0, "graph": graph}}}
               for graph in ({"mode": "replaying", "replays": 48, "capture_seconds": 1.25},
                             {"mode": "replaying", "replays": 48, "capture_seconds": 2.5}, refused)]
    graph = _hook_summary(records)["compute_uwshcu_inv"]["graph"]
    assert graph["ranks"] == {"replaying": 2, "refused": 1} and graph["replays"] == 96
    assert graph["capture_seconds_max"] == 2.5 and "ordinary forward" in graph["refused"]["status"]


def test_the_generated_batch_forward_captures_once_replays_after_and_falls_back() -> None:
    text = (REPO / "native/pi_cam/support/pycam_hooks.F90").read_text()
    table = load_hooks()
    index = table.hook("compute_uwshcu_inv").id
    forward = text[text.index("subroutine pycam_hooks_batch_forward_compute_uwshcu_inv"):
                   text.index("end subroutine pycam_hooks_batch_forward_compute_uwshcu_inv")]
    # captured over the batch's own arrays (twenty inputs, the packed output), at the first batch,
    # with the hook's compiled package when one was given
    assert f"graph_runners({index}) = pycam_torch_graph_open_v2(models({index})%p, graph_package({index})" in forward
    assert (f"if (g_status == 0_c_int) call pycam_torch_graph_compiled_gaps(graph_runners({index}), graph_gap(1, {index})"
            in forward)
    assert "g_in(2) = c_loc(bi_compute_uwshcu_inv_ps0_inv)" in forward and "c_loc(bo_compute_uwshcu_inv)" in forward
    assert f"graph_mode({index}) = merge(2_c_int, 3_c_int, g_status == 0_c_int)" in forward
    # replayed after; a refused graph leaves the ordinary forward answering
    assert f"g_status = pycam_torch_graph_run(graph_runners({index})" in forward
    ordinary = forward.index("call torch_model_forward(models(")
    assert forward.rindex("if (.not. done) then", 0, ordinary) > forward.index("pycam_torch_graph_run")
    # every binding of a model or plugin, and every unbinding, releases a graph first
    for entry in ("pycam_hooks_bind_model_v2", "pycam_hooks_unbind_model_v1", "pycam_hooks_bind_plugin_v1"):
        body = text[text.index(f"function {entry}"):text.index(f"end function {entry}")]
        assert "call reset_graph(hook)" in body, entry
    assert "public" in text and "pycam_hooks_set_graph_v1" in text.split("contains")[0]
    # the first entry is the second without a package; a reset forgets the package and its gap
    v1 = text[text.index("function pycam_hooks_set_graph_v1"):text.index("end function pycam_hooks_set_graph_v1")]
    assert "status = pycam_hooks_set_graph_v2(hook, on, c_null_ptr)" in v1
    reset = text[text.index("subroutine reset_graph"):text.index("end subroutine reset_graph")]
    assert "graph_package(hook) = c_null_ptr; graph_gap(:, hook) = -1.0_c_double" in reset
    assert "pycam_hooks_set_graph_v2" in text.split("contains")[0] and "pycam_hooks_graph_compiled_v1" in text.split("contains")[0]


def test_a_compiled_forward_is_the_package_its_record_names_made_from_this_model(tmp_path: Path) -> None:
    archive = torchscript_archive(tmp_path / "m.pt")
    package = compiled_package(tmp_path, archive)
    with pytest.raises(PhysicsError, match="needs graph=True"):
        NativeModel(archive, device="cuda", compiled=package)
    model = NativeModel(archive, device="cuda", device_index=0, graph=True, compiled=package)
    assert model.compiled_kernel == "compute_uwshcu_inv"
    assert model.key.endswith(f":graph:compiled:{model.compiled_sha256}")
    assert model.describe()["compiled"] == {"file": "model.pt2", "sha256": model.compiled_sha256}
    assert "compiled=" in repr(model)
    # another model's package, a package changed since its record, a CPU package, no record: refused
    with pytest.raises(PhysicsError, match="made from another model"):
        NativeModel(archive, device="cuda", graph=True, compiled=compiled_package(tmp_path, archive, model_sha256="0" * 64))
    package = compiled_package(tmp_path, archive)
    with zipfile.ZipFile(package, "a") as handle:
        handle.writestr("extra", b"changed")
    with pytest.raises(PhysicsError, match="not the package its record describes"):
        NativeModel(archive, device="cuda", graph=True, compiled=package)
    with pytest.raises(PhysicsError, match="compiled for 'cpu'"):
        NativeModel(archive, device="cuda", graph=True, compiled=compiled_package(tmp_path, archive, device="cpu"))
    Path(f"{compiled_package(tmp_path, archive)}.json").unlink()
    with pytest.raises(PhysicsError, match="record model.pt2.json"):
        NativeModel(archive, device="cuda", graph=True, compiled=tmp_path / "model.pt2")


def test_the_image_is_handed_the_package_and_says_how_far_it_answered(tmp_path: Path, monkeypatch) -> None:
    from freecam.physics.cloud_macro_microphysics import CloudMacroMicrophysics
    from freecam.physics.pausable import ShallowConvection
    import freecam.pi_cam.hooks as hooks_module

    monkeypatch.setattr(hooks_module, "_cuda_device_report", lambda: (4, "four devices"))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__="2.14.0+cu126", version=SimpleNamespace(cuda="12.6")))
    archive = torchscript_archive(tmp_path / "v4.pt")
    package = compiled_package(tmp_path, archive)
    stage = ShallowConvection()
    stage.kernels["compute_uwshcu_inv"] = NativeModel(archive, device="cuda", device_index=0, graph=True, compiled=package)
    library = _CompiledLibrary()
    native = SimpleNamespace(library=library, run_action=lambda name, phase=None: None, segment_runner=lambda s: None)
    stage.tend(None, SimpleNamespace(native=native, step=1))
    index = load_hooks().hook("compute_uwshcu_inv").id
    assert library.packages == [(index, 1, str(package))] and library.graphs == []       # through the second entry
    record = read_hook_graph(library, index)
    assert record["forward"] == "compiled" and record["compiled_gap"] == {"median": 1.0e-5, "max": 3.0e-5}
    # a package compiled for another kernel is refused at the binding
    other = CloudMacroMicrophysics()
    other.kernels["instratus_condensate"] = NativeModel(archive, device="cuda", device_index=0, graph=True,
                                                        compiled=package)
    with pytest.raises(PhysicsError, match="compiled for 'compute_uwshcu_inv', not 'instratus_condensate'"):
        other.tend(None, SimpleNamespace(native=SimpleNamespace(library=_CompiledLibrary(), run_action=lambda n, phase=None: None,
                                                                segment_runner=lambda s: None), step=1))
    # an image before compiled forwards refuses a package, and still takes the model's own graph
    with pytest.raises(PICAMConfigurationError, match="predates compiled forwards"):
        set_hook_graph(_GraphLibrary(), index, True, package=str(package))
    set_hook_graph(_GraphLibrary(), index, True)


def test_the_command_line_asks_for_a_graph_only_on_a_gpu(tmp_path: Path) -> None:
    from freecam.pi_cam import cli

    archive = torchscript_archive(tmp_path / "m.pt")
    assert cli._load_kernel_model(archive, device="cuda", graph=True).graph is True
    with pytest.raises(PhysicsError, match="needs device='cuda'"):
        cli._load_kernel_model(archive, device="cpu", graph=True)
    summary = cli._kernel_models_summary({"compute_uwshcu_inv": archive}, device="cuda", graph=True)
    assert summary["compute_uwshcu_inv"]["graph"] is True and summary["compute_uwshcu_inv"]["device"] == "cuda"
    package = compiled_package(tmp_path, archive)
    assert cli._load_kernel_model(archive, device="cuda", graph=True, compiled=package).compiled == package.resolve()
    summary = cli._kernel_models_summary({"compute_uwshcu_inv": archive}, device="cuda", graph=True,
                                         compiled={"compute_uwshcu_inv": package})
    assert summary["compute_uwshcu_inv"]["compiled"]["file"] == "model.pt2"


class _TF32Library(_CompiledLibrary):
    """An image with libtorch's TF32 switch: it keeps the order of what was asked, and the switch's state."""

    def __init__(self, cuda: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.asked: list[tuple[str, int, int]] = []
        self.on = False

        def set_tf32(on):
            if on and not cuda:
                return 1
            self.on = bool(on)
            self.asked.append(("tf32", int(on), 0))
            return 0

        self.pycam_torch_set_tf32 = _Entry(set_tf32)
        self.pycam_torch_tf32 = _Entry(lambda: int(self.on))
        graph_v2 = self.pycam_hooks_set_graph_v2._function

        def set_graph_v2(hook, on, package):
            self.asked.append(("graph", int(hook), int(on)))
            return graph_v2(hook, on, package)

        self.pycam_hooks_set_graph_v2 = _Entry(set_graph_v2)


def test_tf32_is_asked_of_a_gpu_model_and_set_before_its_graph_is_captured(tmp_path: Path, monkeypatch) -> None:
    from freecam.physics.pausable import ShallowConvection
    import freecam.pi_cam.hooks as hooks_module

    archive = torchscript_archive(tmp_path / "v4.pt")
    with pytest.raises(PhysicsError, match="needs device='cuda'"):
        NativeModel(archive, tf32=True)
    model = NativeModel(archive, device="cuda", device_index=0, graph=True, tf32=True,
                        compiled=compiled_package(tmp_path, archive))
    assert model.key.endswith(":tf32") and model.describe()["tf32"] is True and "tf32=True" in repr(model)
    assert NativeModel(archive, device="cuda", device_index=0).key != NativeModel(archive, device="cuda", device_index=0,
                                                                                  tf32=True).key
    monkeypatch.setattr(hooks_module, "_cuda_device_report", lambda: (4, "four devices"))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__="2.14.0+cu126", version=SimpleNamespace(cuda="12.6")))
    stage = ShallowConvection()
    stage.kernels["compute_uwshcu_inv"] = model
    library = _TF32Library()
    native = SimpleNamespace(library=library, run_action=lambda name, phase=None: None, segment_runner=lambda s: None)
    stage.tend(None, SimpleNamespace(native=native, step=1))
    index = load_hooks().hook("compute_uwshcu_inv").id
    assert library.asked == [("tf32", 1, 0), ("graph", index, 1)]               # the switch, then the capture
    counts = next(iter(read_hook_counts(library).values()))                    # the test image names one hook
    assert counts["tf32"] is True and counts["graph"]["forward"] == "compiled"


def test_tf32_is_one_switch_a_process_and_an_old_image_refuses_it(tmp_path: Path) -> None:
    from freecam.pi_cam.hooks import read_torch_tf32, set_hook_tf32, unbind_hook_model

    library = _TF32Library()
    set_hook_tf32(library, 3, True)
    set_hook_tf32(library, 5, True)
    with pytest.raises(PICAMConfigurationError, match="one switch for the process"):
        set_hook_tf32(library, 7, False)                    # another hook's model asks the other way
    unbind_hook_model(library, 3)
    unbind_hook_model(library, 5)
    set_hook_tf32(library, 7, False)                        # nothing bound asks for it any more
    assert read_torch_tf32(library) is False
    with pytest.raises(PICAMConfigurationError, match="predates TF32"):
        set_hook_tf32(_GraphLibrary(), 3, True)
    set_hook_tf32(_GraphLibrary(), 3, False)                # nothing to turn off in an old image
    assert read_torch_tf32(_GraphLibrary()) is None
    with pytest.raises(PICAMConfigurationError, match="has no CUDA"):
        set_hook_tf32(_TF32Library(cuda=False), 3, True)


def test_the_record_counts_the_ranks_on_tf32_and_the_packages_worst_gap() -> None:
    from freecam.pi_cam import cli

    records = [{"hook_counts": {"compute_uwshcu_inv": {"calls": 2, "paused": 0, "tf32": True, "graph": {
        "mode": "replaying", "replays": 48, "capture_seconds": 1.0, "forward": "compiled",
        "compiled_gap": {"median": median, "max": largest}}}}} for median, largest in ((1e-9, 0.2), (3e-9, 0.1))]
    summary = cli._hook_summary(records)["compute_uwshcu_inv"]
    assert summary["tf32_ranks"] == 2 and summary["graph"]["compiled_ranks"] == 2
    assert summary["graph"]["compiled_gap"] == {"median": 3e-9, "max": 0.2}


def test_the_command_line_asks_for_tf32_only_on_a_gpu(tmp_path: Path) -> None:
    from freecam.pi_cam import cli

    archive = torchscript_archive(tmp_path / "m.pt")
    assert cli._load_kernel_model(archive, device="cuda", tf32=True).tf32 is True
    with pytest.raises(PhysicsError, match="needs device='cuda'"):
        cli._load_kernel_model(archive, device="cpu", tf32=True)
    summary = cli._kernel_models_summary({"compute_uwshcu_inv": archive}, device="cuda", tf32=True)
    assert summary["compute_uwshcu_inv"]["tf32"] is True
