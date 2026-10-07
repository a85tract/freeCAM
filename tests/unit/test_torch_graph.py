"""A bound model's batched forward replayed as a CUDA graph: the request, the binding, the record.

The graph itself is captured and replayed in native/pi_cam/support/pycam_torch_graph.cpp on a
GPU; what is tested here is everything around it that runs without one: the generated hook
module's path, the Python requests and refusals, and the run record.
"""

from __future__ import annotations

import ctypes
import sys
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
    # captured over the batch's own arrays (twenty inputs, the packed output), at the first batch
    assert f"graph_runners({index}) = pycam_torch_graph_open(models({index})%p" in forward
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


def test_the_command_line_asks_for_a_graph_only_on_a_gpu(tmp_path: Path) -> None:
    from freecam.pi_cam import cli

    archive = torchscript_archive(tmp_path / "m.pt")
    assert cli._load_kernel_model(archive, device="cuda", graph=True).graph is True
    with pytest.raises(PhysicsError, match="needs device='cuda'"):
        cli._load_kernel_model(archive, device="cpu", graph=True)
    summary = cli._kernel_models_summary({"compute_uwshcu_inv": archive}, device="cuda", graph=True)
    assert summary["compute_uwshcu_inv"]["graph"] is True and summary["compute_uwshcu_inv"]["device"] == "cuda"
