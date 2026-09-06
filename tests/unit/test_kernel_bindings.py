"""One registry for what computes a kernel, and one manifest for where the image can pause.

A kernel's replacement lives in ``stage.kernels[name]`` whichever way it was
installed -- assigned into the mapping, named as ``surrogate=``, or bound
over the method -- so the single-column caller and the model can never
disagree about what runs.  The segment runners are declared in one manifest
the backend, the stages and the builder all read.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from freecam.physics.errors import PhysicsError
from freecam.physics.macrophysics import RETURNED, Macrophysics
from freecam.physics.segments import OriginalKernel
from freecam.physics.stage import MethodKernel
from freecam.pi_cam import segment_runner as runners

REPO = Path(__file__).resolve().parents[2]


# -- the manifest ---------------------------------------------------------------------


def test_the_manifest_declares_the_stage_7_runner_and_the_module_exports_its_entries() -> None:
    specs = runners.load_manifest()
    assert [spec.stage for spec in specs][0] == "cam_run1.cloud_macro_microphysics"
    spec = specs[0]
    assert spec.kernel_names == ("mmacro_pcond", "micro_mg_tend")
    assert spec.kernel_id("mmacro_pcond") == 1 and spec.kernel_id("micro_mg_tend") == 2
    module = (REPO / spec.module).read_text()
    exported = set(re.findall(r"bind\(C,\s*name='([^']+)'\)", module))
    assert set(spec.entries) <= exported, sorted(set(spec.entries) - exported)
    assert (REPO / spec.generator).is_file() and (REPO / spec.descriptors).is_file()
    assert spec.kernel("mmacro_pcond").validated          # every gate record named is in the checkout
    assert spec.kernel("micro_mg_tend").validated         # gate 7331040: the pause, bit-for-bit
    assert spec.kernel("micro_mg_tend").contract == "native/pi_cam/functions/micro_mg_tend.yaml"
    assert runners.runner_kernels()["cam_run1.cloud_macro_microphysics"] == ("mmacro_pcond", "micro_mg_tend")
    assert runners.bindable_kernels()[:2] == ("mmacro_pcond", "micro_mg_tend")
    assert runners.KERNELS == ("mmacro_pcond", "micro_mg_tend") and runners.ENTRIES[0] == "pycam_stage7_create_v1"
    assert runners.runner_spec("cam_run1.radiation") is None


def test_the_manifest_refuses_a_kernel_two_runners_claim_and_a_runner_that_pauses_nowhere(tmp_path) -> None:
    text = (REPO / "native/pi_cam/segment_runners.yaml").read_text()
    twice = tmp_path / "twice.yaml"
    twice.write_text(text + text[text.index("- stage:"):].replace("cam_run1.cloud_macro_microphysics", "cam_run1.other"))
    with pytest.raises(Exception, match="claimed by two runners"):
        runners.load_manifest(twice)
    none = tmp_path / "none.yaml"
    none.write_text(text[:text.index("  kernels:")] + "  kernels: []\n")
    with pytest.raises(Exception, match="pauses at no kernel"):
        runners.load_manifest(none)


def test_the_micro_frame_is_the_contract_s_argument_list_without_the_character_argument() -> None:
    names = runners.frame_names_from_contract(REPO / "native/pi_cam/functions/micro_mg_tend.yaml")
    assert len(names) == 115 and "errstring" not in names
    assert names[:6] == ("microp_uniform", "pcols", "pver", "ncol", "top_lev", "deltatin")
    assert names[-1] == "wtpostlat"
    # the Fortran frame table is the same list, slot for slot
    import sys
    sys.path.insert(0, str(REPO / "tools"))
    import generate_pi_cam_micro_handles as micro

    table = micro.frame_table(micro.PINNED.read_text().splitlines())
    assert tuple(row["dummy"] for row in table) == names
    assert [row["actual"] for row in table][:6] == ["microp_uniform", "mgncol", "nlev", "mgncol", "1", "dtime/num_steps"]
    pointers = [row["dummy"] for row in table if row["kind"] == "pointer"]
    assert pointers == ["tnd_qsnow", "tnd_nsnow", "re_ice", "frzimm", "frzcnt", "frzdep"]
    assert {row["rank"] for row in table if row["kind"] == "array"} == {1, 2, 3}
    assert (REPO / "native/pi_cam/support/pycam_micro_handles.F90").read_text().count("micro_frame_slots = 115") == 1


def test_the_microphysics_answers_the_runner_s_frame_under_the_core_s_names() -> None:
    from freecam.physics.microphysics import (
        PACKED_INPUTS, PACKED_OUTPUTS, PACKED_TO_DUMMY, RETURNED, Microphysics,
    )

    seen: list = []

    def model(batch):
        seen.append(dict(batch))
        rows = batch["t"].shape[0]
        return {name: np.full((rows, 31) if name in ("rflx", "sflx") else (rows,) if name in ("prect", "preci")
                        else (rows, 30), 2.0) for name in PACKED_OUTPUTS}

    model.takes_packed_batch = True
    stage = Microphysics()
    answer = stage.frame_kernel("micro_mg_tend", model, native=None)
    batch = {PACKED_TO_DUMMY.get(name, name): np.ones((5, 30)) for name in PACKED_INPUTS}
    batch.update({"rndst": np.ones((5, 30, 4)), "nacon": np.ones((5, 30, 4)),
                  "ncol": np.int32(5), "pcols": np.int32(5), "pver": np.int32(30), "top_lev": np.int32(1),
                  "deltatin": np.float64(900.0), "microp_uniform": np.int32(0), "do_cldice": np.int32(1),
                  "reff_rain": np.zeros((5, 30)), "reff_snow": np.zeros((5, 30))})
    out = answer(batch)
    # the model saw packed names and the scalars, and answered under the routine's dummies
    assert "tn" not in seen[0] and "t" in seen[0] and seen[0]["deltatin"] == 900.0 and seen[0]["do_cldice"] == 1
    assert set(RETURNED) <= set(out) and out["effc"].shape == (5, 30) and out["rflx"].shape == (5, 31)
    for name in ("effc_fn", "reff_rain", "reff_snow", "drout2", "dsout2"):      # discarded by the driver
        assert name in out and out[name].shape == (5, 30)
    assert np.all(out["reff_rain"] == 0.0)                                       # handed back as it came


def test_an_image_without_the_entries_offers_no_runner() -> None:
    spec = runners.runner_spec("cam_run1.cloud_macro_microphysics")
    assert spec is not None
    assert not runners.image_offers_runner(SimpleNamespace(), spec)
    assert runners.runner_for(SimpleNamespace(), "cam_run1.cloud_macro_microphysics") is None
    assert runners.runner_for(SimpleNamespace(), "cam_run1.radiation") is None


# -- one registry ---------------------------------------------------------------------


def _answer(inputs, parameters=None):
    return {name: np.zeros(30) for name in RETURNED}


def test_binding_over_the_method_fills_the_kernel_slot_so_the_model_sees_the_replacement() -> None:
    stage = Macrophysics()
    assert stage.replacements() == ()
    stage.mmacro_pcond = MethodType(lambda self, inputs, parameters=None: _answer(inputs), stage)
    assert isinstance(stage.kernels["mmacro_pcond"], MethodKernel)
    assert stage.replacements() == ("mmacro_pcond",)              # what decides how the stage runs
    assert stage.binding_kind("mmacro_pcond") == "method"
    result = stage.mmacro_pcond({"t0": np.zeros(30)})               # the single-column caller: the same function
    assert set(result.outputs) | set(result.updated_inputs) == set(RETURNED)
    # a whole stage with this replacement runs segmented where the image pauses at it, never whole
    from freecam.physics.cloud_macro_microphysics import CloudMacroMicrophysics

    composed = CloudMacroMicrophysics()
    assert composed.macro is not None
    composed.macro.mmacro_pcond = MethodType(lambda self, inputs, parameters=None: _answer(inputs), composed.macro)
    assert composed.replacements() == ("mmacro_pcond",)
    covering = SimpleNamespace(segment_runner=lambda stage: SimpleNamespace(kernels=("mmacro_pcond",)))
    assert composed.select_mode(covering) == "segmented"
    assert composed.select_mode(SimpleNamespace(segment_runner=lambda stage: None)) == "legacy-python"
    composed.macro.mmacro_pcond = None                              # back to the original
    assert composed.replacements() == () and composed.select_mode(covering) == "native-whole"


def test_a_bound_function_result_is_flattened_and_a_one_argument_function_is_called_as_such() -> None:
    from freecam.physics.result import FunctionResult

    seen = []

    def one(inputs):
        seen.append(("one", sorted(inputs)))
        return FunctionResult(outputs={"a": 1.0}, updated_inputs={"b": 2.0})

    def two(inputs, parameters):
        seen.append(("two", parameters))
        return {"a": 3.0}

    assert MethodKernel(one)({"x": 0}, {"p": 1}) == {"a": 1.0, "b": 2.0}
    assert MethodKernel(two)({"x": 0}, {"p": 1}) == {"a": 3.0}
    assert seen == [("one", ["x"]), ("two", {"p": 1})]
    assert MethodKernel(one).takes_parameters


def test_the_slot_takes_only_a_callable_the_original_marker_or_none() -> None:
    stage = Macrophysics()
    with pytest.raises(PhysicsError, match="swappable kernel"):
        stage.mmacro_pcond = 3
    stage.mmacro_pcond = OriginalKernel()
    assert stage.binding_kind("mmacro_pcond") == "original-through-python"
    stage.surrogate_marker = 3                                     # any other attribute is an attribute
    assert stage.surrogate_marker == 3


def test_a_surrogate_named_by_path_and_a_callable_are_the_same_kind_of_binding_to_the_registry() -> None:
    stage = Macrophysics(surrogate="somewhere/model.pt")
    assert stage.binding_kind("mmacro_pcond") == "surrogate"
    plain = Macrophysics()
    plain.kernels["mmacro_pcond"] = _answer
    assert plain.binding_kind("mmacro_pcond") == "callable" and plain.replacements() == ("mmacro_pcond",)


# -- the read-only description ------------------------------------------------------


def test_describe_kernels_reports_contract_coverage_binding_and_calls() -> None:
    from freecam.physics.cloud_macro_microphysics import CloudMacroMicrophysics

    stage = CloudMacroMicrophysics()
    rows = {row["kernel"]: row for row in stage.describe_kernels()}
    assert list(rows) == ["mmacro_pcond", "micro_mg_tend"]
    pcond = rows["mmacro_pcond"]
    assert pcond["owner_class"].endswith("macrophysics.Macrophysics")
    assert pcond["stage_action"] == "cam_run1.cloud_macro_microphysics"
    assert pcond["bindable"] and pcond["validated"] and len(pcond["validated_by"]) == 4
    assert pcond["contract"]["path"] == "native/pi_cam/functions/mmacro_pcond.yaml"
    assert "cld" in pcond["contract"]["outputs"] and "t0" in pcond["contract"]["in_place"]
    assert pcond["contract"]["module_state_inputs"]["parameter"] >= 1
    assert pcond["binding"] == "original" and not pcond["replaced"] and pcond["model_calls"] == 0
    micro = rows["micro_mg_tend"]
    assert micro["owner_class"].endswith("microphysics.Microphysics")
    assert micro["bindable"] and micro["validated"] and len(micro["validated_by"]) == 4
    assert micro["contract"]["path"] == "native/pi_cam/functions/micro_mg_tend.yaml"
    stage.kernels["mmacro_pcond"] = _answer
    stage.execution.count_model_call("mmacro_pcond")
    again = {row["kernel"]: row for row in stage.describe_kernels()}
    assert again["mmacro_pcond"]["binding"] == "callable" and again["mmacro_pcond"]["model_calls"] == 1
    assert stage.execution.describe()["python_model_calls_by_kernel"] == {"mmacro_pcond": 1}


def test_the_builder_s_capabilities_come_from_the_manifest() -> None:
    from freecam.pi_cam.workflow_builder.capabilities import kernel_capabilities, validated_through_runner

    assert set(validated_through_runner()) == {"mmacro_pcond", "micro_mg_tend"}
    by_name = {c.kernel: c for c in kernel_capabilities()}
    assert by_name["mmacro_pcond"].bindable and by_name["mmacro_pcond"].validated
    assert by_name["mmacro_pcond"].evidence == runners.runner_spec("cam_run1.cloud_macro_microphysics").kernel("mmacro_pcond").validated_by
    assert by_name["micro_mg_tend"].bindable and by_name["micro_mg_tend"].validated
    assert not by_name["rad_rrtmg_sw"].bindable


# -- decoding the second kernel's frame ------------------------------------------


class _Entry:
    def __init__(self, lib, name):
        self.lib, self.name = lib, name
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        lib = self.lib
        lib.calls.append(self.name)
        if self.name.endswith("_create_v1"):
            args[0]._obj.value = 1; return 0
        if self.name.endswith("_start_v1"):
            lib.mask = list(args[2]); args[3]._obj.value = 1; return 0
        if self.name.endswith("_frame_v1"):
            kernel, index, lchnk, ncol, substep, token, count, ptrs, ndims, shapes, dtypes, intents = args[1:]
            kernel._obj.value = 2; index._obj.value = 5; lchnk._obj.value = 3; ncol._obj.value = 5
            substep._obj.value = 1; token._obj.value = 9
            assert count >= 115
            for i in range(115):
                ptrs[i] = lib.arrays[i].ctypes.data
                ndims[i] = lib.arrays[i].ndim
                for axis, extent in enumerate(lib.arrays[i].shape):
                    shapes[3 * i + axis] = extent
                dtypes[i] = 1 if lib.arrays[i].dtype == np.float64 else 2
                intents[i] = lib.intents[i]
            return 0
        if self.name.endswith("_resume_v1"):
            lib.resumed = (args[1], args[2]); args[3]._obj.value = 0; return 0
        return 0


class _TwoKernelLibrary:
    """A fake stage-7 image paused on the second kernel, micro_mg_tend."""

    def __init__(self) -> None:
        self.calls: list = []
        names = runners.frame_names_from_contract(REPO / "native/pi_cam/functions/micro_mg_tend.yaml")
        self.arrays = [np.zeros((), dtype=np.int32) if n in ("pcols", "pver", "ncol", "top_lev", "microp_uniform", "do_cldice")
                       else np.zeros((), dtype=np.float64) if n == "deltatin"
                       else np.zeros((5, 30, 4), order="F") if n in ("rndst", "nacon")
                       else np.zeros((5,), order="F") if n in ("prect", "preci")
                       else np.zeros((5, 31), order="F") if n in ("rflx", "sflx")
                       else np.zeros((5, 30), order="F") for n in names]
        for name, value in (("pcols", 5), ("ncol", 5), ("pver", 30), ("top_lev", 1), ("do_cldice", 1), ("deltatin", 900.0)):
            self.arrays[names.index(name)][...] = value           # the scalars the runner copies by value
        self.intents = [2 if n in ("qc", "qi", "nc", "ni", "reff_rain", "reff_snow") else 1 if names.index(n) >= 19 and n not in
                        ("naai", "npccnin", "rndst", "nacon", "do_cldice", "tnd_qsnow", "tnd_nsnow", "re_ice", "frzimm", "frzcnt", "frzdep")
                        else 0 for n in names]
        for suffix in runners.ENTRY_SUFFIXES:
            setattr(self, f"pycam_stage7_{suffix}_v1", _Entry(self, f"pycam_stage7_{suffix}_v1"))


def test_the_runner_decodes_a_pause_on_the_second_kernel_by_the_contract_s_names() -> None:
    from freecam.physics.segments import SegmentEvent

    lib = _TwoKernelLibrary()
    spec = runners.runner_spec("cam_run1.cloud_macro_microphysics")
    assert runners.image_offers_runner(lib, spec)
    runner = runners.ImageSegmentRunner(lib, spec)
    assert runner.slots == 115 and len(runner.names["mmacro_pcond"]) == 60
    context = runner.create("cam_run1.cloud_macro_microphysics")
    assert runner.start(context, {"mmacro_pcond": False, "micro_mg_tend": True}) == SegmentEvent.NEEDS_PYTHON_KERNEL
    assert lib.mask == [0, 1]                       # the mask follows the manifest's order
    frame = runner.frame(context)
    assert frame.kernel == "micro_mg_tend" and frame.ncol == 5 and frame.token == 9 and frame.call_index == 5
    assert [a.name for a in frame.arguments][:6] == ["microp_uniform", "pcols", "pver", "ncol", "top_lev", "deltatin"]
    assert frame.argument("rndst").array.shape == (5, 30, 4) and frame.argument("rflx").array.shape == (5, 31)
    assert frame.argument("qc").intent == "inout" and frame.argument("tlat").intent == "out"
    assert frame.argument("tn").array.ctypes.data == lib.arrays[6].ctypes.data       # a view of the Fortran storage
    batch = frame.batch()
    assert "tn" in batch and "tlat" not in batch and int(batch["ncol"]) == 5
    assert runner.resume(context, "micro_mg_tend", frame.token) == SegmentEvent.DONE
    assert lib.resumed == (2, 9)                    # the module's id for the second kernel, and the token
    with pytest.raises(Exception, match="cannot pause at"):
        runner.start(context, {"rad_rrtmg_sw": True})


def test_falling_back_to_the_walk_is_said_out_loud() -> None:
    """A replacement no runner pauses at runs the Python walk; auto never does that silently."""

    import warnings

    from freecam.physics.radiation import Radiation

    stage = Radiation()
    stage.kernels["rad_rrtmg_sw"] = _answer
    native = SimpleNamespace(segment_runner=lambda stage: None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert stage.select_mode(native) == "legacy-python"
    assert any("does not pause at ['rad_rrtmg_sw']" in str(w.message) for w in caught)
    # a stage already on the walk does not repeat itself every step
    stage.execution.mode = "legacy-python"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        stage.select_mode(native)
    assert not caught
