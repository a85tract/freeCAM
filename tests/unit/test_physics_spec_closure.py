"""Function specs for the kernel-API closure: results, optionals, lower bounds, direct layout, mangled bindings."""
from pathlib import Path
import sys

import numpy as np
import pytest

from freecam.physics.column import empty_pool, pack, presence_field, unpack
from freecam.physics.spec import PhysicsSpecError, parse_function_spec

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import build_pi_cam_standalone_function as standalone  # noqa: E402

from freecam.pi_cam.kernel_codegen import generate_direct_kernel_module  # noqa: E402

IMAGE = {"archive_members": ["toy.o"], "stubs": {}, "base_address": 0x50000000}


def _document(**overrides):
    document = {
        "schema_version": 1,
        "function": "toyfun",
        "qualified_name": "toymod::toyfun",
        "routine": "toyfun",
        "source": "toy.F90",
        "module": "toymod",
        "layout": "direct",
        "binding": "mangled",
        "dimensions": {"nlev": 4, "nlevp": 5},
        "arguments": [
            {"name": "n", "role": "structural", "fortran_type": "integer", "dtype": "int32", "rank": 0,
             "intent": "in", "native_shape": [], "value": 4},
            {"name": "prof", "role": "input", "fortran_type": "real", "dtype": "float64", "rank": 1,
             "intent": "in", "native_shape": ["nlevp"], "lower_bounds": [0], "units": "Pa"},
            {"name": "scale", "role": "input", "fortran_type": "real", "dtype": "float64", "rank": 0,
             "intent": "in", "native_shape": [], "optional": True},
            {"name": "flux", "role": "output", "fortran_type": "real", "dtype": "float64", "rank": 1,
             "intent": "out", "native_shape": ["nlevp"], "lower_bounds": [0]},
            {"name": "toyfun", "role": "result", "fortran_type": "real", "dtype": "float64", "rank": 0,
             "intent": "out", "native_shape": []},
        ],
        "image": IMAGE,
    }
    document.update(overrides)
    return document


def test_a_direct_function_spec_carries_result_optional_and_lower_bounds() -> None:
    spec = parse_function_spec(_document())
    assert spec.kind == "function" and spec.result.name == "toyfun" and spec.layout == "direct"
    assert spec.binding == "mangled" and spec.bound_symbol == "toymod_mp_toyfun_"
    assert spec.argument("prof").lower_bounds == (0,) and spec.argument("prof").public_shape == ("nlevp",)
    assert spec.argument("scale").optional and spec.argument("n").lower_bounds == ()
    assert [item.name for item in spec.outputs] == ["flux", "toyfun"]
    assert "toyfun" in spec.describe()


def test_direct_layout_packs_declared_shapes_and_reports_presence() -> None:
    spec = parse_function_spec(_document())
    pool = empty_pool(spec)
    assert pool["toyfun.prof"].shape == (5, 1) and pool[presence_field(spec, spec.argument("scale"))].shape == (1,)
    packed = pack(spec, {"prof": np.arange(5.0)})
    assert packed["toyfun.n"][0] == 4 and packed[presence_field(spec, spec.argument("scale"))][0] == 0
    assert list(packed["toyfun.prof"][:, 0]) == [0.0, 1.0, 2.0, 3.0, 4.0]
    packed = pack(spec, {"prof": np.arange(5.0), "scale": 2.0})
    assert packed[presence_field(spec, spec.argument("scale"))][0] == 1 and packed["toyfun.scale"][0] == 2.0
    packed["toyfun.flux"][:, 0] = 7.0
    packed["toyfun.toyfun"][0] = 42.0
    outputs, updated = unpack(spec, packed)
    assert outputs["result"] == 42.0 and list(outputs["flux"]) == [7.0] * 5 and updated == {}


def test_the_wrapper_assigns_the_result_and_branches_on_the_optional() -> None:
    spec = parse_function_spec(_document())
    kernel = standalone.standalone_kernel(spec)
    assert kernel.kind == "function"
    assert [a.field for a in kernel.arguments] == [
        "toyfun.n", "toyfun.prof", "toyfun.scale", "toyfun.scale.present", "toyfun.flux", "toyfun.toyfun"]
    text = generate_direct_kernel_module((kernel,), module_name="toy")
    assert "bind(C, name='toymod_mp_toyfun_')" in text
    assert "real(c_double), intent(in), optional :: scale" in text
    assert "real(c_double), intent(in) :: prof(*)" in text
    assert "field_6(chunk) = toyfun( &" in text          # the result field takes the function value
    assert "if (field_4(chunk) /= 0_c_int32_t) then" in text and "scale=field_3(chunk)" in text
    assert "else if (field_4(chunk) == 0_c_int32_t) then" in text
    assert "use toymod" not in text                       # a private procedure is not imported


def test_spec_refusals_for_the_new_fields() -> None:
    with pytest.raises(PhysicsSpecError):
        parse_function_spec(_document(layout="sideways"))
    with pytest.raises(PhysicsSpecError):
        parse_function_spec(_document(binding="mangled", module=None))
    bad = _document()
    bad["arguments"][1]["lower_bounds"] = [0, 0]
    with pytest.raises(PhysicsSpecError):
        parse_function_spec(bad)
    bad = _document()
    bad["arguments"].insert(0, bad["arguments"].pop())          # the result not last
    with pytest.raises(PhysicsSpecError):
        parse_function_spec(bad)
    bad = _document()
    bad["arguments"][2]["default"] = 1.0                          # an optional with a default
    with pytest.raises(PhysicsSpecError):
        parse_function_spec(bad)
