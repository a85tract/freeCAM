import ctypes
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from freecam.core.fortran_adapter import PointerTableAdapter
from freecam.pi_cam.in_module_adapter import (
    can_generate_in_module_adapter,
    generate_in_module_source_tree,
)
from freecam.pi_cam.source_catalog import PICAMSourceCatalog

from test_pi_cam_source_catalog import _write_rules


def _compiler() -> str:
    compiler = Path("/opt/cray/pe/gcc/12.2.0/bin/gfortran")
    executable = str(compiler) if compiler.is_file() else shutil.which("gfortran")
    if executable is None:
        pytest.skip("gfortran is unavailable")
    return executable


def test_in_module_adapter_calls_private_derived_type_kernel(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    physics = source_root / "components/cam/src/physics"
    physics.mkdir(parents=True)
    source = physics / "private_state.F90"
    source.write_text(
        """module private_state
  use, intrinsic :: iso_c_binding, only: c_double
  implicit none
  private
  type :: state_t
    real(c_double) :: value
  end type
contains
  subroutine update(state, increment)
    type(state_t), intent(inout) :: state
    real(c_double), intent(in) :: increment
    state%value = state%value + increment
  end subroutine update
end module private_state
"""
    )
    catalog = PICAMSourceCatalog.discover(
        tmp_path,
        source_root=source_root,
        rules_path=_write_rules(tmp_path / "rules.yaml"),
        scan_roots=(physics,),
    )
    procedure = catalog.procedure("private_state::update")
    assert can_generate_in_module_adapter(procedure)

    output_root = tmp_path / "generated"
    adapters = generate_in_module_source_tree(
        (procedure,), source_root=source_root, output_root=output_root
    )
    patched = output_root / procedure.source
    library = tmp_path / "libprivate_state.so"
    subprocess.run(
        [
            _compiler(),
            "-shared",
            "-fPIC",
            "-ffree-line-length-none",
            str(patched),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    state = np.asarray(4.0, dtype=np.float64)
    increment = np.asarray(1.5, dtype=np.float64)
    runtime = PointerTableAdapter(
        ctypes.CDLL(str(library)),
        {
            "update": {
                "symbol": adapters[0].symbol,
                "action_id": 0,
                "arguments": (
                    {"field": "state", "dtype": "float64", "rank": 0},
                    {"field": "increment", "dtype": "float64", "rank": 0},
                ),
            }
        },
        library_name=str(library),
    )
    runtime.call(
        "update", {"state": state, "increment": increment}, fcomm=0
    )
    assert state.item() == 5.5


def test_source_appended_adapter_calls_external_derived_type_kernel(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    physics = source_root / "components/cam/src/physics"
    physics.mkdir(parents=True)
    (physics / "state_types.F90").write_text(
        """module state_types
  use, intrinsic :: iso_c_binding, only: c_double
  type :: state_t
    real(c_double) :: value
  end type
end module state_types
"""
    )
    kernel = physics / "external_update.F90"
    kernel.write_text(
        """subroutine external_update(state, increment)
  use state_types
  use, intrinsic :: iso_c_binding, only: c_double
  type(state_t), intent(inout) :: state
  real(c_double), intent(in) :: increment
  state%value = state%value + increment
end subroutine external_update
"""
    )
    catalog = PICAMSourceCatalog.discover(
        tmp_path,
        source_root=source_root,
        rules_path=_write_rules(tmp_path / "rules.yaml"),
        scan_roots=(physics,),
    )
    procedure = catalog.procedure("external_update::external_update")
    output_root = tmp_path / "generated"
    adapters = generate_in_module_source_tree(
        (procedure,), source_root=source_root, output_root=output_root
    )
    library = tmp_path / "libexternal_state.so"
    subprocess.run(
        [
            _compiler(),
            "-shared",
            "-fPIC",
            "-ffree-line-length-none",
            str(physics / "state_types.F90"),
            str(output_root / procedure.source),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    state = np.asarray(2.0, dtype=np.float64)
    increment = np.asarray(4.0, dtype=np.float64)
    runtime = PointerTableAdapter(
        ctypes.CDLL(str(library)),
        {
            "external_update": {
                "symbol": adapters[0].symbol,
                "action_id": 0,
                "arguments": (
                    {"field": "state", "dtype": "float64", "rank": 0},
                    {"field": "increment", "dtype": "float64", "rank": 0},
                ),
            }
        },
        library_name=str(library),
    )
    runtime.call(
        "external_update",
        {"state": state, "increment": increment},
        fcomm=0,
    )
    assert state.item() == 6.0


def test_in_module_adapter_exposes_function_result_as_output_pointer(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    physics = source_root / "components/cam/src/physics"
    physics.mkdir(parents=True)
    (physics / "private_function.F90").write_text(
        """module private_function
  use, intrinsic :: iso_c_binding, only: c_double
  implicit none
  private
contains
  real(c_double) function square(value)
    real(c_double), intent(in) :: value
    square = value * value
  end function square
end module private_function
"""
    )
    catalog = PICAMSourceCatalog.discover(
        tmp_path,
        source_root=source_root,
        rules_path=_write_rules(tmp_path / "rules.yaml"),
        scan_roots=(physics,),
    )
    procedure = catalog.procedure("private_function::square")
    assert procedure.result is not None
    assert procedure.result.dtype == "float64"
    output_root = tmp_path / "generated"
    adapters = generate_in_module_source_tree(
        (procedure,), source_root=source_root, output_root=output_root
    )
    library = tmp_path / "libprivate_function.so"
    subprocess.run(
        [
            _compiler(),
            "-shared",
            "-fPIC",
            "-ffree-line-length-none",
            str(output_root / procedure.source),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    value = np.asarray(3.0, dtype=np.float64)
    result = np.asarray(0.0, dtype=np.float64)
    runtime = PointerTableAdapter(
        ctypes.CDLL(str(library)),
        {
            "square": {
                "symbol": adapters[0].symbol,
                "action_id": 0,
                "arguments": (
                    {"field": "value", "dtype": "float64", "rank": 0},
                    {"field": "result", "dtype": "float64", "rank": 0},
                ),
            }
        },
        library_name=str(library),
    )
    runtime.call("square", {"value": value, "result": result}, fcomm=0)
    assert result.item() == 9.0


def test_in_module_adapter_preserves_fixed_character_length(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    physics = source_root / "components/cam/src/physics"
    physics.mkdir(parents=True)
    (physics / "labels.F90").write_text(
        """module labels
  implicit none
contains
  subroutine uppercase_label(label)
    character(len=8), intent(inout) :: label
    label = 'UPDATED '
  end subroutine uppercase_label
end module labels
"""
    )
    catalog = PICAMSourceCatalog.discover(
        tmp_path,
        source_root=source_root,
        rules_path=_write_rules(tmp_path / "rules.yaml"),
        scan_roots=(physics,),
    )
    procedure = catalog.procedure("labels::uppercase_label")
    assert procedure.arguments[0].character_length == "8"
    output_root = tmp_path / "generated"
    adapters = generate_in_module_source_tree(
        (procedure,), source_root=source_root, output_root=output_root
    )
    library = tmp_path / "liblabels.so"
    subprocess.run(
        [
            _compiler(),
            "-shared",
            "-fPIC",
            "-ffree-line-length-none",
            str(output_root / procedure.source),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    label = np.asarray(b"before", dtype="S8")
    runtime = PointerTableAdapter(
        ctypes.CDLL(str(library)),
        {
            "uppercase_label": {
                "symbol": adapters[0].symbol,
                "action_id": 0,
                "arguments": (
                    {"field": "label", "dtype": "S8", "rank": 0},
                ),
            }
        },
        library_name=str(library),
    )
    runtime.call("uppercase_label", {"label": label}, fcomm=0)
    assert label.item() == b"UPDATED "


def test_in_module_adapter_uses_itemsize_for_assumed_character_length(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    physics = source_root / "components/cam/src/physics"
    physics.mkdir(parents=True)
    (physics / "labels.F90").write_text(
        """module labels
  implicit none
contains
  subroutine update_label(label)
    character(len=*), intent(inout) :: label
    label = 'runtime'
  end subroutine update_label
end module labels
"""
    )
    catalog = PICAMSourceCatalog.discover(
        tmp_path,
        source_root=source_root,
        rules_path=_write_rules(tmp_path / "rules.yaml"),
        scan_roots=(physics,),
    )
    procedure = catalog.procedure("labels::update_label")
    output_root = tmp_path / "generated"
    adapters = generate_in_module_source_tree(
        (procedure,), source_root=source_root, output_root=output_root
    )
    library = tmp_path / "liblabels.so"
    subprocess.run(
        [
            _compiler(),
            "-shared",
            "-fPIC",
            "-ffree-line-length-none",
            str(output_root / procedure.source),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    label = np.asarray(b"before", dtype="S8")
    runtime = PointerTableAdapter(
        ctypes.CDLL(str(library)),
        {
            "update_label": {
                "symbol": adapters[0].symbol,
                "action_id": 0,
                "arguments": (
                    {
                        "field": "label",
                        "dtype": "S8",
                        "rank": 0,
                        "character": True,
                    },
                ),
            }
        },
        library_name=str(library),
    )
    runtime.call("update_label", {"label": label}, fcomm=0)
    assert label.item() == b"runtime "


def test_in_module_adapter_omits_optional_procedure_dummy(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    physics = source_root / "components/cam/src/physics"
    physics.mkdir(parents=True)
    (physics / "callbacks.F90").write_text(
        """module callbacks
  use, intrinsic :: iso_c_binding, only: c_double
  implicit none
contains
  subroutine update(value, callback)
    real(c_double), intent(inout) :: value
    optional :: callback
    interface
      subroutine callback(value)
        import :: c_double
        real(c_double), intent(inout) :: value
      end subroutine callback
    end interface
    if (present(callback)) call callback(value)
    value = value + 2.0_c_double
  end subroutine update
end module callbacks
"""
    )
    catalog = PICAMSourceCatalog.discover(
        tmp_path,
        source_root=source_root,
        rules_path=_write_rules(tmp_path / "rules.yaml"),
        scan_roots=(physics,),
    )
    procedure = catalog.procedure("callbacks::update")
    callback = procedure.arguments[1]
    assert callback.procedure and callback.optional
    output_root = tmp_path / "generated"
    adapters = generate_in_module_source_tree(
        (procedure,), source_root=source_root, output_root=output_root
    )
    library = tmp_path / "libcallbacks.so"
    subprocess.run(
        [
            _compiler(),
            "-shared",
            "-fPIC",
            "-ffree-line-length-none",
            str(output_root / procedure.source),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    value = np.asarray(3.0, dtype=np.float64)
    runtime = PointerTableAdapter(
        ctypes.CDLL(str(library)),
        {
            "update": {
                "symbol": adapters[0].symbol,
                "action_id": 0,
                "arguments": (
                    {"field": "value", "dtype": "float64", "rank": 0},
                ),
            }
        },
        library_name=str(library),
    )
    runtime.call("update", {"value": value}, fcomm=0)
    assert value.item() == 5.0


def test_external_adapter_uses_assumed_size_for_context_dependent_extent(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    physics = source_root / "components/cam/src/physics"
    physics.mkdir(parents=True)
    (physics / "state_types.F90").write_text(
        """module state_types
  use, intrinsic :: iso_c_binding, only: c_double
  type :: state_t
    integer :: n
  end type
end module state_types
"""
    )
    kernel = physics / "context_extent.F90"
    kernel.write_text(
        """subroutine context_extent(state, values)
  use state_types, only: state_t
  use, intrinsic :: iso_c_binding, only: c_double
  type(state_t), intent(in) :: state
  real(c_double), intent(inout) :: values(state%n)
  values = values + 2.0_c_double
end subroutine context_extent
"""
    )
    catalog = PICAMSourceCatalog.discover(
        tmp_path,
        source_root=source_root,
        rules_path=_write_rules(tmp_path / "rules.yaml"),
        scan_roots=(physics,),
    )
    procedure = catalog.procedure("context_extent::context_extent")
    output_root = tmp_path / "generated"
    generate_in_module_source_tree(
        (procedure,), source_root=source_root, output_root=output_root
    )
    generated = (output_root / procedure.source).read_text()
    assert "values(*)" in generated
    library = tmp_path / "libcontext_extent.so"
    subprocess.run(
        [
            _compiler(),
            "-shared",
            "-fPIC",
            "-ffree-line-length-none",
            str(physics / "state_types.F90"),
            str(output_root / procedure.source),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_in_module_adapter_resolves_procedure_local_kind_parameter(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    physics = source_root / "components/cam/src/physics"
    physics.mkdir(parents=True)
    source = physics / "local_kind.F90"
    source.write_text(
        """module local_kind
contains
  subroutine update(values)
    integer, parameter :: kr8 = selected_real_kind(15, 300)
    real(kind=kr8), intent(inout) :: values(:)
    values = values + 1.0_kr8
  end subroutine update
end module local_kind
"""
    )
    catalog = PICAMSourceCatalog.discover(
        tmp_path,
        source_root=source_root,
        rules_path=_write_rules(tmp_path / "rules.yaml"),
        scan_roots=(physics,),
    )
    procedure = catalog.procedure("local_kind::update")
    output_root = tmp_path / "generated"
    generate_in_module_source_tree(
        (procedure,), source_root=source_root, output_root=output_root
    )
    generated = (output_root / procedure.source).read_text()
    assert "real(c_double), pointer :: arg_values(:)" in generated
    subprocess.run(
        [
            _compiler(),
            "-shared",
            "-fPIC",
            "-ffree-line-length-none",
            str(output_root / procedure.source),
            "-o",
            str(tmp_path / "liblocal_kind.so"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
