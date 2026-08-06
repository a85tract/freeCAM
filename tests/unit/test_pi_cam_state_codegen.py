from __future__ import annotations

from pathlib import Path

from pycam_sima.pi_cam.state_codegen import (
    generate_fortran_include,
    instrument_cam_comp,
    load_state_bridge,
)


def test_state_bridge_discovers_inline_pointer_and_allocatable_fields(tmp_path: Path) -> None:
    source = tmp_path / "state.F90"
    source.write_text(
        """
module state_mod
  type cam_in_t
    integer :: ncol
    real(8) :: temperature(pcols,pver)
    real(8), pointer, dimension(:) :: optional_flux
  end type cam_in_t
  type physics_state
    real(8), dimension(:,:), allocatable :: t, u
  end type physics_state
end module state_mod
"""
    )
    description = tmp_path / "bridge.yaml"
    description.write_text(
        """
schema_version: 1
owners:
  - name: cam_in
    type: cam_in_t
    source: state.F90
  - name: phys_state
    type: physics_state
    source: state.F90
exclude: []
"""
    )

    bridge = load_state_bridge(description, tmp_path)

    assert [field.name for field in bridge.fields] == [
        "cam_in.ncol",
        "cam_in.temperature",
        "cam_in.optional_flux",
        "phys_state.t",
        "phys_state.u",
    ]
    assert bridge.fields[1].source_dimensions == ("pcols", "pver")
    assert bridge.fields[2].pointer
    assert bridge.fields[3].allocatable
    generated = generate_fortran_include(bridge)
    assert "integer function cam_python_state_count()" in generated
    assert "count = 5" in generated
    assert "associated(cam_in(c)%optional_flux)" in generated
    assert "allocated(phys_state(c)%t)" in generated
    assert "reshape(phys_state(c)%t" in generated


def test_cam_comp_instrumentation_adds_public_generated_bridge() -> None:
    source = """
module cam_comp
  private
   public cam_final     ! CAM Finalization
contains
subroutine cam_final()
end subroutine
end module cam_comp
"""

    result = instrument_cam_comp(source, "generated_state.inc")

    assert "public cam_python_state_count" in result
    assert "include 'generated_state.inc'" in result
    assert result.index("include 'generated_state.inc'") < result.index("end module")
